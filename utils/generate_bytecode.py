

import subprocess
import sys
import py_compile
import platform
import os
import re
import shutil
from pathlib import Path

from utils.version import PythonVersion

UV_VERSIONS = {PythonVersion((3, x)) for x in range(8, 16)}  # 3.8–3.15 have uv-managed builds


class CompileError(Exception):
    success = False
    bc_a = None


class PyenvError(Exception):
    pass


class WorkerUnavailable(Exception):
    """The persistent compile worker could not be used (failed to start, or died); the
    caller should fall back to the legacy per-compile subprocess path."""


# --- Persistent per-version compile worker pool -------------------------------------------
# A cross-version compile normally spawns a fresh `uvx --python <ver> python` subprocess per
# call. In the semantic-repair hot loop that is dozens-to-hundreds of spawns per file. When
# SEMANTIC_COMPILE_WORKER is enabled, compile_version routes those compiles through one
# long-lived interpreter per version (see utils.compile_worker) whose output is byte-identical
# to the subprocess path. Any worker failure falls back to the exact legacy uv/pyenv behaviour,
# so the flag is a pure optimisation and never changes what gets compiled.
_WORKER_POOL: dict = {}  # version tuple -> CompileWorker | None (None == negative cache)


def _worker_enabled() -> bool:
    return os.getenv("SEMANTIC_COMPILE_WORKER", "").strip().lower() in ("1", "true", "yes", "on")


def _worker_launch_cmd(version: "PythonVersion") -> list:
    return ["uvx", "--no-config", "--python", version.as_str(), "python"]


def _worker_env() -> dict:
    return {
        **os.environ,
        "PYTHONWARNINGS": "ignore",
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str((Path("/tmp") / "uv-cache").resolve())),
        "UV_TOOL_DIR": os.environ.get("UV_TOOL_DIR", str((Path("/tmp") / "uv-tools").resolve())),
    }


def _get_compile_worker(version):
    from utils.compile_worker import CompileWorker, WorkerStartError

    version = PythonVersion(version)
    key = version.as_tuple()
    if key in _WORKER_POOL:
        existing = _WORKER_POOL[key]
        if existing is None:
            raise WorkerUnavailable(f"compile worker for {version} previously failed to start")
        if existing.alive:
            return existing
        # died between files -> respawn below
    try:
        worker = CompileWorker(_worker_launch_cmd(version), version, env=_worker_env()).start()
    except (WorkerStartError, OSError) as exc:
        _WORKER_POOL[key] = None  # negative cache: don't retry for the rest of the process
        raise WorkerUnavailable(str(exc)) from exc
    _WORKER_POOL[key] = worker
    return worker


def _drop_worker(version) -> None:
    key = PythonVersion(version).as_tuple()
    worker = _WORKER_POOL.pop(key, None)
    if worker is not None and hasattr(worker, "close"):
        worker.close()


def _compile_worker(py_file: str, out_file: str, version: "PythonVersion") -> None:
    worker = _get_compile_worker(version)
    try:
        worker.compile(py_file, out_file)
    except CompileError:
        if not worker.alive:
            # the worker process died mid-request (not a genuine compile failure); drop it so
            # the next file respawns, and signal a fallback for this compile.
            _drop_worker(version)
            raise WorkerUnavailable(f"compile worker for {version} died mid-request")
        raise


def _compile_native(py_file: str, out_file: str):
    try:
        py_compile.compile(py_file, cfile=out_file, doraise=True, optimize=0)
    except py_compile.PyCompileError as e:
        raise CompileError(str(e))
    return


def _clean_compile_stderr(stderr: str) -> str:
    """Strip HARMLESS noise from a uv/pyenv compile subprocess's stderr so a genuinely
    successful compile is not mistaken for a failure. Removes uv download/bytecode
    progress AND broken ``*.pth`` site-startup errors emitted by uv-managed interpreters
    (notably a missing ``_virtualenv`` module in the 3.10 toolchain: the .pth's
    sitecustomize import fails at startup with 'Error processing line N of <...>.pth: ...
    Remainder of file ignored', but ``py_compile`` still writes the correct .pyc). Real
    compile errors (SyntaxError, or the version-guard AssertionError) are preserved so a
    true failure still raises. Without this, EVERY Python 3.10 candidate is rejected
    despite compiling correctly -> 0 deterministic fixes on 3.10."""
    s = re.sub(r"\s*Download(ing|ed)\s.+\n", "", stderr)
    s = re.sub(r"\s*Bytecode compiled \d+ files? in \d+ms\n", "", s)
    s = re.sub(r"Error processing line \d+ of [^\n]*\.pth:.*?Remainder of file ignored\n?",
               "", s, flags=re.DOTALL)
    return s.strip()


def _compile_uv(py_file: str, out_file: str, version: PythonVersion):
    # print(f"Compiling {py_file} to {out_file} using uv with Python {version.as_str()}")
    compile_cmd = f"import py_compile, sys; assert sys.version_info[:2] == {version.as_tuple()!r}; py_compile.compile({py_file!r}, cfile={out_file!r})"

    cmd = ["uvx", "--no-config", "--python", version.as_str(), "python", "-c", compile_cmd]

    env = {
        **os.environ,
        "PYTHONWARNINGS": "ignore",
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str((Path("/tmp") / "uv-cache").resolve())),
        "UV_TOOL_DIR": os.environ.get("UV_TOOL_DIR", str((Path("/tmp") / "uv-tools").resolve())),
    }
    output = subprocess.run(cmd, shell=False, capture_output=True, text=True, env=env)

    # Ignore uv download progress AND broken .pth site-startup noise (the compile still
    # succeeds); real errors (SyntaxError / version-guard AssertionError) survive.
    stderr = _clean_compile_stderr(output.stderr)
    if stderr:
        raise CompileError(stderr)


def _compile_pyenv(py_file: str, out_file: str, version: PythonVersion):

    which_pyenv = shutil.which("pyenv")
    if not which_pyenv:
        raise PyenvError(f"Could not find pyenv installation to compile in version {version.as_str()}. Try running with --init-pyenv to enable verification for end-of-life Python versions.")

    version_win = None
    if platform.system() == "Windows":  # workaround for pyenv-win being bugged when passing versions like 3.x not 3.x.y

        def try_get_version_patch(version_str) -> int | None:
            try:
                return int(version_str.split(".", 2)[2])
            except IndexError:
                return None
            except ValueError:
                return None

        pyenv_versions_cmd = [which_pyenv, *"versions --bare".split()]
        pyenv_versions_output = subprocess.run(pyenv_versions_cmd, shell=False, capture_output=True, text=True)
        if pyenv_versions_output.stderr:
            raise CompileError(pyenv_versions_output.stderr)
        # get lastest pyenv version for the correct major.minor version
        pyenv_version_strings = list(filter(lambda x: x.startswith(f"{version.as_str()}"), pyenv_versions_output.stdout.splitlines()))
        if not any(x == version.as_str() for x in pyenv_version_strings):
            pyenv_real_versions = list(filter(lambda x: x is not None, map(try_get_version_patch, pyenv_version_strings)))
            if len(pyenv_real_versions) == 0:
                raise PyenvError(f"Could not find pyenv version for {version.as_str()}")
            version_win = f"{version.as_str()}.{max(pyenv_real_versions)}"

    compile_cmd = f"import py_compile, sys; assert sys.version_info[:2] == {version.as_tuple()!r}; py_compile.compile({py_file!r}, cfile={out_file!r})"

    cmd = [which_pyenv, *"exec python -c".split(), compile_cmd]

    output = subprocess.run(cmd, shell=False, capture_output=True, text=True, env={**os.environ, "PYENV_VERSION": version_win if version_win else version.as_str(), "PYTHONWARNINGS": "ignore"})

    stderr = _clean_compile_stderr(output.stderr)
    if stderr:
        raise CompileError(stderr)


def compile_version(py_file, out_file, version):
    py_file = str(py_file)
    out_file = str(out_file)
    version = PythonVersion(version)

    if version == sys.version_info:
        _compile_native(py_file=py_file, out_file=out_file)
    elif version in UV_VERSIONS:
        if _worker_enabled():
            try:
                _compile_worker(py_file, out_file, version)
                return
            except WorkerUnavailable:
                pass  # worker can't be used -> legacy path (and negative-cached for the run)
            except CompileError:
                pass  # worker reported a failure -> re-verify through the legacy uv/pyenv path
        try:
            _compile_uv(py_file=py_file, out_file=out_file, version=version)
        except CompileError:
            _compile_pyenv(py_file=py_file, out_file=out_file, version=version)
    else:
        _compile_pyenv(py_file=py_file, out_file=out_file, version=version)
