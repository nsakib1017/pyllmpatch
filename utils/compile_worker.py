"""Persistent per-version compile worker.

The semantic-repair hot loop compiles a candidate ``.py`` to ``.pyc`` once per candidate
per iteration. For any target version that is not the host interpreter, the legacy path
(:func:`utils.generate_bytecode._compile_uv`) spawns a fresh ``uvx --python <ver> python``
subprocess *every* call — paying interpreter startup + uv resolution dozens-to-hundreds of
times per file. This worker spawns that interpreter ONCE and streams compile requests to it
over stdin/stdout, so the startup cost is amortised across the whole file.

Output is byte-identical to :func:`utils.generate_bytecode._compile_native` because the
worker runs the same ``py_compile.compile(..., doraise=True, optimize=0)`` call in the same
target interpreter — there is no change to what gets compiled or how, only to process reuse.
The launch command is injected so tests can drive the worker with ``sys.executable`` while
production injects the ``uvx --python <ver> python`` launcher.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from utils.generate_bytecode import CompileError
from utils.version import PythonVersion

# Runs inside the target interpreter. Emits one JSON handshake line, then loops: read one
# JSON request per line, compile, write one JSON response per line. stderr is discarded by
# the parent, so harmless site-startup noise (e.g. the 3.10 _virtualenv.pth error) can never
# be mistaken for a compile failure — real failures come back as {"ok": false, "error": ...}.
_WORKER_SRC = r"""
import sys, json, py_compile
def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()
_emit({"ready": True, "version": list(sys.version_info[:2])})
for _line in sys.stdin:
    _line = _line.strip()
    if not _line:
        continue
    try:
        _req = json.loads(_line)
    except Exception:
        continue
    if _req.get("cmd") == "shutdown":
        break
    try:
        py_compile.compile(_req["py_file"], cfile=_req["out_file"], doraise=True, optimize=0)
        _emit({"ok": True})
    except py_compile.PyCompileError as _e:
        _emit({"ok": False, "error": str(_e)})
    except Exception as _e:
        _emit({"ok": False, "error": repr(_e)})
"""


class WorkerStartError(RuntimeError):
    """The worker interpreter failed to start or reported the wrong version."""


class CompileWorker:
    def __init__(self, launch_cmd: Sequence[str], version, env: dict | None = None):
        self.launch_cmd = list(launch_cmd)
        self.version = PythonVersion(version)
        self._env = env
        self._proc: subprocess.Popen | None = None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> "CompileWorker":
        cmd = [*self.launch_cmd, "-c", _WORKER_SRC]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=self._env,
        )
        handshake = self._proc.stdout.readline()
        if not handshake:
            self.close()
            raise WorkerStartError(f"compile worker for {self.version} produced no handshake")
        try:
            info = json.loads(handshake)
        except json.JSONDecodeError as exc:
            self.close()
            raise WorkerStartError(f"compile worker handshake was not JSON: {handshake!r}") from exc
        got = tuple(info.get("version") or ())
        if got != self.version.as_tuple():
            self.close()
            raise WorkerStartError(
                f"compile worker reported Python {got}, expected {self.version.as_tuple()}"
            )
        return self

    def compile(self, py_file, out_file) -> None:
        if not self.alive:
            raise CompileError("compile worker is not running")
        req = json.dumps({"py_file": str(py_file), "out_file": str(out_file)})
        try:
            self._proc.stdin.write(req + "\n")
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline()
        except (BrokenPipeError, ValueError) as exc:
            raise CompileError(f"compile worker died: {exc}") from exc
        if not resp_line:
            raise CompileError("compile worker died mid-request")
        resp = json.loads(resp_line)
        if not resp.get("ok"):
            raise CompileError(resp.get("error", "unknown compile error"))

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, ValueError, OSError):
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
