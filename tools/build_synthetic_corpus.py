"""Manufacture synthetic GT-bytecode-reconstruction training pairs from arbitrary Python.

The oracle GT-reconstruction corpus (``tools/build_gt_reconstruction_corpus.py``) is bounded
by the ~386 PyPi/3.10 rows that carry a genuine hand-written GT source. This generator lifts
both limits -- volume AND python-version diversity -- by *manufacturing* cases from any correct
``.py`` module:

    for each correct module, for each target version in {3.10, 3.11, 3.12, 3.13}:
      1. compile   correct.py  -> GT.pyc            (utils.generate_bytecode.compile_version)
      2. decompile GT.pyc      -> derived.py        (INJECTABLE decompile_fn; default = PyLingual)
      3. recompile derived.py  -> derived.pyc
      4. compare_pyc(GT.pyc, derived.pyc): keep ONLY if it diverges in the target regime
         (a failing object that is control_flow / missing -- the brief's regime), then
      5. REUSE the corpus builder's OracleCaptureFragmentFixer + real repair loop to emit
         brief-ON prompt -> correct-source rows, with ``correct.py`` as the GT-source label.

The emitted rows are byte-for-byte in the finetuner schema (``prompt.messages`` +
``replacement_text``) so they APPEND to / match ``dataset/gt_reconstruction_corpus.jsonl``.

The DECOMPILE step is injectable so tests run CPU-only with a fake decompiler (no GPU). The
default decompile_fn is ``pylingual.decompiler.decompile`` -- a GPU/model job the parent runs
separately, never here.

The oracle is used ONLY as a label source and is never weakened. No LLM, no network, no hosted
API. Compiling non-host bytecode needs ``uvx`` on PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

# Allow `python tools/build_synthetic_corpus.py` (repo root + pylingual on sys.path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "pylingual") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "pylingual"))

# Importing the corpus builder pulls in the shared machinery we reuse verbatim: the
# OracleCaptureFragmentFixer, the holdout loader, and (transitively) the pylingual sys.path
# injection performed by pipeline.code_object_repair_loop.
from tools.build_gt_reconstruction_corpus import (  # noqa: E402
    OracleCaptureFragmentFixer,
    _load_holdout_hashes,
)
from pipeline.code_object_repair_loop import CodeObjectRepairLoop  # noqa: E402
from utils.generate_bytecode import CompileError, compile_version  # noqa: E402
from utils.rebuild_semantic_prompt_refresh_dataset import _write_jsonl  # noqa: E402
from pylingual.equivalence_check import compare_pyc  # noqa: E402

DecompileFn = Callable[..., Any]

_DEFAULT_SOURCE_DIR = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
_DEFAULT_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
_DEFAULT_OUT = _REPO_ROOT / "dataset" / "synthetic_corpus.jsonl"
_DEFAULT_HOLDOUT = [_REPO_ROOT / "dataset" / "pyllmpatch_semantic_hybrid100.csv"]


# --------------------------------------------------------------------------------------
# Default decompiler (GPU / model-based). Imported lazily so tests never load PyLingual.
# --------------------------------------------------------------------------------------
def _default_decompile_fn(pyc: Path, *, save_to: Path, version: str) -> Any:
    from pylingual.decompiler import decompile  # heavy import; deferred to call time

    return decompile(pyc, save_to=save_to, version=version)


# --------------------------------------------------------------------------------------
# Module enumeration + hashing.
# --------------------------------------------------------------------------------------
def _module_file_hash(path: Path) -> str:
    """Stable content hash used for holdout exclusion and row provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _iter_source_files(source_dir: Path) -> Iterator[Path]:
    """Every ``*.py`` under source_dir, in a deterministic order."""
    for path in sorted(Path(source_dir).rglob("*.py")):
        if path.is_file():
            yield path


def _compiles_clean(path: Path) -> bool:
    try:
        source = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    try:
        compile(source, str(path), "exec")
    except (SyntaxError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------------------
# Divergence gate -- mirrors the corpus builder's capture regime (control_flow / missing).
#
# _should_capture_target keeps a target iff the GT reconstruction brief renders: a missing
# object (unconditional), a "Different control flow" failure, or a "Different bytecode"
# failure with no single failed offset. This TestResult-level gate is the compare_pyc
# projection of that same regime: it skips cases that could never produce a captured row,
# so we don't spin up the (expensive) repair loop for them.
# --------------------------------------------------------------------------------------
def _has_target_regime_divergence(results: list) -> bool:
    for test in results:
        if getattr(test, "success", True):
            continue
        message = getattr(test, "message", None)
        if message == "Missing bytecode":
            return True
        if message == "Different control flow":
            return True
        if message == "Different bytecode" and getattr(test, "failed_offset", None) is None:
            return True
    return False


# --------------------------------------------------------------------------------------
# Decompile-step adapter: turn whatever the injectable decompile_fn returns into source text.
# --------------------------------------------------------------------------------------
def _derive_source_text(
    decompile_fn: DecompileFn, gt_pyc: Path, save_to: Path, version: str
) -> str:
    """Call decompile_fn and resolve the derived source text.

    Accepts the pylingual.decompiler.decompile contract (returns a DecompilerResult with
    ``.decompiled_source`` and writes ``save_to``), a plain string, or a fn that only writes
    ``save_to``. The generator always re-writes the resolved text to ``save_to`` so the
    recompile step reads exactly what was scored.
    """
    result = decompile_fn(gt_pyc, save_to=save_to, version=version)
    text: str | None = None
    if hasattr(result, "decompiled_source"):
        text = result.decompiled_source
    elif isinstance(result, str):
        text = result
    elif Path(save_to).exists():
        text = Path(save_to).read_text(encoding="utf-8")
    if text is None:
        raise ValueError("decompile_fn produced no source text")
    Path(save_to).write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------------------
# Per-(module, version) case manufacture.
# --------------------------------------------------------------------------------------
def _manufacture_case(
    *,
    correct_py: Path,
    file_hash: str,
    version: str,
    decompile_fn: DecompileFn,
    sink: list[dict],
    seen: set,
    stats: dict[str, int],
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        gt_pyc = tmp_dir / "gt.pyc"
        derived_py = tmp_dir / "derived.py"
        derived_pyc = tmp_dir / "derived.pyc"

        try:
            compile_version(correct_py, gt_pyc, version)
        except Exception as exc:  # noqa: BLE001
            stats["cases_gt_compile_failed"] += 1
            _log_case(file_hash, version, f"gt-compile failed: {type(exc).__name__}: {exc}")
            return

        try:
            _derive_source_text(decompile_fn, gt_pyc, derived_py, version)
        except Exception as exc:  # noqa: BLE001
            stats["cases_decompile_failed"] += 1
            _log_case(file_hash, version, f"decompile failed: {type(exc).__name__}: {exc}")
            return

        try:
            compile_version(derived_py, derived_pyc, version)
        except Exception as exc:  # noqa: BLE001
            stats["cases_derived_compile_failed"] += 1
            _log_case(file_hash, version, f"derived-compile failed: {type(exc).__name__}: {exc}")
            return

        try:
            results = compare_pyc(gt_pyc, derived_pyc)
        except Exception as exc:  # noqa: BLE001
            stats["cases_compare_failed"] += 1
            _log_case(file_hash, version, f"compare_pyc failed: {type(exc).__name__}: {exc}")
            return

        if not _has_target_regime_divergence(results):
            stats["cases_no_target_divergence"] += 1
            return

        stats["cases_diverged"] += 1
        before = len(sink)
        fixer = OracleCaptureFragmentFixer(
            gt_pyc=gt_pyc,
            gt_source=correct_py,
            source_file_hash=file_hash,
            sink=sink,
            seen=seen,
        )
        loop = CodeObjectRepairLoop(fixer)
        with tempfile.TemporaryDirectory() as run_tmp:
            try:
                loop.run(
                    gt_pyc=gt_pyc,
                    derived_pyc=derived_pyc,
                    derived_source=derived_py,
                    output_dir=Path(run_tmp),
                    file_hash=file_hash,
                    verify_with_pylingual=True,
                    verify_each_step_with_pylingual=True,
                    reject_non_improving_candidates=True,
                    max_iterations=1,
                )
            except Exception as exc:  # noqa: BLE001 -- one bad case must not sink the build
                stats["cases_repair_errored"] += 1
                _log_case(file_hash, version, f"repair loop errored: {type(exc).__name__}: {exc}")
                return
        emitted = len(sink) - before
        if emitted > 0:
            stats["cases_emitting_rows"] += 1
        # tag emitted rows with the target version for downstream diversity accounting
        for row in sink[before:]:
            row.setdefault("bytecode_version", version)


def _log_case(file_hash: str, version: str, message: str) -> None:
    print(
        f"[synthetic_corpus] {file_hash[:16]} @ {version}: {message}",
        flush=True,
    )


# --------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------
def build_synthetic_corpus(
    *,
    source_dir: Path,
    versions: list[str],
    limit: int | None,
    holdout_paths: list[Path],
    decompile_fn: DecompileFn = _default_decompile_fn,
) -> tuple[list[dict], dict[str, int]]:
    """Manufacture the synthetic corpus. Returns (rows, stats).

    ``limit`` caps the number of *modules processed* (compilable, non-holdout). Dedup is
    scoped PER VERSION so an identical source span reconstructed under different bytecode
    versions yields distinct rows (their briefs differ) -- the version diversity this tool
    exists to add. ``decompile_fn`` is injected so tests run CPU-only with a fake.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    holdout = _load_holdout_hashes(holdout_paths)

    sink: list[dict] = []
    seen_by_version: dict[str, set] = {version: set() for version in versions}
    stats: dict[str, int] = {
        "modules_seen": 0,
        "modules_skipped_uncompilable": 0,
        "modules_excluded_by_holdout": 0,
        "modules_processed": 0,
        "cases_gt_compile_failed": 0,
        "cases_decompile_failed": 0,
        "cases_derived_compile_failed": 0,
        "cases_compare_failed": 0,
        "cases_no_target_divergence": 0,
        "cases_diverged": 0,
        "cases_repair_errored": 0,
        "cases_emitting_rows": 0,
    }

    for py_file in _iter_source_files(source_dir):
        if limit is not None and stats["modules_processed"] >= limit:
            break
        stats["modules_seen"] += 1
        if not _compiles_clean(py_file):
            stats["modules_skipped_uncompilable"] += 1
            continue
        try:
            file_hash = _module_file_hash(py_file)
        except OSError:
            stats["modules_skipped_uncompilable"] += 1
            continue
        if file_hash in holdout:
            stats["modules_excluded_by_holdout"] += 1
            continue

        stats["modules_processed"] += 1
        for version in versions:
            _manufacture_case(
                correct_py=py_file,
                file_hash=file_hash,
                version=version,
                decompile_fn=decompile_fn,
                sink=sink,
                seen=seen_by_version[version],
                stats=stats,
            )

    return sink, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manufacture synthetic GT-bytecode-reconstruction training rows from arbitrary "
            "correct Python (enlarges + version-diversifies the fine-tuning corpus)."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=_DEFAULT_SOURCE_DIR,
        help="Directory to enumerate *.py from (default: this env's site-packages).",
    )
    parser.add_argument(
        "--versions",
        type=str,
        default=",".join(_DEFAULT_VERSIONS),
        help="Comma-separated target python versions (default: 3.10,3.11,3.12,3.13).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max modules to process.")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        nargs="+",
        default=_DEFAULT_HOLDOUT,
        help="One or more holdout CSVs whose file_hashes are excluded (union).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    versions = [v.strip() for v in str(args.versions).split(",") if v.strip()]

    sink, stats = build_synthetic_corpus(
        source_dir=args.source_dir,
        versions=versions,
        limit=args.limit,
        holdout_paths=list(args.holdout_csv),
        decompile_fn=_default_decompile_fn,
    )

    print("[synthetic_corpus] funnel:", flush=True)
    for key, value in stats.items():
        print(f"    {key}: {value}", flush=True)

    by_version: dict[str, int] = {}
    for row in sink:
        version = str(row.get("bytecode_version"))
        by_version[version] = by_version.get(version, 0) + 1
    print("[synthetic_corpus] rows by version:", flush=True)
    for version, count in sorted(by_version.items()):
        print(f"    {version}: {count}", flush=True)

    out_path = Path(args.out).expanduser().resolve()
    _write_jsonl(out_path, sink)
    print(f"[synthetic_corpus] wrote {len(sink)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
