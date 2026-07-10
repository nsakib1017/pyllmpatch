"""Build an oracle-labelled GT-bytecode-reconstruction fine-tuning corpus.

This tool teaches the local Qwen to reconstruct control_flow / missing code objects
from the GT-bytecode reconstruction brief. For every control_flow / missing target the
real repair driver surfaces, it emits one training row whose

    INPUT  = the exact brief-ON prompt the model sees at inference
             (build_semantic_repair_messages with SEMANTIC_RECONSTRUCTION_BRIEF=true and
              telemetry/action guidance forced to '' -> the NON-retrieval strategy prompt,
              byte-for-byte with inference), and
    TARGET = the verbatim oracle GT source span (OracleFragmentFixer).

Restricted to PyPi rows -- the only rows with a genuine, hand-written GT source distinct
from the decompiled derived source. PyLingual.IO rows have gt_source == derived_source, so
they are unusable as labels and are dropped by a gt != derived guard.

The oracle is used ONLY as a label source and is never weakened. No LLM, no network, no
hosted API. Any run that compiles 3.10 bytecode needs uvx on PATH.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

# Allow `python tools/build_gt_reconstruction_corpus.py` (repo root on sys.path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from pipeline.code_object_repair_loop import (
    CodeObjectRepairLoop,
    OracleFragmentFixer,
    build_semantic_repair_messages,
    _filter_semantic_dataset_rows,
    _resolve_dataset_repair_paths,
    _replacement_hash,
    _replacement_structural_hash,
)
from utils.rebuild_semantic_prompt_refresh_dataset import _write_jsonl

BRIEF_ENV = "SEMANTIC_RECONSTRUCTION_BRIEF"
_DEFAULT_DATASET = _REPO_ROOT / "dataset" / "pyllmpatch_dataset.csv"
_DEFAULT_HOLDOUT = [
    _REPO_ROOT / "dataset" / "pyllmpatch_semantic_hybrid100.csv",
    _REPO_ROOT / "dataset" / "pyllmpatch_semantic_briefab.csv",
]
_DEFAULT_OUT = _REPO_ROOT / "dataset" / "gt_reconstruction_corpus.jsonl"


# --------------------------------------------------------------------------------------
# Capture predicate -- mirrors _format_gt_reconstruction_brief's render gate exactly.
# --------------------------------------------------------------------------------------
def _should_capture_target(repair_context: dict | None) -> bool:
    """Keep a target iff the GT reconstruction brief would actually render for it.

    Missing targets (derived_code_object is None -> target_is_missing) render the brief
    unconditionally. Present targets render it only under the same structural gate
    _format_gt_reconstruction_brief uses: a PyLingual failure message of
    'Different control flow', or 'Different bytecode' with no single failed offset.
    """
    if not repair_context:
        return False
    if repair_context.get("target_is_missing"):
        return True
    failed = repair_context.get("pylingual_failed_result") or {}
    message = failed.get("message") if isinstance(failed, dict) else None
    return message == "Different control flow" or (
        message == "Different bytecode" and repair_context.get("failed_offset") is None
    )


def _repair_operation_for_target(repair_context: dict | None) -> str:
    if repair_context and repair_context.get("target_is_missing"):
        return "insert_missing"
    return "control_flow"


def _target_kind_for(repair_context: dict | None, qualname: str) -> str:
    if repair_context and repair_context.get("target_kind"):
        return str(repair_context["target_kind"])
    return "module_statement" if qualname == "<module>" else "code_object_fragment"


@contextlib.contextmanager
def _brief_enabled() -> Iterator[None]:
    previous = os.environ.get(BRIEF_ENV)
    os.environ[BRIEF_ENV] = "true"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(BRIEF_ENV, None)
        else:
            os.environ[BRIEF_ENV] = previous


def _emit_corpus_row(
    *,
    source_file_hash: str,
    qualname: str,
    target_kind: str,
    repair_operation: str,
    messages: list[dict],
    gt_fragment: str,
) -> dict:
    """Assemble one finetuner-shaped row.

    The trainer's build_chat_samples reads ONLY prompt.messages (>=2 dicts, each a
    non-empty role+content) and top-level replacement_text; every other key is audit
    decoration used for dedup / holdout / telemetry.
    """
    prompt_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    return {
        "prompt": {"messages": prompt_messages},
        "replacement_text": gt_fragment,
        "source_file_hash": source_file_hash,
        "qualname": qualname,
        "target_kind": target_kind,
        "repair_operation": repair_operation,
    }


# --------------------------------------------------------------------------------------
# Capture fixer -- wraps OracleFragmentFixer, labels every target, captures the ones
# whose brief renders. Returns the oracle GT candidate so the loop ACCEPTS and advances,
# surfacing the full missing / extra / control_flow cascade.
# --------------------------------------------------------------------------------------
class OracleCaptureFragmentFixer:
    def __init__(
        self,
        *,
        gt_pyc: Path,
        gt_source: Path | None,
        source_file_hash: str,
        sink: list[dict],
        seen: set,
    ) -> None:
        self._oracle = OracleFragmentFixer(gt_pyc, gt_source)
        self._source_file_hash = source_file_hash
        self._sink = sink
        self._seen = seen

    # The loop calls set_prompt_output_dir if present; make it a no-op so we can be
    # driven by CodeObjectRepairLoop without writing prompt artifacts.
    def set_prompt_output_dir(self, prompt_dir: Path | None) -> None:  # noqa: D401
        return None

    def _oracle_label(self, qualname: str) -> str:
        return self._oracle.generate_candidate(
            qualname=qualname,
            gt_code_object=None,
            derived_code_object=None,
            derived_source_fragment="",
            repair_context=None,
        )

    def _maybe_capture(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None,
        gt_fragment: str,
    ) -> None:
        if not _should_capture_target(repair_context):
            return
        structural = _replacement_structural_hash(gt_fragment)
        dedup_key = (self._source_file_hash, qualname, structural)
        if dedup_key in self._seen:
            return
        with _brief_enabled():
            messages = build_semantic_repair_messages(
                qualname=qualname,
                gt_code_object=gt_code_object,
                derived_code_object=derived_code_object,
                derived_source_fragment=derived_source_fragment,
                repair_context=repair_context,
                telemetry_guidance="",
                action_pattern_guidance="",
            )
        row = _emit_corpus_row(
            source_file_hash=self._source_file_hash,
            qualname=qualname,
            target_kind=_target_kind_for(repair_context, qualname),
            repair_operation=_repair_operation_for_target(repair_context),
            messages=messages,
            gt_fragment=gt_fragment,
        )
        self._seen.add(dedup_key)
        self._sink.append(row)

    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        gt_fragment = self._oracle_label(qualname)
        self._maybe_capture(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
            repair_context=repair_context,
            gt_fragment=gt_fragment,
        )
        return gt_fragment

    def generate_candidates(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> list[dict[str, Any]]:
        gt_fragment = self._oracle_label(qualname)
        self._maybe_capture(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
            repair_context=repair_context,
            gt_fragment=gt_fragment,
        )
        return [
            {
                "candidate_index": 0,
                "strategy": "oracle_capture",
                "text": gt_fragment,
                "replacement_hash": _replacement_hash(gt_fragment),
                "replacement_structural_hash": _replacement_structural_hash(gt_fragment),
            }
        ]


# --------------------------------------------------------------------------------------
# Holdout handling.
# --------------------------------------------------------------------------------------
def _load_holdout_hashes(paths: Iterable[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str)
        if "file_hash" not in frame.columns:
            continue
        for value in frame["file_hash"].dropna().tolist():
            hashes.add(str(value))
    return hashes


def _row_file_hash(row: Any) -> Any:
    if isinstance(row, dict):
        return row.get("file_hash")
    return getattr(row, "file_hash", None)


def _filter_holdout_rows(rows: Iterable[Any], holdout: set[str]) -> Iterator[Any]:
    for row in rows:
        if str(_row_file_hash(row)) in holdout:
            continue
        yield row


# --------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------
def _resolve_row(row: Any, dataset_base_dir: Path):
    try:
        return _resolve_dataset_repair_paths(row, dataset_base_dir)
    except Exception:
        return None, None, None, None


def _gt_differs_from_derived(gt_source: Path, derived_source: Path) -> bool:
    """Guard: keep the row only when the GT source genuinely differs from the derived
    source (PyLingual.IO rows have gt_source == derived_source and are unusable)."""
    try:
        gt_text = Path(gt_source).read_text(encoding="utf-8", errors="replace")
        derived_text = Path(derived_source).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return gt_text != derived_text


def build_corpus(
    *,
    dataset_path: Path,
    source: str,
    holdout_paths: list[Path],
    limit: int | None,
    row_range: str | None,
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    dataset_path = dataset_path.expanduser().resolve()
    holdout = _load_holdout_hashes(holdout_paths)

    frame = pd.read_csv(dataset_path)
    filtered = _filter_semantic_dataset_rows(
        frame, source=source, row_range=row_range, limit=limit
    )
    row_items = list(filtered.itertuples(index=False))
    survivors = list(_filter_holdout_rows(row_items, holdout))

    sink: list[dict] = []
    seen: set = set()
    stats = {
        "rows_considered": len(row_items),
        "rows_after_holdout": len(survivors),
        "rows_unresolved": 0,
        "rows_gt_equals_derived": 0,
        "rows_processed": 0,
        "rows_errored": 0,
    }

    for row in survivors:
        file_hash = str(_row_file_hash(row))
        gt_source, gt_pyc, derived_pyc, derived_source = _resolve_row(row, dataset_path.parent)
        if gt_source is None or gt_pyc is None or derived_pyc is None or derived_source is None:
            stats["rows_unresolved"] += 1
            continue
        if not all(Path(p).exists() for p in (gt_source, gt_pyc, derived_pyc, derived_source)):
            stats["rows_unresolved"] += 1
            continue
        if not _gt_differs_from_derived(gt_source, derived_source):
            stats["rows_gt_equals_derived"] += 1
            continue

        fixer = OracleCaptureFragmentFixer(
            gt_pyc=gt_pyc,
            gt_source=gt_source,
            source_file_hash=file_hash,
            sink=sink,
            seen=seen,
        )
        loop = CodeObjectRepairLoop(fixer)
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                loop.run(
                    gt_pyc=Path(gt_pyc),
                    derived_pyc=Path(derived_pyc),
                    derived_source=Path(derived_source),
                    output_dir=Path(tmp_dir),
                    file_hash=file_hash,
                    verify_with_pylingual=True,
                    verify_each_step_with_pylingual=True,
                    reject_non_improving_candidates=True,
                    max_iterations=1,
                )
                stats["rows_processed"] += 1
            except Exception as exc:  # noqa: BLE001 -- one bad row must not sink the build
                stats["rows_errored"] += 1
                print(
                    f"[gt_reconstruction_corpus] row {file_hash[:16]} errored: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    by_operation: dict[str, int] = {}
    for record in sink:
        op = str(record.get("repair_operation"))
        by_operation[op] = by_operation.get(op, 0) + 1

    return sink, by_operation, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the oracle-labelled GT-bytecode reconstruction corpus (PyPi only)."
    )
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--source", type=str, default="PyPi")
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        nargs="+",
        default=_DEFAULT_HOLDOUT,
        help="One or more holdout CSVs whose file_hashes are excluded (union).",
    )
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--row-range", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    sink, by_operation, stats = build_corpus(
        dataset_path=args.dataset,
        source=args.source,
        holdout_paths=list(args.holdout_csv),
        limit=args.limit,
        row_range=args.row_range,
    )

    print("[gt_reconstruction_corpus] row funnel:", flush=True)
    for key, value in stats.items():
        print(f"    {key}: {value}", flush=True)
    print(
        f"[gt_reconstruction_corpus] captured {len(sink)} rows; by repair_operation:",
        flush=True,
    )
    for op, count in sorted(by_operation.items()):
        print(f"    {op}: {count}", flush=True)

    out_path = Path(args.out).expanduser().resolve()
    _write_jsonl(out_path, sink)
    print(f"[gt_reconstruction_corpus] wrote {len(sink)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
