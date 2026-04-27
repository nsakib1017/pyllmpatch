from __future__ import annotations

import argparse
import csv
import dis
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import BASE_DATASET_PATH, build_run_paths, current_run_timestamp, now_iso
from pipeline.logging_utils import append_log
from utils.file_helpers import fetch_pyllmpatch_repair_paths, strip_code_fences
from utils.providers import find_llm_config, make_llm_call_from_config
from utils.reattach_source_code_object import (
    _find_target_row,
    extract_source_segment,
    infer_source_from_pyc,
    repair_mismatching_code_objects,
)

SEMANTIC_REPAIR_SYSTEM_PROMPT = """You are a Python bytecode-aware semantic repair specialist.

You will receive:
- the target ground-truth Python code object metadata and either disassembly or a localized instruction window
- the current derived Python code object metadata and either disassembly or a localized instruction window
- the source fragment from the derived source file that produced the derived code object

Edit only the provided source fragment. Preserve the same public names, function/class boundary, decorators, parameters, and return shape unless the bytecode evidence clearly requires a small change. Make the minimal source changes that best align the derived code object with the ground-truth code object. Keep relative indentation valid. Return only the repaired Python source fragment, with no markdown fences and no explanation.
"""


def _format_code_object_for_prompt(code_object: Any) -> str:
    if code_object is None:
        return "<missing>"

    fields = [
        f"co_name: {getattr(code_object, 'co_name', None)}",
        f"co_qualname: {getattr(code_object, 'co_qualname', None)}",
        f"co_argcount: {getattr(code_object, 'co_argcount', None)}",
        f"co_posonlyargcount: {getattr(code_object, 'co_posonlyargcount', None)}",
        f"co_kwonlyargcount: {getattr(code_object, 'co_kwonlyargcount', None)}",
        f"co_nlocals: {getattr(code_object, 'co_nlocals', None)}",
        f"co_stacksize: {getattr(code_object, 'co_stacksize', None)}",
        f"co_flags: {getattr(code_object, 'co_flags', None)}",
        f"co_varnames: {getattr(code_object, 'co_varnames', ())}",
        f"co_names: {getattr(code_object, 'co_names', ())}",
        f"co_freevars: {getattr(code_object, 'co_freevars', ())}",
        f"co_cellvars: {getattr(code_object, 'co_cellvars', ())}",
        f"co_consts: {getattr(code_object, 'co_consts', ())}",
    ]
    try:
        disassembly = dis.Bytecode(code_object).dis()
    except Exception as exc:
        disassembly = f"<disassembly unavailable: {type(exc).__name__}: {exc}>"
    return "\n".join(fields) + "\n\nDisassembly:\n" + disassembly


def _format_code_object_summary_for_prompt(code_object: Any) -> str:
    if code_object is None:
        return "<missing>"
    fields = [
        f"co_name: {getattr(code_object, 'co_name', None)}",
        f"co_qualname: {getattr(code_object, 'co_qualname', None)}",
        f"co_firstlineno: {getattr(code_object, 'co_firstlineno', None)}",
        f"co_argcount: {getattr(code_object, 'co_argcount', None)}",
        f"co_kwonlyargcount: {getattr(code_object, 'co_kwonlyargcount', None)}",
        f"co_nlocals: {getattr(code_object, 'co_nlocals', None)}",
        f"co_stacksize: {getattr(code_object, 'co_stacksize', None)}",
        f"co_flags: {getattr(code_object, 'co_flags', None)}",
        f"co_varnames: {getattr(code_object, 'co_varnames', ())}",
        f"co_names: {getattr(code_object, 'co_names', ())}",
    ]
    return "\n".join(fields)


def _format_repair_context(repair_context: dict | None) -> str:
    if not repair_context:
        return ""
    instruction_context = repair_context.get("localized_instruction_context", {})
    rejected_attempts = repair_context.get("rejected_attempts") or []
    rejected_text = "<none>"
    if rejected_attempts:
        rejected_text = "\n\n".join(
            (
                f"Attempt {item.get('attempt')} rejected: {item.get('acceptance_reason')}\n"
                f"Replacement:\n```python\n{item.get('replacement_text', '')}\n```"
            )
            for item in rejected_attempts
        )
    failed = repair_context.get("pylingual_failed_result") or {}
    return f"""Localized repair context:
- target_kind: {repair_context.get("target_kind")}
- localized_line_number: {repair_context.get("localized_line_number")}
- failed_offset: {instruction_context.get("failed_offset")}
- alignment_tag: {instruction_context.get("alignment_tag")}
- pylingual_message: {failed.get("message")}

Ground-truth instruction window:
```text
{instruction_context.get("gt_instruction_window", "<unavailable>")}
```

Current derived instruction window:
```text
{instruction_context.get("derived_instruction_window", "<unavailable>")}
```

Previous rejected attempts:
{rejected_text}
"""


def build_semantic_repair_messages(
    *,
    qualname: str,
    gt_code_object: Any,
    derived_code_object: Any,
    derived_source_fragment: str,
    repair_context: dict | None = None,
) -> list[dict]:
    if qualname == "<module>":
        task_text = "Edit only the localized top-level module source statement. Preserve valid Python syntax and make the smallest source change that best aligns the module bytecode with the ground-truth module bytecode."
        source_label = "Current derived top-level module statement"
        gt_code_object_text = _format_code_object_summary_for_prompt(gt_code_object)
        derived_code_object_text = _format_code_object_summary_for_prompt(derived_code_object)
    elif derived_code_object is None:
        task_text = "The target code object is missing from the derived bytecode. Synthesize only the missing Python source fragment that should be inserted into the derived source."
        source_label = "Derived insertion context"
        gt_code_object_text = _format_code_object_for_prompt(gt_code_object)
        derived_code_object_text = _format_code_object_for_prompt(derived_code_object)
    else:
        task_text = "Edit only the current derived source fragment."
        source_label = "Current derived source fragment"
        gt_code_object_text = _format_code_object_for_prompt(gt_code_object)
        derived_code_object_text = _format_code_object_for_prompt(derived_code_object)

    user_prompt = f"""Task: {task_text}

Target qualname: {qualname}

Ground-truth code object:
{gt_code_object_text}

Current derived code object:
{derived_code_object_text}

{_format_repair_context(repair_context)}

{source_label}:
```python
{derived_source_fragment}
```

Return only the repaired source fragment."""
    return [
        {"role": "system", "content": SEMANTIC_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class FragmentFixer(ABC):
    @abstractmethod
    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        raise NotImplementedError


class OracleFragmentFixer(FragmentFixer):
    def __init__(self, gt_pyc: Path):
        self.gt_pyc = gt_pyc.expanduser().resolve()
        self.gt_source = infer_source_from_pyc(self.gt_pyc)
        self.gt_source_text = self.gt_source.read_text(encoding="utf-8")

    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        del gt_code_object
        del derived_code_object
        del derived_source_fragment
        del repair_context
        row = _find_target_row(self.gt_source, self.gt_pyc, qualname, strict_map=True)
        return extract_source_segment(self.gt_source_text, row)


class LLMFragmentFixer(FragmentFixer):
    def __init__(self, *, provider: str = "Google", model: str = "gemini-2.5-flash-lite"):
        self.llm_config = find_llm_config(provider, model)
        self.calls: list[dict] = []

    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        messages = build_semantic_repair_messages(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
            repair_context=repair_context,
        )
        started = time.perf_counter()
        response = make_llm_call_from_config(messages, self.llm_config)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        content = strip_code_fences(response)
        self.calls.append(
            {
                "provider": self.llm_config["provider"],
                "model": self.llm_config["name"],
                "latency_ms": elapsed_ms,
                "usage": None if response is None else str(response.get("usage")),
                "qualname": qualname,
            }
        )
        return content or derived_source_fragment


class CodeObjectRepairLoop:
    def __init__(self, fixer: FragmentFixer):
        self.fixer = fixer

    def run(
        self,
        *,
        gt_pyc: Path,
        derived_pyc: Path,
        derived_source: Path,
        output_dir: Path | None = None,
        strict_map: bool = True,
        verify_with_pylingual: bool = True,
        verify_each_step_with_pylingual: bool = True,
        reject_non_improving_candidates: bool = True,
        max_iterations: int = 1,
    ) -> dict:
        result = repair_mismatching_code_objects(
            gt_pyc=gt_pyc,
            derived_pyc=derived_pyc,
            derived_source=derived_source,
            output_dir=output_dir,
            fragment_fixer=self._fix_fragment,
            strict_map=strict_map,
            verify_with_pylingual=verify_with_pylingual,
            verify_each_step_with_pylingual=verify_each_step_with_pylingual,
            reject_non_improving_candidates=reject_non_improving_candidates,
            max_iterations=max_iterations,
        )
        if hasattr(self.fixer, "calls"):
            result["llm_calls"] = getattr(self.fixer, "calls")
        return result

    def _fix_fragment(
        self,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        return self.fixer.generate_candidate(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
            repair_context=repair_context,
        )


def _dataset_fieldnames() -> list[str]:
    return [
        "file_hash",
        "source",
        "error_type",
        "status",
        "gt_pyc",
        "derived_pyc",
        "derived_source",
        "initial_combined_distance",
        "final_combined_distance",
        "initial_gt_code_object_count",
        "initial_derived_code_object_count",
        "final_gt_code_object_count",
        "final_derived_code_object_count",
        "repair_target_count",
        "accepted_step_count",
        "pylingual_all_equal",
        "error_message",
        "result_json",
    ]


def _dataset_result_row(row, result: dict, result_json_path: Path) -> dict:
    accepted_steps = sum(1 for step in result["steps"] if step["accepted"])
    verification = result.get("pylingual_verification")
    return {
        "file_hash": row.file_hash,
        "source": row.source,
        "error_type": row.error_type,
        "status": "repaired",
        "gt_pyc": result["gt_pyc"],
        "derived_pyc": result["derived_pyc"],
        "derived_source": result["derived_source"],
        "initial_combined_distance": result["initial_summary"]["combined_distance"],
        "final_combined_distance": result["final_summary"]["combined_distance"],
        "initial_gt_code_object_count": result["initial_summary"]["gt_code_object_count"],
        "initial_derived_code_object_count": result["initial_summary"]["derived_code_object_count"],
        "final_gt_code_object_count": result["final_summary"]["gt_code_object_count"],
        "final_derived_code_object_count": result["final_summary"]["derived_code_object_count"],
        "repair_target_count": len(result["repair_targets"]),
        "accepted_step_count": accepted_steps,
        "pylingual_all_equal": None if verification is None else verification["all_equal"],
        "error_message": None,
        "result_json": str(result_json_path),
    }


def _dataset_error_row(row, gt_pyc: Path | None, derived_pyc: Path | None, derived_source: Path | None, message: str) -> dict:
    return {
        "file_hash": row.file_hash,
        "source": row.source,
        "error_type": row.error_type,
        "status": "failed",
        "gt_pyc": str(gt_pyc) if gt_pyc else None,
        "derived_pyc": str(derived_pyc) if derived_pyc else None,
        "derived_source": str(derived_source) if derived_source else None,
        "initial_combined_distance": None,
        "final_combined_distance": None,
        "initial_gt_code_object_count": None,
        "initial_derived_code_object_count": None,
        "final_gt_code_object_count": None,
        "final_derived_code_object_count": None,
        "repair_target_count": None,
        "accepted_step_count": None,
        "pylingual_all_equal": None,
        "error_message": message,
        "result_json": None,
    }


def run_dataset_repair_loop(
    *,
    fixer_name: str,
    dataset_path: Path = BASE_DATASET_PATH,
    output_dir: Path | None = None,
    limit: int | None = None,
    file_hash: str | None = None,
    strict_map: bool = True,
    verify_with_pylingual: bool = True,
    verify_each_step_with_pylingual: bool = True,
    reject_non_improving_candidates: bool = True,
    max_iterations: int = 1,
    llm_provider: str = "Google",
    llm_model: str = "gemini-2.5-flash-lite",
) -> dict:
    dataset_path = dataset_path.expanduser().resolve()
    if output_dir is None:
        run_id, log_base, log_file = build_run_paths(current_run_timestamp())
    else:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = output_dir.name
        log_base = output_dir
        log_file = output_dir / f"run_log_{run_id}_{dataset_path.stem}.jsonl"
    results_csv = log_base / f"semantic_repair_results_{dataset_path.stem}.csv"

    df = pd.read_csv(dataset_path)
    semantic_df = df[df["error_type"] == "semantic_error"].copy()
    if file_hash is not None:
        semantic_df = semantic_df[semantic_df["file_hash"].astype(str) == str(file_hash)]
    if limit is not None:
        semantic_df = semantic_df.head(limit)

    processed = 0
    repaired = 0
    failed = 0

    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_dataset_fieldnames())
        writer.writeheader()

        for row in semantic_df.itertuples(index=False):
            processed += 1
            gt_pyc, derived_pyc, derived_source = fetch_pyllmpatch_repair_paths(row.file_hash, row.source)
            if gt_pyc is None or derived_pyc is None or derived_source is None:
                error_row = _dataset_error_row(row, gt_pyc, derived_pyc, derived_source, "Could not resolve gt_pyc, derived_pyc, and/or derived_source")
                writer.writerow(error_row)
                append_log(
                    log_file,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "mode": "semantic_repair",
                        **error_row,
                    },
                )
                failed += 1
                continue

            row_output_dir = log_base / "semantic_repair" / str(row.source) / str(row.file_hash)
            row_output_dir.mkdir(parents=True, exist_ok=True)
            result_json_path = row_output_dir / "result.json"
            try:
                if fixer_name not in {"oracle", "llm"}:
                    raise ValueError(f"Unsupported fixer backend: {fixer_name}")
                fixer = (
                    OracleFragmentFixer(gt_pyc)
                    if fixer_name == "oracle"
                    else LLMFragmentFixer(provider=llm_provider, model=llm_model)
                )
                loop = CodeObjectRepairLoop(fixer)
                result = loop.run(
                    gt_pyc=gt_pyc,
                    derived_pyc=derived_pyc,
                    derived_source=derived_source,
                    output_dir=row_output_dir,
                    strict_map=strict_map,
                    verify_with_pylingual=verify_with_pylingual,
                    verify_each_step_with_pylingual=verify_each_step_with_pylingual,
                    reject_non_improving_candidates=reject_non_improving_candidates,
                    max_iterations=max_iterations,
                )
                result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                result_row = _dataset_result_row(row, result, result_json_path)
                writer.writerow(result_row)
                append_log(
                    log_file,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "mode": "semantic_repair",
                        **result_row,
                    },
                )
                repaired += 1
            except Exception as exc:
                error_row = _dataset_error_row(row, gt_pyc, derived_pyc, derived_source, f"{type(exc).__name__}: {exc}")
                writer.writerow(error_row)
                append_log(
                    log_file,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "mode": "semantic_repair",
                        **error_row,
                    },
                )
                failed += 1

    return {
        "dataset_path": str(dataset_path),
        "output_dir": str(log_base),
        "results_csv": str(results_csv),
        "run_log": str(log_file),
        "run_id": run_id,
        "processed_rows": processed,
        "repaired_rows": repaired,
        "failed_rows": failed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the per-code-object repair loop with a pluggable fragment fixer."
    )
    parser.add_argument("gt_pyc", type=Path, nargs="?", help="Ground-truth .pyc path")
    parser.add_argument("derived_pyc", type=Path, nargs="?", help="Derived .pyc path")
    parser.add_argument("derived_source", type=Path, nargs="?", help="Derived source .py path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for intermediate repaired files and fragments",
    )
    parser.add_argument(
        "--strict-map",
        action="store_true",
        help="Require strict source-to-pyc mapping for span lookup",
    )
    parser.add_argument(
        "--skip-pylingual-verification",
        action="store_true",
        help="Disable final and per-step PyLingual equivalence checks",
    )
    parser.add_argument(
        "--skip-step-verification",
        action="store_true",
        help="Disable per-step PyLingual checks while keeping final verification enabled",
    )
    parser.add_argument(
        "--keep-non-improving",
        action="store_true",
        help="Retain candidates even when they do not improve the measured state",
    )
    parser.add_argument(
        "--fixer",
        choices=("oracle", "llm"),
        default="oracle",
        help="Fragment fixer backend to use",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="Google",
        help="Provider from utils.providers for --fixer llm",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Model name from utils.providers for --fixer llm",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum semantic repair iterations over recomputed mismatch targets",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full loop result as JSON",
    )
    parser.add_argument(
        "--dataset-mode",
        action="store_true",
        help="Run semantic repair for semantic_error rows in the env-configured dataset",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=BASE_DATASET_PATH,
        help=f"Dataset CSV path. Defaults to {BASE_DATASET_PATH}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for dataset-mode rows",
    )
    parser.add_argument(
        "--file-hash",
        type=str,
        default=None,
        help="Optional file hash filter for dataset-mode",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.fixer not in {"oracle", "llm"}:
        raise ValueError(f"Unsupported fixer backend: {args.fixer}")

    if args.dataset_mode:
        if args.output_dir is None:
            raise ValueError("--output-dir is required in --dataset-mode")
        result = run_dataset_repair_loop(
            fixer_name=args.fixer,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            limit=args.limit,
            file_hash=args.file_hash,
            strict_map=args.strict_map,
            verify_with_pylingual=not args.skip_pylingual_verification,
            verify_each_step_with_pylingual=not args.skip_step_verification,
            reject_non_improving_candidates=not args.keep_non_improving,
            max_iterations=args.max_iterations,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
        )
    else:
        if args.gt_pyc is None or args.derived_pyc is None or args.derived_source is None:
            raise ValueError("gt_pyc, derived_pyc, and derived_source are required unless --dataset-mode is used")
        fixer = (
            OracleFragmentFixer(args.gt_pyc)
            if args.fixer == "oracle"
            else LLMFragmentFixer(provider=args.llm_provider, model=args.llm_model)
        )
        loop = CodeObjectRepairLoop(fixer)
        result = loop.run(
            gt_pyc=args.gt_pyc,
            derived_pyc=args.derived_pyc,
            derived_source=args.derived_source,
            output_dir=args.output_dir,
            strict_map=args.strict_map,
            verify_with_pylingual=not args.skip_pylingual_verification,
            verify_each_step_with_pylingual=not args.skip_step_verification,
            reject_non_improving_candidates=not args.keep_non_improving,
            max_iterations=args.max_iterations,
        )

    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
