from __future__ import annotations

import argparse
import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import BASE_DATASET_PATH, build_run_paths, current_run_timestamp, now_iso
from pipeline.logging_utils import append_log
from utils.file_helpers import fetch_pyllmpatch_repair_paths
from utils.reattach_source_code_object import (
    _find_target_row,
    extract_source_segment,
    infer_source_from_pyc,
    repair_mismatching_code_objects,
)

#TODO: Insert LLM here for repairing
class FragmentFixer(ABC):
    @abstractmethod
    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
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
    ) -> str:
        del gt_code_object
        del derived_code_object
        del derived_source_fragment
        row = _find_target_row(self.gt_source, self.gt_pyc, qualname, strict_map=True)
        return extract_source_segment(self.gt_source_text, row)


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
    ) -> dict:
        return repair_mismatching_code_objects(
            gt_pyc=gt_pyc,
            derived_pyc=derived_pyc,
            derived_source=derived_source,
            output_dir=output_dir,
            fragment_fixer=self._fix_fragment,
            strict_map=strict_map,
            verify_with_pylingual=verify_with_pylingual,
            verify_each_step_with_pylingual=verify_each_step_with_pylingual,
            reject_non_improving_candidates=reject_non_improving_candidates,
        )

    def _fix_fragment(
        self,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
    ) -> str:
        return self.fixer.generate_candidate(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
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
                if fixer_name != "oracle":
                    raise ValueError(f"Unsupported fixer backend: {fixer_name}")
                loop = CodeObjectRepairLoop(OracleFragmentFixer(gt_pyc))
                result = loop.run(
                    gt_pyc=gt_pyc,
                    derived_pyc=derived_pyc,
                    derived_source=derived_source,
                    output_dir=row_output_dir,
                    strict_map=strict_map,
                    verify_with_pylingual=verify_with_pylingual,
                    verify_each_step_with_pylingual=verify_each_step_with_pylingual,
                    reject_non_improving_candidates=reject_non_improving_candidates,
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
        choices=("oracle",),
        default="oracle",
        help="Fragment fixer backend to use",
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

    if args.fixer != "oracle":
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
        )
    else:
        if args.gt_pyc is None or args.derived_pyc is None or args.derived_source is None:
            raise ValueError("gt_pyc, derived_pyc, and derived_source are required unless --dataset-mode is used")
        loop = CodeObjectRepairLoop(OracleFragmentFixer(args.gt_pyc))
        result = loop.run(
            gt_pyc=args.gt_pyc,
            derived_pyc=args.derived_pyc,
            derived_source=args.derived_source,
            output_dir=args.output_dir,
            strict_map=args.strict_map,
            verify_with_pylingual=not args.skip_pylingual_verification,
            verify_each_step_with_pylingual=not args.skip_step_verification,
            reject_non_improving_candidates=not args.keep_non_improving,
        )

    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
