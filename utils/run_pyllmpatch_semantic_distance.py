from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.file_helpers import fetch_pyllmpatch_pyc_paths
from utils.pyc_code_object_distance import compare_code_object_distances, summarize_results

DEFAULT_DATASET_PATH = REPO_ROOT / "dataset" / "pyllmpatch_dataset.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "dataset" / "results" / "pyllmpatch_semantic_error_distance.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bytecode distance calculation for semantic_error rows in pyllmpatch_dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the input dataset CSV. Defaults to {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write the summary CSV. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of semantic_error rows to process.",
    )
    return parser


def _summary_fieldnames() -> list[str]:
    return [
        "file_hash",
        "source",
        "error_type",
        "status",
        "gt_pyc",
        "derived_pyc",
        "gt_code_object_count",
        "derived_code_object_count",
        "total_gt_inst_count",
        "total_derived_inst_count",
        "total_instruction_distance",
        "total_control_flow_distance",
        "total_interaction_penalty",
        "total_unmatched_penalty",
        "mean_normalized_instruction_distance",
        "mean_normalized_cfg_distance",
        "mean_normalized_interaction_penalty",
        "mean_normalized_unmatched_penalty",
        "control_flow_weight",
        "total_combined_distance",
        "normalized_combined_distance",
        "error_message",
    ]


def _unresolved_row(file_hash: str, source: str, error_type: str, gt_pyc: Path | None, derived_pyc: Path | None) -> dict:
    return {
        "file_hash": file_hash,
        "source": source,
        "error_type": error_type,
        "status": "unresolved_pyc",
        "gt_pyc": str(gt_pyc) if gt_pyc else None,
        "derived_pyc": str(derived_pyc) if derived_pyc else None,
        "gt_code_object_count": None,
        "derived_code_object_count": None,
        "total_gt_inst_count": None,
        "total_derived_inst_count": None,
        "total_instruction_distance": None,
        "total_control_flow_distance": None,
        "total_interaction_penalty": None,
        "total_unmatched_penalty": None,
        "mean_normalized_instruction_distance": None,
        "mean_normalized_cfg_distance": None,
        "mean_normalized_interaction_penalty": None,
        "mean_normalized_unmatched_penalty": None,
        "control_flow_weight": None,
        "total_combined_distance": None,
        "normalized_combined_distance": None,
        "error_message": "Could not resolve gt_pyc and/or derived_pyc",
    }


def _exception_row(file_hash: str, source: str, error_type: str, gt_pyc: Path, derived_pyc: Path, exc: Exception) -> dict:
    return {
        "file_hash": file_hash,
        "source": source,
        "error_type": error_type,
        "status": "comparison_failed",
        "gt_pyc": str(gt_pyc),
        "derived_pyc": str(derived_pyc),
        "gt_code_object_count": None,
        "derived_code_object_count": None,
        "total_gt_inst_count": None,
        "total_derived_inst_count": None,
        "total_instruction_distance": None,
        "total_control_flow_distance": None,
        "total_interaction_penalty": None,
        "total_unmatched_penalty": None,
        "mean_normalized_instruction_distance": None,
        "mean_normalized_cfg_distance": None,
        "mean_normalized_interaction_penalty": None,
        "mean_normalized_unmatched_penalty": None,
        "control_flow_weight": None,
        "total_combined_distance": None,
        "normalized_combined_distance": None,
        "error_message": f"{type(exc).__name__}: {exc}",
    }


def _summary_row(file_hash: str, source: str, error_type: str, gt_pyc: Path, derived_pyc: Path) -> dict:
    results = compare_code_object_distances(gt_pyc, derived_pyc)
    summary = summarize_results(results)
    return {
        "file_hash": file_hash,
        "source": source,
        "error_type": error_type,
        "status": summary["status"],
        "gt_pyc": str(gt_pyc),
        "derived_pyc": str(derived_pyc),
        "gt_code_object_count": summary["gt_code_object_count"],
        "derived_code_object_count": summary["derived_code_object_count"],
        "total_gt_inst_count": summary["gt_inst_count"],
        "total_derived_inst_count": summary["derived_inst_count"],
        "total_instruction_distance": summary["instruction_distance"],
        "total_control_flow_distance": summary["control_flow_distance"],
        "total_interaction_penalty": summary["interaction_penalty"],
        "total_unmatched_penalty": summary["unmatched_penalty"],
        "mean_normalized_instruction_distance": f"{summary['normalized_instruction_distance']:.6f}",
        "mean_normalized_cfg_distance": f"{summary['normalized_cfg_distance']:.6f}",
        "mean_normalized_interaction_penalty": f"{summary['normalized_interaction_penalty']:.6f}",
        "mean_normalized_unmatched_penalty": f"{summary['normalized_unmatched_penalty']:.6f}",
        "control_flow_weight": summary["control_flow_weight"],
        "total_combined_distance": summary["combined_distance"],
        "normalized_combined_distance": f"{summary['normalized_combined_distance']:.6f}",
        "error_message": None,
    }


def run_dataset(dataset_path: Path, csv_out: Path, limit: int | None = None) -> dict:
    dataset_path = dataset_path.expanduser().resolve()
    csv_out = csv_out.expanduser().resolve()

    df = pd.read_csv(dataset_path)
    semantic_df = df[df["error_type"] == "semantic_error"].copy()
    if limit is not None:
        semantic_df = semantic_df.head(limit)

    csv_out.parent.mkdir(parents=True, exist_ok=True)

    total_rows = len(semantic_df)
    resolved_rows = 0
    unresolved_rows = 0
    failed_rows = 0

    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_summary_fieldnames())
        writer.writeheader()

        for index, row in enumerate(semantic_df.itertuples(index=False), start=1):
            print(f"[{index}/{total_rows}] file_hash={row.file_hash}")
            gt_pyc, derived_pyc = fetch_pyllmpatch_pyc_paths(row.file_hash, row.source)

            if gt_pyc is None or derived_pyc is None:
                writer.writerow(_unresolved_row(row.file_hash, row.source, row.error_type, gt_pyc, derived_pyc))
                unresolved_rows += 1
                continue

            try:
                writer.writerow(_summary_row(row.file_hash, row.source, row.error_type, gt_pyc, derived_pyc))
                resolved_rows += 1
            except Exception as exc:
                writer.writerow(_exception_row(row.file_hash, row.source, row.error_type, gt_pyc, derived_pyc, exc))
                failed_rows += 1

    return {
        "dataset_path": str(dataset_path),
        "csv_out": str(csv_out),
        "semantic_error_rows": total_rows,
        "resolved_rows": resolved_rows,
        "unresolved_rows": unresolved_rows,
        "failed_rows": failed_rows,
    }


def main() -> int:
    args = build_parser().parse_args()
    summary = run_dataset(args.dataset_path, args.csv_out, args.limit)

    print("\nRun summary")
    print(f"dataset_path: {summary['dataset_path']}")
    print(f"csv_out: {summary['csv_out']}")
    print(f"semantic_error_rows: {summary['semantic_error_rows']}")
    print(f"resolved_rows: {summary['resolved_rows']}")
    print(f"unresolved_rows: {summary['unresolved_rows']}")
    print(f"failed_rows: {summary['failed_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
