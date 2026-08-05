from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.file_helpers import fetch_pyllmpatch_pyc_paths
from utils.pyc_code_object_distance import (
    _ged_control_flow_distance_from_graphs,
    _heuristic_control_flow_distance,
    _is_control_flow_equivalent_serialized,
    _iter_matched_code_objects,
    levenshtein_distance,
    prepare_pyc,
)


DEFAULT_DATASET_PATH = REPO_ROOT / "dataset" / "pyllmpatch_dataset.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "dataset" / "results" / "cfg_heuristic_vs_ged.csv"
DEFAULT_SAMPLE_TIMEOUT_SECONDS = 600


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare heuristic CFG distance against GED on matched code-object pairs."
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
        help=f"Path to write comparison CSV. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--error-type",
        type=str,
        default="semantic_error",
        help="Dataset error_type to filter. Defaults to semantic_error.",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Optional limit on dataset rows to process.",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=None,
        help="Optional limit on matched code-object pairs to emit.",
    )
    parser.add_argument(
        "--skip-equivalent",
        action="store_true",
        help="Skip pairs whose CFGs are exactly equivalent.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_SAMPLE_TIMEOUT_SECONDS,
        help=f"Per-sample timeout for CFG preparation/comparison. Defaults to {DEFAULT_SAMPLE_TIMEOUT_SECONDS} seconds.",
    )
    return parser


class SampleTimeoutError(TimeoutError):
    pass


@contextmanager
def _sample_timeout(timeout_seconds: int):
    if timeout_seconds <= 0:
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise SampleTimeoutError(f"sample exceeded timeout of {timeout_seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _safe_float(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.6f}"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + end - 1) / 2 + 1
        for idx in range(start, end):
            ranks[indexed[idx][0]] = avg_rank
        start = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _comparison_fieldnames() -> list[str]:
    return [
        "file_hash",
        "source",
        "gt_pyc",
        "derived_pyc",
        "code_object",
        "gt_name",
        "derived_name",
        "gt_inst_count",
        "derived_inst_count",
        "gt_block_count",
        "derived_block_count",
        "gt_block_edge_count",
        "derived_block_edge_count",
        "instruction_distance",
        "normalized_instruction_distance",
        "equivalent_cfg",
        "heuristic_cfg_distance",
        "ged_cfg_distance",
        "distance_delta",
    ]


def _row_for_pair(file_hash: str, source: str, gt_pyc: Path, derived_pyc: Path, gt_obj: dict, derived_obj: dict) -> dict:
    gt_graph = gt_obj["block_graph"]
    derived_graph = derived_obj["block_graph"]
    instruction_distance = levenshtein_distance(gt_obj["instruction_signatures"], derived_obj["instruction_signatures"])
    instruction_basis = max(gt_obj["inst_count"], derived_obj["inst_count"], 1)
    gt_block_count = sum(1 for _, attrs in gt_graph.nodes(data=True) if attrs["category"] == "block")
    derived_block_count = sum(1 for _, attrs in derived_graph.nodes(data=True) if attrs["category"] == "block")
    gt_edge_count = sum(
        1
        for src, dst in gt_graph.edges
        if gt_graph.nodes[src]["category"] == "block" and gt_graph.nodes[dst]["category"] == "block"
    )
    derived_edge_count = sum(
        1
        for src, dst in derived_graph.edges
        if derived_graph.nodes[src]["category"] == "block" and derived_graph.nodes[dst]["category"] == "block"
    )
    heuristic = _heuristic_control_flow_distance(gt_graph, derived_graph)
    ged = _ged_control_flow_distance_from_graphs(
        gt_graph,
        derived_graph,
        gt_obj["annotated_block_graph"],
        derived_obj["annotated_block_graph"],
    )
    return {
        "file_hash": file_hash,
        "source": source,
        "gt_pyc": str(gt_pyc),
        "derived_pyc": str(derived_pyc),
        "code_object": gt_obj["name"] if gt_obj["name"] == derived_obj["name"] else f"{gt_obj['name']} <-> {derived_obj['name']}",
        "gt_name": gt_obj["name"],
        "derived_name": derived_obj["name"],
        "gt_inst_count": gt_obj["inst_count"],
        "derived_inst_count": derived_obj["inst_count"],
        "gt_block_count": gt_block_count,
        "derived_block_count": derived_block_count,
        "gt_block_edge_count": gt_edge_count,
        "derived_block_edge_count": derived_edge_count,
        "instruction_distance": instruction_distance,
        "normalized_instruction_distance": f"{instruction_distance / instruction_basis:.6f}",
        "equivalent_cfg": _is_control_flow_equivalent_serialized(gt_graph, derived_graph),
        "heuristic_cfg_distance": heuristic,
        "ged_cfg_distance": ged,
        "distance_delta": heuristic - ged,
    }


def run_comparison(
    dataset_path: Path,
    csv_out: Path,
    error_type: str = "semantic_error",
    file_limit: int | None = None,
    pair_limit: int | None = None,
    skip_equivalent: bool = False,
    timeout_seconds: int = DEFAULT_SAMPLE_TIMEOUT_SECONDS,
) -> dict:
    dataset_path = dataset_path.expanduser().resolve()
    csv_out = csv_out.expanduser().resolve()
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    subset_df = df[df["error_type"] == error_type].copy()
    if file_limit is not None:
        subset_df = subset_df.head(file_limit)

    rows_written = 0
    files_seen = 0
    unresolved_files = 0
    failed_files = 0
    timed_out_files = 0
    heuristic_values: list[float] = []
    ged_values: list[float] = []
    zero_case_matches = 0
    total_pairs_checked = 0

    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_comparison_fieldnames())
        writer.writeheader()

        for row in subset_df.itertuples(index=False):
            if pair_limit is not None and rows_written >= pair_limit:
                break

            files_seen += 1
            print(f"[file {files_seen}] file_hash={row.file_hash}")
            gt_pyc, derived_pyc = fetch_pyllmpatch_pyc_paths(row.file_hash, row.source)
            if gt_pyc is None or derived_pyc is None:
                unresolved_files += 1
                continue

            try:
                with _sample_timeout(timeout_seconds):
                    prepared_gt = prepare_pyc(gt_pyc)
                    prepared_derived = prepare_pyc(derived_pyc)

                    for gt_obj, derived_obj in _iter_matched_code_objects(prepared_gt, prepared_derived):
                        if pair_limit is not None and rows_written >= pair_limit:
                            break
                        if gt_obj is None or derived_obj is None:
                            continue

                        comparison_row = _row_for_pair(row.file_hash, row.source, gt_pyc, derived_pyc, gt_obj, derived_obj)
                        total_pairs_checked += 1

                        if skip_equivalent and comparison_row["equivalent_cfg"]:
                            continue

                        heuristic_values.append(comparison_row["heuristic_cfg_distance"])
                        ged_values.append(comparison_row["ged_cfg_distance"])
                        if (comparison_row["heuristic_cfg_distance"] == 0) == (comparison_row["ged_cfg_distance"] == 0):
                            zero_case_matches += 1

                        writer.writerow(comparison_row)
                        rows_written += 1

                        if pair_limit is not None and rows_written >= pair_limit:
                            break
            except SampleTimeoutError:
                timed_out_files += 1
                print(f"  skipped: exceeded timeout of {timeout_seconds} seconds")
                continue
            except Exception:
                failed_files += 1
                continue

            if pair_limit is not None and rows_written >= pair_limit:
                break

    pearson = _pearson(heuristic_values, ged_values)
    spearman = _spearman(heuristic_values, ged_values)

    return {
        "dataset_path": str(dataset_path),
        "csv_out": str(csv_out),
        "error_type": error_type,
        "files_seen": files_seen,
        "unresolved_files": unresolved_files,
        "failed_files": failed_files,
        "timed_out_files": timed_out_files,
        "rows_written": rows_written,
        "pairs_checked": total_pairs_checked,
        "zero_case_agreement": _safe_float(zero_case_matches / total_pairs_checked) if total_pairs_checked else None,
        "pearson_correlation": _safe_float(pearson),
        "spearman_correlation": _safe_float(spearman),
    }


def main() -> int:
    args = build_parser().parse_args()
    summary = run_comparison(
        dataset_path=args.dataset_path,
        csv_out=args.csv_out,
        error_type=args.error_type,
        file_limit=args.file_limit,
        pair_limit=args.pair_limit,
        skip_equivalent=args.skip_equivalent,
        timeout_seconds=args.timeout_seconds,
    )

    print("\nComparison summary")
    print(f"dataset_path: {summary['dataset_path']}")
    print(f"csv_out: {summary['csv_out']}")
    print(f"error_type: {summary['error_type']}")
    print(f"files_seen: {summary['files_seen']}")
    print(f"unresolved_files: {summary['unresolved_files']}")
    print(f"failed_files: {summary['failed_files']}")
    print(f"timed_out_files: {summary['timed_out_files']}")
    print(f"rows_written: {summary['rows_written']}")
    print(f"pairs_checked: {summary['pairs_checked']}")
    print(f"zero_case_agreement: {summary['zero_case_agreement']}")
    print(f"pearson_correlation: {summary['pearson_correlation']}")
    print(f"spearman_correlation: {summary['spearman_correlation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
