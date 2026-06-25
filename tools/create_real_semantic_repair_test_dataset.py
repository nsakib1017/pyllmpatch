from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.file_helpers import fetch_pyllmpatch_repair_paths, fetch_pyllmpatch_source_path
from utils.reattach_source_code_object import (
    compare_code_object_distances,
    infer_semantic_repair_python_version,
    summarize_results,
)


DEFAULT_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
PYLLMPATCH_COLUMNS = [
    "file_hash",
    "error_type",
    "error",
    "error_message",
    "error_description",
    "bytecode_version",
    "source",
]


def _empty_if_na(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value)
    return "" if text.lower() == "nan" else text


def _source_for_hash(file_hash: str, virustotal_hashes: set[str]) -> str:
    if file_hash in virustotal_hashes:
        return "VirusTotal"
    return "PyLingual.IO"


def _load_virustotal_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    if "identifier" not in df.columns:
        return set()
    return {str(value) for value in df["identifier"].dropna()}


def _candidate_rows(summary: pd.DataFrame, version: str) -> pd.DataFrame:
    rows = summary[
        (summary["error_type"] == "semantic_error")
        & (summary["bytecode_version"].astype(str) == version)
    ].copy()
    return rows.sample(frac=1, random_state=42).reset_index(drop=True)


def _verify_candidate(row: Any, source: str, version: str) -> dict[str, Any] | None:
    gt_source = fetch_pyllmpatch_source_path(row.file_hash, source)
    gt_pyc, derived_pyc, derived_source = fetch_pyllmpatch_repair_paths(row.file_hash, source)
    paths = {
        "gt_source": gt_source,
        "gt_pyc": gt_pyc,
        "derived_pyc": derived_pyc,
        "derived_source": derived_source,
    }
    if any(path is None or not Path(path).exists() for path in paths.values()):
        return None

    pyc_version = infer_semantic_repair_python_version(gt_pyc, derived_pyc).as_str()
    if pyc_version != version:
        return None

    distance_rows = compare_code_object_distances(gt_pyc, derived_pyc)
    summary = summarize_results(distance_rows)
    combined_distance = int(summary.get("combined_distance") or 0)
    if combined_distance <= 0:
        return None

    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "initial_summary": summary,
        "resolved_python_version": pyc_version,
    }


def build_dataset(
    *,
    summary_path: Path,
    output_path: Path,
    metadata_path: Path,
    versions: list[str],
    samples_per_version: int,
    virustotal_mapping_path: Path,
    require_local_files: bool,
) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path, dtype={"bytecode_version": str, "file_hash": str})
    virustotal_hashes = _load_virustotal_hashes(virustotal_mapping_path.expanduser().resolve())

    output_rows: list[dict[str, str]] = []
    selected_metadata: list[dict[str, Any]] = []
    unavailable_versions: dict[str, str] = {}

    for version in versions:
        candidates = _candidate_rows(summary, version)
        if candidates.empty:
            unavailable_versions[version] = "no semantic_error rows in summary"
            continue

        selected_for_version = 0
        examined = 0
        for row in candidates.itertuples(index=False):
            examined += 1
            file_hash = str(row.file_hash)
            source = _source_for_hash(file_hash, virustotal_hashes)
            if require_local_files:
                verification = _verify_candidate(row, source, version)
                if verification is None:
                    continue
            else:
                verification = {
                    "paths": None,
                    "initial_summary": None,
                    "resolved_python_version": None,
                }

            output_rows.append(
                {
                    "file_hash": file_hash,
                    "error_type": "semantic_error",
                    "error": _empty_if_na(getattr(row, "syntactic_error_word", "")),
                    "error_message": _empty_if_na(getattr(row, "precessed_error_message", "")),
                    "error_description": _empty_if_na(getattr(row, "syntactic_error_description", "")),
                    "bytecode_version": version,
                    "source": source,
                }
            )
            selected_metadata.append(
                {
                    "file_hash": file_hash,
                    "source": source,
                    "bytecode_version": version,
                    "summary_semantic_error_lines": _empty_if_na(getattr(row, "semantic_error_lines", "")),
                    **verification,
                }
            )
            selected_for_version += 1
            if selected_for_version >= samples_per_version:
                break

        if selected_for_version < samples_per_version:
            unavailable_versions[version] = (
                f"selected {selected_for_version}/{samples_per_version} after examining "
                f"{examined} candidate semantic rows"
            )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PYLLMPATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    result = {
        "summary_path": str(summary_path),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "requested_versions": versions,
        "samples_per_version": samples_per_version,
        "require_local_files": require_local_files,
        "row_count": len(output_rows),
        "counts_by_version": pd.Series([row["bytecode_version"] for row in output_rows]).value_counts().sort_index().to_dict()
        if output_rows
        else {},
        "unavailable_versions": unavailable_versions,
        "selected_rows": selected_metadata,
    }
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a small real semantic-repair dataset from available PyLingual summary rows."
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "pylingual" / "pylingual_dataset_summary_pylingual.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "pyllmpatch_semantic_real_test_3_10_3_14.csv",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "pyllmpatch_semantic_real_test_3_10_3_14_metadata.json",
    )
    parser.add_argument("--versions", nargs="*", default=DEFAULT_VERSIONS)
    parser.add_argument("--samples-per-version", type=int, default=3)
    parser.add_argument(
        "--no-require-local-files",
        action="store_true",
        help="Sample rows from the summary without verifying that local .py/.pyc artifacts exist.",
    )
    parser.add_argument(
        "--virustotal-mapping-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "mappings" / "virustotal-malicious.csv",
    )
    args = parser.parse_args()

    result = build_dataset(
        summary_path=args.summary_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        versions=args.versions,
        samples_per_version=max(1, args.samples_per_version),
        virustotal_mapping_path=args.virustotal_mapping_path,
        require_local_files=not args.no_require_local_files,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "selected_rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
