from __future__ import annotations

import pandas as pd

from pipeline.config import BASE_DATASET_PATH
from utils.file_helpers import read_csv_file, fetch_pyllmpatch_source_path


def filter_dataset_rows(
    df: pd.DataFrame,
    *,
    source: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    result = df.copy()
    if source is not None:
        result = result[result["source"].astype(str).str.lower() == source.lower()]
    if limit is not None:
        result = result.head(limit)
    return result


def resolve_syntax_dataset_source_path(row: pd.Series) -> str | None:
    file_hash = str(row.get("file_hash") or "").strip()
    source = str(row.get("source") or "").strip()
    if not file_hash or not source:
        return None
    path = fetch_pyllmpatch_source_path(file_hash, source)
    return str(path) if path is not None else None


def prepare_snippets_for_repair(previous_run_log_path: str = "", only_previously_failed: bool = False):
    if not previous_run_log_path:
        return read_csv_file(f"{BASE_DATASET_PATH}")

    df_previous_log = pd.read_json(previous_run_log_path, lines=True)
    df_balanced = read_csv_file(f"{BASE_DATASET_PATH}")

    df_previous_log["file_hash"] = df_previous_log["file_hash"].astype(str)
    df_balanced["file_hash"] = df_balanced["file_hash"].astype(str)

    if only_previously_failed:
        success_by_file = df_previous_log.groupby("file_hash")["compiled_success"].any()
        failed_hashes = success_by_file[~success_by_file].index
        mask = df_balanced["file_hash"].isin(set(failed_hashes))
    else:
        prev_hashes = df_previous_log["file_hash"].dropna().drop_duplicates()
        mask = ~df_balanced["file_hash"].isin(set(prev_hashes))

    return df_balanced.loc[mask].copy()
