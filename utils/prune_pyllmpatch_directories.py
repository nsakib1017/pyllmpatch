#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


@dataclass(frozen=True)
class RootSpec:
    name: str
    path: Path


def load_dataset(dataset_csv: Path) -> pd.DataFrame:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    df = pd.read_csv(dataset_csv, dtype={"file_hash": str, "source": str})
    required = {"file_hash", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset CSV is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["file_hash", "source"]).copy()
    df["file_hash"] = df["file_hash"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df = df[(df["file_hash"] != "") & (df["source"] != "")]
    return df


def resolve_root(path_value: str | None, label: str) -> Path:
    if not path_value:
        raise ValueError(f"Missing required environment variable for {label}")
    return Path(path_value).expanduser().resolve()


def build_root_specs() -> dict[str, RootSpec]:
    root_for_files = resolve_root(os.getenv("ROOT_FOR_FILES"), "ROOT_FOR_FILES")
    pypi_name = os.getenv("BASE_DIR_PYTHON_FILES_PYPI")
    pylingual_name = os.getenv("BASE_DIR_PYTHON_FILES_PYLINGUAL")

    if not pypi_name:
        raise ValueError("Missing required environment variable BASE_DIR_PYTHON_FILES_PYPI")
    if not pylingual_name:
        raise ValueError("Missing required environment variable BASE_DIR_PYTHON_FILES_PYLINGUAL")

    return {
        "PyPi": RootSpec("PyPi", (root_for_files / pypi_name).resolve()),
        "PyLingual.IO": RootSpec("PyLingual.IO", (root_for_files / pylingual_name).resolve()),
        "VirusTotal": RootSpec("VirusTotal", (root_for_files / pylingual_name).resolve()),
    }


def collect_keep_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    keep: dict[str, set[str]] = {
        "PyPi": set(),
        "shared_pylingual": set(),
    }

    for source, group in df.groupby("source"):
        hashes = set(group["file_hash"].tolist())
        if source == "PyPi":
            keep["PyPi"].update(hashes)
        elif source in {"PyLingual.IO", "VirusTotal"}:
            keep["shared_pylingual"].update(hashes)

    return keep


def direct_child_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def prune_root(root: Path, keep_hashes: set[str], *, dry_run: bool) -> tuple[list[Path], list[Path]]:
    kept: list[Path] = []
    removed: list[Path] = []

    for child in direct_child_dirs(root):
        if child.name in keep_hashes:
            kept.append(child)
            continue

        removed.append(child)
        if not dry_run:
            shutil.rmtree(child)

    return kept, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove unlisted hash subdirectories from the PyPi and PyLingual roots using pyllmpatch_dataset.csv."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help="Path to dataset/pyllmpatch_dataset.csv. Defaults to PROJECT_ROOT_DIR/dataset/pyllmpatch_dataset.csv.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories. Without this flag the script performs a dry run.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    project_root = resolve_root(os.getenv("PROJECT_ROOT_DIR"), "PROJECT_ROOT_DIR")
    dataset_csv = args.dataset_csv or (project_root / "dataset" / "pyllmpatch_dataset.csv")
    dry_run = not args.apply

    df = load_dataset(dataset_csv)
    root_specs = build_root_specs()
    keep_sets = collect_keep_sets(df)

    print(f"Dataset: {dataset_csv}")
    print(f"Rows: {len(df)}")
    print(f"Mode: {'dry-run' if dry_run else 'apply'}")
    print()

    total_removed = 0

    pypi_root = root_specs["PyPi"]
    kept, removed = prune_root(pypi_root.path, keep_sets["PyPi"], dry_run=dry_run)

    print(f"[PyPi] root: {pypi_root.path}")
    print(f"  keep: {len(kept)}")
    print(f"  remove: {len(removed)}")

    if removed:
        preview = ", ".join(p.name for p in removed[:10])
        suffix = " ..." if len(removed) > 10 else ""
        print(f"  preview removals: {preview}{suffix}")

    total_removed += len(removed)
    print()

    shared_root = root_specs["PyLingual.IO"]
    kept, removed = prune_root(shared_root.path, keep_sets["shared_pylingual"], dry_run=dry_run)

    print(f"[PyLingual.IO + VirusTotal] root: {shared_root.path}")
    print(f"  keep: {len(kept)}")
    print(f"  remove: {len(removed)}")

    if removed:
        preview = ", ".join(p.name for p in removed[:10])
        suffix = " ..." if len(removed) > 10 else ""
        print(f"  preview removals: {preview}{suffix}")

    total_removed += len(removed)
    print()

    if dry_run:
        print("Dry run complete. Re-run with --apply to delete the listed directories.")
    else:
        print(f"Deletion complete. Removed {total_removed} directories.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
