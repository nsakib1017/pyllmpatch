#!/usr/bin/env python3
# remove_by_extension.py
# Recursively delete files with a given extension on Linux/WSL.

import argparse
import os
from pathlib import Path

def normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        raise ValueError("Extension cannot be empty.")
    # Accept "tmp" or ".tmp" => ".tmp"
    if not ext.startswith("."):
        ext = "." + ext
    return ext

def matches(name: str, ext: str, case_insensitive: bool) -> bool:
    if case_insensitive:
        return name.lower().endswith(ext.lower())
    return name.endswith(ext)

def main():
    p = argparse.ArgumentParser(
        description="Traverse a directory and remove all files with the given extension."
    )
    p.add_argument("root", nargs="?", default=".", help="Root directory to scan (default: .)")
    p.add_argument("--ext", required=True,
                   help="Target extension (e.g., 'tmp' or '.tmp' or 'Zone.Identifier').")
    p.add_argument("--dry-run", "-n", action="store_true",
                   help="Show what would be deleted without removing.")
    p.add_argument("--case-insensitive", "-i", action="store_true",
                   help="Match extension case-insensitively.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    ext = normalize_ext(args.ext)

    total = 0
    deleted = 0
    errors = 0
    would_delete = 0

    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            total += 1
            fpath = Path(dirpath, fname)
            if matches(fname, ext, args.case_insensitive):
                if args.dry_run:
                    print(f"[DRY] Would delete: {fpath}")
                    would_delete += 1
                else:
                    try:
                        fpath.unlink()
                        print(f"[OK ] Deleted: {fpath}")
                        deleted += 1
                    except Exception as e:
                        print(f"[ERR] Could not delete {fpath}: {e}")
                        errors += 1

    print("\n=== Summary ===")
    print(f"Scanned files: {total}")
    if args.dry_run:
        print(f"Would delete:  {would_delete} file(s) with extension '{ext}'")
    else:
        print(f"Deleted:       {deleted}")
    print(f"Errors:        {errors}")

if __name__ == "__main__":
    main()
