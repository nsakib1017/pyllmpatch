"""Cross-shard result aggregation for the parallel scale run (speedup step 4).

Each worker writes result.json trees under its own OUTDIR; this globs every shard's results,
dedups by file_hash (preferring the perfect result), and reports the file-perfect headline +
per-version breakdown. Pure functions here are unit-tested; `load_shard_results` does the glob.
"""
from __future__ import annotations

import glob
import json
import sys


def file_hash_of(result: dict) -> str:
    """file_hash is the parent dir of the gt pyc path (…/<file_hash>/gt.pyc)."""
    return (result.get("gt_pyc") or "").split("/")[-2] if result.get("gt_pyc") else ""


def is_perfect(result: dict) -> bool:
    return bool((result.get("pylingual_verification") or {}).get("all_equal") is True)


def merge_results(pairs) -> dict:
    """Fold (file_hash, result) pairs into a dict, preferring a perfect result over a
    non-perfect one for the same hash (a resumed/rerun row can appear twice)."""
    merged: dict[str, dict] = {}
    for fh, result in pairs:
        if not fh:
            continue
        if fh not in merged or (is_perfect(result) and not is_perfect(merged[fh])):
            merged[fh] = result
    return merged


def summarize(results_by_hash: dict) -> dict:
    per_version: dict[str, dict] = {}
    file_perfect = 0
    for result in results_by_hash.values():
        ver = str(result.get("target_python_version") or "?")
        pv = per_version.setdefault(ver, {"total": 0, "perfect": 0})
        pv["total"] += 1
        if is_perfect(result):
            pv["perfect"] += 1
            file_perfect += 1
    return {
        "total": len(results_by_hash),
        "file_perfect": file_perfect,
        "per_version": per_version,
    }


def load_shard_results(base_dir: str) -> dict:
    """Glob every result.json under base_dir (all shard OUTDIRs live inside it) and merge."""
    def _pairs():
        for path in glob.glob(f"{base_dir}/**/result.json", recursive=True):
            try:
                result = json.load(open(path))
            except Exception:
                continue
            yield file_hash_of(result), result
    return merge_results(_pairs())


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m tools.aggregate_shards <base_dir>", file=sys.stderr)
        return 2
    results = load_shard_results(argv[1])
    s = summarize(results)
    n = s["total"] or 1
    print(f"FILE-PERFECT: {s['file_perfect']}/{s['total']} ({100 * s['file_perfect'] / n:.1f}%)")
    for ver in sorted(s["per_version"]):
        pv = s["per_version"][ver]
        d = pv["total"] or 1
        print(f"  {ver}: {pv['perfect']}/{pv['total']} ({100 * pv['perfect'] / d:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
