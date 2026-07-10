"""Semantic-repair benchmark harness (M1).

Runs the PyLingual equivalence oracle over a dataset of semantic_error rows and
reports the *perfect-decompilation rate* — the fraction of files where every
code object passes the oracle (`compare_pyc` -> all success) — plus a per-object
pass rate and a regime breakdown of the remaining failures.

This measures the AS-DECOMPILED state (derived .pyc vs ground-truth .pyc); it
does NOT run any repair. It is the frozen baseline view that gates every later
milestone. Re-run after each change and diff the JSON.

Usage:
  python tools/benchmark_semantic_repair.py \
      --dataset dataset/pyllmpatch_semantic_summary_test_3_10_3_14.csv \
      --label baseline \
      --json-out results/semantic_repair_benchmark/baseline.json

The oracle is version-agnostic (xdis reads any .pyc), so this works for all
versions regardless of the host interpreter — no recompilation involved.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# `pylingual` must resolve to the INNER package (REPO_ROOT/pylingual/pylingual),
# so put REPO_ROOT/pylingual ahead of REPO_ROOT — otherwise `import pylingual`
# binds to the outer namespace dir. (Mirrors pyc_code_object_distance.py:20.)
for p in (str(REPO_ROOT), str(REPO_ROOT / "pylingual")):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pylingual"))

from utils.file_helpers import fetch_pyllmpatch_repair_paths  # noqa: E402
from utils.reattach_source_code_object import run_pylingual_verification  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "dataset" / "pyllmpatch_semantic_summary_test_3_10_3_14.csv"


def classify_regime(result: dict) -> str:
    """Map a failing oracle result to its repair regime (the routing signal)."""
    message = result.get("message")
    if message == "Different bytecode":
        # failed_offset None ==> exception-table-only divergence (checked before
        # the instruction loop in compare_bytecode) -> structural, not leaf.
        return "Different bytecode (offset)" if result.get("failed_offset") is not None \
            else "Different bytecode (no-offset / exc-table)"
    return message or "Unknown"


def benchmark_row(file_hash: str, source: str) -> dict:
    """Resolve one row and run the oracle. Returns a structured record."""
    gt, der, src = fetch_pyllmpatch_repair_paths(file_hash, source)
    rec: dict = {
        "file_hash": file_hash,
        "source": source,
        "gt_pyc": str(gt) if gt else None,
        "derived_pyc": str(der) if der else None,
        "derived_source": str(src) if src else None,
        "resolved": bool(gt and der and src and Path(gt).exists()
                         and Path(der).exists() and Path(src).exists()),
    }
    if not rec["resolved"]:
        rec["error"] = "unresolved artifacts"
        return rec
    try:
        verification = run_pylingual_verification(Path(gt), Path(der))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        return rec

    results = verification["results"]
    n_objects = len(results)
    n_success = sum(1 for r in results if r["success"])
    regimes = Counter(classify_regime(r) for r in results if not r["success"])
    rec.update({
        "n_objects": n_objects,
        "n_success": n_success,
        "perfect": bool(verification["all_equal"]),
        "regimes": dict(regimes),
        "failing_objects": [
            {
                "names": r.get("names"),
                "message": r.get("message"),
                "failed_offset": r.get("failed_offset"),
                "failed_line_number": r.get("failed_line_number"),
                "regime": classify_regime(r),
            }
            for r in results if not r["success"]
        ],
    })
    return rec


def run_benchmark(dataset_path: Path) -> dict:
    with open(dataset_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("file_hash")]

    records = []
    for r in rows:
        rec = benchmark_row(r["file_hash"], r.get("source", ""))
        rec["bytecode_version"] = r.get("bytecode_version")
        records.append(rec)

    # ---- aggregate ----
    by_version: dict = defaultdict(lambda: {"rows": 0, "perfect": 0,
                                            "n_objects": 0, "n_success": 0,
                                            "errors": 0})
    overall_regimes: Counter = Counter()
    total_rows = len(records)
    total_perfect = 0
    total_objects = 0
    total_success = 0
    total_errors = 0

    for rec in records:
        ver = rec.get("bytecode_version") or "?"
        bv = by_version[ver]
        bv["rows"] += 1
        if rec.get("error"):
            bv["errors"] += 1
            total_errors += 1
            continue
        if rec.get("perfect"):
            bv["perfect"] += 1
            total_perfect += 1
        bv["n_objects"] += rec.get("n_objects", 0)
        bv["n_success"] += rec.get("n_success", 0)
        total_objects += rec.get("n_objects", 0)
        total_success += rec.get("n_success", 0)
        overall_regimes.update(rec.get("regimes", {}))

    return {
        "dataset": str(dataset_path),
        "total_rows": total_rows,
        "perfect_rows": total_perfect,
        "perfect_rate": (total_perfect / total_rows) if total_rows else 0.0,
        "object_pass_rate": (total_success / total_objects) if total_objects else 0.0,
        "total_objects": total_objects,
        "total_success": total_success,
        "errors": total_errors,
        "by_version": {k: dict(v) for k, v in sorted(by_version.items())},
        "regimes": dict(overall_regimes.most_common()),
        "records": records,
    }


def print_scoreboard(report: dict, label: str) -> None:
    print("\n" + "=" * 72)
    print(f"  SEMANTIC-REPAIR BENCHMARK  [{label}]")
    print(f"  dataset: {report['dataset']}")
    print("=" * 72)
    print(f"  perfect-decompilation rate : {report['perfect_rows']}/{report['total_rows']}"
          f"  ({report['perfect_rate']*100:.1f}%)")
    print(f"  object pass rate           : {report['total_success']}/{report['total_objects']}"
          f"  ({report['object_pass_rate']*100:.1f}%)")
    if report["errors"]:
        print(f"  rows with errors           : {report['errors']}")
    print("\n  per version:")
    print(f"    {'ver':5} {'perfect':>9} {'objs pass':>12} {'err':>4}")
    print("    " + "-" * 32)
    for ver, v in report["by_version"].items():
        objs = f"{v['n_success']}/{v['n_objects']}" if v["n_objects"] else "-"
        print(f"    {ver:5} {v['perfect']:>3}/{v['rows']:<5} {objs:>12} {v['errors']:>4}")
    print("\n  failing-object regime breakdown:")
    if report["regimes"]:
        for regime, count in report["regimes"].items():
            print(f"    {count:>4}  {regime}")
    else:
        print("    (none)")
    # per-row detail
    print("\n  per-row:")
    print(f"    {'ver':5} {'hash':10} {'objs':>8} {'perfect':>8}  regimes")
    print("    " + "-" * 60)
    for rec in report["records"]:
        ver = rec.get("bytecode_version") or "?"
        h = rec["file_hash"][:8]
        if rec.get("error"):
            print(f"    {ver:5} {h:10} {'ERR':>8} {'-':>8}  {rec['error']}")
            continue
        objs = f"{rec['n_success']}/{rec['n_objects']}"
        perfect = "YES" if rec["perfect"] else "no"
        regimes = ", ".join(f"{k}×{v}" for k, v in rec.get("regimes", {}).items())
        print(f"    {ver:5} {h:10} {objs:>8} {perfect:>8}  {regimes}")
    print("=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--label", default="baseline",
                    help="label for this run (used in the JSON filename if --json-out omitted)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    report = run_benchmark(args.dataset)
    print_scoreboard(report, args.label)

    json_out = args.json_out or (
        REPO_ROOT / "results" / "semantic_repair_benchmark" / f"{args.label}.json"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  wrote {json_out}\n")


if __name__ == "__main__":
    main()
