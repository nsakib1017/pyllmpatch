"""Summarize a semantic-repair *loop* run (post-repair), for milestone A/Bs.

Unlike tools/benchmark_semantic_repair.py (which measures the as-decompiled
state), this reads a completed dataset-mode run directory and reports what the
loop actually achieved: post-repair perfect-decompilation rate, object pass
rate, regime breakdown of the remaining failures, accepted steps, and LLM-call
counts. Each per-row result.json carries `pylingual_verification` (final),
`target_python_version`, `llm_calls`, and `steps`.

Usage:
  python tools/summarize_repair_run.py results/semantic_repair_baseline_run --label distance
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.oracle_state import classify_regime  # noqa: E402


def _version_of(result: dict) -> str:
    v = result.get("target_python_version")
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return f"{v[0]}.{v[1]}"
    return str(v)


def summarize_run(run_dir: Path) -> dict:
    result_files = sorted(glob.glob(str(run_dir / "semantic_repair" / "*" / "*" / "result.json")))
    records = []
    overall_regimes: Counter = Counter()
    by_version: dict = defaultdict(lambda: {"rows": 0, "perfect": 0, "objs": 0, "pass": 0})
    total_llm_calls = 0
    total_accepted = 0

    for rf in result_files:
        try:
            d = json.load(open(rf))
        except Exception:
            continue
        verification = d.get("pylingual_verification") or {}
        results = verification.get("results", [])
        perfect = bool(verification.get("all_equal"))
        n_obj = len(results)
        n_pass = sum(1 for r in results if r.get("success"))
        ver = _version_of(d)
        regimes = Counter(classify_regime(r) for r in results if not r.get("success"))
        overall_regimes.update(regimes)
        llm_calls = len(d.get("llm_calls", []) or [])
        accepted = sum(1 for s in (d.get("steps") or []) if s.get("accepted"))
        total_llm_calls += llm_calls
        total_accepted += accepted

        bv = by_version[ver]
        bv["rows"] += 1
        bv["perfect"] += int(perfect)
        bv["objs"] += n_obj
        bv["pass"] += n_pass

        fs = d.get("final_summary") or {}
        is_ = d.get("initial_summary") or {}
        records.append({
            "hash": Path(rf).parent.name[:10],
            "version": ver,
            "perfect": perfect,
            "n_objects": n_obj,
            "n_pass": n_pass,
            "initial_combined": is_.get("combined_distance"),
            "final_combined": fs.get("combined_distance"),
            "accepted_steps": accepted,
            "llm_calls": llm_calls,
            "regimes": dict(regimes),
        })

    total_rows = len(records)
    total_perfect = sum(1 for r in records if r["perfect"])
    total_objs = sum(r["n_objects"] for r in records)
    total_pass = sum(r["n_pass"] for r in records)
    return {
        "run_dir": str(run_dir),
        "rows_with_result": total_rows,
        "perfect_rows": total_perfect,
        "perfect_rate": (total_perfect / total_rows) if total_rows else 0.0,
        "object_pass_rate": (total_pass / total_objs) if total_objs else 0.0,
        "total_objects": total_objs,
        "total_pass": total_pass,
        "total_accepted_steps": total_accepted,
        "total_llm_calls": total_llm_calls,
        "by_version": {k: dict(v) for k, v in sorted(by_version.items())},
        "regimes": dict(overall_regimes.most_common()),
        "records": records,
    }


def print_summary(rep: dict, label: str) -> None:
    print("\n" + "=" * 72)
    print(f"  REPAIR-RUN SUMMARY  [{label}]   {rep['run_dir']}")
    print("=" * 72)
    print(f"  post-repair perfect rate : {rep['perfect_rows']}/{rep['rows_with_result']}"
          f"  ({rep['perfect_rate']*100:.1f}%)")
    print(f"  object pass rate         : {rep['total_pass']}/{rep['total_objects']}"
          f"  ({rep['object_pass_rate']*100:.1f}%)")
    print(f"  accepted steps           : {rep['total_accepted_steps']}")
    print(f"  LLM calls                : {rep['total_llm_calls']}")
    print("\n  per version:")
    for ver, v in rep["by_version"].items():
        print(f"    {ver:5} perfect {v['perfect']}/{v['rows']:<3}  objs {v['pass']}/{v['objs']}")
    print("\n  remaining failing-object regimes:")
    for regime, count in rep["regimes"].items():
        print(f"    {count:>4}  {regime}")
    print("\n  per-row (initial->final combined, perfect):")
    for r in sorted(rep["records"], key=lambda x: (x["version"], x["hash"])):
        p = "YES" if r["perfect"] else "no"
        print(f"    {r['version']:5} {r['hash']:11} {str(r['initial_combined']):>7}->{str(r['final_combined']):<6}"
              f"  obj {r['n_pass']}/{r['n_objects']:<3} acc={r['accepted_steps']} llm={r['llm_calls']}  {p}")
    print("=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--label", default="run")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    rep = summarize_run(args.run_dir)
    print_summary(rep, args.label)
    out = args.json_out or (REPO_ROOT / "results" / "semantic_repair_benchmark" / f"run_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
