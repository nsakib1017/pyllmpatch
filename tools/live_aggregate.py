"""Live progress aggregator for the multi-day winnable-subset run.

Loops: glob every shard's result.json under BASE_DIR, dedup by file_hash, and atomically write a
live_progress.json (safe to read anytime) with overall + per-version + per-verdict progress, the
file-perfect rate, throughput and ETA. Also writes summary.json (the durable final aggregate, in
the same shape as prior experiment outputs). Stops when a STOP sentinel appears, when done==total,
or when killed.

Usage:
  python tools/live_aggregate.py --base results/semantic_repair_winnable_final \
      --csv <winnable_run.csv> --fixability <fixability_full.json> --interval 60
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path


def _load_targets(csv_path):
    """file_hash -> version from the run CSV (the denominator)."""
    out = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            out[r["file_hash"]] = r.get("bytecode_version", "?")
    return out


def _load_verdicts(fixability_path):
    """file_hash -> classifier file_verdict (for per-verdict progress), best-effort."""
    if not fixability_path or not os.path.exists(fixability_path):
        return {}
    try:
        pf = json.load(open(fixability_path)).get("per_file", [])
        return {r["file_hash"]: r.get("file_verdict") for r in pf if r.get("file_hash")}
    except Exception:
        return {}


def _scan(base):
    """Latest result.json per file_hash across all shards -> (ae, improved, ver)."""
    latest = {}
    for p in glob.glob(f"{base}/**/result.json", recursive=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        # Key off the result DIR name (the <hash>_<ver> case_id), NOT gt_pyc — the pipeline
        # resolves the adapter symlink to the raw versionless hash, which would collide across
        # versions and undercount `done`.
        fh = Path(p).parent.name
        if not fh:
            continue
        ae = (d.get("pylingual_verification") or {}).get("all_equal") is True
        ib = (d.get("initial_summary") or {}).get("combined_distance")
        fb = (d.get("final_summary") or {}).get("combined_distance")
        improved = isinstance(ib, int) and isinstance(fb, int) and fb < ib
        prev = latest.get(fh)
        # prefer a perfect result if the same file was retried
        if prev is None or (ae and not prev[0]):
            latest[fh] = (ae, improved, d.get("target_python_version") or "?")
    return latest


def _atomic_write(path, obj):
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def build_snapshot(base, targets, verdicts, t0):
    latest = _scan(base)
    total = len(targets)
    done = len(latest)
    perfect = sum(1 for v in latest.values() if v[0])
    improved = sum(1 for v in latest.values() if v[1])
    per_ver = defaultdict(lambda: {"total": 0, "done": 0, "perfect": 0})
    for fh, ver in targets.items():
        per_ver[ver]["total"] += 1
    for fh, (ae, _, ver) in latest.items():
        tv = targets.get(fh, ver)
        per_ver[tv]["done"] += 1
        if ae:
            per_ver[tv]["perfect"] += 1
    per_verdict = defaultdict(lambda: {"done": 0, "perfect": 0})
    for fh, (ae, _, _) in latest.items():
        vd = verdicts.get(fh, "unknown")
        per_verdict[vd]["done"] += 1
        if ae:
            per_verdict[vd]["perfect"] += 1
    elapsed = max(1.0, time.time() - t0)
    rate_hr = done / (elapsed / 3600.0)
    remaining = max(0, total - done)
    eta_hr = (remaining / rate_hr) if rate_hr > 0 else None
    return {
        "updated_unix": int(time.time()),
        "elapsed_hours": round(elapsed / 3600.0, 2),
        "total": total,
        "done": done,
        "remaining": remaining,
        "file_perfect": perfect,
        "file_perfect_rate_of_done": round(perfect / done, 4) if done else 0,
        "file_perfect_rate_of_total": round(perfect / total, 4) if total else 0,
        "improved_not_perfect": improved - perfect if improved >= perfect else improved,
        "throughput_files_per_hour": round(rate_hr, 1),
        "eta_hours": round(eta_hr, 1) if eta_hr is not None else None,
        "eta_days": round(eta_hr / 24, 2) if eta_hr is not None else None,
        "per_version": {k: dict(v) for k, v in sorted(per_ver.items())},
        "per_verdict": {k: dict(v) for k, v in sorted(per_verdict.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fixability", default=None)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--t0", type=float, default=None, help="run start unix time (default: now)")
    ap.add_argument("--once", action="store_true", help="write one snapshot and exit")
    args = ap.parse_args()

    targets = _load_targets(args.csv)
    verdicts = _load_verdicts(args.fixability)
    t0 = args.t0 or time.time()
    base = args.base
    live_path = f"{base}/live_progress.json"
    summary_path = f"{base}/summary.json"
    stop_sentinel = f"{base}/STOP_AGGREGATE"
    Path(base).mkdir(parents=True, exist_ok=True)

    while True:
        snap = build_snapshot(base, targets, verdicts, t0)
        _atomic_write(live_path, snap)
        _atomic_write(summary_path, snap)  # durable final aggregate (same shape)
        print(f"[live] done={snap['done']}/{snap['total']} perfect={snap['file_perfect']} "
              f"({100*snap['file_perfect_rate_of_total']:.1f}% of total) "
              f"{snap['throughput_files_per_hour']}/hr eta={snap['eta_days']}d", flush=True)
        if args.once or snap["done"] >= snap["total"] or os.path.exists(stop_sentinel):
            break
        # interruptible sleep
        for _ in range(args.interval):
            if os.path.exists(stop_sentinel):
                break
            time.sleep(1)
    print("[live] aggregator exiting", flush=True)


if __name__ == "__main__":
    main()
