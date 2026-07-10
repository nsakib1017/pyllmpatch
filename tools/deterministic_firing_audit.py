"""EMPIRICAL deterministic-FIRING audit (companion to tests/test_deterministic_dispatch_coverage.py).

The dispatch-coverage test proves every operator is WIRED. This tool proves they actually
FIRE on real matching objects and quantifies the realization rate — the user's ask: "make
sure the deterministic layer's fixes/logics are indeed firing for the matching objects and
are not missed."

For a labeled sample of 3.10+ semantic-error rows it runs the REAL engine with a
NullFragmentFixer (no LLM, no GPU) so ONLY the deterministic prepass can act, then for every
initially-failing code object records:
  * regime   -- leaf (Different bytecode + offset) / control_flow (Different control flow)
                / diffbc (Different bytecode, no offset); exactly the prepass dispatch's own
                routing, so "regime" == "an object the deterministic layer is asked to handle".
  * fired    -- a deterministic operator produced an ACCEPTED edit for this object (joined to
                the accepted step by qualname), and which operator.
  * equal    -- the object reached Equal after the prepass.
  * deferred -- no operator fired (SAFE: the operator gates declined an ambiguous case and
                the object is routed to the LLM; NOT a bug).

Output: per-regime {objects, fired, equal, deferred}, per-operator firing counts, and a
sanity flag for any regime that saw matchable objects but ZERO operator firings (a real miss
worth investigating). CPU-only, parallel, per-row SIGALRM. Usage:
  python tools/deterministic_firing_audit.py <dataset.csv> --source PyLingual.IO \
     --sample 200 --workers 8 --out firing_audit.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual"))
sys.path.insert(0, str(REPO))
csv.field_size_limit(sys.maxsize)

# deterministic prepass must be ON (it is default-on, but pin it so the audit is unambiguous)
os.environ.setdefault("SEMANTIC_DETERMINISTIC_OPERATORS", "1")

ROW_TIMEOUT = 90


def _regime(message, failed_offset) -> str:
    if message == "Different control flow":
        return "control_flow"
    if message == "Different bytecode":
        return "leaf" if failed_offset is not None else "diffbc_no_offset"
    if message in ("Missing bytecode", "Extra bytecode"):
        return "missing_extra"
    return f"other:{message}"


def _row_timeout(signum, frame):
    raise TimeoutError()


def worker(chunk):
    signal.signal(signal.SIGALRM, _row_timeout)
    from pipeline.code_object_repair_loop import CodeObjectRepairLoop, NullFragmentFixer
    from utils.file_helpers import fetch_pyllmpatch_repair_paths
    from pylingual.equivalence_check import compare_pyc

    # per-regime object tallies and per-operator firing counts
    regime = defaultdict(lambda: {"objects": 0, "fired": 0, "equal": 0, "deferred": 0})
    ops = Counter()
    fired_regime = defaultdict(Counter)   # operator -> regime -> count
    b = {"rows": 0, "resolved": 0, "timeout": 0, "error": 0, "rows_with_firing": 0}
    fixer = NullFragmentFixer()

    for fh, ver, source in chunk:
        b["rows"] += 1
        out = None
        try:
            signal.alarm(ROW_TIMEOUT)
            gt, der, src = fetch_pyllmpatch_repair_paths(fh, source)
            if not (gt and der and src and Path(gt).exists() and Path(der).exists() and Path(src).exists()):
                signal.alarm(0)
                continue
            b["resolved"] += 1
            # regime of every initially-failing object (the "matching set")
            init = {t.names(): _regime(t.message, t.failed_offset)
                    for t in compare_pyc(Path(gt), Path(der)) if not t.success}
            out = tempfile.mkdtemp(prefix="detfire_")
            res = CodeObjectRepairLoop(fixer).run(
                gt_pyc=Path(gt), derived_pyc=Path(der), derived_source=Path(src),
                output_dir=Path(out), max_iterations=1,
                sample_hard_timeout_seconds=ROW_TIMEOUT - 5,
            )
            signal.alarm(0)
            # which objects a deterministic operator fired on, and which operator
            fired_by_qual = {}
            for s in res.get("steps") or []:
                if s.get("accepted") and "deterministic" in (s.get("repair_operation") or ""):
                    q = s.get("qualname")
                    opname = s.get("selected_candidate_strategy") or s.get("repair_operation")
                    fired_by_qual.setdefault(q, opname)
                    ops[opname] += 1
            if fired_by_qual:
                b["rows_with_firing"] += 1
            # which initially-failing objects reached Equal
            final_fail = {r.get("names") for r in
                          (res.get("pylingual_verification") or {}).get("results", [])
                          if not r.get("success")}
            for qual, reg in init.items():
                r = regime[reg]
                r["objects"] += 1
                if qual in fired_by_qual:
                    r["fired"] += 1
                    fired_regime[fired_by_qual[qual]][reg] += 1
                else:
                    r["deferred"] += 1
                if qual not in final_fail:
                    r["equal"] += 1
        except TimeoutError:
            signal.alarm(0)
            b["timeout"] += 1
        except BaseException:  # noqa: BLE001
            signal.alarm(0)
            b["error"] += 1
        finally:
            if out:
                shutil.rmtree(out, ignore_errors=True)

    return b, {k: dict(v) for k, v in regime.items()}, dict(ops), \
        {k: dict(v) for k, v in fired_regime.items()}


def _vt(s):
    try:
        return tuple(int(x) for x in str(s).split(".")[:2])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--source", default="PyLingual.IO")
    ap.add_argument("--min-version", default="3.10")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    mv = _vt(a.min_version)

    def keep(r):
        if (r.get("error_type") or "").strip() != "semantic_error":
            return False
        t = _vt(r.get("bytecode_version") or "")
        return not (mv and (t is None or t < mv))

    rows = []
    for r in csv.DictReader(open(a.dataset)):
        if not keep(r):
            continue
        fh = r.get("file_hash") or r.get("hash")
        if not fh:
            continue
        rows.append((fh, r.get("bytecode_version") or "?", r.get("source") or a.source))
        if len(rows) >= a.sample:
            break
    print(f"rows={len(rows)} workers={a.workers}", flush=True)

    # chunk across workers
    import multiprocessing as mp
    nw = max(1, a.workers)
    chunks = [rows[i::nw] for i in range(nw)]
    chunks = [c for c in chunks if c]

    tot = Counter()
    regime = defaultdict(lambda: Counter())
    ops = Counter()
    fired_regime = defaultdict(Counter)
    with mp.Pool(len(chunks)) as pool:
        for b, reg, op, fr in pool.imap_unordered(worker, chunks):
            for k, v in b.items():
                tot[k] += v
            for rname, d in reg.items():
                for k, v in d.items():
                    regime[rname][k] += v
            for k, v in op.items():
                ops[k] += v
            for opn, d in fr.items():
                for rn, v in d.items():
                    fired_regime[opn][rn] += v

    # sanity: regimes with matchable objects but zero firing
    zero_fire = [rn for rn, d in regime.items()
                 if d.get("objects", 0) > 0 and d.get("fired", 0) == 0
                 and rn in ("leaf", "control_flow", "diffbc_no_offset")]

    result = {
        "totals": dict(tot),
        "per_regime": {k: dict(v) for k, v in regime.items()},
        "operator_firings": dict(ops.most_common()),
        "operator_by_regime": {k: dict(v) for k, v in fired_regime.items()},
        "regimes_matchable_but_zero_firing": zero_fire,
    }
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    print("\n=== FIRING SUMMARY (per regime: objects | fired | equal | deferred) ===")
    for rn in ("leaf", "control_flow", "diffbc_no_offset", "missing_extra"):
        d = regime.get(rn)
        if not d:
            continue
        o = d.get("objects", 0)
        fired = d.get("fired", 0)
        print(f"  {rn:18s} objects={o:5d}  fired={fired:5d} ({fired/max(o,1)*100:4.1f}%)  "
              f"equal={d.get('equal',0):5d}  deferred(->LLM)={d.get('deferred',0):5d}")
    if zero_fire:
        print(f"\n!! REGIMES WITH MATCHABLE OBJECTS BUT ZERO OPERATOR FIRINGS (investigate): {zero_fire}")
    else:
        print("\nOK: every deterministic regime with matchable objects saw operator firings.")


if __name__ == "__main__":
    main()
