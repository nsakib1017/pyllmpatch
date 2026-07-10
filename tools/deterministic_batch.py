"""Large-scale DETERMINISTIC-only repair pass over ALL resolvable 3.10+ semantic-error
files in the summary (no LLM, no GPU). Runs the real engine with a NullFragmentFixer so
only the deterministic pre-pass + operator seam act, and aggregates how much the mechanical
layer improves the whole 3.10+ corpus: object-level fixes, files improved/perfected,
per-version, and which operators fire.

In-process (each worker imports the pipeline ONCE and processes a chunk) so there is no
per-row subprocess/import overhead; per-row SIGALRM guard; incremental JSON dump.

Env: PER_VERSION_CAP (cap rows/version, default 0 = all), NPROC, ROW_TIMEOUT, HOLDOUT_CSV
(exclude these file_hashes — the hybrid eval set).
Usage: python tools/deterministic_batch.py
"""
import os, sys, csv, json, time, signal, tempfile, shutil
from collections import Counter, defaultdict
from pathlib import Path
from multiprocessing import Pool

os.environ["SEMANTIC_DETERMINISTIC_OPERATORS"] = "true"
os.environ["SEMANTIC_ACCEPTANCE_MODE"] = "oracle"
csv.field_size_limit(sys.maxsize)
R = Path("/home/mxs220189/pylingual_collaboration/pylingual_download/code")
sys.path.insert(0, str(R / "pylingual")); sys.path.insert(0, str(R))

SUMMARY = R / "dataset" / "pylingual" / "pylingual_dataset_summary_pylingual.csv"
OUT = Path("/home/mxs220189/.claude/jobs/c42cbf21/tmp/deterministic_batch_result.json")
NPROC = int(os.getenv("NPROC", "14"))
ROW_TIMEOUT = int(os.getenv("ROW_TIMEOUT", "90"))
PER_VERSION_CAP = int(os.getenv("PER_VERSION_CAP", "0"))  # 0 = all
HOLDOUT_CSV = os.getenv("HOLDOUT_CSV", "")


def _is_310_plus(v: str) -> bool:
    try:
        return tuple(map(int, v.split(".")[:2])) >= (3, 10)
    except Exception:
        return False


def _row_timeout(signum, frame):
    raise TimeoutError("row exceeded budget")


def worker(chunk):
    signal.signal(signal.SIGALRM, _row_timeout)
    from pipeline.code_object_repair_loop import CodeObjectRepairLoop, NullFragmentFixer
    from utils.file_helpers import fetch_pyllmpatch_repair_paths
    from pylingual.equivalence_check import compare_pyc

    b = {"n": 0, "resolved": 0, "improved": 0, "perfect": 0, "obj_fixed": 0,
         "cd_before": 0, "cd_after": 0, "timeout": 0, "error": 0}
    ops = Counter()
    per_ver = defaultdict(lambda: {"n": 0, "improved": 0, "perfect": 0, "obj_fixed": 0})
    fixer = NullFragmentFixer()
    for fh, ver in chunk:
        b["n"] += 1
        out = None
        try:
            signal.alarm(ROW_TIMEOUT)
            gt, der, src = fetch_pyllmpatch_repair_paths(fh, "PyLingual.IO")
            if not (gt and der and src and Path(gt).exists() and Path(der).exists() and Path(src).exists()):
                signal.alarm(0); continue
            b["resolved"] += 1
            pv = per_ver[ver]; pv["n"] += 1
            init_fail = [t for t in compare_pyc(Path(gt), Path(der)) if not t.success]
            n0 = len(init_fail)
            out = tempfile.mkdtemp(prefix="detb_")
            res = CodeObjectRepairLoop(fixer).run(
                gt_pyc=Path(gt), derived_pyc=Path(der), derived_source=Path(src),
                output_dir=Path(out), max_iterations=1,
                sample_hard_timeout_seconds=ROW_TIMEOUT - 5,
            )
            signal.alarm(0)
            ib = (res.get("initial_summary") or {}).get("combined_distance")
            fb = (res.get("final_summary") or {}).get("combined_distance")
            perfect = bool((res.get("pylingual_verification") or {}).get("all_equal"))
            fin_fail = [r for r in (res.get("pylingual_verification") or {}).get("results", []) if not r.get("success")]
            n1 = len(fin_fail)
            if isinstance(ib, int) and isinstance(fb, int):
                b["cd_before"] += ib; b["cd_after"] += fb
                if fb < ib:
                    b["improved"] += 1; pv["improved"] += 1
            fixed = max(0, n0 - n1)
            b["obj_fixed"] += fixed; pv["obj_fixed"] += fixed
            if perfect:
                b["perfect"] += 1; pv["perfect"] += 1
            for s in res.get("steps") or []:
                if s.get("accepted") and "deterministic" in (s.get("repair_operation") or ""):
                    ops[s.get("selected_candidate_strategy") or s.get("repair_operation")] += 1
        except TimeoutError:
            signal.alarm(0); b["timeout"] += 1
        except BaseException:  # noqa: BLE001
            signal.alarm(0); b["error"] += 1
        finally:
            if out:
                shutil.rmtree(out, ignore_errors=True)
    return b, dict(ops), {k: dict(v) for k, v in per_ver.items()}


def main():
    holdout = set()
    if HOLDOUT_CSV and Path(HOLDOUT_CSV).exists():
        holdout = {r["file_hash"] for r in csv.DictReader(open(HOLDOUT_CSV))}
    rows = []
    per_ver_count = Counter()
    with open(SUMMARY, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("error_type") != "semantic_error":
                continue
            v = r.get("bytecode_version", "?")
            if not _is_310_plus(v) or r["file_hash"] in holdout:
                continue
            if PER_VERSION_CAP and per_ver_count[v] >= PER_VERSION_CAP:
                continue
            per_ver_count[v] += 1
            rows.append((r["file_hash"], v))
    print(f"3.10+ semantic rows to process: {len(rows)}  by_ver={dict(sorted(per_ver_count.items()))}", flush=True)
    chunks = [rows[i::NPROC * 6] for i in range(NPROC * 6)]
    tot = Counter(); tot_ops = Counter(); tot_pv = defaultdict(lambda: defaultdict(int))
    t0 = time.time(); done = 0

    def dump(final=False):
        agg = {"chunks_done": done, "chunks_total": len(chunks), "final": final,
               "elapsed_s": round(time.time() - t0), "totals": dict(tot),
               "operators": dict(tot_ops.most_common()),
               "per_version": {k: dict(v) for k, v in tot_pv.items()}}
        OUT.write_text(json.dumps(agg, indent=2))

    with Pool(NPROC) as pool:
        for b, ops, pv in pool.imap_unordered(worker, chunks):
            tot.update(b); tot_ops.update(ops)
            for k, v in pv.items():
                for kk, vv in v.items():
                    tot_pv[k][kk] += vv
            done += 1
            dump()
            print(f"  chunk {done}/{len(chunks)} processed={tot['n']} resolved={tot['resolved']} "
                  f"improved={tot['improved']} perfect={tot['perfect']} obj_fixed={tot['obj_fixed']} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    dump(final=True)
    print(f"DONE processed={tot['n']} resolved={tot['resolved']} improved={tot['improved']} "
          f"perfect={tot['perfect']} obj_fixed={tot['obj_fixed']} ops={dict(tot_ops)}", flush=True)


if __name__ == "__main__":
    main()
