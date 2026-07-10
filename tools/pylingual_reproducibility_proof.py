"""TIER 3 — PyLingual reproducibility proof. For each hybrid100 residual (non-perfect) file,
re-decompile the GROUND-TRUTH .pyc with the PyLingual CLI, recompile that fresh output, and
compare to GT. If PyLingual reproduces the SAME failing objects as the original derived
source, the failure is PROVABLY a fundamental PyLingual decompilation limitation (deterministic
mis-decompilation — repair can't help, the fix must be in PyLingual). Otherwise it's not
fundamental (fluke/stale/non-determinism). Aggregates a per-construct decompiler-bug catalog.

Verdict per originally-failing object:
  * pylingual_bound   -- fresh re-decompile fails the SAME object the same way (reproduced).
  * not_fundamental   -- fresh re-decompile makes that object Equal (repairable in principle).
  * fresh_diverged    -- fresh fails but differently (decoder non-determinism / different bug).
GPU-based (CLI ~11s/file), SEQUENTIAL. Usage:
  python tools/pylingual_reproducibility_proof.py <hybrid_runs_dir> [--limit N] [--out out.json]
"""
from __future__ import annotations
import argparse, json, glob, subprocess, tempfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual")); sys.path.insert(0, str(REPO))
from pylingual.equivalence_check import compare_pyc  # noqa
from utils.reattach_source_code_object import compile_source_to_pyc  # noqa
from utils.pyc_code_object_distance import compare_code_object_distances, _iter_matched_code_objects, prepare_pyc  # noqa

CLI = "/home/mxs220189/.local/bin/pylingual"


def _construct_tags(gt_bc, der_bc):
    """Coarse construct tag for a matched failing object (reuse the audit's opcode markers)."""
    if not gt_bc or not der_bc:
        return ["missing_or_extra_object"]
    gi = [s[0] for s in gt_bc["instruction_signatures"]]
    di = [s[0] for s in der_bc["instruction_signatures"]]
    tags = set()
    for t, i1, i2, j1, j2 in SequenceMatcher(a=gi, b=di, autojunk=False).get_opcodes():
        if t == "equal":
            continue
        s = set(gi[i1:i2]) | set(di[j1:j2])
        if s & {"FOR_ITER", "GET_ITER", "SEND"}: tags.add("loop")
        if s & {"POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_FALSE"}: tags.add("branch/if")
        if s & {"PUSH_EXC_INFO", "CHECK_EXC_MATCH", "SETUP_WITH", "WITH_EXCEPT_START", "RERAISE"}: tags.add("try/with")
        if s & {"CALL", "CALL_KW", "KW_NAMES", "CALL_FUNCTION_EX"}: tags.add("call")
        if s & {"MATCH_CLASS", "MATCH_MAPPING", "MATCH_SEQUENCE"}: tags.add("match")
        if s & {"YIELD_VALUE", "GET_AWAITABLE"}: tags.add("yield/await")
        if s & {"BUILD_MAP", "BUILD_CONST_KEY_MAP", "MAP_ADD", "BUILD_LIST", "BUILD_SET"}: tags.add("container")
    return sorted(tags) or ["other"]


def redecompile(gt_pyc: Path, version: str, workdir: Path) -> Path | None:
    workdir.mkdir(parents=True, exist_ok=True)
    v = ".".join(str(version).split(".")[:2])
    try:
        subprocess.run([CLI, "-o", str(workdir), "-v", v, "--force", "--timeout", "90", "-q", str(gt_pyc)],
                       capture_output=True, timeout=180)
    except Exception:
        return None
    outs = list(workdir.glob("decompiled_*.py"))
    return outs[0] if outs else None


def analyze(gt_pyc: Path, derived_pyc: Path, version: str) -> dict:
    orig = {t.names(): t.message for t in compare_pyc(gt_pyc, derived_pyc) if not t.success}
    if not orig:
        return {"status": "already_equal"}
    work = Path(tempfile.mkdtemp(prefix="tier3_"))
    fresh_src = redecompile(gt_pyc, version, work)
    if fresh_src is None:
        return {"status": "redecompile_failed"}
    try:
        fresh_pyc = compile_source_to_pyc(fresh_src, work / "fresh.pyc", tuple(int(x) for x in str(version).split(".")[:2]))
    except Exception as e:
        return {"status": "recompile_failed", "err": str(e)[:80]}
    fresh = {t.names(): t.message for t in compare_pyc(gt_pyc, fresh_pyc) if not t.success}
    # construct tags for reproduced objects (from the ORIGINAL derived divergence)
    pg, pd = prepare_pyc(gt_pyc), prepare_pyc(derived_pyc)
    bc = {}
    for drow, (gb, db) in zip(compare_code_object_distances(pg, pd), _iter_matched_code_objects(pg, pd)):
        bc[drow["gt_name"] or drow["derived_name"]] = (gb, db)
    objs = []
    for name in orig:
        if name in fresh:
            verdict = "pylingual_bound" if fresh[name] == orig[name] else "fresh_diverged"
        else:
            verdict = "not_fundamental"
        gb, db = bc.get(name, (None, None))
        objs.append({"obj": name, "verdict": verdict, "constructs": _construct_tags(gb, db)})
    return {"status": "ok", "objs": objs, "fresh_equal": len(fresh) == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir"); ap.add_argument("--limit", type=int, default=1000); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(str(Path(a.runs_dir) / "**" / "result.json"), recursive=True))
    verdict_ct = Counter(); construct_bound = Counter(); n_files = n_bound_files = 0; per = []
    done = 0
    for p in files:
        if done >= a.limit:
            break
        d = json.load(open(p))
        if (d.get("pylingual_verification") or {}).get("all_equal") is True:
            continue  # only the residual (non-perfect) files
        gt = Path(d.get("gt_pyc") or ""); der = Path(d.get("derived_pyc") or "")
        ver = d.get("target_python_version") or "3.12"
        if not gt.exists() or not der.exists():
            continue
        done += 1
        res = analyze(gt, der, ver)
        if res.get("status") != "ok":
            per.append({"gt": gt.parent.name[:10], "ver": ver, "status": res.get("status")})
            print(f"  [{done}] {gt.parent.name[:10]} py{ver}: {res.get('status')}", flush=True)
            continue
        n_files += 1
        file_bound = False
        for o in res["objs"]:
            verdict_ct[o["verdict"]] += 1
            if o["verdict"] == "pylingual_bound":
                file_bound = True
                for c in o["constructs"]:
                    construct_bound[c] += 1
        if file_bound:
            n_bound_files += 1
        per.append({"gt": gt.parent.name[:10], "ver": ver, "objs": res["objs"], "fresh_equal": res["fresh_equal"]})
        print(f"  [{done}] {gt.parent.name[:10]} py{ver}: "
              f"{[(o['obj'][:20], o['verdict']) for o in res['objs']]}", flush=True)
    out = {"files_analyzed": n_files, "files_with_pylingual_bound_object": n_bound_files,
           "object_verdicts": dict(verdict_ct), "pylingual_bound_constructs": dict(construct_bound.most_common()),
           "per_file": per}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    tot = sum(verdict_ct.values()) or 1
    print(f"\n=== TIER-3 PYLINGUAL REPRODUCIBILITY PROOF ({n_files} residual files) ===")
    print(f"files with >=1 provably-PyLingual-bound object: {n_bound_files}/{n_files}")
    print("OBJECT VERDICTS: " + ", ".join(f"{k}={v} ({100*v/tot:.0f}%)" for k, v in verdict_ct.most_common()))
    print("PROVABLY-PYLINGUAL-BOUND by construct:", dict(construct_bound.most_common()))


if __name__ == "__main__":
    main()
