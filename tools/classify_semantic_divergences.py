"""Classify the bytecode divergences of failing semantic-error objects across a dataset.

For every failing code object (combined_distance > 0, or missing/extra), align the GT vs
derived instruction streams and bucket each divergence point into a taxonomy, then give the
object a three-way verdict:

  * det_fixable  -- the whole divergence is injective (GT bytecode uniquely determines a
                    local source edit): literal-value/form, container-construction, call-shape,
                    operator, name-ref, jump-sense, fstring, or an extra (deletable) object.
  * makes_easier -- the object has BOTH mechanical divergences AND genuine control-flow
                    restructuring; a mechanical pre-edit shrinks what the LLM must reconstruct.
  * llm_only     -- only control-flow restructuring / a missing object; non-injective.

CPU-only, no model. Reuses the engine's object matching + instruction signatures.
Usage: python tools/classify_semantic_divergences.py <dataset.csv> [--limit N] [--out out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual"))
sys.path.insert(0, str(REPO))

from utils.pyc_code_object_distance import (  # noqa: E402
    compare_code_object_distances,
    prepare_pyc,
    _iter_matched_code_objects,
)
from utils.file_helpers import fetch_pyllmpatch_repair_paths  # noqa: E402

# ---- opcode families ------------------------------------------------------
CONST_OPS = {"LOAD_CONST", "RETURN_CONST"}
NAME_LOAD_OPS = {"LOAD_FAST", "LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF", "LOAD_FAST_CHECK",
                 "LOAD_FAST_AND_CLEAR", "STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}
CONTAINER_OPS = {"BUILD_MAP", "BUILD_CONST_KEY_MAP", "DICT_UPDATE", "DICT_MERGE", "BUILD_SET",
                 "BUILD_LIST", "BUILD_TUPLE", "LIST_EXTEND", "LIST_APPEND", "SET_UPDATE", "SET_ADD",
                 "MAP_ADD", "BUILD_STRING", "LIST_TO_TUPLE", "BUILD_SLICE"}
CALL_OPS = {"CALL", "CALL_FUNCTION", "CALL_FUNCTION_KW", "CALL_FUNCTION_EX", "CALL_METHOD",
            "KW_NAMES", "PRECALL", "PUSH_NULL", "CALL_KW", "LOAD_METHOD", "CALL_INTRINSIC_1",
            "CALL_INTRINSIC_2"}
OPERATOR_OPS = {"BINARY_OP", "COMPARE_OP", "IS_OP", "CONTAINS_OP", "BINARY_SUBSCR", "UNARY_OP",
                "UNARY_NEGATIVE", "UNARY_NOT", "UNARY_INVERT", "BINARY_ADD", "BINARY_MULTIPLY"}
FSTRING_OPS = {"FORMAT_VALUE", "FORMAT_SIMPLE", "FORMAT_WITH_SPEC", "CONVERT_VALUE"}
# jump ops whose "sense" can flip (mechanical) vs whose targets/insertion is structural
JUMP_TRUE = {"POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_TRUE", "POP_JUMP_BACKWARD_IF_TRUE",
             "JUMP_IF_TRUE_OR_POP"}
JUMP_FALSE = {"POP_JUMP_IF_FALSE", "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_BACKWARD_IF_FALSE",
              "JUMP_IF_FALSE_OR_POP"}
JUMP_NONE = {"POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE", "POP_JUMP_FORWARD_IF_NONE",
             "POP_JUMP_FORWARD_IF_NOT_NONE"}
STRUCTURAL_OPS = {"SETUP_FINALLY", "SETUP_WITH", "SETUP_EXCEPT", "SETUP_CLEANUP", "SETUP_ASYNC_WITH",
                  "FOR_ITER", "GET_ITER", "JUMP_FORWARD", "JUMP_BACKWARD", "JUMP_ABSOLUTE",
                  "RERAISE", "PUSH_EXC_INFO", "POP_EXCEPT", "CHECK_EXC_MATCH", "WITH_EXCEPT_START",
                  "SEND", "END_FOR", "END_SEND", "CLEANUP_THROW"}

MECHANICAL = {"literal_value", "literal_form", "container_construction", "call_shape",
              "operator", "name_ref", "fstring", "jump_sense"}


def _opset(sigs):
    return [s[0] for s in sigs]


def _argvals(sigs):
    return [s[-1] for s in sigs]


def _classify_divergence(tag, gchunk, dchunk) -> str:
    """Map one aligned divergence (replace/insert/delete) to a taxonomy label."""
    gops = set(_opset(gchunk))
    dops = set(_opset(dchunk))
    allops = gops | dops

    # jump-sense flip: a true<->false (or none-variant) swap at the same spot
    if (gops & JUMP_TRUE and dops & JUMP_FALSE) or (gops & JUMP_FALSE and dops & JUMP_TRUE) \
       or (gops & JUMP_NONE and dops & JUMP_NONE):
        return "jump_sense"

    # A chunk that adds/removes a branch or loop op is a control-flow structural change,
    # even when it also carries a COMPARE_OP / LOAD_CONST (a deleted `if x == 'rb':` is
    # COMPARE_OP + POP_JUMP). Check this BEFORE the operator/call/container families so a
    # deleted conditional is not mis-claimed as a mechanical operator swap. (A clean
    # true<->false jump-sense flip already returned above.)
    if allops & (STRUCTURAL_OPS | JUMP_TRUE | JUMP_FALSE | JUMP_NONE):
        return "control_flow_structural"

    # container construction touched on either side
    if allops & CONTAINER_OPS:
        return "container_construction"

    # cross-opcode literal form: name-load <-> const (the c4d3 quoting class)
    if (gops & CONST_OPS and dops & NAME_LOAD_OPS) or (gops & NAME_LOAD_OPS and dops & CONST_OPS):
        return "literal_form"

    # same-position const value swap
    if tag == "replace" and gops <= CONST_OPS and dops <= CONST_OPS:
        return "literal_value"

    if allops & CALL_OPS:
        return "call_shape"
    if allops & FSTRING_OPS:
        return "fstring"
    if allops & OPERATOR_OPS:
        return "operator"

    # pure name/store rename (same opcodes, arg differs) or name-load swaps
    if tag == "replace" and gops <= NAME_LOAD_OPS and dops <= NAME_LOAD_OPS:
        return "name_ref"

    # structural control-flow ops inserted/deleted/retargeted
    if allops & STRUCTURAL_OPS or tag in ("insert", "delete"):
        return "control_flow_structural"

    return "other"


def classify_object(gt_bc, derived_bc, row) -> dict:
    status = row["status"]
    if status == "extra":
        return {"verdict": "det_fixable", "classes": ["extra_object"], "cfg": 0,
                "combined": row["combined_distance"]}
    if status == "missing":
        return {"verdict": "llm_only", "classes": ["missing_object"], "cfg": row["control_flow_distance"],
                "combined": row["combined_distance"]}

    gt = gt_bc["instruction_signatures"]
    der = derived_bc["instruction_signatures"]
    cfg = row["control_flow_distance"]

    sm = SequenceMatcher(a=_opset(gt), b=_opset(der), autojunk=False)
    classes = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # even "equal" opname runs can differ by argval -> const/name value swap
            for gi, dj in zip(range(i1, i2), range(j1, j2)):
                if gt[gi][-1] != der[dj][-1]:
                    op = gt[gi][0]
                    if op in CONST_OPS:
                        classes.append("literal_value")
                    elif op in NAME_LOAD_OPS:
                        classes.append("name_ref")
                    elif op in OPERATOR_OPS:
                        classes.append("operator")
                    elif op in FSTRING_OPS:
                        classes.append("fstring")
                    else:
                        classes.append("arg_value")
            continue
        classes.append(_classify_divergence(tag, gt[i1:i2], der[j1:j2]))

    cset = set(classes)
    has_struct = "control_flow_structural" in cset or "other" in cset
    mech = cset & MECHANICAL
    if "arg_value" in cset:
        mech = mech | {"arg_value"}  # treat generic arg swaps as mechanical-ish

    if not has_struct and cset and cset <= (MECHANICAL | {"arg_value"}):
        verdict = "det_fixable"
    elif has_struct and cset == {"control_flow_structural"} and cfg > 0:
        verdict = "llm_only"
    elif has_struct and cset == {"other"}:
        verdict = "llm_only"
    elif mech and has_struct:
        verdict = "makes_easier"
    elif not cset:
        verdict = "det_fixable"  # divergence only in exc-table / non-instruction terms
    else:
        verdict = "llm_only"

    return {"verdict": verdict, "classes": sorted(cset), "cfg": cfg,
            "combined": row["combined_distance"]}


_SOURCE_OVERRIDE = None  # set in main(); used by workers


def resolve_paths(row):
    gp = (row.get("gt_pyc") or "").strip()
    dp = (row.get("derived_pyc") or "").strip()
    if gp and dp and Path(gp).exists() and Path(dp).exists():
        return Path(gp), Path(dp)
    fh = row.get("file_hash") or row.get("hash")
    src = _SOURCE_OVERRIDE or row.get("source") or "PyPi"
    if not fh:
        return None, None
    g, d, _ = fetch_pyllmpatch_repair_paths(fh, src)
    return g, d


_PER_FILE_TIMEOUT = 12  # seconds; pathologically-large auto-generated modules blow up CFG build


class _RowTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _RowTimeout()


def classify_row(row):
    """Worker: classify one dataset row. Returns a compact per-file dict or None.
    A per-file SIGALRM cap skips pathological files (giant modules) as 'timeout'."""
    gp, dp = resolve_paths(row)
    if not gp or not dp or not gp.exists() or not dp.exists():
        return {"status": "unresolved"}
    import signal
    have_alarm = hasattr(signal, "SIGALRM")
    if have_alarm:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(_PER_FILE_TIMEOUT)
    try:
        pg = prepare_pyc(gp)
        pd = prepare_pyc(dp)
        drows = compare_code_object_distances(pg, pd)
        pairs = list(_iter_matched_code_objects(pg, pd))
    except _RowTimeout:
        return {"status": "timeout"}
    except Exception:
        return {"status": "error"}
    finally:
        if have_alarm:
            signal.alarm(0)
    full_fh = row.get("file_hash") or row.get("hash") or "?"
    fh = full_fh[:12]
    ver = row.get("bytecode_version") or row.get("python_version") or "?"
    obj_verdicts = []
    class_counts = Counter()
    objs = []
    for drow, (gt_bc, der_bc) in zip(drows, pairs):
        if drow["combined_distance"] <= 0 and drow["status"] == "matched":
            continue
        info = classify_object(gt_bc, der_bc, drow)
        for c in info["classes"]:
            class_counts[c] += 1
        obj_verdicts.append(info["verdict"])
        objs.append({"classes": info["classes"], "verdict": info["verdict"],
                     "obj": drow["code_object"], "cd": info["combined"]})
    if not obj_verdicts:
        fv = "already_perfect"
    elif all(v == "det_fixable" for v in obj_verdicts):
        fv = "fully_det_fixable"
    elif all(v in ("det_fixable", "makes_easier") for v in obj_verdicts):
        fv = "det_plus_easier"
    elif any(v in ("det_fixable", "makes_easier") for v in obj_verdicts):
        fv = "mixed_needs_llm"
    else:
        fv = "llm_only"
    return {"status": "ok", "file": fh, "file_hash": full_fh, "ver": ver, "file_verdict": fv,
            "obj_verdicts": obj_verdicts, "class_counts": dict(class_counts), "objs": objs}


def main():
    global _SOURCE_OVERRIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default=None, help="override source for path resolution (e.g. pylingual)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0, help="random sample N rows (0 = all)")
    ap.add_argument("--semantic-only", action="store_true", help="keep only error_type==semantic_error")
    ap.add_argument("--min-version", default="3.10", help="keep rows with bytecode_version >= this")
    ap.add_argument("--progress", type=int, default=2000)
    args = ap.parse_args()
    _SOURCE_OVERRIDE = args.source

    def _vtuple(s):
        try:
            return tuple(int(x) for x in s.strip().split(".")[:2])
        except Exception:
            return None

    min_v = _vtuple(args.min_version) if args.min_version else None

    def keep(row):
        if args.semantic_only and (row.get("error_type") or "").strip() != "semantic_error":
            return False
        vt = _vtuple(row.get("bytecode_version") or "")
        if min_v is not None and (vt is None or vt < min_v):
            return False
        return True

    csv.field_size_limit(sys.maxsize)
    rows = [r for r in csv.DictReader(open(args.dataset)) if keep(r)][: args.limit]
    if args.sample and args.sample < len(rows):
        import random
        random.seed(1234)
        rows = random.sample(rows, args.sample)

    class_hist = Counter()
    verdict_obj = Counter()
    file_verdict = Counter()
    ver_file_verdict = defaultdict(Counter)
    per_file = []
    n_ok = n_unresolved = n_err = n_timeout = 0
    examples = defaultdict(list)

    def consume(res, i):
        nonlocal n_ok, n_unresolved, n_err, n_timeout
        st = res.get("status")
        if st == "unresolved":
            n_unresolved += 1; return
        if st == "error":
            n_err += 1; return
        if st == "timeout":
            n_timeout += 1; return
        if st != "ok":
            n_err += 1; return
        n_ok += 1
        for v in res["obj_verdicts"]:
            verdict_obj[v] += 1
        for c, k in res["class_counts"].items():
            class_hist[c] += k
        file_verdict[res["file_verdict"]] += 1
        ver_file_verdict[res["ver"]][res["file_verdict"]] += 1
        for o in res["objs"]:
            key = tuple(o["classes"])
            if len(examples[key]) < 3:
                examples[key].append({"file": res["file"], "ver": res["ver"],
                                      "obj": o["obj"], "verdict": o["verdict"], "cd": o["cd"]})
        per_file.append({"file": res["file"], "file_hash": res.get("file_hash"), "ver": res["ver"],
                         "file_verdict": res["file_verdict"], "objs": res["obj_verdicts"]})

    print(f"rows to classify: {len(rows)}  workers={args.workers}  source={args.source}", flush=True)
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(classify_row, rows, chunksize=16)):
                consume(res, i)
                if (i + 1) % args.progress == 0:
                    print(f"  {i+1}/{len(rows)}  ok={n_ok} unresolved={n_unresolved} err={n_err}", flush=True)
    else:
        for i, r in enumerate(rows):
            consume(classify_row(r), i)
            if (i + 1) % args.progress == 0:
                print(f"  {i+1}/{len(rows)}  ok={n_ok} unresolved={n_unresolved} err={n_err}", flush=True)

    out = {
        "dataset": args.dataset,
        "files_analyzed": n_ok,
        "files_unresolved": n_unresolved,
        "files_error": n_err,
        "files_timeout": n_timeout,
        "object_verdicts": dict(verdict_obj),
        "divergence_class_histogram": dict(class_hist.most_common()),
        "file_verdicts": dict(file_verdict),
        "file_verdicts_by_version": {k: dict(v) for k, v in sorted(ver_file_verdict.items())},
        "examples_by_class": {", ".join(k) if k else "(none)": v for k, v in
                              sorted(examples.items(), key=lambda kv: -len(kv[1]))[:40]},
        "per_file": per_file,  # [{file, ver, file_verdict, ...}] for downstream subset selection
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    tot_obj = sum(verdict_obj.values()) or 1
    print(f"\n=== dataset={args.dataset} ===")
    print(f"files: analyzed={n_ok} unresolved={n_unresolved} error={n_err} timeout={n_timeout}")
    print(f"OBJECT VERDICTS ({tot_obj}): " + ", ".join(
        f"{k}={v} ({100*v/tot_obj:.0f}%)" for k, v in verdict_obj.most_common()))
    tot_f = sum(file_verdict.values()) or 1
    print(f"FILE VERDICTS ({tot_f}): " + ", ".join(
        f"{k}={v} ({100*v/tot_f:.0f}%)" for k, v in file_verdict.most_common()))
    print("DIVERGENCE CLASSES:", dict(class_hist.most_common()))
    print("BY VERSION (file verdicts):")
    for v, c in sorted(ver_file_verdict.items()):
        print(f"  {v}: {dict(c.most_common())}")


if __name__ == "__main__":
    main()
