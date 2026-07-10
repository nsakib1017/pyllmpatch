"""Full-corpus DETERMINISTIC-COVERAGE audit: for every failing code object, decide whether
its GT-vs-derived divergence is a LOCAL, unambiguous inverse (a deterministic operator can
fix it) or needs real source synthesis (decompilation -> LLM), tag the exact constructs, and
flag whether an operator already covers it. Output = the definitive gap map that says whether
the deterministic layer is at max capacity (residual == needs-decompilation) or has holes.

Classification per failing MATCHED object (by normalized instruction-signature multisets):
  * pure_reorder     -- identical instruction multiset, only order/nesting differs
                        (re-nest / re-parent / reorder territory; fully invertible).
  * local_edit       -- small symmetric diff of RECONSTRUCTABLE ops (LOAD_CONST/name/STORE/
                        operator/container/call/exc-structure) -> a leaf/container/call/exc
                        operator family can invert it.
  * needs_decompile  -- GT carries substantial content (expressions/statements/nested code
                        objects) absent from derived, OR the diff is large/opaque -> real
                        reconstruction, LLM territory.
missing/extra objects handled separately.
CPU-only, parallel, per-file SIGALRM timeout. Usage:
  python tools/deterministic_coverage_audit.py <dataset.csv> --source pylingual --semantic-only \
     --min-version 3.11 --workers 8 --out out.json
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual")); sys.path.insert(0, str(REPO))
csv.field_size_limit(sys.maxsize)
from utils.pyc_code_object_distance import prepare_pyc, compare_code_object_distances, _iter_matched_code_objects  # noqa
from utils.file_helpers import fetch_pyllmpatch_repair_paths  # noqa

# opcodes whose SOURCE is locally reconstructable from bytecode (a simple operator can invert)
RECONSTRUCTABLE = {
    "LOAD_CONST", "RETURN_CONST", "LOAD_NAME", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_DEREF",
    "LOAD_FAST_CHECK", "LOAD_FAST_AND_CLEAR", "STORE_NAME", "STORE_FAST", "STORE_GLOBAL",
    "STORE_DEREF", "LOAD_ATTR", "STORE_ATTR", "LOAD_METHOD", "COMPARE_OP", "IS_OP", "CONTAINS_OP",
    "BINARY_OP", "UNARY_NEGATIVE", "UNARY_NOT", "UNARY_INVERT", "TO_BOOL", "BINARY_SUBSCR",
    "STORE_SUBSCR", "BUILD_MAP", "BUILD_CONST_KEY_MAP", "BUILD_LIST", "BUILD_SET", "BUILD_TUPLE",
    "BUILD_STRING", "MAP_ADD", "LIST_APPEND", "SET_ADD", "LIST_EXTEND", "SET_UPDATE", "DICT_UPDATE",
    "DICT_MERGE", "FORMAT_VALUE", "FORMAT_SIMPLE", "CONVERT_VALUE", "KW_NAMES", "CALL", "CALL_KW",
    "PUSH_NULL", "COPY", "SWAP", "NOP", "POP_TOP", "CACHE", "RESUME", "PRECALL", "CALL_METHOD",
    "DELETE_NAME", "DELETE_FAST", "DELETE_GLOBAL", "DELETE_ATTR", "RETURN_VALUE",
}
# constructs -> whether an operator family already covers the deterministic fix
CONSTRUCT_COVERED = {
    "literal_value": True, "name_ref": True, "operator": True, "container": True,
    "literal_form": True, "fstring": True, "jump_sense": True, "call_kw": True,
    "try_except_with_finally": True, "if_for_flatten": True, "dropped_scalar": True,
    # WAVE 2 + 3 operators (now covered):
    "unary_tobool": True, "del": True, "import": True, "assert": True, "global_nonlocal": True,
    "decorator_default": True, "starred_splat_unpack": True, "bool_shortcircuit": True,
    "ternary": True, "comprehension": True, "extra_object": True,  # extra_object handled by deletion
    # REMAINING GAPS (no operator; yield/await + nested-try buildable, rest = needs-decompile-adjacent):
    "yield_await": False, "nested_try_or_multihandler": False,
    "match_case": False, "arbitrary_reparent": False, "other_structural": False,
}


def _norm(sig):
    # sig = (opname, opcode, optype, real_size, has_arg, has_ext, is_jump_target, argval)
    av = sig[-1]
    if isinstance(av, tuple) and av and av[0] == "code":
        return (sig[0], ("code",))  # nested code object identity-agnostic
    return (sig[0], av if isinstance(av, (str, int, float, bool, type(None), bytes, tuple)) else str(av))


def _constructs(gops, dops):
    """Tag the specific constructs a divergence chunk touches. gops/dops are the GT / derived
    opcode lists for ONE aligned non-equal chunk (so scope-flips can be detected precisely)."""
    gs, ds = set(gops), set(dops); s = gs | ds; out = set()
    # --- covered leaf/arrangement families (so leaf swaps aren't miscounted as gaps) ---
    if s & {"LOAD_CONST", "RETURN_CONST"}: out.add("literal_value")
    if s & {"LOAD_NAME", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_DEREF", "STORE_NAME", "STORE_FAST", "LOAD_ATTR", "STORE_ATTR"}: out.add("name_ref")
    if s & {"COMPARE_OP", "IS_OP", "CONTAINS_OP", "BINARY_OP", "BINARY_SUBSCR"}: out.add("operator")
    if s & {"KW_NAMES", "CALL_KW"}: out.add("call_kw")
    if s & {"FORMAT_VALUE", "FORMAT_SIMPLE", "FORMAT_WITH_SPEC", "CONVERT_VALUE", "BUILD_STRING"}: out.add("fstring")
    if s & {"BUILD_MAP", "BUILD_CONST_KEY_MAP", "MAP_ADD", "BUILD_SET", "BUILD_LIST", "SET_ADD", "LIST_APPEND", "DICT_UPDATE", "SET_UPDATE"}: out.add("container")
    if s & {"SETUP_WITH", "BEFORE_WITH", "WITH_EXCEPT_START", "PUSH_EXC_INFO", "CHECK_EXC_MATCH"}: out.add("try_except_with_finally")
    # --- GAP constructs (no operator yet) ---
    if s & {"IMPORT_NAME", "IMPORT_FROM", "IMPORT_STAR"}: out.add("import")
    if s & {"LOAD_ASSERTION_ERROR"}: out.add("assert")
    if s & {"UNPACK_EX", "CALL_FUNCTION_EX", "UNPACK_SEQUENCE", "DICT_MERGE", "LIST_EXTEND"}: out.add("starred_splat_unpack")
    if s & {"DELETE_NAME", "DELETE_FAST", "DELETE_GLOBAL", "DELETE_ATTR", "DELETE_SUBSCR"}: out.add("del")
    if s & {"JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP"}: out.add("bool_shortcircuit")
    if s & {"YIELD_VALUE", "GET_AWAITABLE", "SEND", "GET_YIELD_FROM_ITER"}: out.add("yield_await")
    if s & {"MATCH_CLASS", "MATCH_MAPPING", "MATCH_SEQUENCE", "MATCH_KEYS"}: out.add("match_case")
    if s & {"UNARY_NEGATIVE", "UNARY_NOT", "UNARY_INVERT", "TO_BOOL"}: out.add("unary_tobool")
    if s & {"MAKE_FUNCTION"}: out.add("decorator_default")
    # precise global/nonlocal: a store-SCOPE flip in the SAME chunk (GT global/deref vs derived local)
    if (gs & {"STORE_GLOBAL", "STORE_DEREF"} and ds & {"STORE_NAME", "STORE_FAST"}) or \
       (ds & {"STORE_GLOBAL", "STORE_DEREF"} and gs & {"STORE_NAME", "STORE_FAST"}):
        out.add("global_nonlocal")
    return out


def classify_object(gt_bc, der_bc, row):
    status = row["status"]
    if status == "missing":
        # a whole GT code object with no derived counterpart: reconstructable only if tiny/const
        return {"class": "needs_decompile", "constructs": ["missing_object"], "covered": False}
    if status == "extra":
        return {"class": "local_edit", "constructs": ["extra_object"], "covered": True}
    gt = gt_bc["instruction_signatures"]; der = der_bc["instruction_signatures"]
    gn = [_norm(s) for s in gt]; dn = [_norm(s) for s in der]
    gm, dm = Counter(gn), Counter(dn)
    gt_only = gm - dm; der_only = dm - gm
    cfg = row["control_flow_distance"]
    gops = [s[0] for s in gt]; dops = [s[0] for s in der]
    constructs = set()
    for t, i1, i2, j1, j2 in SequenceMatcher(a=gops, b=dops, autojunk=False).get_opcodes():
        if t == "equal":
            continue
        constructs |= _constructs(gops[i1:i2], dops[j1:j2])
    # injective classification
    total_only = sum(gt_only.values()) + sum(der_only.values())
    gt_only_ops = {k[0] for k in gt_only}
    gt_only_reconstructable = gt_only_ops <= RECONSTRUCTABLE
    has_code_obj = any(k[1] == ("code",) for k in gt_only)
    if total_only == 0:
        cls = "pure_reorder"          # identical content, only arrangement/nesting differs
        constructs = constructs or {"reorder"}
    elif total_only <= 8 and gt_only_reconstructable and not has_code_obj:
        cls = "local_edit"            # small reconstructable symmetric diff
    else:
        cls = "needs_decompile"       # substantial/opaque content difference
        if has_code_obj:
            constructs.add("comprehension")
    covered = None
    if cls in ("pure_reorder", "local_edit"):
        cov = [CONSTRUCT_COVERED.get(c, False) for c in constructs] or [cls == "pure_reorder"]
        covered = all(cov)  # every construct in the divergence has an operator
    return {"class": cls, "constructs": sorted(constructs), "covered": bool(covered) if covered is not None else None,
            "cfg": cfg}


_TIMEOUT = 12
class _TO(Exception): pass
def _alarm(s, f): raise _TO()


def analyze_row(row):
    import signal
    g = row.get("file_hash") or row.get("hash"); src = row.get("source") or "pylingual"
    gp, dp, _ = fetch_pyllmpatch_repair_paths(g, src) if g else (None, None, None)
    if not gp or not dp or not gp.exists() or not dp.exists():
        return {"status": "unresolved"}
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(_TIMEOUT)
    try:
        pg, pd = prepare_pyc(gp), prepare_pyc(dp)
        rows = compare_code_object_distances(pg, pd)
        pairs = list(_iter_matched_code_objects(pg, pd))
    except _TO:
        return {"status": "timeout"}
    except Exception:
        return {"status": "error"}
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    objs = []
    for drow, (gb, db) in zip(rows, pairs):
        if drow["combined_distance"] <= 0 and drow["status"] == "matched":
            continue
        objs.append(classify_object(gb, db, drow))
    return {"status": "ok", "ver": (row.get("bytecode_version") or "?"), "objs": objs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--source", default="pylingual")
    ap.add_argument("--semantic-only", action="store_true"); ap.add_argument("--min-version", default="3.11")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--sample", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def vt(s):
        try: return tuple(int(x) for x in s.split(".")[:2])
        except Exception: return None
    mv = vt(a.min_version)
    def keep(r):
        if a.semantic_only and (r.get("error_type") or "").strip() != "semantic_error": return False
        t = vt(r.get("bytecode_version") or "")
        return not (mv and (t is None or t < mv))
    rows = [r for r in csv.DictReader(open(a.dataset)) if keep(r)]
    for r in rows: r.setdefault("source", a.source)
    rows = rows[: a.limit]
    if a.sample and a.sample < len(rows):
        import random; random.seed(3); rows = random.sample(rows, a.sample)

    cls_ct = Counter(); constr_ct = Counter(); constr_files = Counter()
    covered_local = Counter(); n_ok = n_un = n_err = n_to = 0; n_obj = 0
    ver_cls = defaultdict(Counter)
    def consume(res):
        nonlocal n_ok, n_un, n_err, n_to, n_obj
        st = res.get("status")
        if st != "ok":
            if st == "unresolved": n_un += 1
            elif st == "timeout": n_to += 1
            else: n_err += 1
            return
        n_ok += 1
        for o in res["objs"]:
            n_obj += 1; cls_ct[o["class"]] += 1; ver_cls[res["ver"]][o["class"]] += 1
            for c in o["constructs"]: constr_ct[c] += 1
            if o["class"] in ("pure_reorder", "local_edit"):
                covered_local["covered" if o["covered"] else "GAP_uncovered"] += 1
    print(f"rows={len(rows)} workers={a.workers}", flush=True)
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(analyze_row, rows, chunksize=16)):
                consume(res)
                if (i+1) % 3000 == 0: print(f"  {i+1}/{len(rows)} ok={n_ok} objs={n_obj}", flush=True)
    else:
        for r in rows: consume(analyze_row(r))
    tot = n_obj or 1
    out = {"files_ok": n_ok, "unresolved": n_un, "error": n_err, "timeout": n_to, "objects": n_obj,
           "injective_class": dict(cls_ct), "construct_freq": dict(constr_ct.most_common()),
           "local_coverage": dict(covered_local), "by_version_class": {k: dict(v) for k, v in sorted(ver_cls.items())}}
    if a.out: Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nfiles ok={n_ok} unresolved={n_un} err={n_err} timeout={n_to}; failing objects={n_obj}")
    print("INJECTIVE CLASS (share of failing objects):")
    for k in ("pure_reorder", "local_edit", "needs_decompile"):
        print(f"  {cls_ct.get(k,0)/tot*100:5.1f}%  {k}")
    dl = sum(covered_local.values()) or 1
    print(f"LOCAL-INVERTIBLE COVERAGE (of pure_reorder+local_edit objects): covered={covered_local.get('covered',0)} ({covered_local.get('covered',0)/dl*100:.0f}%), GAP_uncovered={covered_local.get('GAP_uncovered',0)} ({covered_local.get('GAP_uncovered',0)/dl*100:.0f}%)")
    print("CONSTRUCT FREQ (share of objects whose diff touches it) + covered?:")
    for c, n in constr_ct.most_common():
        print(f"  {n/tot*100:5.1f}%  {c:26s}  {'covered' if CONSTRUCT_COVERED.get(c) else 'GAP'}")


if __name__ == "__main__":
    main()
