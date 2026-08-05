from __future__ import annotations

import ast
import textwrap
from difflib import SequenceMatcher

from utils.semantic_operators.leaf import _instructions

# CALL-family terminators. 3.11/3.12 emit CALL (optionally preceded by PRECALL); 3.13 emits
# CALL_KW for a keyword call. CALL_FUNCTION* are the <=3.10 forms (kept for completeness; the
# KW_NAMES/CALL_KW keyword signal they lack means those simply never produce a divergence here).
_CALL_OPCODES = {"CALL", "CALL_KW", "CALL_FUNCTION", "CALL_FUNCTION_KW"}


def _real_instructions(bytecode) -> list:
    return [i for i in _instructions(bytecode) if getattr(i, "opname", None) != "CACHE"]


def _argint(ins):
    a = getattr(ins, "arg", None)
    if isinstance(a, int):
        return a
    v = getattr(ins, "argval", None)
    return v if isinstance(v, int) else None


def _valid_names(names) -> bool:
    if not isinstance(names, tuple) or not names:
        return False
    if not all(isinstance(n, str) and n.isidentifier() for n in names):
        return False
    return len(set(names)) == len(names)


def _call_total_after(insts: list, i: int) -> int | None:
    for j in range(i + 1, len(insts)):
        op = getattr(insts[j], "opname", None)
        if op in _CALL_OPCODES:
            return _argint(insts[j])
        if op in ("PRECALL", "CACHE"):
            continue
        break  # any other opcode between KW_NAMES and its CALL -> not the clean signal
    return None


def _callkw_names(insts: list, i: int):
    if i - 1 < 0:
        return None
    prev = insts[i - 1]
    if getattr(prev, "opname", None) != "LOAD_CONST":
        return None
    v = getattr(prev, "argval", None)
    return v if isinstance(v, tuple) else None


def _find_call_shape_divergences(gt: list, der: list) -> list[dict]:
    sm = SequenceMatcher(a=[getattr(i, "opname", None) for i in gt],
                         b=[getattr(i, "opname", None) for i in der], autojunk=False)
    divs: list[dict] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for i in range(i1, i2):
            op = getattr(gt[i], "opname", None)
            if op == "KW_NAMES":
                names = getattr(gt[i], "argval", None)
                total = _call_total_after(gt, i)
            elif op == "CALL_KW":
                names = _callkw_names(gt, i)
                total = _argint(gt[i])
            else:
                continue
            if not _valid_names(names) or total is None:
                continue
            n_positional = total - len(names)
            if n_positional < 0:
                continue
            divs.append({"names": tuple(names), "total": total, "n_positional": n_positional})
    return divs


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _positional_call_candidates(tree: ast.AST, total: int) -> list[ast.Call]:
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if node.keywords:  # already has keyword args -> not the fully-positional case we own
            continue
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue  # *args: positional count does not map to the recovered split
        if len(node.args) != total:
            continue
        if _dotted_name(node.func) is None:
            continue
        out.append(node)
    return out


def _replace_call_span(fragment: str, dedented: str, node: ast.AST, new_text: str) -> str | None:
    if None in (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset):
        return None
    orig = fragment.splitlines(keepends=True)
    ded = dedented.splitlines(keepends=True)
    li0, li1 = node.lineno - 1, node.end_lineno - 1
    if li0 < 0 or li1 >= len(orig) or li0 >= len(ded) or li1 >= len(ded) or li0 > li1:
        return None
    if not orig[li0].isascii() or not orig[li1].isascii():
        return None
    start = node.col_offset + (len(orig[li0]) - len(ded[li0]))
    end = node.end_col_offset + (len(orig[li1]) - len(ded[li1]))
    prefix = orig[li0][:start]
    suffix = orig[li1][end:]
    return "".join(orig[:li0]) + prefix + new_text + suffix + "".join(orig[li1 + 1:])


def call_shape_candidate(gt_code_object, derived_code_object, fragment: str,
                         repair_context: dict | None) -> dict | None:
    try:
        return _call_shape_candidate(gt_code_object, derived_code_object, fragment, repair_context)
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None


def _call_shape_candidate(gt_code_object, derived_code_object, fragment: str, repair_context: dict | None) -> dict | None:
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None

    gt = _real_instructions(gt_code_object)
    der = _real_instructions(derived_code_object)
    if not gt or not der:
        return None

    # Require EXACTLY ONE call-shape divergence: 0 = nothing to do / same shape, >1 =
    # ambiguous which call to touch (the oracle gate cannot localize for us) -> defer.
    divs = _find_call_shape_divergences(gt, der)
    if len(divs) != 1:
        return None
    div = divs[0]
    names = div["names"]
    total = div["total"]
    n_positional = div["n_positional"]

    try:
        dedented = textwrap.dedent(fragment)
        tree = ast.parse(dedented)
    except SyntaxError:
        return None

    # Locate the UNIQUE derived call that passes ``total`` args positionally; >1 (or 0)
    # means we cannot attribute the divergence to a single call site -> defer.
    candidates = _positional_call_candidates(tree, total)
    if len(candidates) != 1:
        return None
    node = candidates[0]

    kept_positional = node.args[:n_positional]
    to_convert = node.args[n_positional:]
    if len(to_convert) != len(names):
        return None  # split does not line up -> defer

    callee = _dotted_name(node.func)
    new_keywords = [ast.keyword(arg=nm, value=val) for nm, val in zip(names, to_convert)]
    new_call = ast.Call(func=node.func, args=list(kept_positional), keywords=new_keywords)
    try:
        new_text = ast.unparse(new_call)
    except Exception:  # noqa: BLE001
        return None

    edited = _replace_call_span(fragment, dedented, node, new_text)
    if edited is None or edited == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(edited))
    except SyntaxError:
        return None

    opname = "CALL_KW" if any(getattr(i, "opname", None) == "CALL_KW" for i in gt) else "KW_NAMES"
    return {
        "text": edited,
        "operator": f"positional->keyword {callee}({', '.join(names)})",
        "opname": opname,
        "kind": "call_shape",
        "confidence": "unique",
    }
