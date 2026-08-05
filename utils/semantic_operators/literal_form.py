from __future__ import annotations

import ast
import textwrap
from difflib import SequenceMatcher
from keyword import iskeyword

from utils.semantic_operators.leaf import _instructions, _replace_node_span, _round_trips

# The two opcode families whose XOR is the quoting signature. A const *load* on one side,
# a bare value-name load on the other. RETURN_CONST / RETURN_VALUE (tail returns) are
# intentionally excluded — those are owned by leaf's tail_return / const_to_name operators.
_CONST_LOAD_OPCODES = {"LOAD_CONST"}
_NAME_LOAD_OPCODES = {"LOAD_NAME", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_DEREF"}


def _real_instructions(bytecode) -> list:
    return [i for i in _instructions(bytecode) if getattr(i, "opname", None) != "CACHE"]


def _name_of(ins) -> str | None:
    v = getattr(ins, "argval", None)
    if isinstance(v, str) and v.isidentifier():
        return v
    r = (getattr(ins, "argrepr", "") or "").strip()
    if "+" in r:  # 3.11 LOAD_GLOBAL "NULL + x"
        r = r.split("+")[-1].strip()
    return r if r.isidentifier() else None


def _find_quoting_divergences(gt: list, der: list) -> list[dict]:
    sm = SequenceMatcher(a=[i.opname for i in gt], b=[i.opname for i in der], autojunk=False)
    divs: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
            continue
        g, d = gt[i1], der[j1]
        gop, dop = getattr(g, "opname", None), getattr(d, "opname", None)
        if gop in _CONST_LOAD_OPCODES and dop in _NAME_LOAD_OPCODES:
            cval = getattr(g, "argval", None)
            ident = _name_of(d)
            if isinstance(cval, str) and ident is not None and cval == ident:
                divs.append({"direction": "quote", "identifier": ident, "const_value": cval})
        elif gop in _NAME_LOAD_OPCODES and dop in _CONST_LOAD_OPCODES:
            cval = getattr(d, "argval", None)
            ident = _name_of(g)
            if isinstance(cval, str) and ident is not None and cval == ident:
                divs.append({"direction": "unquote", "identifier": ident, "const_value": cval})
    return divs


def literal_form_candidate(gt_code_object, derived_code_object, fragment: str,
                           repair_context: dict | None) -> dict | None:
    try:
        return _literal_form_candidate(gt_code_object, derived_code_object, fragment, repair_context)
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None


def _literal_form_candidate(gt_code_object, derived_code_object, fragment: str, repair_context: dict | None) -> dict | None:
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None

    gt = _real_instructions(gt_code_object)
    der = _real_instructions(derived_code_object)
    if not gt or not der:
        return None

    # Require EXACTLY ONE quoting divergence — defer on 0 (nothing to do / not our case)
    # or >1 (ambiguous which leaf to touch; the oracle gate can't localize for us).
    divs = _find_quoting_divergences(gt, der)
    if len(divs) != 1:
        return None
    div = divs[0]
    ident = div["identifier"]
    const_value = div["const_value"]

    try:
        dedented = textwrap.dedent(fragment)
        tree = ast.parse(dedented)
    except SyntaxError:
        return None

    if div["direction"] == "quote":
        # bare name -> string literal. repr() must round-trip so we never inline a
        # value that would re-parse to something else.
        if not _round_trips(const_value):
            return None
        nodes = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == ident]
        if len(nodes) != 1:
            return None
        new_text = repr(const_value)
        opname = "LOAD_CONST"
        label = f"quote name {ident} -> {const_value!r}"
    else:  # unquote: string literal -> bare name
        if not ident.isidentifier() or iskeyword(ident):
            return None
        nodes = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == const_value]
        if len(nodes) != 1:
            return None
        new_text = ident
        opname = "LOAD_NAME"
        label = f"unquote {const_value!r} -> name {ident}"

    edited = _replace_node_span(fragment, dedented, nodes[0], new_text)
    if edited is None or edited == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(edited))
    except SyntaxError:
        return None
    return {
        "text": edited,
        "operator": label,
        "opname": opname,
        "kind": "literal_form",
        "confidence": "unique",
    }
