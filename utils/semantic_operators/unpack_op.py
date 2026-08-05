from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.container import _replace_span, _top_level_displays
from utils.semantic_operators.leaf import _instructions

# --- opcode families ------------------------------------------------------------------
_NAME_LOAD_OPS = {"LOAD_NAME", "LOAD_GLOBAL", "LOAD_FAST", "LOAD_DEREF"}
_SINGLE_LOAD_OPS = _NAME_LOAD_OPS | {"LOAD_CONST", "LOAD_ATTR", "LOAD_METHOD", "BINARY_SUBSCR"}
# A positional argument built from MORE than a single iterable (BUILD_LIST/…/LIST_TO_TUPLE):
# its presence right before CALL_FUNCTION_EX means the call is NOT a lone ``*single`` splat.
_POS_MULTI_OPS = {"BUILD_LIST", "BUILD_TUPLE", "LIST_EXTEND", "LIST_APPEND", "CALL_INTRINSIC_1",
                  "BUILD_SET", "SET_UPDATE", "BUILD_MAP", "DICT_MERGE", "DICT_UPDATE"}
_FINALIZE_OPS = {"STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "STORE_DEREF",
                 "POP_TOP", "RETURN_VALUE", "YIELD_VALUE"}

_OPAQUE = ("opaque",)


def _argint(ins):
    a = getattr(ins, "arg", None)
    if isinstance(a, int):
        return a
    v = getattr(ins, "argval", None)
    return v if isinstance(v, int) else None


def _load_name(ins) -> str | None:
    v = getattr(ins, "argval", None)
    if isinstance(v, str) and v.isidentifier():
        return v
    r = (getattr(ins, "argrepr", "") or "").strip()
    if "+" in r:  # 3.11 LOAD_GLOBAL "NULL + x"
        r = r.split("+")[-1].strip()
    return r if r.isidentifier() else None


def _real_instructions(bytecode) -> list:
    return [i for i in _instructions(bytecode) if getattr(i, "opname", None) != "CACHE"]


def _node_source(fragment: str, dedented: str, node: ast.AST) -> str | None:
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
    if li0 == li1:
        return orig[li0][start:end]
    return orig[li0][start:] + "".join(orig[li0 + 1:li1]) + orig[li1][:end]


# starred-assignment recovery:  a, b = it   ->   a, *b = it   (GT UNPACK_EX)

def _unpack_candidate(gt: list, der: list, fragment: str) -> dict | None:
    gt_ux = [i for i in gt if getattr(i, "opname", None) == "UNPACK_EX"]
    if len(gt_ux) != 1:
        return None
    # Derived already carrying an UNPACK_EX is either already-starred or wrongly-starred:
    # repositioning a star is riskier than the flat case, so we defer.
    if any(getattr(i, "opname", None) == "UNPACK_EX" for i in der):
        return None
    arg = _argint(gt_ux[0])
    if arg is None or arg < 0:
        return None
    before, after = arg & 0xFF, arg >> 8
    total = before + 1 + after

    try:
        dedented = textwrap.dedent(fragment)
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    # Store-context tuple/list targets with the recovered element count and no existing star.
    targets = [n for n in ast.walk(tree)
               if isinstance(n, (ast.Tuple, ast.List)) and isinstance(n.ctx, ast.Store)
               and len(n.elts) == total
               and not any(isinstance(e, ast.Starred) for e in n.elts)]
    if len(targets) != 1:
        return None
    elt = targets[0].elts[before]
    src = _node_source(fragment, dedented, elt)
    if src is None:
        return None
    edited = _replace_span(fragment, dedented, elt, "*" + src)
    if edited is None or edited == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(edited))
    except SyntaxError:
        return None
    return {
        "text": edited,
        "operator": f"star assignment target (*{src.strip()}) from UNPACK_EX",
        "opname": "UNPACK_EX",
        "kind": "unpack",
        "confidence": "unique",
    }


# call-splat recovery:  f(args) -> f(*args)  /  f(kw) -> f(**kw)   (GT CALL_FUNCTION_EX)

def _callex_star(gt: list) -> str | None:
    idx = [i for i, ins in enumerate(gt) if getattr(ins, "opname", None) == "CALL_FUNCTION_EX"]
    if len(idx) != 1:
        return None
    k = idx[0]
    flag = _argint(gt[k])
    if flag == 0:
        prev = gt[k - 1] if k - 1 >= 0 else None
        if prev is not None and getattr(prev, "opname", None) not in _POS_MULTI_OPS:
            return "*"
        return None
    if flag == 1 and k - 4 >= 0:
        i1, i2, i3, i4 = gt[k - 1], gt[k - 2], gt[k - 3], gt[k - 4]
        if (getattr(i1, "opname", None) in ("DICT_MERGE", "DICT_UPDATE") and _argint(i1) == 1
                and getattr(i2, "opname", None) in _SINGLE_LOAD_OPS
                and getattr(i3, "opname", None) == "BUILD_MAP" and (_argint(i3) or 0) == 0
                and getattr(i4, "opname", None) == "LOAD_CONST" and getattr(i4, "argval", 1) == ()):
            return "**"
    return None


def _callex_candidate(gt: list, der: list, fragment: str) -> dict | None:
    if sum(1 for i in gt if getattr(i, "opname", None) == "CALL_FUNCTION_EX") != 1:
        return None
    if any(getattr(i, "opname", None) == "CALL_FUNCTION_EX" for i in der):
        return None  # derived already splatted -> nothing to attribute -> defer
    star = _callex_star(gt)
    if star is None:
        return None

    try:
        dedented = textwrap.dedent(fragment)
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    # The unique call passing exactly one positional argument, no keywords, not already starred.
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and len(n.args) == 1 and not n.keywords
             and not isinstance(n.args[0], ast.Starred)]
    if len(calls) != 1:
        return None
    arg = calls[0].args[0]
    src = _node_source(fragment, dedented, arg)
    if src is None:
        return None
    edited = _replace_span(fragment, dedented, arg, star + src)
    if edited is None or edited == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(edited))
    except SyntaxError:
        return None
    return {
        "text": edited,
        "operator": f"call splat {star}{src.strip()} from CALL_FUNCTION_EX",
        "opname": "CALL_FUNCTION_EX",
        "kind": "unpack",
        "confidence": "unique",
    }


# display-unpack recovery:  {**base, 'k': 1} / [*a, b] / {*a, b}  reconstructed from GT

class _Disp:
    __slots__ = ("kind", "parts", "has_spread", "ok")

    def __init__(self, kind):
        self.kind, self.parts, self.has_spread, self.ok = kind, [], False, True


def _is_const(e) -> bool:
    return isinstance(e, tuple) and len(e) == 2 and e[0] == "const"


def _is_name(e) -> bool:
    return isinstance(e, tuple) and len(e) == 2 and e[0] == "name"


def _elem_text(e) -> str | None:
    if _is_const(e):
        return repr(e[1])
    if _is_name(e):
        return e[1]
    return None


def _reconstruct_spread_displays(insts: list) -> list[tuple[str, str]]:
    stack: list = []
    out: list[tuple[str, str]] = []

    def pop():
        return stack.pop() if stack else _OPAQUE

    def finalize(entry):
        if isinstance(entry, _Disp) and entry.ok and entry.has_spread and entry.parts:
            rendered = _render(entry)
            if rendered is not None:
                out.append((entry.kind, rendered))

    for ins in insts:
        op = getattr(ins, "opname", None)
        if op is None or op == "CACHE":
            continue

        # A branch crossing a live display makes an element conditional -> poison it.
        if getattr(ins, "is_jump", False) or "JUMP" in op or op in ("FOR_ITER", "SEND"):
            for e in stack:
                if isinstance(e, _Disp):
                    e.ok = False
            if op.startswith("POP_JUMP"):
                pop()
            continue

        if op == "LOAD_CONST":
            stack.append(("const", getattr(ins, "argval", None)))
            continue
        if op in _NAME_LOAD_OPS:
            nm = _load_name(ins)
            stack.append(("name", nm) if nm else _OPAQUE)
            continue
        if op in ("LOAD_ATTR", "LOAD_METHOD"):
            base = pop()
            attr = (getattr(ins, "argrepr", "") or "").strip()
            stack.append(("name", f"{base[1]}.{attr}") if _is_name(base) and attr.isidentifier()
                         else _OPAQUE)
            continue

        if op in ("BUILD_LIST", "BUILD_SET"):
            n = _argint(ins) or 0
            elems = [pop() for _ in range(n)][::-1]
            disp = _Disp("list" if op == "BUILD_LIST" else "set")
            for e in elems:
                t = _elem_text(e)
                if t is None:
                    disp.ok = False
                else:
                    disp.parts.append(("elem", t))
            stack.append(disp)
            continue
        if op == "BUILD_MAP":
            n = _argint(ins) or 0
            raw = [pop() for _ in range(2 * n)][::-1]
            disp = _Disp("dict")
            for i in range(0, len(raw) - 1, 2):
                k, v = raw[i], raw[i + 1]
                if _is_const(k) and _is_const(v):
                    disp.parts.append(("kv", repr(k[1]), repr(v[1])))
                else:
                    disp.ok = False
            stack.append(disp)
            continue
        if op == "BUILD_CONST_KEY_MAP":
            n = _argint(ins) or 0
            keys = pop()
            vals = [pop() for _ in range(n)][::-1]
            disp = _Disp("dict")
            if _is_const(keys) and isinstance(keys[1], tuple) and len(keys[1]) == n:
                for key, v in zip(keys[1], vals):
                    if _is_const(v):
                        disp.parts.append(("kv", repr(key), repr(v[1])))
                    else:
                        disp.ok = False
            else:
                disp.ok = False
            stack.append(disp)
            continue

        if op in ("LIST_EXTEND", "SET_UPDATE", "DICT_UPDATE", "DICT_MERGE"):
            src = pop()
            tgt = stack[-1] if stack else None
            if not isinstance(tgt, _Disp):
                continue
            if _is_name(src):
                tgt.parts.append(("spread", src[1]))
                tgt.has_spread = True
            elif isinstance(src, _Disp) and src.ok and src.kind == tgt.kind:
                tgt.parts.extend(src.parts)
                tgt.has_spread = tgt.has_spread or src.has_spread
            else:
                tgt.ok = False
            continue
        if op in ("LIST_APPEND", "SET_ADD"):
            src = pop()
            tgt = stack[-1] if stack else None
            if isinstance(tgt, _Disp):
                t = _elem_text(src)
                if t is None:
                    tgt.ok = False
                else:
                    tgt.parts.append(("elem", t))
            continue
        if op == "MAP_ADD":
            v = pop()
            k = pop()
            tgt = stack[-1] if stack else None
            if isinstance(tgt, _Disp):
                if _is_const(k) and _is_const(v):
                    tgt.parts.append(("kv", repr(k[1]), repr(v[1])))
                else:
                    tgt.ok = False
            continue

        if op in _FINALIZE_OPS:
            finalize(pop())
            continue
        if op == "RETURN_CONST":
            continue

        # any other value-producing load pushes an opaque; everything else is left alone.
        if op.startswith("LOAD_") or op in ("PUSH_NULL", "IMPORT_FROM", "COPY", "CALL",
                                            "CALL_FUNCTION", "BINARY_SUBSCR"):
            stack.append(_OPAQUE)

    return out


def _render(disp: _Disp) -> str | None:
    segs: list[str] = []
    for p in disp.parts:
        if p[0] == "spread":
            segs.append(("**" if disp.kind == "dict" else "*") + p[1])
        elif p[0] == "kv":
            segs.append(f"{p[1]}: {p[2]}")
        elif p[0] == "elem":
            segs.append(p[1])
        else:
            return None
    if disp.kind == "list":
        return "[" + ", ".join(segs) + "]"
    return "{" + ", ".join(segs) + "}"  # dict and set both use braces (non-empty here)


_DISPLAY_CLS = {"dict": ast.Dict, "list": ast.List, "set": ast.Set}
_DISPLAY_OPNAME = {"dict": "DICT_UPDATE", "list": "LIST_EXTEND", "set": "SET_UPDATE"}


def _display_candidate(gt: list, der: list, fragment: str) -> dict | None:
    reconstructed = _reconstruct_spread_displays(gt)
    if not reconstructed:
        return None
    try:
        dedented = textwrap.dedent(fragment)
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    for kind in ("dict", "list", "set"):
        texts = [t for (k, t) in reconstructed if k == kind]
        if len(texts) != 1:
            continue
        displays = _top_level_displays(tree, _DISPLAY_CLS[kind])
        if len(displays) != 1:
            continue
        edited = _replace_span(fragment, dedented, displays[0], texts[0])
        if edited is None or edited == fragment:
            continue
        try:
            ast.parse(textwrap.dedent(edited))
        except SyntaxError:
            continue
        return {
            "text": edited,
            "operator": f"reconstruct {kind} unpack {texts[0]} from GT",
            "opname": _DISPLAY_OPNAME[kind],
            "kind": "unpack",
            "confidence": "unique",
        }
    return None


# top-level dispatch

def unpack_op_candidate(gt_code_object, derived_code_object, fragment: str,
                        repair_context: dict | None) -> dict | None:
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") != "Different bytecode":
            return None

        gt = _real_instructions(gt_code_object)
        der = _real_instructions(derived_code_object)
        if not gt:
            return None

        for detector in (_unpack_candidate, _callex_candidate, _display_candidate):
            out = detector(gt, der, fragment)
            if out is not None:
                return out
        return None
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
