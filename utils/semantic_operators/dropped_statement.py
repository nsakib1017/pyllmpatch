from __future__ import annotations

import ast
import textwrap
from collections import Counter

from utils.semantic_operators.leaf import _instructions, _round_trips
from utils.semantic_operators.placeholder import (
    _ANY_STORE_OPCODES,
    _store_name,
    _unique_assign_target,
)

# --- opcode groups ---------------------------------------------------------
_NAME_LOADS = {"LOAD_NAME", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_CLASSDEREF",
               "LOAD_FAST_CHECK"}
_ATTR_LOADS = {"LOAD_ATTR", "LOAD_METHOD"}
_STORE_OPS = {"STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "STORE_DEREF"}
_LOCAL_STORE_OPS = {"STORE_FAST", "STORE_NAME"}
_DELETE_SIMPLE = {"DELETE_NAME", "DELETE_FAST", "DELETE_GLOBAL", "DELETE_DEREF"}
_IMPORT_TAIL_OPS = {"IMPORT_FROM", "STORE_NAME", "STORE_FAST", "STORE_GLOBAL",
                    "STORE_DEREF", "POP_TOP", "CALL_INTRINSIC_1"}
# expression-only opcodes the assert/target mini-machine understands
_EXPR_OPS = _NAME_LOADS | _ATTR_LOADS | {"LOAD_CONST", "COMPARE_OP", "IS_OP", "CONTAINS_OP"}
_ASSERT_TRUE_JUMPS = {"POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_TRUE",
                      "POP_JUMP_BACKWARD_IF_TRUE"}
_ASSERT_NONE_JUMPS = {"POP_JUMP_IF_NONE", "POP_JUMP_FORWARD_IF_NONE",
                      "POP_JUMP_BACKWARD_IF_NONE"}
_ASSERT_NOT_NONE_JUMPS = {"POP_JUMP_IF_NOT_NONE", "POP_JUMP_FORWARD_IF_NOT_NONE",
                          "POP_JUMP_BACKWARD_IF_NOT_NONE"}

_DEFER = object()  # sentinel: a message/test is present but not cleanly reconstructable


# --- small primitives ------------------------------------------------------
def _ident(v) -> str | None:
    return v if isinstance(v, str) and v.isidentifier() else None


def _name_of(ins) -> str | None:
    r = _ident(getattr(ins, "argval", None))
    if r:
        return r
    ar = (getattr(ins, "argrepr", "") or "").strip()
    ar = ar.split("+")[-1].strip()
    return _ident(ar)


def _const_repr(value) -> str | None:
    return repr(value) if _round_trips(value) else None


# --- del reconstruction ----------------------------------------------------
def _recover_load_chain(insts: list, end_idx: int) -> str | None:
    if end_idx < 0 or end_idx >= len(insts):
        return None
    attrs: list[str] = []
    i = end_idx
    while i >= 0 and getattr(insts[i], "opname", None) in _ATTR_LOADS:
        a = _name_of(insts[i])
        if not a:
            return None
        attrs.append(a)
        i -= 1
    if i < 0 or getattr(insts[i], "opname", None) not in _NAME_LOADS:
        return None
    base = _name_of(insts[i])
    if not base:
        return None
    return base + "".join("." + a for a in reversed(attrs))


def _recover_del(insts: list, idx: int) -> str | None:
    ins = insts[idx]
    op = ins.opname
    if op in _DELETE_SIMPLE:
        name = _name_of(ins)
        return f"del {name}" if name else None
    if op == "DELETE_ATTR":
        attr = _name_of(ins)
        obj = _recover_load_chain(insts, idx - 1)
        if attr and obj:
            return f"del {obj}.{attr}"
        return None
    if op == "DELETE_SUBSCR":
        if idx < 2:
            return None
        kins = insts[idx - 1]
        if kins.opname in _NAME_LOADS:
            key = _name_of(kins)
        elif kins.opname == "LOAD_CONST":
            key = _const_repr(getattr(kins, "argval", None))
        else:
            key = None
        if key is None:
            return None
        obj = _recover_load_chain(insts, idx - 2)
        if obj is None:
            return None
        return f"del {obj}[{key}]"
    return None


def _gt_dels(insts: list) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for i, ins in enumerate(insts):
        if ins.opname in _DELETE_SIMPLE or ins.opname in ("DELETE_ATTR", "DELETE_SUBSCR"):
            text = _recover_del(insts, i)
            if text is not None:
                out.append((text, i))
    return out


# --- import reconstruction -------------------------------------------------
def _reconstruct_import(insts: list, import_idx: int) -> str | None:
    ins = insts[import_idx]
    module = getattr(ins, "argval", None)
    module = module if isinstance(module, str) else ""
    if import_idx < 2:
        return None
    lv_ins, fl_ins = insts[import_idx - 2], insts[import_idx - 1]
    if lv_ins.opname != "LOAD_CONST" or fl_ins.opname != "LOAD_CONST":
        return None
    level = lv_ins.argval if isinstance(lv_ins.argval, int) else 0
    fromlist = fl_ins.argval
    full = ("." * level) + module

    tail: list = []
    k = import_idx + 1
    while k < len(insts) and insts[k].opname in _IMPORT_TAIL_OPS:
        tail.append(insts[k])
        k += 1

    if fromlist is None:  # plain ``import a[.b] [as x]``
        stores = [t for t in tail if t.opname in _STORE_OPS]
        if len(stores) != 1 or not full:
            return None
        bound = _name_of(stores[0])
        if not bound:
            return None
        has_from = any(t.opname == "IMPORT_FROM" for t in tail)
        if has_from:
            return f"import {full} as {bound}"
        if bound == module or module.split(".")[0] == bound:
            return f"import {full}"
        return f"import {full} as {bound}"

    if isinstance(fromlist, tuple) and fromlist == ("*",):  # ``from m import *``
        return f"from {full} import *" if full else None
    if not isinstance(fromlist, tuple) or not full:
        return None

    items: list[str] = []
    i = 0
    while i < len(tail):
        t = tail[i]
        if t.opname == "IMPORT_FROM":
            name = _name_of(t)
            if not name or i + 1 >= len(tail) or tail[i + 1].opname not in _STORE_OPS:
                return None
            bound = _name_of(tail[i + 1])
            if not bound:
                return None
            items.append(name if bound == name else f"{name} as {bound}")
            i += 2
            continue
        i += 1
    if not items:
        return None
    return f"from {full} import {', '.join(items)}"


def _all_imports(insts: list) -> list[str]:
    out: list[str] = []
    for i, ins in enumerate(insts):
        if ins.opname == "IMPORT_NAME":
            text = _reconstruct_import(insts, i)
            if text is not None:
                out.append(text)
    return out


# --- assert reconstruction -------------------------------------------------
def _recover_expr(slice_insts: list) -> str | None:
    stack: list[str] = []
    for i in slice_insts:
        op = i.opname
        if op in _NAME_LOADS:
            n = _name_of(i)
            if not n:
                return None
            stack.append(n)
        elif op == "LOAD_CONST":
            r = _const_repr(getattr(i, "argval", None))
            if r is None:
                return None
            stack.append(r)
        elif op in _ATTR_LOADS:
            if not stack:
                return None
            a = _name_of(i)
            if not a:
                return None
            stack[-1] = f"{stack[-1]}.{a}"
        elif op == "COMPARE_OP":
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            sym = (getattr(i, "argrepr", "") or "").strip()
            if not sym:
                return None
            stack.append(f"{a} {sym} {b}")
        elif op == "IS_OP":
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            stack.append(f"{a} is not {b}" if getattr(i, "arg", 0) else f"{a} is {b}")
        elif op == "CONTAINS_OP":
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            stack.append(f"{a} not in {b}" if getattr(i, "arg", 0) else f"{a} in {b}")
        else:
            return None
    return stack[0] if len(stack) == 1 else None


def _recover_assert_msg(insts: list, ae_idx: int):
    between: list = []
    k = ae_idx + 1
    limit = min(len(insts), ae_idx + 8)
    while k < limit and insts[k].opname != "RAISE_VARARGS":
        between.append(insts[k])
        k += 1
    if k >= len(insts) or (k < len(insts) and insts[k].opname != "RAISE_VARARGS"):
        return _DEFER  # no terminating raise in window -> not a clean assert
    if not between:
        return None
    if (len(between) == 2 and between[0].opname == "LOAD_CONST"
            and between[1].opname.startswith("CALL")):
        r = _const_repr(getattr(between[0], "argval", None))
        return r if r is not None else _DEFER
    return _DEFER


def _extract_asserts(insts: list) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for ae, ins in enumerate(insts):
        if ins.opname != "LOAD_ASSERTION_ERROR" or ae == 0:
            continue
        jump = insts[ae - 1]
        jop = jump.opname
        if jop in _ASSERT_TRUE_JUMPS:
            mode = "true"
        elif jop in _ASSERT_NONE_JUMPS:
            mode = "none"
        elif jop in _ASSERT_NOT_NONE_JUMPS:
            mode = "not_none"
        else:
            continue  # POP_JUMP_IF_FALSE etc. -> negated / not an assert guard -> defer
        # backward-collect the test expression run (up to, excluding, the guard jump)
        j = ae - 2
        expr_insts: list = []
        while j >= 0 and insts[j].opname in _EXPR_OPS:
            expr_insts.append(insts[j])
            j -= 1
        expr_insts.reverse()
        expr = _recover_expr(expr_insts)
        if expr is None:
            continue
        if mode == "none":
            cond = f"{expr} is None"
        elif mode == "not_none":
            cond = f"{expr} is not None"
        else:
            cond = expr
        msg = _recover_assert_msg(insts, ae)
        if msg is _DEFER:
            continue
        text = f"assert {cond}" if msg is None else f"assert {cond}, {msg}"
        try:
            ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        out.append((text, ae))
    return out


# --- global reconstruction -------------------------------------------------
def _global_flips(gt: list, der: list) -> set[str]:
    gt_glob, gt_loc, der_glob, der_loc = set(), set(), set(), set()
    for i in gt:
        n = _store_name(i)
        if n is None:
            continue
        if i.opname == "STORE_GLOBAL":
            gt_glob.add(n)
        elif i.opname in _LOCAL_STORE_OPS:
            gt_loc.add(n)
    for i in der:
        n = _store_name(i)
        if n is None:
            continue
        if i.opname == "STORE_GLOBAL":
            der_glob.add(n)
        elif i.opname in _LOCAL_STORE_OPS:
            der_loc.add(n)
    return {n for n in gt_glob if n not in gt_loc and n in der_loc and n not in der_glob}


# --- splicing --------------------------------------------------------------
def _neighbor_splice(gt: list, marker_idx: int, fragment: str, tree: ast.AST,
                     stmt_text: str) -> str | None:
    prec = foll = None
    for i, ins in enumerate(gt):
        if ins.opname in _ANY_STORE_OPCODES:
            n = _store_name(ins)
            if n is None:
                continue
            if i < marker_idx:
                prec = n
            elif i > marker_idx and foll is None:
                foll = n
    anchor = None
    after = True
    if prec is not None:
        a = _unique_assign_target(tree, prec)
        if a is not None:
            anchor, after = a, True
    if anchor is None and foll is not None:
        a = _unique_assign_target(tree, foll)
        if a is not None:
            anchor, after = a, False
    if anchor is None or anchor.lineno is None or anchor.end_lineno is None:
        return None
    orig_lines = fragment.splitlines(keepends=True)
    ref_idx = anchor.lineno - 1
    if ref_idx < 0 or anchor.end_lineno > len(orig_lines):
        return None
    ref_line = orig_lines[ref_idx]
    indent = ref_line[:len(ref_line) - len(ref_line.lstrip())]
    new_line = f"{indent}{stmt_text}\n"
    cut = anchor.end_lineno if after else anchor.lineno - 1
    before, after_l = list(orig_lines[:cut]), list(orig_lines[cut:])
    if before and not before[-1].endswith("\n"):
        before[-1] += "\n"
    return "".join(before + [new_line] + after_l)


def _top_of_body_splice(fragment: str, tree: ast.AST, stmt_text: str) -> str | None:
    body = tree.body
    if len(tree.body) == 1 and isinstance(
            tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = tree.body[0].body
    if not body:
        return None
    orig_lines = fragment.splitlines(keepends=True)
    insert_before = None
    docstring_end = None
    for k, st in enumerate(body):
        if (k == 0 and isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
                and isinstance(st.value.value, str)):
            docstring_end = st.end_lineno
            continue
        insert_before = st
        break
    if insert_before is not None:
        ref_idx = insert_before.lineno - 1
        if ref_idx < 0 or ref_idx >= len(orig_lines):
            return None
        ref_line = orig_lines[ref_idx]
        indent = ref_line[:len(ref_line) - len(ref_line.lstrip())]
        cut = insert_before.lineno - 1
    else:
        if docstring_end is None:
            return None
        d_idx = body[0].lineno - 1
        if d_idx < 0 or d_idx >= len(orig_lines):
            return None
        d_line = orig_lines[d_idx]
        indent = d_line[:len(d_line) - len(d_line.lstrip())]
        cut = docstring_end
    new_line = f"{indent}{stmt_text}\n"
    before, after_l = list(orig_lines[:cut]), list(orig_lines[cut:])
    if before and not before[-1].endswith("\n"):
        before[-1] += "\n"
    return "".join(before + [new_line] + after_l)


def _finalize(new_fragment: str | None, fragment: str, stmt_text: str, opname: str,
              kind_label: str) -> dict | None:
    if new_fragment is None or new_fragment == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(new_fragment))
    except (SyntaxError, ValueError):
        return None
    preview = stmt_text if len(stmt_text) <= 48 else stmt_text[:45] + "..."
    return {
        "text": new_fragment,
        "operator": f"insert dropped {kind_label}: {preview}",
        "opname": opname,
        "kind": "dropped_statement",
        "confidence": "unique",
    }


# --- operator --------------------------------------------------------------
def dropped_statement_candidate(gt_code_object, derived_code_object, fragment: str,
                                repair_context: dict | None) -> dict | None:
    try:
        return _dropped_statement_candidate(gt_code_object, derived_code_object,
                                            fragment, repair_context)
    except Exception:  # noqa: BLE001 -- contract: NEVER raise; defer instead.
        return None


def _dropped_statement_candidate(gt_code_object, derived_code_object, fragment: str,
                                 repair_context: dict | None) -> dict | None:
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    message = failed.get("message")
    # del/import/global need a bytecode divergence; assert typically surfaces as a
    # control-flow divergence (the raise branch), so accept that message too.
    if message == "Different bytecode":
        allow_bytecode_kinds = True
    elif message == "Different control flow":
        allow_bytecode_kinds = False  # only assert is eligible
    else:
        return None

    gt = _instructions(gt_code_object)
    der = _instructions(derived_code_object)
    if not gt:
        return None

    # --- collect the dropped statements of each kind (GT minus derived) -----
    del_drops: list[dict] = []
    import_drops: list[dict] = []
    global_drops: list[dict] = []
    if allow_bytecode_kinds:
        gt_dels = _gt_dels(gt)
        der_del_texts = [t for t, _ in _gt_dels(der)]
        drop = Counter(t for t, _ in gt_dels) - Counter(der_del_texts)
        used: set[int] = set()
        for text, cnt in drop.items():
            for _ in range(cnt):
                idx = next((i for (t, i) in gt_dels if t == text and i not in used), None)
                if idx is None:
                    break
                used.add(idx)
                del_drops.append({"text": text, "idx": idx})

        drop_imp = Counter(_all_imports(gt)) - Counter(_all_imports(der))
        for text, cnt in drop_imp.items():
            for _ in range(cnt):
                import_drops.append({"text": text})

        global_drops = [{"name": n} for n in sorted(_global_flips(gt, der))]

    # asserts (eligible in both regimes)
    gt_asserts = _extract_asserts(gt)
    der_assert_texts = [t for t, _ in _extract_asserts(der)]
    assert_drops: list[dict] = []
    drop_asrt = Counter(t for t, _ in gt_asserts) - Counter(der_assert_texts)
    used_a: set[int] = set()
    for text, cnt in drop_asrt.items():
        for _ in range(cnt):
            idx = next((i for (t, i) in gt_asserts if t == text and i not in used_a), None)
            if idx is None:
                break
            used_a.add(idx)
            assert_drops.append({"text": text, "idx": idx})

    total = len(del_drops) + len(import_drops) + len(global_drops) + len(assert_drops)
    if total != 1:  # exactly one dropped statement of one kind, or defer
        return None

    try:
        tree = ast.parse(textwrap.dedent(fragment))
    except (SyntaxError, ValueError):
        return None

    if del_drops:
        d = del_drops[0]
        new = _neighbor_splice(gt, d["idx"], fragment, tree, d["text"])
        return _finalize(new, fragment, d["text"], gt[d["idx"]].opname, "del")

    if import_drops:
        text = import_drops[0]["text"]
        new = _top_of_body_splice(fragment, tree, text)
        return _finalize(new, fragment, text, "IMPORT_NAME", "import")

    if global_drops:
        name = global_drops[0]["name"]
        text = f"global {name}"
        new = _top_of_body_splice(fragment, tree, text)
        return _finalize(new, fragment, text, "STORE_GLOBAL", "global")

    d = assert_drops[0]
    new = _neighbor_splice(gt, d["idx"], fragment, tree, d["text"])
    return _finalize(new, fragment, d["text"], "LOAD_ASSERTION_ERROR", "assert")
