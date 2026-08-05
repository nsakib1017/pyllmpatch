from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.control_flow import _const_repr
from utils.semantic_operators.leaf import _instructions, _replace_node_span
from utils.semantic_operators.structural import _clean_name

# Pseudo / prologue instructions that carry no value or structure for our purposes.
_INERT = {
    "CACHE", "RESUME", "PRECALL", "MAKE_CELL", "COPY_FREE_VARS", "EXTENDED_ARG",
    "NOP", "GEN_START", "RETURN_GENERATOR", "PUSH_NULL",
}
_NAME_LOADS = {
    "LOAD_FAST", "LOAD_NAME", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_CLASSDEREF",
    "LOAD_FAST_CHECK", "LOAD_FAST_BORROW",
}
# Ops a simple operand may be built from (also the "glue" allowed between boundaries).
_OPERAND_OPS = _NAME_LOADS | {
    "LOAD_CONST", "LOAD_ATTR", "LOAD_METHOD", "COMPARE_OP", "IS_OP", "CONTAINS_OP",
}
_STORE_OPS = {"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}

# Value-producing short-circuit boundary opcodes.
_ORPOP_FALSE = {"JUMP_IF_FALSE_OR_POP"}   # `and` (3.10-3.11)
_ORPOP_TRUE = {"JUMP_IF_TRUE_OR_POP"}     # `or`  (3.10-3.11)
_POP_JUMP_FALSE = {"POP_JUMP_IF_FALSE", "POP_JUMP_FORWARD_IF_FALSE",
                   "POP_JUMP_BACKWARD_IF_FALSE"}   # `and` (3.12+, with COPY/POP_TOP)
_POP_JUMP_TRUE = {"POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_TRUE",
                  "POP_JUMP_BACKWARD_IF_TRUE"}      # `or`  (3.12+, with COPY/POP_TOP)


# Small instruction helpers.
def _filtered(bc) -> list:
    return [i for i in _instructions(bc)
            if getattr(i, "opname", None) not in _INERT and getattr(i, "opname", None) is not None]


def _arg_is_one(ins) -> bool:
    return getattr(ins, "arg", None) == 1 or getattr(ins, "argval", None) == 1


def _jump_target(ins):
    v = getattr(ins, "argval", None)
    if isinstance(v, int):
        return v
    r = (getattr(ins, "argrepr", "") or "").strip()
    if r.startswith("to "):
        try:
            return int(r[3:].strip())
        except ValueError:
            return None
    return None


def _store_name(ins) -> str | None:
    v = getattr(ins, "argval", None)
    if isinstance(v, str) and v.isidentifier():
        return v
    name = _clean_name(getattr(ins, "argrepr", ""))
    return name if name and name.isidentifier() else None


# Operand recovery: a simple stack machine over one operand's loads.
def _recover_expr(insts: list) -> str | None:
    stack: list[str] = []
    for i in insts:
        op = getattr(i, "opname", None)
        if op in _INERT:
            continue
        if op in _NAME_LOADS:
            name = _clean_name(getattr(i, "argrepr", ""))
            if not name:
                v = getattr(i, "argval", None)
                name = v if isinstance(v, str) and v.isidentifier() else None
            if not name:
                return None
            stack.append(name)
        elif op == "LOAD_CONST":
            rep = _const_repr(getattr(i, "argval", None))
            if rep is None:
                return None
            stack.append(rep)
        elif op in ("LOAD_ATTR", "LOAD_METHOD"):
            if not stack:
                return None
            attr = _clean_name(getattr(i, "argrepr", ""))
            if not attr:
                v = getattr(i, "argval", None)
                attr = v if isinstance(v, str) and v.isidentifier() else None
            if not attr:
                return None
            stack[-1] = f"{stack[-1]}.{attr}"
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
    if len(stack) != 1:
        return None
    return stack[0]


# Boundary + region detection.
def _boundaries(filtered: list) -> list[dict]:
    out: list[dict] = []
    n = len(filtered)
    for idx, ins in enumerate(filtered):
        op = getattr(ins, "opname", None)
        if op in _ORPOP_FALSE:
            out.append({"lo": idx, "hi": idx, "kind": "and", "target": _jump_target(ins)})
        elif op in _ORPOP_TRUE:
            out.append({"lo": idx, "hi": idx, "kind": "or", "target": _jump_target(ins)})
        elif op in _POP_JUMP_FALSE or op in _POP_JUMP_TRUE:
            prev = filtered[idx - 1] if idx > 0 else None
            nxt = filtered[idx + 1] if idx + 1 < n else None
            if (prev is not None and getattr(prev, "opname", None) == "COPY" and _arg_is_one(prev)
                    and nxt is not None and getattr(nxt, "opname", None) == "POP_TOP"):
                kind = "and" if op in _POP_JUMP_FALSE else "or"
                out.append({"lo": idx - 1, "hi": idx + 1, "kind": kind, "target": _jump_target(ins)})
            # else: a bare POP_JUMP (if-condition / loop) — not our value-producing form.
    return out


def _all_operand_ops(insts: list) -> bool:
    return all(getattr(i, "opname", None) in _OPERAND_OPS or getattr(i, "opname", None) in _INERT
               for i in insts)


def _build_region_unit(filtered: list, group: list[dict]) -> dict:
    invalid = {"valid": False}

    kinds = {b["kind"] for b in group}
    targets = {b["target"] for b in group}
    if len(kinds) != 1 or len(targets) != 1:
        return invalid  # mixed and/or, or divergent targets -> precedence-ambiguous
    kind = next(iter(kinds))
    target = next(iter(targets))
    if target is None:
        return invalid

    # The consumer is the instruction sitting at the common branch target.
    consumer_idx = next((k for k, i in enumerate(filtered)
                         if getattr(i, "offset", None) == target), None)
    if consumer_idx is None or consumer_idx <= group[-1]["hi"]:
        return invalid
    consumer = filtered[consumer_idx]
    cop = getattr(consumer, "opname", None)
    if cop == "RETURN_VALUE":
        consumer_kind = ("return", None)
    elif cop in _STORE_OPS:
        name = _store_name(consumer)
        if not name:
            return invalid
        consumer_kind = ("assign", name)
    else:
        return invalid  # feeds a call / bigger expression / if-condition -> defer

    # operand0: walk left over operand ops from the first boundary.
    lo0 = group[0]["lo"]
    s = lo0 - 1
    while s >= 0 and getattr(filtered[s], "opname", None) in _OPERAND_OPS:
        s -= 1
    segments = [filtered[s + 1:lo0]]
    # middle operands: between successive boundaries.
    for a, b in zip(group, group[1:]):
        segments.append(filtered[a["hi"] + 1:b["lo"]])
    # last operand: after the final boundary, up to the consumer.
    segments.append(filtered[group[-1]["hi"] + 1:consumer_idx])

    operands: list[str] = []
    for seg in segments:
        if not _all_operand_ops(seg):
            return invalid
        expr = _recover_expr(seg)
        if expr is None:
            return invalid
        operands.append(expr)
    if len(operands) < 2:
        return invalid

    joiner = " and " if kind == "and" else " or "
    return {
        "valid": True,
        "kind": kind,
        "operands": operands,
        "expr": joiner.join(operands),
        "consumer": consumer_kind,
    }


def _find_units(bc) -> list[dict]:
    filtered = _filtered(bc)
    bnds = _boundaries(filtered)
    if not bnds:
        return []
    # Group boundaries that are connected: only operand loads sit between them.
    groups: list[list[dict]] = [[bnds[0]]]
    for b in bnds[1:]:
        between = filtered[groups[-1][-1]["hi"] + 1:b["lo"]]
        if _all_operand_ops(between):
            groups[-1].append(b)
        else:
            groups.append([b])
    return [_build_region_unit(filtered, g) for g in groups]


# Derived-side location: find the unique node the boolean feeds.
def _locate_value_node(tree: ast.AST, consumer_kind: tuple):
    kind, name = consumer_kind
    if kind == "return":
        nodes = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Return) and n.value is not None]
    else:  # assign to a plain name
        nodes = []
        for n in ast.walk(tree):
            if (isinstance(n, ast.Assign) and n.value is not None and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name):
                nodes.append(n.value)
            elif (isinstance(n, ast.AnnAssign) and n.value is not None
                  and isinstance(n.target, ast.Name) and n.target.id == name):
                nodes.append(n.value)
        return nodes[0] if len(nodes) == 1 else None
    if kind == "return":
        vals = [n.value for n in nodes]
        return vals[0] if len(vals) == 1 else None
    return None


# Operator entry point.
def bool_shortcircuit_candidate(gt_code_object, derived_code_object, fragment: str,
                                repair_context: dict | None) -> dict | None:
    try:
        return _bool_shortcircuit_candidate(gt_code_object, derived_code_object, fragment, repair_context)
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None


def _bool_shortcircuit_candidate(gt_code_object, derived_code_object, fragment: str, repair_context: dict | None) -> dict | None:
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") not in ("Different bytecode", "Different control flow"):
        return None

    # Exactly one region, and it must be fireable.
    units = _find_units(gt_code_object)
    if len(units) != 1:
        return None
    unit = units[0]
    if not unit.get("valid"):
        return None

    expr = unit["expr"]
    # Never emit an expression that does not parse as a boolean expression.
    try:
        parsed = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed.body, ast.BoolOp):
        return None

    try:
        dedented = textwrap.dedent(fragment)
        tree = ast.parse(dedented)
    except SyntaxError:
        return None

    value_node = _locate_value_node(tree, unit["consumer"])
    if value_node is None:
        return None

    edited = _replace_node_span(fragment, dedented, value_node, expr)
    if edited is None or edited == fragment:
        return None
    # Never emit a fragment that does not parse.
    try:
        ast.parse(textwrap.dedent(edited))
    except SyntaxError:
        return None

    n = len(unit["operands"])
    consumer_kind = unit["consumer"][0]
    return {
        "text": edited,
        "operator": (f"recover short-circuit `{expr}` "
                     f"({unit['kind']} chain, {n} operands) into {consumer_kind}"),
        "opname": "POP_JUMP_IF_FALSE" if unit["kind"] == "and" else "POP_JUMP_IF_TRUE",
        "kind": "bool_shortcircuit",
        "confidence": "unique",
    }
