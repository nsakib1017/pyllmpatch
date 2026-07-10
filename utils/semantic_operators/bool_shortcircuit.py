"""Boolean short-circuit operator: deterministically recover a SHORT-CIRCUIT boolean
expression (``a and b`` / ``p or q`` and single-operator chains) from GROUND-TRUTH
bytecode when the decompiler rendered it wrong.

Python compiles a *value-producing* short-circuit boolean to a self-describing bytecode
shape, so the correct boolean structure is bytecode-determined:

  * 3.10 - 3.11:  the operand loads are separated by ``JUMP_IF_FALSE_OR_POP`` (``and``)
                  or ``JUMP_IF_TRUE_OR_POP`` (``or``); every branch targets the offset
                  where the boolean's value is consumed.
  * 3.12+:        that opcode was split; each operand boundary is the triple
                  ``COPY 1 ; POP_JUMP_IF_FALSE/TRUE ; POP_TOP`` — the ``COPY`` preserves
                  the operand as the possible short-circuit result, the ``POP_JUMP`` tests
                  a duplicate, and the ``POP_TOP`` discards it to evaluate the next operand.
                  ``POP_JUMP_IF_FALSE`` => ``and``, ``POP_JUMP_IF_TRUE`` => ``or``.

The ``COPY``/``POP_TOP`` pair is the load-bearing signature: it is present ONLY when the
boolean's *value* is used (a ``return`` value, an assignment RHS). An ``if a and b:``
*condition* discards the value and compiles to a bare ``POP_JUMP`` chain that is
indistinguishable from nested ``if`` statements — that case is owned by the control-flow
operator / the LLM and is DEFERRED here.

What this operator does, all unique-or-defer, reading truth only from GT:

  1. Guard the regime (``Different bytecode`` / ``Different control flow``).
  2. Structure the GT stream into short-circuit *regions*. A region is a maximal run of
     operand loads glued together by boundaries. A region is *fireable* only when every
     boundary shares one operator (a pure ``and`` chain or pure ``or`` chain), every
     branch targets the same consumer offset, that consumer is a ``RETURN_VALUE`` or a
     simple ``STORE`` (name assignment), and EVERY operand recovers as a simple load
     (name / attribute chain / constant / comparison / identity / membership).
  3. Require EXACTLY ONE region and require it fireable (0, >1, mixed ``and``/``or``,
     a call/nested operand, or a non-value consumer all DEFER).
  4. Recover the operand texts verbatim from GT bytecode and join them with the recovered
     operator, then locate the UNIQUE derived source node the boolean feeds (the sole
     ``return`` value, or the sole assignment to the recovered target name) and splice the
     recovered expression over its value span.

Every emitted fragment is re-parsed; a no-op (derived already matches GT) defers. The
repair loop's recompile + oracle accept gate is the backstop — a wrong splice cannot land
because acceptance requires the combined distance to strictly drop. This operator never
raises (top-level try/except -> None).

Honest scope: single-operator ``and``/``or`` chains feeding a return or a plain-name
assignment. Mixed ``a and b or c`` (precedence-ambiguous to reconstruct), booleans that
feed an ``if`` condition / a call argument / a bigger expression, and any operand richer
than a simple load are all DEFERRED.
"""
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


# ---------------------------------------------------------------------------
# Small instruction helpers.
# ---------------------------------------------------------------------------
def _filtered(bc) -> list:
    """The instruction stream with inert prologue/pseudo ops removed, so offsets of real
    ops still line up with jump targets (which point at kept instructions)."""
    return [i for i in _instructions(bc)
            if getattr(i, "opname", None) not in _INERT and getattr(i, "opname", None) is not None]


def _arg_is_one(ins) -> bool:
    return getattr(ins, "arg", None) == 1 or getattr(ins, "argval", None) == 1


def _jump_target(ins):
    """Absolute target offset of a branch, from ``argval`` (an int) or the ``to N``
    ``argrepr`` fallback. None if unavailable."""
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


# ---------------------------------------------------------------------------
# Operand recovery: a simple stack machine over one operand's loads.
# ---------------------------------------------------------------------------
def _recover_expr(insts: list) -> str | None:
    """Reconstruct a SIMPLE value expression (name / attribute chain / constant /
    comparison / identity / membership) from a run of load instructions, or None to
    defer. Calls, arithmetic, nested boolean/if, subscript, and anything that does not
    reduce to exactly one expression all defer."""
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


# ---------------------------------------------------------------------------
# Boundary + region detection.
# ---------------------------------------------------------------------------
def _boundaries(filtered: list) -> list[dict]:
    """Every value-producing short-circuit boundary, as ``{lo, hi, kind, target}`` where
    (lo, hi) are the inclusive filtered-index span the boundary occupies (a single OR_POP
    jump, or the COPY..POP_TOP triple)."""
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
    """Turn a connected group of boundaries into a unit descriptor. ``valid`` is True only
    for a clean, fireable pure-``and``/pure-``or`` short-circuit feeding a return or a
    simple name assignment with all operands recoverable."""
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
    """Detect every short-circuit region in ``bc`` (a code object / instruction iterable),
    returning one unit descriptor per region. Regions with no boundary are omitted;
    unfireable regions surface as ``{"valid": False}`` so the caller can DEFER on them."""
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


# ---------------------------------------------------------------------------
# Derived-side location: find the unique node the boolean feeds.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Operator entry point.
# ---------------------------------------------------------------------------
def bool_shortcircuit_candidate(gt_code_object, derived_code_object, fragment: str,
                                repair_context: dict | None) -> dict | None:
    """Recover a dropped/mis-rendered short-circuit boolean from GT bytecode and splice it
    into the derived fragment. Returns ``{"text", "operator", "opname", "kind",
    "confidence"}`` or None to defer. Never raises."""
    try:
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
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
