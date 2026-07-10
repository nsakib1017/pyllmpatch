"""Ternary (conditional-expression) operator: deterministically recover a
``X if COND else Y`` *value* expression from GROUND-TRUTH bytecode when the decompiler
mangled it (rendered only one branch, stranded it as dead code, or wrote an ``if``
statement in its place).

A ternary is a value-producing branch, so the correct shape is bytecode-determined when
the three sub-expressions are simple. In CPython 3.10-3.14 it takes one of two shapes:

  * duplicated-tail form (the branch tail ends in a terminator -> the tail is copied into
    both arms)::

        <COND>                 # a value pushed for the test
        POP_JUMP_IF_FALSE  L   # bare pop-jump (NO `COPY 1` before / `POP_TOP` after,
        <X>                    #   which is what a short-circuit boolean value uses)
        <TAIL>                 # e.g. RETURN_VALUE  /  STORE x; LOAD x; RETURN_VALUE
     L: <Y>
        <TAIL>                 # identical duplicated tail

  * convergent form (the consumer is a normal, non-terminating instruction)::

        <COND>
        POP_JUMP_IF_FALSE  L
        <X>
        JUMP_FORWARD       M   # then-arm jumps over the else-arm to the merge
     L: <Y>                    # else-arm falls through to the merge
     M: <CONSUMER>             # one shared value consumer (STORE / a call, ...)

The load-bearing invariant that distinguishes a ternary *expression* from an ``if``
*statement* is: **neither arm contains a STORE inside its own value** — the consumer
(a ``RETURN`` / ``STORE_*`` / a single call argument) is *shared* (either the duplicated
tail or the convergence point). A real ``if`` statement whose arms run several statements
puts a ``STORE`` inside an arm, so ``_recover_value_expr`` rejects it and we DEFER (that
is the control-flow operator's / the LLM's job).

Everything is unique-or-defer and reads truth only from GT:

  1. Guard the regime (``Different bytecode`` / ``Different control flow``).
  2. Require EXACTLY ONE conditional branch (``POP_JUMP_IF_*``) in the object — this single
     gate simultaneously enforces "exactly one ternary", "no nested ternary", and "simple
     (non short-circuit) condition"; a compound ``a and b`` / another ``if`` / a nested
     ternary all add a second conditional jump and DEFER.
  3. Recover ``COND`` (reusing ``control_flow._recover_if_condition``) and the two arm
     values ``X`` / ``Y`` (a small operand stack machine over name / attribute / constant /
     comparison / identity / membership / simple-call loads). Any arm that is not a simple
     recoverable value (a store, a bigger expression, a nested branch) DEFERS.
  4. Assemble ``X if COND else Y`` and locate the UNIQUE derived value position it feeds
     (the sole ``return`` / assignment-to-target / call-argument node whose text equals the
     arm the decompiler rendered), and splice the ternary over it.

Every emitted fragment is re-parsed; a no-op (derived already matches GT) DEFERS. The
repair loop's recompile + oracle accept gate is the backstop — a wrong splice cannot land
because acceptance requires the combined distance to strictly drop. This never raises
(top-level try/except -> None).

Honest scope: single non-nested ternary whose condition + both arms are simple loads /
attribute chains / constants / comparisons / simple calls, feeding a return, a plain-name
assignment, or a single call argument. Compound / negated (``POP_JUMP_IF_TRUE``) conditions,
nested ternaries, arms that are richer expressions, and value positions that are not
unique all DEFER. A single-statement ``if c: x = a`` / ``else: x = b`` compiles byte-for-byte
identically to the assign-ternary, so firing on it is a bytecode-equivalent (distance-0)
rewrite, not an error.
"""
from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.control_flow import (
    _HEADER_JUMPS,
    _const_repr,
    _recover_if_condition,
)
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
_STORE_OPS = {"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}
_CALL_OPS = {"CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_FUNCTION_KW"}
# Ops a simple value expression may be built from (also the backward "run" for the
# condition, so a call/attr/compare condition is captured before it is isolated).
_VALUE_OPS = _NAME_LOADS | {
    "LOAD_CONST", "LOAD_ATTR", "LOAD_METHOD", "COMPARE_OP", "IS_OP", "CONTAINS_OP",
} | _CALL_OPS
# Conditional branch opcodes. Exactly one of these in the object is our core gate; the
# OR_POP forms are value-producing short-circuit booleans (bool_shortcircuit's job).
_ORPOP = {"JUMP_IF_FALSE_OR_POP", "JUMP_IF_TRUE_OR_POP"}
_CONDITIONAL_JUMPS = _HEADER_JUMPS | _ORPOP
# Terminators (a branch tail that ends in one of these got its tail duplicated).
_TERMINATORS = {"RETURN_VALUE", "RETURN_CONST", "RAISE_VARARGS", "RERAISE"}
# Unconditional forward jumps (the then-arm's jump over the else-arm in convergent form).
_UNCOND_FWD = {"JUMP_FORWARD", "JUMP_ABSOLUTE", "JUMP"}


# ---------------------------------------------------------------------------
# Small instruction helpers.
# ---------------------------------------------------------------------------
def _filtered(bc) -> list:
    """The instruction stream with inert prologue/pseudo ops removed."""
    return [i for i in _instructions(bc)
            if getattr(i, "opname", None) is not None and getattr(i, "opname", None) not in _INERT]


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


def _index_at_offset(filtered: list, off) -> int | None:
    if off is None:
        return None
    for idx, i in enumerate(filtered):
        if getattr(i, "offset", None) == off:
            return idx
    return None


def _arg_is_one(ins) -> bool:
    return getattr(ins, "arg", None) == 1 or getattr(ins, "argval", None) == 1


def _name_arg(ins) -> str | None:
    """The identifier a name/attr load references (argval when a clean identifier, else a
    cleaned argrepr that strips 3.11+ ``NULL + x`` / ``NULL|self + x`` prefixes)."""
    v = getattr(ins, "argval", None)
    if isinstance(v, str) and v.isidentifier():
        return v
    name = _clean_name(getattr(ins, "argrepr", ""))
    return name if name and name.isidentifier() else None


def _call_argcount(ins) -> int | None:
    for attr in ("arg", "argval"):
        v = getattr(ins, attr, None)
        if isinstance(v, int) and v >= 0:
            return v
    return None


def _sig(ins):
    """A version-stable (opname, operand-key) signature for suffix matching."""
    op = getattr(ins, "opname", None)
    key = None
    if op in _NAME_LOADS or op in _STORE_OPS or op in ("LOAD_ATTR", "LOAD_METHOD"):
        key = _name_arg(ins)
    elif op == "LOAD_CONST":
        try:
            key = repr(getattr(ins, "argval", None))
        except Exception:  # noqa: BLE001
            key = None
    elif op in _CALL_OPS:
        key = _call_argcount(ins)
    elif op in ("COMPARE_OP", "IS_OP", "CONTAINS_OP", "BINARY_OP"):
        key = (getattr(ins, "argrepr", "") or "").strip()
    return (op, key)


def _tail_sig(insts: list) -> list:
    return [_sig(i) for i in insts]


# ---------------------------------------------------------------------------
# Value recovery: a small stack machine (simple loads + simple calls).
# ---------------------------------------------------------------------------
def _recover_value_expr(insts: list) -> str | None:
    """Reconstruct a SIMPLE value expression from a run of load instructions, or None to
    defer. Handles name / attribute chain / constant / comparison / identity / membership /
    a simple call of already-simple arguments. A STORE, arithmetic, subscript, container
    build, keyword argument, nested branch, or anything that does not reduce to exactly one
    expression all DEFER (return None) — which is exactly how a real multi-statement ``if``
    arm is rejected (its value contains a STORE)."""
    stack: list[str] = []
    for i in insts:
        op = getattr(i, "opname", None)
        if op in _INERT:
            continue
        if op in _NAME_LOADS:
            name = _name_arg(i)
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
            attr = _name_arg(i)
            if not attr:
                return None
            stack[-1] = f"{stack[-1]}.{attr}"
        elif op == "COMPARE_OP":
            if len(stack) < 2:
                return None
            sym = (getattr(i, "argrepr", "") or "").strip()
            if not sym:
                return None
            b, a = stack.pop(), stack.pop()
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
        elif op in _CALL_OPS:
            n = _call_argcount(i)
            if n is None or len(stack) < n + 1:
                return None
            args = stack[len(stack) - n:] if n else []
            del stack[len(stack) - n:]
            callee = stack.pop()
            stack.append(f"{callee}({', '.join(args)})")
        else:
            return None
    if len(stack) != 1:
        return None
    return stack[0]


def _longest_value_prefix(insts: list) -> tuple:
    """Return ``(expr, consumer_index)`` where ``insts[:consumer_index]`` is the LONGEST
    prefix that recovers to EXACTLY ONE value expression, or ``(None, None)``. The
    "longest" rule keeps an arm's own trailing call (``io.BytesIO()`` stays whole) while a
    call that consumes the arm as an *argument* underflows (its callable lives below the
    arm) and so is correctly left as the consumer, not folded into the value."""
    best_expr, best_idx = None, None
    for p in range(1, len(insts) + 1):
        expr = _recover_value_expr(insts[:p])
        if expr is not None:
            best_expr, best_idx = expr, p
    return (best_expr, best_idx)


def _isolate_condition(filtered: list, j: int) -> list | None:
    """The instructions computing the branch condition: the largest suffix of the value-op
    run immediately preceding the jump ``j`` that reduces to EXACTLY ONE expression. The
    "largest suffix" rule drops any leftover stack value pushed *before* the condition (e.g.
    the callable of a call-argument ternary) while keeping a real compound value operand."""
    s = j
    while s - 1 >= 0 and getattr(filtered[s - 1], "opname", None) in _VALUE_OPS:
        s -= 1
    run = filtered[s:j]
    for start in range(len(run)):
        sub = run[start:]
        if sub and _recover_value_expr(sub) is not None:
            return sub
    return None


def _classify_consumer(ins) -> tuple | None:
    op = getattr(ins, "opname", None)
    if op == "RETURN_VALUE":
        return ("return", None)
    if op in _STORE_OPS:
        name = _name_arg(ins)
        return ("assign", name) if name else None
    if op in _CALL_OPS:
        return ("call", None)
    return None


# ---------------------------------------------------------------------------
# Ternary detection.
# ---------------------------------------------------------------------------
def _detect_single_ternary(bc) -> dict | None:
    """Detect the object's single ternary and recover its parts, or None to defer.
    Returns ``{"cond", "then", "else", "consumer"}`` where ``then``/``else`` are the
    recovered arm expression texts and ``consumer`` is ``("return"|"assign"|"call", name)``.
    """
    filtered = _filtered(bc)
    if not filtered:
        return None

    cond_jumps = [idx for idx, i in enumerate(filtered)
                  if getattr(i, "opname", None) in _CONDITIONAL_JUMPS]
    if len(cond_jumps) != 1:
        return None  # 0 -> nothing; >1 -> compound / nested / another if -> defer
    j = cond_jumps[0]
    jump = filtered[j]
    if getattr(jump, "opname", None) not in _HEADER_JUMPS:
        return None  # an OR_POP short-circuit boolean value -> bool_shortcircuit's job

    # Exclude the 3.12+ short-circuit boolean triple (COPY 1 ; POP_JUMP ; POP_TOP).
    prev = filtered[j - 1] if j > 0 else None
    nxt = filtered[j + 1] if j + 1 < len(filtered) else None
    if prev is not None and getattr(prev, "opname", None) == "COPY" and _arg_is_one(prev):
        return None
    if nxt is not None and getattr(nxt, "opname", None) == "POP_TOP":
        return None

    target = _jump_target(jump)
    k = _index_at_offset(filtered, target)
    if k is None or k <= j:
        return None  # need a forward else-target
    then_raw = filtered[j + 1:k]
    if not then_raw:
        return None

    if getattr(then_raw[-1], "opname", None) in _UNCOND_FWD:
        # Convergent form: then-arm jumps over the else-arm to a single shared consumer.
        merge = _jump_target(then_raw[-1])
        m = _index_at_offset(filtered, merge)
        if m is None or m <= k:
            return None
        x_insts = then_raw[:-1]
        y_insts = filtered[k:m]
        consumer_inst = filtered[m]
        if getattr(consumer_inst, "opname", None) in _TERMINATORS:
            return None  # a terminating merge would be the duplicated-tail form, not this
        then_expr = _recover_value_expr(x_insts)
        else_expr = _recover_value_expr(y_insts)
    else:
        # Duplicated-tail form: both arms produce a value, then share an identical
        # terminating tail (consumer + tail copied). Split each arm at the LONGEST value
        # prefix; the op after it is the (shared) consumer, and the tails must match.
        if getattr(then_raw[-1], "opname", None) not in _TERMINATORS:
            return None
        else_raw = filtered[k:]
        then_expr, x_idx = _longest_value_prefix(then_raw)
        else_expr, y_idx = _longest_value_prefix(else_raw)
        if x_idx is None or y_idx is None or x_idx >= len(then_raw) or y_idx >= len(else_raw):
            return None
        if _tail_sig(then_raw[x_idx:]) != _tail_sig(else_raw[y_idx:]):
            return None  # arms must converge on an identical consumer + duplicated tail
        consumer_inst = then_raw[x_idx]

    if then_expr is None or else_expr is None:
        return None
    consumer = _classify_consumer(consumer_inst)
    if consumer is None:
        return None

    cond_insts = _isolate_condition(filtered, j)
    if cond_insts is None:
        return None
    cond = _recover_if_condition(list(cond_insts) + [jump])
    if cond is None:
        return None

    return {"cond": cond, "then": then_expr, "else": else_expr, "consumer": consumer}


# ---------------------------------------------------------------------------
# Derived-side location: the unique value position the ternary feeds.
# ---------------------------------------------------------------------------
def _norm_dump(expr_text: str) -> str | None:
    try:
        return ast.dump(ast.parse(expr_text, mode="eval").body)
    except (SyntaxError, ValueError):
        return None


def _locate_position(tree: ast.AST, consumer: tuple, arm_dumps: set) -> ast.AST | None:
    """The UNIQUE derived expression node that (a) sits in the consumer position implied by
    GT and (b) has the same AST as one of the rendered arms — i.e. the value the decompiler
    kept from one branch. None on zero or multiple matches."""
    kind, name = consumer
    candidates: list[ast.AST] = []
    for node in ast.walk(tree):
        if kind == "return":
            if isinstance(node, ast.Return) and node.value is not None:
                if ast.dump(node.value) in arm_dumps:
                    candidates.append(node.value)
        elif kind == "assign":
            if (isinstance(node, ast.Assign) and node.value is not None and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
                if ast.dump(node.value) in arm_dumps:
                    candidates.append(node.value)
            elif (isinstance(node, ast.AnnAssign) and node.value is not None
                  and isinstance(node.target, ast.Name) and node.target.id == name):
                if ast.dump(node.value) in arm_dumps:
                    candidates.append(node.value)
        elif kind == "call":
            if isinstance(node, ast.Call):
                for arg in node.args:
                    if not isinstance(arg, ast.Starred) and ast.dump(arg) in arm_dumps:
                        candidates.append(arg)
    return candidates[0] if len(candidates) == 1 else None


# ---------------------------------------------------------------------------
# Operator entry point.
# ---------------------------------------------------------------------------
def ternary_op_candidate(gt_code_object, derived_code_object, fragment: str,
                         repair_context: dict | None) -> dict | None:
    """Recover a mangled ``X if COND else Y`` value from GT bytecode and splice it into the
    derived fragment. Returns ``{"text", "operator", "opname", "kind", "confidence"}`` or
    None to defer. Never raises (top-level try/except -> None)."""
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") not in ("Different bytecode", "Different control flow"):
            return None

        unit = _detect_single_ternary(gt_code_object)
        if unit is None:
            return None

        tern = f"{unit['then']} if {unit['cond']} else {unit['else']}"
        # Never emit anything that does not parse as a genuine conditional expression.
        try:
            parsed = ast.parse(tern, mode="eval")
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed.body, ast.IfExp):
            return None

        try:
            dedented = textwrap.dedent(fragment)
            tree = ast.parse(dedented)
        except SyntaxError:
            return None

        arm_dumps = {d for d in (_norm_dump(unit["then"]), _norm_dump(unit["else"])) if d}
        if not arm_dumps:
            return None
        node = _locate_position(tree, unit["consumer"], arm_dumps)
        if node is None:
            return None

        edited = _replace_node_span(fragment, dedented, node, tern)
        if edited is None or edited == fragment:
            return None
        # Never emit a fragment that does not parse.
        try:
            ast.parse(textwrap.dedent(edited))
        except SyntaxError:
            return None

        kind = unit["consumer"][0]
        return {
            "text": edited,
            "operator": f"recover ternary `{tern}` into {kind}",
            "opname": "POP_JUMP_IF_FALSE",
            "kind": "ternary",
            "confidence": "unique",
        }
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
