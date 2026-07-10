"""Control-flow RE-NESTING operator (M4+): deterministic re-parenting of a contiguous
run of statements that the ground truth nests inside ONE construct (``if`` / ``for``)
but the decompiler emitted at a SHALLOWER nesting depth.

This generalizes the narrow ``control_flow.control_flow_candidate`` (which handles only a
dropped simple-``if`` around a fully-flat derived body) along two axes:

  1. **Construct kinds** — it re-nests a dropped ``for`` loop as well as a dropped ``if``
     (``while`` / ``with`` / ``try`` are detected but DEFERRED — their headers are not
     recovered here, so the operator declines cleanly rather than guess).
  2. **A PEP 657 co_positions PROOF of re-parenting** — instead of inferring "these
     statements were flattened" purely from structure, it reads the *column* offsets that
     CPython 3.11+ records per instruction (``code.co_positions()``) and proves, statement
     by statement, that the derived run sits exactly one indentation level SHALLOWER than
     GT says it should. Comparing the SAME statement's start-column in GT vs derived
     cancels the keyword/expression offset that otherwise confounds a raw column reading
     (e.g. ``return x`` positions at the ``return`` keyword but ``a = ...`` at the target),
     leaving a clean, keyword-invariant nesting-depth delta.

Why co_positions and not just structure? Two reasons, both verified against real bytecode:

  * The re-parenting direction is *provable*: a positive, constant per-statement column
    delta (GT deeper than derived) is exactly the signature of "GT wraps a run the
    decompiler flattened". A zero delta (e.g. a trailing ``return`` that stays at the outer
    level) correctly identifies the statements that must be LEFT outside the wrapper.
  * Re-nesting changes variable BINDING. A ``for it in items:`` loop variable is a local
    (``LOAD_FAST``) in GT, but the same ``it`` in the flattened derived body is an unbound
    global (``LOAD_GLOBAL``). Statement matching therefore must be binding-insensitive
    (``_renest_signature`` collapses all name-load opcodes to one family), or the run can
    never be located — and once re-nested, recompilation restores ``LOAD_FAST`` and the
    oracle distance collapses to 0.

Every step is unique-or-defer and reads truth only from GT bytecode + GT/derived columns.
Anything ambiguous — no columns (3.10), a non-flat derived body, more than one differing
construct, an unrecoverable header, a non-unique body match, an inconsistent column delta —
returns None. The repair loop still recompiles the result and accepts it only if the oracle
distance strictly drops, so a wrong re-nest cannot land; and this operator NEVER raises.

Honest scope: converts the "wrapper dropped, body flattened to a straight-line derived
body" sub-pattern for ``if`` and ``for``. Arbitrary re-parenting (statements mis-nested
UNDER a wrong parent in a derived body that itself has control flow) is DEFERRED — see the
module-level assessment in the test file for why that case needs the LLM.
"""
from __future__ import annotations

import ast

from utils.semantic_operators.leaf import _instructions
from utils.semantic_operators.structural import _clean_name, _match_indent_convention
from utils.semantic_operators.control_flow import (
    _SKIP_OPS,
    _NAME_LOADS,
    _STORE_OPS,
    _TERMINATORS,
    _bc_to_cft,
    _members,
    _unwrap_body,
    _is_flat,
    _get_member,
    _const_repr,
    _recover_if_condition,
    _unique_contiguous_match,
    _statement_groups,
    _covered_group_ordinals,
    _locate_body,
)

_IF_CLASSES = {"IfThen", "IfElse", "IfJumpElse", "IfElseJump"}
_FOR_CLASSES = {"ForLoop"}
_FLAT_CLASSES = {"BlockTemplate", "InstTemplate", "JumpTemplate", "NopTemplate"}
# Ops that participate in an expression the recoverer can rebuild; anything else defers.
_EXPR_SKIP = _SKIP_OPS | {"PUSH_NULL", "PRECALL", "GET_ITER", "KW_NAMES"}


# ---------------------------------------------------------------------------
# co_positions (PEP 657) column reader.
# ---------------------------------------------------------------------------
def _raw_code(bc):
    """The raw code object behind an EditableBytecode (native ``code`` on the host
    version, xdis ``Code311`` cross-version), or None."""
    return getattr(bc, "codeobj", None) or getattr(bc, "code", None)


def _positions(bc):
    """``list(code.co_positions())`` if COLUMN offsets are available, else None.

    Returns None when the code object is missing, ``co_positions`` is unavailable, or
    every start-column is None (CPython 3.10 predates PEP 657) — the caller defers.
    """
    co = _raw_code(bc)
    if co is None:
        return None
    try:
        pos = list(co.co_positions())
    except Exception:  # noqa: BLE001 - odd/foreign code objects
        return None
    if not pos:
        return None
    if all(sc is None for (_sl, _el, sc, _ec) in pos):
        return None
    return pos


def _cols_by_line(bc):
    """Map source line -> minimum start-column over every code unit on that line, or None
    if columns are unavailable. CACHE code units carry their parent instruction's position,
    so they never lower the true statement indent."""
    pos = _positions(bc)
    if pos is None:
        return None
    d: dict[int, list[int]] = {}
    for (sl, _el, sc, _ec) in pos:
        if sl is None or sc is None:
            continue
        d.setdefault(sl, []).append(sc)
    return {ln: min(cs) for ln, cs in d.items()}


# ---------------------------------------------------------------------------
# GT construct detection (reuses control_flow's CFT navigation).
# ---------------------------------------------------------------------------
def _find_single_construct(cft):
    """Return the unique top-level control construct in the GT tree, or None. The body may
    carry flat (non-control) statements before/after it, but exactly ONE control construct
    may differ; more than one, or a construct nested at the top level, defers."""
    node = _unwrap_body(cft)
    cls = type(node).__name__
    if cls in _IF_CLASSES or cls in _FOR_CLASSES:
        return node
    if cls == "BlockTemplate":
        members = _members(node) or []
        constructs = [m for m in members if type(m).__name__ not in _FLAT_CLASSES]
        if len(constructs) == 1:
            return constructs[0]
    return None


# ---------------------------------------------------------------------------
# Binding-insensitive statement signature (see module docstring).
# ---------------------------------------------------------------------------
def _renest_signature(insts: list) -> list[tuple]:
    """One (family, key) tuple per instruction (``_SKIP_OPS`` already removed by callers),
    collapsing all name-load opcodes to a single ``"LOAD"`` family and all store opcodes to
    ``"STORE"`` so a run can be matched regardless of whether a moved name binds locally or
    globally in the two contexts."""
    out = []
    for i in insts:
        op = i.opname
        if op in _NAME_LOADS:
            out.append(("LOAD", _clean_name(getattr(i, "argrepr", ""))))
        elif op in _STORE_OPS:
            out.append(("STORE", _clean_name(getattr(i, "argrepr", ""))))
        elif op in ("LOAD_ATTR", "LOAD_METHOD"):
            out.append(("LOAD_ATTR", _clean_name(getattr(i, "argrepr", ""))))
        elif op == "LOAD_CONST":
            try:
                out.append(("LOAD_CONST", repr(getattr(i, "argval", None))))
            except Exception:  # noqa: BLE001
                out.append(("LOAD_CONST", None))
        else:
            out.append((op, None))
    return out


# ---------------------------------------------------------------------------
# Expression / for-header recovery from GT bytecode.
# ---------------------------------------------------------------------------
def _recover_expr(insts: list) -> str | None:
    """Rebuild a simple expression's source text from its evaluating instructions, or None.

    Supports names, attribute chains, simple constants, subscripts, binary ops, and single
    calls whose arguments are themselves simple. Anything richer (comprehensions, keyword
    args, starred args, unpackings) defers."""
    stack: list[str] = []
    for i in insts:
        op = i.opname
        if op in _EXPR_SKIP:
            continue
        if op in _NAME_LOADS:
            name = _clean_name(getattr(i, "argrepr", ""))
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
                return None
            stack[-1] = f"{stack[-1]}.{attr}"
        elif op == "BINARY_SUBSCR":
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            stack.append(f"{a}[{b}]")
        elif op == "BINARY_OP":
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            stack.append(f"{a} {getattr(i, 'argrepr', '')} {b}")
        elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD"):
            arg = getattr(i, "arg", None)
            if arg is None or len(stack) < arg + 1:
                return None
            args = [stack.pop() for _ in range(arg)][::-1]
            callee = stack.pop()
            stack.append(f"{callee}({', '.join(args)})")
        else:
            return None
    if len(stack) != 1:
        return None
    expr = stack[0]
    try:
        ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    return expr


def _recover_for_header(gt_insts: list) -> tuple[str, str] | None:
    """Recover ``(target, iterable)`` source text for a single ``for TARGET in ITER:`` from
    GT bytecode, or None. Only a single simple STORE target and a recoverable iterable
    expression are accepted; tuple/starred targets and rich iterables defer."""
    idx = next((k for k, i in enumerate(gt_insts) if i.opname == "FOR_ITER"), None)
    if idx is None:
        return None
    # target: the single STORE immediately after FOR_ITER
    target = None
    for i in gt_insts[idx + 1:]:
        if i.opname in _SKIP_OPS:
            continue
        if i.opname in _STORE_OPS:
            target = _clean_name(getattr(i, "argrepr", ""))
        break  # first real instruction must be the target store; else not a simple target
    if not target:
        return None
    # iterable: the statement group ending just before GET_ITER (which precedes FOR_ITER)
    giter = None
    for k in range(idx - 1, -1, -1):
        if gt_insts[k].opname in _SKIP_OPS:
            continue
        giter = k if gt_insts[k].opname == "GET_ITER" else None
        break
    if giter is None:
        return None
    start = giter
    for k in range(giter - 1, -1, -1):
        start = k
        if getattr(gt_insts[k], "starts_line", None) is not None:
            break
    itr = _recover_expr(gt_insts[start:giter])
    if itr is None:
        return None
    return target, itr


def _for_body_content(insts: list) -> list:
    """The real body instructions of a ForLoop node: drop noise, the leading loop-target
    store(s), and the trailing loop-back jump, leaving only the statements the derived body
    should contain."""
    ins = [i for i in insts if i.opname not in _SKIP_OPS]
    k = 0
    while k < len(ins) and (ins[k].opname in _STORE_OPS or ins[k].opname == "UNPACK_SEQUENCE"):
        k += 1
    ins = ins[k:]
    while ins and ins[-1].opname in ("JUMP_BACKWARD", "JUMP_ABSOLUTE", "JUMP_FORWARD"):
        ins = ins[:-1]
    return ins


# ---------------------------------------------------------------------------
# co_positions re-parenting proof.
# ---------------------------------------------------------------------------
def _ordered_lines(insts: list) -> list[int]:
    """Distinct source lines of ``insts`` in first-seen order (skips missing lines)."""
    out: list[int] = []
    for i in insts:
        ln = getattr(i, "starts_line", None)
        if ln is not None and (not out or out[-1] != ln):
            if ln not in out:
                out.append(ln)
    return out


def _proves_reparenting(gt_cols, der_cols, gt_body_insts, der_matched) -> bool:
    """True iff every GT body statement sits at a CONSTANT, POSITIVE column delta above the
    matching derived statement — the keyword-invariant signature of "GT nests this run one
    level deeper than the derived output placed it". Pairs statements positionally (both
    runs are the same statements in source order), so the per-statement expression offset
    cancels and only the indentation difference remains."""
    gt_lines = _ordered_lines(gt_body_insts)
    der_lines = _ordered_lines(der_matched)
    if not gt_lines or len(gt_lines) != len(der_lines):
        return False
    delta = None
    for gl, dl in zip(gt_lines, der_lines):
        if gl not in gt_cols or dl not in der_cols:
            return False
        d = gt_cols[gl] - der_cols[dl]
        if d <= 0:
            return False
        if delta is None:
            delta = d
        elif d != delta:
            return False
    return delta is not None


# ---------------------------------------------------------------------------
# Operator entry point.
# ---------------------------------------------------------------------------
def control_flow_renest_candidate(gt_code_object, derived_code_object, fragment: str,
                                  repair_context: dict | None) -> dict | None:
    """Deterministically re-nest a contiguous run of derived statements inside the single
    ``if`` / ``for`` construct GT wraps them in but the decompiler flattened away, using
    PEP 657 columns to prove the re-parenting. Returns
    ``{"text", "operator", "opname", "kind", "confidence"}`` or None to defer. Never raises.
    """
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") != "Different control flow":
            return None

        gt_insts = _instructions(gt_code_object)
        der_insts = _instructions(derived_code_object)
        if not gt_insts or not der_insts:
            return None

        # (2) Require PEP 657 COLUMN offsets on both sides (defer on 3.10 / no columns).
        gt_cols = _cols_by_line(gt_code_object)
        der_cols = _cols_by_line(derived_code_object)
        if gt_cols is None or der_cols is None:
            return None

        # (2/3) Structure both objects: GT must reduce to ONE top-level if/for; the derived
        # body must be straight-line so bytecode-statement ordinals map 1:1 onto derived AST
        # statements (a non-flat derived body is the arbitrary-re-parenting case -> defer).
        gt_cft = _bc_to_cft(gt_code_object)
        der_cft = _bc_to_cft(derived_code_object)
        if gt_cft is None or der_cft is None:
            return None
        if not _is_flat(der_cft):
            return None
        node = _find_single_construct(gt_cft)
        if node is None:
            return None
        cls = type(node).__name__

        # Recover the construct header + its body-content instructions from GT truth.
        builder = None  # ("if", cond) | ("for", (target, iter))
        require_terminator = False
        if cls in _IF_CLASSES:
            header = _get_member(node, "if_header")
            if_body = _get_member(node, "if_body")
            if header is None or if_body is None:
                return None
            try:
                cond = _recover_if_condition(header.get_instructions())
                body_insts = if_body.get_instructions()
            except Exception:  # noqa: BLE001
                return None
            if cond is None:
                return None
            content = [i for i in body_insts if i.opname not in _SKIP_OPS]
            builder = ("if", cond)
            require_terminator = True  # a trailing run would otherwise imply a real else
        elif cls in _FOR_CLASSES:
            for_body = _get_member(node, "for_body")
            if for_body is None:
                return None
            hdr = _recover_for_header(gt_insts)
            if hdr is None:
                return None
            try:
                content = _for_body_content(for_body.get_instructions())
            except Exception:  # noqa: BLE001
                return None
            builder = ("for", hdr)
        else:
            return None  # while / with / try headers are not recovered here -> defer
        if not content:
            return None

        # (3) Locate the body run as a UNIQUE contiguous match in the derived stream
        # (binding-insensitive), then map to derived AST statement ordinals.
        der_nonskip = [i for i in der_insts if i.opname not in _SKIP_OPS]
        needle = _renest_signature(content)
        haystack = _renest_signature(der_nonskip)
        start = _unique_contiguous_match(haystack, needle)
        if start is None:
            return None
        der_matched = der_nonskip[start:start + len(needle)]

        # (2/3) PROVE re-parenting from columns: the matched derived run must be a constant,
        # positive indentation level shallower than GT nests it. This is the load-bearing
        # co_positions gate — no proof, no re-nest.
        if not _proves_reparenting(gt_cols, der_cols, content, der_matched):
            return None

        matched_offsets = {getattr(i, "offset", None) for i in der_matched}
        groups = _statement_groups(derived_code_object)
        covered = _covered_group_ordinals(groups, matched_offsets)
        if not covered or covered != list(range(covered[0], covered[-1] + 1)):
            return None
        lo, hi = covered[0], covered[-1]

        # (4) Parse the derived fragment and re-nest exactly those statements.
        try:
            import textwrap
            tree = ast.parse(textwrap.dedent(fragment))
        except SyntaxError:
            return None
        stmts, container = _locate_body(tree)
        if lo < 0 or hi >= len(stmts):
            return None
        # ``if`` soundness: if statements TRAIL the wrapped run, leaving them un-wrapped is
        # equivalent to a GT ``else`` only when the body cannot fall through (its last real
        # instruction terminates control); otherwise a real else may be involved -> defer.
        if require_terminator and hi < len(stmts) - 1:
            if not content or content[-1].opname not in _TERMINATORS:
                return None

        wrap = list(stmts[lo:hi + 1])
        try:
            if builder[0] == "if":
                test = ast.parse(builder[1], mode="eval").body
                new_node = ast.If(test=test, body=wrap, orelse=[])
            else:  # for
                target_name, iter_src = builder[1]
                iter_node = ast.parse(iter_src, mode="eval").body
                new_node = ast.For(
                    target=ast.Name(id=target_name, ctx=ast.Store()),
                    iter=iter_node, body=wrap, orelse=[],
                )
        except (SyntaxError, ValueError):
            return None

        container.body = stmts[:lo] + [new_node] + stmts[hi + 1:]
        ast.fix_missing_locations(tree)
        try:
            unparsed = ast.unparse(tree)
        except Exception:  # noqa: BLE001
            return None
        reindented = _match_indent_convention(fragment, unparsed)
        if reindented is None or reindented.strip() == fragment.strip():
            return None
        # Never emit a fragment that doesn't parse.
        try:
            import textwrap
            ast.parse(textwrap.dedent(reindented))
        except SyntaxError:
            return None

        n = hi - lo + 1
        if builder[0] == "if":
            detail = f"re-nest {n} statement{'s' if n != 1 else ''} into GT `if {builder[1]}:`"
            opname = "POP_JUMP_IF_FALSE"
        else:
            detail = f"re-nest {n} statement{'s' if n != 1 else ''} into GT `for {builder[1][0]} in {builder[1][1]}:`"
            opname = "FOR_ITER"
        return {
            "text": reindented,
            "operator": detail,
            "opname": opname,
            "kind": "control_flow_renest",
            "confidence": "unique",
        }
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
