"""Comprehension operator: deterministically reconstruct a SIMPLE list/set/dict
comprehension or generator expression from ground-truth bytecode when the decompiler
rendered it wrong, then splice it over the unique matching comprehension display in
the derived source.

A comprehension is fully determined by bytecode, so — for the tightly-scoped simple
case — the correct source is recoverable without any model. Two regimes appear,
depending on which code object diverges (this is what pylingual actually hands us):

  * INLINED (PEP 709, CPython 3.12+). A list/set/dict comprehension compiles *inline*
    into its enclosing code object: ``<iter> GET_ITER; LOAD_FAST_AND_CLEAR <t>...;
    BUILD_LIST/SET/MAP 0; FOR_ITER; STORE <t>; [filter]; <element>; LIST_APPEND/
    SET_ADD/MAP_ADD; JUMP_BACKWARD; END_FOR``. When the enclosing object diverges the
    whole comprehension — element, target, iterable, optional filter — is recovered
    verbatim from *its own* bytecode. (``bc_a`` is the enclosing object.)

  * NESTED code object. A generator expression ALWAYS compiles to a separate
    ``<genexpr>`` code object (even on 3.12); pre-3.12 list/set/dict comprehensions do
    too. Its OUTERMOST iterable is evaluated eagerly in the ENCLOSING frame and passed
    in as the ``.0`` argument, so when only the comprehension *body* diverges pylingual
    reports the failure on the nested record and hands the body directly as ``bc_a``.
    Element/target/filter are recovered from that body; the iterable — which is not in
    the body — is reused verbatim from the derived source's own comprehension display
    (sound because the enclosing frame, which owns that iterable, matched GT).

Both regimes read truth from GT bytecode and are unique-or-defer. The element/iterable
sub-expressions are rebuilt by a small stack machine that FULLY PARENTHESISES every
composed sub-expression in bytecode-evaluation order: parentheses emit no bytecode, so
the reconstructed parse tree is isomorphic to the bytecode-implied tree and recompiles
to IDENTICAL bytecode — no operator-precedence guessing. Anything richer than the tight
simple grammar (nested/multi-``for`` comprehensions, complex elements, call chains,
unrecoverable sub-exprs, unusual filters, async) DEFERS by returning None.

Honest scope note: rebuilding the element/iterable expression is, in miniature,
decompilation — but a *bounded, deterministic* one (fixed grammar, verbatim operands,
no search). The insertion-time recompile + oracle distance gate is the final backstop:
a wrong reconstruction never lowers the distance and is rejected, so this operator can
only ever LAND a fix that provably reduces divergence. It never raises.
"""
from __future__ import annotations

import ast
import dis
import textwrap

from utils.semantic_operators.container import _replace_span, _top_level_displays
from utils.semantic_operators.control_flow import _const_repr
from utils.semantic_operators.leaf import _instructions
from utils.semantic_operators.structural import _clean_name

# ---------------------------------------------------------------------------
# Opcode vocabulary.
# ---------------------------------------------------------------------------
_VALUE_NAME_LOADS = {
    "LOAD_FAST", "LOAD_NAME", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_CLASSDEREF",
    "LOAD_FAST_CHECK",
}
_STORE_OPS = {"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}
_ADD_KIND = {"LIST_APPEND": "list", "SET_ADD": "set", "MAP_ADD": "dict"}
_BUILD_KIND = {"BUILD_LIST": "list", "BUILD_SET": "set", "BUILD_MAP": "dict"}
_COMP_CONAMES = {"<listcomp>": "list", "<setcomp>": "set", "<dictcomp>": "dict",
                 "<genexpr>": "gen"}
_KIND_DELIMS = {"list": ("[", "]"), "set": ("{", "}"), "dict": ("{", "}"),
                "gen": ("(", ")")}
_ANCHOR_CLS = {"list": ast.ListComp, "set": ast.SetComp, "dict": ast.DictComp,
               "gen": ast.GeneratorExp}

# comprehension filters compile to POP_JUMP_IF_TRUE -> element (keep-if-true).
_TRUE_JUMPS = {"POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_TRUE", "POP_JUMP_BACKWARD_IF_TRUE"}
_ANY_COND_JUMP = _TRUE_JUMPS | {
    "POP_JUMP_IF_FALSE", "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_BACKWARD_IF_FALSE",
    "POP_JUMP_IF_NONE", "POP_JUMP_FORWARD_IF_NONE", "POP_JUMP_BACKWARD_IF_NONE",
    "POP_JUMP_IF_NOT_NONE", "POP_JUMP_FORWARD_IF_NOT_NONE", "POP_JUMP_BACKWARD_IF_NOT_NONE",
    "JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP",
}
_BACK_JUMPS = {"JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "JUMP_ABSOLUTE"}
# async / structurally-hard markers that force a defer wherever they appear.
_ASYNC_OPS = {"GET_AITER", "GET_ANEXT", "END_ASYNC_FOR", "SEND"}
# skipped for both stack simulation and structure scanning (net zero, no value).
_SKIP = {"CACHE", "RESUME", "PRECALL", "NOP", "EXTENDED_ARG", "COPY_FREE_VARS",
         "MAKE_CELL", "RETURN_GENERATOR", "GEN_START"}

_BINOP_SYMS = {"+", "-", "*", "/", "//", "%", "**", "@", "<<", ">>", "&", "|", "^"}
_COMPARE_SYMS = {"<", ">", "==", "!=", "<=", ">=", "<>"}
_UNARY = {"UNARY_NEGATIVE": ("-", ""), "UNARY_INVERT": ("~", ""), "UNARY_NOT": ("not ", "")}

_NULL = object()  # PUSH_NULL / method-load self slot sentinel.


class _E:
    """A recovered sub-expression: its source text plus whether it is an ``atom``
    (name / const / trailer) that needs no parentheses when used as an operand."""
    __slots__ = ("text", "atom")

    def __init__(self, text: str, atom: bool):
        self.text = text
        self.atom = atom


def _wrap(entry: "_E") -> str:
    """Operand text, parenthesised unless it is an atom. Fully parenthesising every
    composed operand makes the parse tree isomorphic to the bytecode-implied tree."""
    return entry.text if entry.atom else f"({entry.text})"


def _name_of(ins) -> str | None:
    name = _clean_name(getattr(ins, "argrepr", "") or "")
    if not name:
        av = getattr(ins, "argval", None)
        name = av if isinstance(av, str) else ""
    name = name.strip()
    return name or None


def _is_method_attr(ins) -> bool:
    """True iff a LOAD_ATTR/LOAD_METHOD is the *method* form (pushes a NULL/self slot
    below the callable) — 3.12 marks it with a low arg bit and a 'NULL|self' argrepr."""
    if ins.opname == "LOAD_METHOD":
        return True
    argrepr = getattr(ins, "argrepr", "") or ""
    if "NULL" in argrepr:
        return True
    arg = getattr(ins, "arg", None)
    return isinstance(arg, int) and bool(arg & 1)


# ---------------------------------------------------------------------------
# Expression stack machine — the only "mini-decompilation" surface, kept to a
# fixed simple grammar; anything outside it returns None (defer).
# ---------------------------------------------------------------------------
def _recover_stack(insts: list) -> list | None:
    """Simulate ``insts`` as a value stack and return the final list of ``_E`` entries,
    or None if any instruction falls outside the supported simple grammar. Rejects
    call *chains* (allows at most one CALL). Never raises."""
    try:
        stack: list = []
        calls = 0
        for ins in insts:
            op = getattr(ins, "opname", None)
            if op is None or op in _SKIP:
                continue
            if op == "PUSH_NULL":
                stack.append(_NULL)
                continue
            if op in _VALUE_NAME_LOADS:
                nm = _name_of(ins)
                if not nm or nm.startswith(".") or not nm.isidentifier():
                    return None
                stack.append(_E(nm, True))
                continue
            if op == "LOAD_CONST":
                rep = _const_repr(getattr(ins, "argval", None))
                if rep is None:
                    return None
                stack.append(_E(rep, True))
                continue
            if op in ("LOAD_ATTR", "LOAD_METHOD"):
                if not stack or stack[-1] is _NULL:
                    return None
                base = stack.pop()
                attr = _name_of(ins)
                if not attr or not attr.isidentifier():
                    return None
                text = f"{_wrap(base)}.{attr}"
                if _is_method_attr(ins):
                    stack.append(_NULL)          # self / null slot
                    stack.append(_E(text, True))  # the callable
                else:
                    stack.append(_E(text, True))
                continue
            if op == "BINARY_OP":
                if len(stack) < 2:
                    return None
                b, a = stack.pop(), stack.pop()
                if a is _NULL or b is _NULL:
                    return None
                sym = (getattr(ins, "argrepr", "") or "").strip()
                if sym not in _BINOP_SYMS:
                    return None
                stack.append(_E(f"{_wrap(a)} {sym} {_wrap(b)}", False))
                continue
            if op in ("BINARY_SUBSCR", "BINARY_SUBSCR_LIST_INT"):
                if len(stack) < 2:
                    return None
                b, a = stack.pop(), stack.pop()
                if a is _NULL or b is _NULL:
                    return None
                stack.append(_E(f"{_wrap(a)}[{b.text}]", True))
                continue
            if op == "COMPARE_OP":
                if len(stack) < 2:
                    return None
                b, a = stack.pop(), stack.pop()
                if a is _NULL or b is _NULL:
                    return None
                sym = (getattr(ins, "argrepr", "") or "").strip()
                if sym not in _COMPARE_SYMS:
                    return None
                stack.append(_E(f"{_wrap(a)} {sym} {_wrap(b)}", False))
                continue
            if op == "IS_OP":
                if len(stack) < 2:
                    return None
                b, a = stack.pop(), stack.pop()
                if a is _NULL or b is _NULL:
                    return None
                word = "is not" if getattr(ins, "arg", 0) else "is"
                stack.append(_E(f"{_wrap(a)} {word} {_wrap(b)}", False))
                continue
            if op == "CONTAINS_OP":
                if len(stack) < 2:
                    return None
                b, a = stack.pop(), stack.pop()
                if a is _NULL or b is _NULL:
                    return None
                word = "not in" if getattr(ins, "arg", 0) else "in"
                stack.append(_E(f"{_wrap(a)} {word} {_wrap(b)}", False))
                continue
            if op in _UNARY:
                if not stack or stack[-1] is _NULL:
                    return None
                a = stack.pop()
                pre, _post = _UNARY[op]
                stack.append(_E(f"{pre}{_wrap(a)}", False))
                continue
            if op in ("CALL", "CALL_FUNCTION"):
                n = _argint(ins)
                # CALL (3.11+) pops n args + callable + null/self; CALL_FUNCTION (<=3.10)
                # pops n args + callable (no null slot).
                need = n + 2 if op == "CALL" else n + 1
                if len(stack) < need:
                    return None
                args = [stack.pop() for _ in range(n)][::-1]
                callable_e = stack.pop()
                if op == "CALL":
                    nullself = stack.pop()
                    if nullself is not _NULL:
                        return None
                if callable_e is _NULL or any(a is _NULL for a in args):
                    return None
                calls += 1
                if calls > 1:  # reject call chains / nested calls
                    return None
                argtext = ", ".join(_wrap(a) for a in args)
                stack.append(_E(f"{_wrap(callable_e)}({argtext})", True))
                continue
            # anything else (BUILD_*, FORMAT_VALUE, jumps, MAKE_FUNCTION, ...) is outside
            # the simple grammar -> defer.
            return None
        return stack
    except Exception:  # noqa: BLE001 - defensively never raise
        return None


def _argint(ins) -> int:
    a = getattr(ins, "arg", None)
    if isinstance(a, int) and a >= 0:
        return a
    v = getattr(ins, "argval", None)
    return v if isinstance(v, int) and v >= 0 else 0


def _stack_effect(ins) -> int | None:
    """dis.stack_effect for a pylingual instruction, or None if unknown to this host."""
    op = getattr(ins, "opname", None)
    if op in _SKIP or op is None:
        return 0
    try:
        code = dis.opmap[op]
    except KeyError:
        return None
    arg = getattr(ins, "arg", None)
    try:
        return dis.stack_effect(code, arg if isinstance(arg, int) else None)
    except (ValueError, TypeError):
        return None


def _expr_span_before(insts: list, idx: int) -> list | None:
    """The minimal contiguous instruction span ending just before ``idx`` whose net
    stack effect is +1 — i.e. the single sub-expression consumed at ``idx`` (used to
    isolate the inlined comprehension's iterable, which ends at GET_ITER). None if it
    cannot be balanced with host-known opcodes."""
    depth = 0
    j = idx - 1
    while j >= 0:
        se = _stack_effect(insts[j])
        if se is None:
            return None
        depth += se
        if depth == 1:
            return insts[j:idx]
        if depth > 1 or depth < -8:
            return None
        j -= 1
    return None


# ---------------------------------------------------------------------------
# Structure recovery: target, filter+element.
# ---------------------------------------------------------------------------
def _parse_target(insts: list, start: int):
    """Recover the loop target starting at ``insts[start]``. Returns ``(target_text,
    next_index)`` or None. Handles a single name or a FLAT tuple of names."""
    if start >= len(insts):
        return None
    first = insts[start]
    if first.opname == "UNPACK_SEQUENCE":
        n = _argint(first)
        if n < 1 or start + n >= len(insts):
            return None
        names = []
        for k in range(1, n + 1):
            st = insts[start + k]
            if st.opname not in _STORE_OPS:  # nested unpack / extended -> defer
                return None
            nm = _name_of(st)
            if not nm or not nm.isidentifier():
                return None
            names.append(nm)
        return ", ".join(names), start + n + 1
    if first.opname in _STORE_OPS:
        nm = _name_of(first)
        if not nm or not nm.isidentifier():
            return None
        return nm, start + 1
    return None


def _recover_filter_and_element(body: list, kind: str):
    """From the loop-body slice AFTER the target stores and BEFORE the ADD/YIELD, return
    ``(cond_text_or_None, element_stack)`` or None. At most one keep-if-true filter is
    supported; any other conditional-jump shape (ternary, boolean-op element, multiple
    filters, inverted sense) defers."""
    cond_positions = [i for i, ins in enumerate(body) if ins.opname in _ANY_COND_JUMP]
    if not cond_positions:
        return None, _recover_stack(body)
    if len(cond_positions) != 1:
        return None
    cj = cond_positions[0]
    jump = body[cj]
    if jump.opname not in _TRUE_JUMPS:  # only keep-if-true filters
        return None
    # the fall-through immediately after the keep test must be the loop skip.
    if cj + 1 >= len(body) or body[cj + 1].opname not in _BACK_JUMPS:
        return None
    cond_stack = _recover_stack(body[:cj])
    if not cond_stack or len(cond_stack) != 1:
        return None
    element = _recover_stack(body[cj + 2:])
    return cond_stack[0].text, element


def _element_text(element_stack, kind: str):
    """Render the recovered element stack into (list/set/gen) a single element string or
    (dict) a ``key: value`` string. None if the arity is wrong."""
    if element_stack is None:
        return None
    if kind == "dict":
        if len(element_stack) != 2:
            return None
        return f"{element_stack[0].text}: {element_stack[1].text}"
    if len(element_stack) != 1:
        return None
    return element_stack[0].text


# ---------------------------------------------------------------------------
# Regime 1: INLINED comprehension in the enclosing object.
# ---------------------------------------------------------------------------
def _recover_inlined(insts: list) -> dict | None:
    """Recover a simple inlined list/set/dict comprehension (element, target, iterable,
    optional filter) from the enclosing object's bytecode, or None to defer."""
    if any(i.opname in _ASYNC_OPS for i in insts):
        return None
    lfac = [j for j, i in enumerate(insts) if i.opname == "LOAD_FAST_AND_CLEAR"]
    if not lfac:
        return None
    adds = [j for j, i in enumerate(insts) if i.opname in _ADD_KIND]
    if len(adds) != 1:  # unique-or-defer: exactly one comprehension
        return None
    add_idx = adds[0]
    kind = _ADD_KIND[insts[add_idx].opname]

    fors = [j for j, i in enumerate(insts) if i.opname == "FOR_ITER"]
    inner_fors = [j for j in fors if lfac[0] < j < add_idx]
    if len(inner_fors) != 1:  # nested / multi-for have >1 FOR_ITER in the region
        return None
    for_idx = inner_fors[0]
    # reject a nested comprehension: a second BUILD_*/LOAD_FAST_AND_CLEAR inside the loop.
    for j in range(for_idx + 1, add_idx):
        if insts[j].opname in _BUILD_KIND or insts[j].opname == "LOAD_FAST_AND_CLEAR":
            return None

    # iterable: the sub-expression consumed by the GET_ITER that precedes the first
    # LOAD_FAST_AND_CLEAR.
    gi = lfac[0] - 1
    if gi < 0 or insts[gi].opname != "GET_ITER":
        return None
    iter_span = _expr_span_before(insts, gi)
    if iter_span is None:
        return None
    iter_stack = _recover_stack(iter_span)
    if not iter_stack or len(iter_stack) != 1:
        return None
    iter_text = iter_stack[0].text

    tgt = _parse_target(insts, for_idx + 1)
    if tgt is None:
        return None
    target_text, elt_start = tgt

    fe = _recover_filter_and_element(insts[elt_start:add_idx], kind)
    if fe is None:
        return None
    cond, element_stack = fe
    element = _element_text(element_stack, kind)
    if element is None:
        return None
    return {"kind": kind, "target": target_text, "iter": iter_text,
            "element": element, "cond": cond, "opname": "LOAD_FAST_AND_CLEAR"}


# ---------------------------------------------------------------------------
# Regime 2: NESTED comprehension body (genexpr on 3.12; pre-3.12 comps).
# ---------------------------------------------------------------------------
def _recover_nested_body(insts: list) -> dict | None:
    """Recover element/target/(optional filter) from a comprehension's own body code
    object (iterating the ``.0`` argument). Kind comes from the body's ADD/YIELD. The
    iterable is intentionally absent (it lives in the enclosing frame). None to defer."""
    if any(i.opname in _ASYNC_OPS for i in insts):
        return None
    if any(getattr(getattr(i, "argval", None), "co_name", None) for i in insts):
        return None  # a nested code object inside -> nested comprehension -> defer
    if any(i.opname == "LOAD_FAST_AND_CLEAR" for i in insts):
        return None  # inlined sub-comprehension -> defer

    fors = [j for j, i in enumerate(insts) if i.opname == "FOR_ITER"]
    if len(fors) != 1:
        return None
    for_idx = fors[0]

    # the loop must iterate the eager .0 argument (LOAD_FAST '.0' shortly before FOR_ITER).
    pre = [i for i in insts[:for_idx] if i.opname not in _SKIP]
    if not pre or pre[-1].opname not in ("LOAD_FAST", "GET_ITER"):
        return None
    dot_zero = any(i.opname in ("LOAD_FAST", "LOAD_FAST_CHECK")
                   and (_name_of(i) or "") == ".0" for i in insts[:for_idx])
    if not dot_zero:
        return None

    yields = [j for j, i in enumerate(insts) if i.opname == "YIELD_VALUE"]
    adds = [j for j, i in enumerate(insts) if i.opname in _ADD_KIND]
    if yields and not adds:
        kind = "gen"
        end_idx = yields[0]
    elif len(adds) == 1 and not yields:
        kind = _ADD_KIND[insts[adds[0]].opname]
        end_idx = adds[0]
    else:
        return None
    if end_idx <= for_idx:
        return None

    tgt = _parse_target(insts, for_idx + 1)
    if tgt is None:
        return None
    target_text, elt_start = tgt

    fe = _recover_filter_and_element(insts[elt_start:end_idx], "list" if kind == "gen" else kind)
    if fe is None:
        return None
    cond, element_stack = fe
    element = _element_text(element_stack, "list" if kind == "gen" else kind)
    if element is None:
        return None
    return {"kind": kind, "target": target_text, "iter": None,
            "element": element, "cond": cond, "opname": "MAKE_FUNCTION"}


# ---------------------------------------------------------------------------
# Text assembly + splice.
# ---------------------------------------------------------------------------
def _build_comprehension_text(desc: dict) -> str:
    open_, close = _KIND_DELIMS[desc["kind"]]
    cond = f" if {desc['cond']}" if desc.get("cond") else ""
    inner = f"{desc['element']} for {desc['target']} in {desc['iter']}{cond}"
    return f"{open_}{inner}{close}"


def _unique_anchor(tree: ast.AST, kind: str):
    anchors = _top_level_displays(tree, _ANCHOR_CLS[kind])
    return anchors[0] if len(anchors) == 1 else None


def _derived_iterable(anchor) -> str | None:
    """The source of the derived comprehension's single, non-async for-clause iterable
    (reused for the nested regime because the enclosing frame owns it and matched GT)."""
    gens = getattr(anchor, "generators", None)
    if not gens or len(gens) != 1:
        return None
    gen0 = gens[0]
    if getattr(gen0, "is_async", 0):
        return None
    return gen0  # the comprehension node; iterable text resolved from dedented source


# ---------------------------------------------------------------------------
# Operator entry point.
# ---------------------------------------------------------------------------
def comprehension_op_candidate(gt_code_object, derived_code_object, fragment: str,
                               repair_context: dict | None) -> dict | None:
    """Reconstruct a simple comprehension / generator expression from GT bytecode and
    splice it over the unique matching derived display. Returns the standard operator
    dict (``kind == "comprehension"``) or None to defer. Never raises."""
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") != "Different bytecode":
            return None

        gt = _instructions(gt_code_object)
        if not gt:
            return None

        # choose regime from GT bytecode shape (unique-or-defer).
        inlined_present = any(i.opname == "LOAD_FAST_AND_CLEAR" for i in gt)
        desc = _recover_inlined(gt) if inlined_present else _recover_nested_body(gt)
        if desc is None:
            return None

        try:
            dedented = textwrap.dedent(fragment)
            tree = ast.parse(dedented)
        except SyntaxError:
            return None

        anchor = _unique_anchor(tree, desc["kind"])
        if anchor is None:
            return None

        # nested regime: the iterable is not in the body — reuse the derived display's
        # single for-clause iterable source verbatim.
        if desc["iter"] is None:
            gen0 = _derived_iterable(anchor)
            if gen0 is None:
                return None
            iter_src = ast.get_source_segment(dedented, gen0.iter)
            if not iter_src:
                return None
            desc["iter"] = iter_src.strip()

        new_text = _build_comprehension_text(desc)
        try:
            ast.parse(new_text, mode="eval")  # reconstructed expression must parse
        except SyntaxError:
            return None

        new_fragment = _replace_span(fragment, dedented, anchor, new_text)
        if new_fragment is None or new_fragment == fragment:
            return None
        try:
            ast.parse(textwrap.dedent(new_fragment))  # re-parse guard on the whole fragment
        except SyntaxError:
            return None

        pretty = "generator expression" if desc["kind"] == "gen" else f"{desc['kind']} comprehension"
        return {
            "text": new_fragment,
            "operator": f"reconstruct {pretty} `{new_text}` from GT bytecode",
            "opname": desc["opname"],
            "kind": "comprehension",
            "confidence": "unique",
        }
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
