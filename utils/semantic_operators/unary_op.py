"""Unary-form operator: deterministically fix a SINGLE leaf whose UNARY form the
decompiler got wrong, reading the correct form straight from ground-truth bytecode.

The decompiler frequently gets the operand right but the unary *wrapper* wrong at one
instruction: it drops a ``-``/``not``/``~`` the source really had, adds one it did not,
or emits the wrong one. In bytecode this is a single ``UNARY_NEGATIVE`` (``-x``),
``UNARY_NOT`` (``not x``), or ``UNARY_INVERT`` (``~x``) appearing / disappearing /
changing over the SAME operand. ``TO_BOOL`` (3.13's explicit-truthiness opcode) is the
same shape for an explicit ``bool(...)`` conversion.

This is deliberately the cross-opcode case ``leaf.py`` defers ("A DIFFERENT opcode at the
divergence ... is not an operand swap and is deferred") — and it is *not* the same-opcode
``LOAD_CONST`` value swap ``leaf.swap_literal`` owns: a unary over a compile-time constant
folds into the constant (``-1`` -> ``LOAD_CONST -1``), so it never surfaces here. Only a
unary over a *runtime* value (a bare name load) reaches us, which is exactly the tightest,
self-verifying signature.

Mechanism, mirroring :mod:`literal_form`:
  * align the GT vs derived OPNAME streams with :class:`difflib.SequenceMatcher`;
  * require the unary mismatch to be the ONE-AND-ONLY non-``equal`` chunk (a wider or
    second divergence is a compound bug we cannot attribute to a single unary);
  * classify it as add (GT dropped a unary), remove (derived added one), swap (both a
    single unary of different kind), or an explicit ``bool()`` (dropped ``TO_BOOL``);
  * read the correct form from GT, locate the UNIQUE matching operand node in the derived
    AST, and splice via leaf's ASCII-guarded single-line span replace.

Unique-or-defer everywhere; a complex (non-leaf) operand defers; every edit re-parses and
is still gated by the loop's recompile + oracle accept, so a wrong guess can never land.
This never emits a value it did not read from GT and it NEVER raises (top-level try/except
-> None).
"""
from __future__ import annotations

import ast
import textwrap
from difflib import SequenceMatcher

from utils.semantic_operators.leaf import _replace_node_span
from utils.semantic_operators.literal_form import _name_of, _real_instructions

# unary opcode <-> ast unary-op class <-> surface symbol. ``not`` keeps its trailing space
# so ``f"({sym}{operand})"`` renders ``(not x)`` (a valid atom), never ``(notx)``.
_UNARY_OPCODE_TO_AST: dict[str, type] = {
    "UNARY_NEGATIVE": ast.USub,    # -x
    "UNARY_NOT": ast.Not,          # not x
    "UNARY_INVERT": ast.Invert,    # ~x
}
_AST_TO_SYMBOL: dict[type, str] = {ast.USub: "-", ast.Not: "not ", ast.Invert: "~"}
_ARITH_UNARY_OPCODES = frozenset(_UNARY_OPCODE_TO_AST)
_TO_BOOL_OPCODES = frozenset({"TO_BOOL"})           # 3.13 explicit truthiness
_UNARY_FAMILY = _ARITH_UNARY_OPCODES | _TO_BOOL_OPCODES
# Operand loads that map 1:1 to a bare ``ast.Name`` (Load). LOAD_CONST is EXCLUDED on
# purpose: a unary over a compile-time constant folds into the constant, so it never
# appears as a UNARY_* op — that case is leaf's swap_literal, not ours.
_NAME_LOAD_OPCODES = frozenset({"LOAD_FAST", "LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"})


def _disp(sym: str) -> str:
    """Human-readable operator token (``not `` -> ``not``, ``-`` -> ``-``)."""
    return sym.strip() or sym


def _find_unary_divergence(gt: list, der: list) -> dict | None:
    """Align GT vs derived opname streams; if the SOLE divergence is a single unary-op
    mismatch, return a descriptor, else None.

    Descriptors:
      * ``{"mode": "add",         "unary": <UNARY_* opcode>, "ident": <operand name>}``
      * ``{"mode": "to_bool_add", "unary": "TO_BOOL",        "ident": <operand name>}``
      * ``{"mode": "remove",      "der_unary": <UNARY_* opcode>}``
      * ``{"mode": "swap",        "unary": <gt opcode>, "der_unary": <derived opcode>}``
    """
    sm = SequenceMatcher(a=[i.opname for i in gt], b=[i.opname for i in der], autojunk=False)
    non_equal = [c for c in sm.get_opcodes() if c[0] != "equal"]
    # Exactly ONE non-equal chunk: the unary must be the only divergence, otherwise this is
    # a compound bug and we cannot claim a single-leaf fix reaches distance 0.
    if len(non_equal) != 1:
        return None
    tag, i1, i2, j1, j2 = non_equal[0]
    gt_seg, der_seg = gt[i1:i2], der[j1:j2]

    # (a) dropped unary: GT carries one extra UNARY_*/TO_BOOL the derived stream lacks.
    if tag == "delete" and len(gt_seg) == 1 and gt_seg[0].opname in _UNARY_FAMILY:
        if i1 - 1 < 0:
            return None  # nothing loads the operand -> not our shape
        # The operand is whatever sits on TOS just before the unary — the immediately
        # preceding instruction. Requiring it to be a plain name load both localizes the
        # operand unambiguously AND defers every complex operand (`-(a+b)`, `-f()`, `-a.b`,
        # and the precedence trap `-x**2`, whose UNARY_NEGATIVE trails the BINARY_OP).
        operand = gt[i1 - 1]
        if getattr(operand, "opname", None) not in _NAME_LOAD_OPCODES:
            return None
        ident = _name_of(operand)
        if ident is None:
            return None
        unary = gt_seg[0].opname
        mode = "to_bool_add" if unary in _TO_BOOL_OPCODES else "add"
        return {"mode": mode, "unary": unary, "ident": ident}

    # derived added a spurious UNARY_* the GT lacks -> strip it back off.
    if tag == "insert" and len(der_seg) == 1 and der_seg[0].opname in _ARITH_UNARY_OPCODES:
        return {"mode": "remove", "der_unary": der_seg[0].opname}

    # both sides a single unary of DIFFERENT kind -> rewrite to GT's form.
    if tag == "replace" and len(gt_seg) == 1 and len(der_seg) == 1:
        g, d = gt_seg[0].opname, der_seg[0].opname
        if g in _ARITH_UNARY_OPCODES and d in _ARITH_UNARY_OPCODES and g != d:
            return {"mode": "swap", "unary": g, "der_unary": d}
    return None


def unary_op_candidate(gt_code_object, derived_code_object, fragment: str,
                       repair_context: dict | None) -> dict | None:
    """Fix a single mis-rendered unary by reading the correct form from GT bytecode.

    Returns ``{"text", "operator", "opname", "kind": "unary", "confidence"}`` for an
    accepted source edit, or None to defer. Never raises.
    """
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") != "Different bytecode":
            return None

        gt = _real_instructions(gt_code_object)
        der = _real_instructions(derived_code_object)
        if not gt or not der:
            return None
        div = _find_unary_divergence(gt, der)
        if div is None:
            return None

        try:
            dedented = textwrap.dedent(fragment)
            tree = ast.parse(dedented)
        except SyntaxError:
            return None

        mode = div["mode"]

        if mode in ("add", "to_bool_add"):
            # Wrap the UNIQUE bare-name operand in the derived source. Uniqueness over the
            # whole fragment (ast.walk spans nested scopes) is the anti-ambiguity guard: two
            # matching Load names -> defer. Parenthesizing keeps grouping exact regardless of
            # the surrounding operator precedence (`a ** (-x)`, `(-x) ** 2`, ...).
            ident = div["ident"]
            nodes = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == ident]
            if len(nodes) != 1:
                return None
            node = nodes[0]
            if mode == "add":
                sym = _AST_TO_SYMBOL[_UNARY_OPCODE_TO_AST[div["unary"]]]
                new_text = f"({sym}{ident})"
                label = f"add unary {_disp(sym)} on {ident}"
            else:  # to_bool_add: recover an explicit bool() the decompiler dropped
                new_text = f"bool({ident})"
                label = f"add bool() on {ident}"
            opname = div["unary"]

        elif mode == "remove":
            der_cls = _UNARY_OPCODE_TO_AST[div["der_unary"]]
            nodes = [n for n in ast.walk(tree)
                     if isinstance(n, ast.UnaryOp) and isinstance(n.op, der_cls)]
            if len(nodes) != 1:
                return None
            node = nodes[0]
            new_text = ast.unparse(node.operand)  # splice the operand back in, unwrapped
            opname = div["der_unary"]
            label = f"drop unary {_disp(_AST_TO_SYMBOL[der_cls])}"

        elif mode == "swap":
            der_cls = _UNARY_OPCODE_TO_AST[div["der_unary"]]
            gt_cls = _UNARY_OPCODE_TO_AST[div["unary"]]
            nodes = [n for n in ast.walk(tree)
                     if isinstance(n, ast.UnaryOp) and isinstance(n.op, der_cls)]
            if len(nodes) != 1:
                return None
            node = nodes[0]
            sym = _AST_TO_SYMBOL[gt_cls]
            new_text = f"({sym}{ast.unparse(node.operand)})"
            opname = div["unary"]
            label = f"swap unary {_disp(_AST_TO_SYMBOL[der_cls])} -> {_disp(sym)}"
        else:
            return None

        edited = _replace_node_span(fragment, dedented, node, new_text)
        if edited is None or edited == fragment:
            return None
        try:
            ast.parse(textwrap.dedent(edited))  # re-parse guard: never emit broken source
        except SyntaxError:
            return None
        return {
            "text": edited,
            "operator": label,
            "opname": opname,
            "kind": "unary",
            "confidence": "unique",
        }
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything -> defer.
        return None
