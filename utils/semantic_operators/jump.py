"""Jump-sense operator (M4): deterministic, no-LLM recovery of a single branch-sense
inversion the decompiler emits as a dead ``if COND: pass else: BODY`` idiom.

Signal (read straight from GT bytecode): at the divergence the derived and GT
streams carry a MATCHED ``POP_JUMP_IF_*`` pair (also the 3.11 ``POP_JUMP_FORWARD_IF_*``
/ ``POP_JUMP_BACKWARD_IF_*`` variant names) whose polarity is OPPOSITE (TRUE<->FALSE)
but whose jump TARGET (``argval``) is identical, and every other instruction is
opname-identical — a PURE single flip. That is exactly the shape produced when the
decompiler renders a branch whose true/false sense it inverted: it moves the real
body into an ``else`` and leaves a dead ``pass`` then-branch. Inside the enclosing
loop/branch, the dead ``pass`` and the ``else`` body both fall through to the same
join, so the ONLY bytecode difference is the jump polarity.

Fix (source): drop the dead ``pass`` then-branch and promote the ``else`` body under
the SAME condition — ``if COND: pass else: BODY`` -> ``if COND: BODY``. This flips
ONLY the jump polarity and leaves the condition's own instructions (e.g.
``CONTAINS_OP``/``COMPARE_OP`` operands) byte-identical to GT.

NOTE — we deliberately do NOT negate the condition (``if not COND: BODY``). Negation
is the De-Morgan equivalent of the *derived* (buggy) program, so it preserves the
derived jump polarity rather than flipping it to GT's; and for membership/identity
tests CPython folds the ``not`` into the comparison operand (``CONTAINS_OP`` 1 -> 0),
which would introduce a NEW divergence against GT's operand. Promoting the ``else``
body under the unchanged condition is the transform that reproduces GT verbatim.

Correctness: fires only on the exact pure-flip signal, only when the source has
EXACTLY ONE ``if COND: pass else: <non-empty>`` node (else None), and the rewritten
source must re-parse. Every candidate still passes the loop's recompile + PyLingual
oracle + combined-distance accept gate, so a wrong collapse cannot land.
"""
from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.leaf import _instructions
from utils.semantic_operators.structural import _match_indent_convention, _only_pass_or_dead

# The conditional-pop-jump opcodes across supported versions. 3.10/3.12+ use the
# short names; 3.11 splits them into FORWARD/BACKWARD directional variants.
_POP_JUMP_OPCODES = {
    "POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE",
    "POP_JUMP_FORWARD_IF_TRUE", "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_TRUE", "POP_JUMP_BACKWARD_IF_FALSE",
}


def _jump_sense(opname: str) -> str | None:
    """Return "TRUE"/"FALSE" for a conditional pop-jump, else None."""
    if opname not in _POP_JUMP_OPCODES:
        return None
    return "TRUE" if opname.endswith("IF_TRUE") else "FALSE"


def _is_pure_jump_flip(gt_bc, der_bc, failed_offset: int) -> str | None:
    """True-flip detector. Returns the DERIVED pop-jump opname when the two streams
    differ by EXACTLY a matched opposite-polarity same-target pop-jump at
    ``failed_offset`` and are otherwise opname-identical; else None."""
    der = _instructions(der_bc)
    gt = _instructions(gt_bc)
    if not der or not gt or len(der) != len(gt):
        return None
    idx = next((i for i, ins in enumerate(der)
                if getattr(ins, "offset", None) == failed_offset), None)
    if idx is None:
        return None
    d_ins, g_ins = der[idx], gt[idx]
    d_op = getattr(d_ins, "opname", None)
    g_op = getattr(g_ins, "opname", None)
    d_sense, g_sense = _jump_sense(d_op), _jump_sense(g_op)
    if d_sense is None or g_sense is None or d_sense == g_sense:
        return None  # not a matched opposite-polarity pop-jump pair
    if getattr(d_ins, "argval", object()) != getattr(g_ins, "argval", object()):
        return None  # different jump target -> a restructure, not a sense flip
    # Every OTHER instruction must be opname-identical (a pure single flip).
    for j, (a, b) in enumerate(zip(der, gt)):
        if j == idx:
            continue
        if getattr(a, "opname", None) != getattr(b, "opname", None):
            return None
    return d_op


def _dead_pass_else_ifs(tree: ast.AST) -> list[ast.If]:
    """``if COND: pass else: <non-empty, non-dead>`` nodes (the recoverable idiom)."""
    out: list[ast.If] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
            continue
        if not node.orelse or _only_pass_or_dead(node.orelse):
            continue
        out.append(node)
    return out


class _Collapse(ast.NodeTransformer):
    """Collapse the ONE target ``if COND: pass else: BODY`` into ``if COND: BODY``
    (identity-matched), keeping the condition unchanged."""

    def __init__(self, target: ast.If):
        self.target = target

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if node is self.target:
            node.body = node.orelse
            node.orelse = []
        return node


def collapse_dead_pass_else_candidate(gt_code_object, derived_code_object, fragment: str,
                                      repair_context: dict | None) -> dict | None:
    """Recover a single jump-sense inversion. Returns
    ``{"text", "operator", "opname", "kind", "confidence"}`` or None to defer."""
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None
    failed_offset = repair_context.get("failed_offset")
    if failed_offset is None:
        return None

    der_opname = _is_pure_jump_flip(gt_code_object, derived_code_object, int(failed_offset))
    if der_opname is None:
        return None

    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    targets = _dead_pass_else_ifs(tree)
    if len(targets) != 1:
        return None  # ambiguity (or nothing to collapse) -> defer

    _Collapse(targets[0]).visit(tree)
    ast.fix_missing_locations(tree)
    try:
        new_dedented = ast.unparse(tree)
    except Exception:  # noqa: BLE001
        return None
    reindented = _match_indent_convention(fragment, new_dedented)
    if reindented is None or reindented.strip() == fragment.strip():
        return None
    try:
        ast.parse(textwrap.dedent(reindented))  # never emit unparseable source
    except SyntaxError:
        return None
    return {
        "text": reindented,
        "operator": "collapse dead pass-else",
        "opname": der_opname,
        "kind": "jump_sense",
        "confidence": "unique",
    }
