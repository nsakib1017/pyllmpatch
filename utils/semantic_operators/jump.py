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
    if opname not in _POP_JUMP_OPCODES:
        return None
    return "TRUE" if opname.endswith("IF_TRUE") else "FALSE"


def _is_pure_jump_flip(gt_bc, der_bc, failed_offset: int) -> str | None:
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
