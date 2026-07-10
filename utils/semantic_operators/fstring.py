"""F-string spurious-segment operator: deterministically drop a trailing literal
segment PyLingual hallucinated onto an f-string, read straight from ground-truth
bytecode.

Pattern: the decompiler emits an f-string with one EXTRA trailing literal piece
the ground truth does not have — the classic ``f"_{name}0"`` for a GT
``f"_{name}"`` (a stray ``LOAD_CONST '0'`` immediately before ``BUILD_STRING``,
whose piece count is exactly one higher than GT's). Because the surplus
instruction shifts every downstream offset, this regime usually surfaces with
``tr.failed_offset is None``; we therefore SCAN the two instruction streams
rather than keying off a single offset.

Deterministic and conservative. We fire only when:
  * GT and the derived object have the same number of ``BUILD_STRING`` ops, paired
    by order, and EXACTLY ONE pair has ``der_count == gt_count + 1`` (all others
    equal) — a single +1 surplus;
  * the surplus is a TRAILING string-``LOAD_CONST`` directly before that
    ``BUILD_STRING``, with no matching literal at GT's trailing position, and GT
    does not contain that literal anywhere as a ``LOAD_CONST`` operand (never
    delete a segment GT keeps);
  * the source has EXACTLY ONE ``JoinedStr`` of the right piece count whose final
    value is an ``ast.Constant`` str equal to the surplus literal.

We drop that trailing ``Constant`` and re-render, requiring a clean AST
round-trip. Anything else — more than one surplus, an interior/non-Constant
surplus, an ambiguous or non-round-tripping localization — defers (returns None).
The loop's recompile + PyLingual oracle + combined-distance accept gate is the
final backstop; this operator never emits a value it did not read verbatim from
the bytecode.
"""
from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.leaf import _instructions, _replace_node_span


def _build_string_count(ins) -> int | None:
    v = getattr(ins, "argval", None)
    return v if isinstance(v, int) else None


def fstring_segment_candidate(gt_code_object, derived_code_object, fragment: str,
                              repair_context: dict | None) -> dict | None:
    """Drop a hallucinated trailing f-string literal segment. Returns
    ``{"text", "operator", "opname", "kind", "confidence"}`` or None to defer."""
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None

    der = _instructions(derived_code_object)
    gt = _instructions(gt_code_object)
    if not der or not gt:
        return None

    gt_bs = [i for i, ins in enumerate(gt) if getattr(ins, "opname", None) == "BUILD_STRING"]
    der_bs = [i for i, ins in enumerate(der) if getattr(ins, "opname", None) == "BUILD_STRING"]
    # Equal count of BUILD_STRING, paired by order. A whole added/removed f-string is
    # a different regime (unclear alignment) -> defer.
    if not der_bs or len(der_bs) != len(gt_bs):
        return None

    # Exactly one paired BUILD_STRING is +1 in piece count; all others equal. Any
    # other delta (e.g. +2 = two surplus pieces, or -1) -> defer.
    surplus = None  # (der_bs_index, gt_bs_index, der_count)
    for di, gi in zip(der_bs, gt_bs):
        dc = _build_string_count(der[di])
        gc = _build_string_count(gt[gi])
        if dc is None or gc is None:
            return None
        if dc == gc:
            continue
        if dc == gc + 1:
            if surplus is not None:
                return None  # more than one +1 divergence -> ambiguous
            surplus = (di, gi, dc)
        else:
            return None
    if surplus is None:
        return None
    di, gi, der_count = surplus
    if der_count < 1 or di < 1:
        return None

    # The surplus must be a TRAILING string-Constant LOAD_CONST directly feeding the
    # derived BUILD_STRING.
    surplus_ins = der[di - 1]
    if getattr(surplus_ins, "opname", None) != "LOAD_CONST":
        return None
    literal = getattr(surplus_ins, "argval", None)
    if not isinstance(literal, str):
        return None
    # GT's trailing piece must differ from the surplus (else the divergence is an
    # interior insertion, or a segment GT actually keeps).
    if gi >= 1:
        gt_prev = gt[gi - 1]
        if (getattr(gt_prev, "opname", None) == "LOAD_CONST"
                and getattr(gt_prev, "argval", None) == literal):
            return None
    # Never delete a literal GT contains anywhere as a LOAD_CONST operand.
    if any(getattr(ins, "opname", None) == "LOAD_CONST"
           and isinstance(getattr(ins, "argval", None), str)
           and getattr(ins, "argval", None) == literal
           for ins in gt):
        return None

    # Localize the unique source JoinedStr: der_count top-level pieces (one AST value
    # each), final value an ast.Constant str == literal. Any ambiguity -> defer.
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    matches: list[ast.JoinedStr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        values = node.values
        if len(values) != der_count:
            continue
        if not all(isinstance(v, (ast.Constant, ast.FormattedValue)) for v in values):
            continue
        last = values[-1]
        if isinstance(last, ast.Constant) and isinstance(last.value, str) and last.value == literal:
            matches.append(node)
    if len(matches) != 1:
        return None
    node = matches[0]

    # Drop the trailing Constant and re-render; require a clean AST round-trip so we
    # never splice text that does not parse back to the intended f-string.
    new_node = ast.JoinedStr(values=list(node.values[:-1]))
    if not new_node.values:
        return None
    ast.fix_missing_locations(new_node)
    try:
        rendered = ast.unparse(new_node)
    except Exception:  # noqa: BLE001
        return None
    try:
        reparsed = ast.parse(rendered, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(reparsed, ast.JoinedStr) or ast.dump(reparsed) != ast.dump(new_node):
        return None

    edited = _replace_node_span(fragment, dedented, node, rendered)
    if edited is None or edited == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(edited))
    except SyntaxError:
        return None
    return {
        "text": edited,
        "operator": f"drop f-string segment {literal!r}",
        "opname": "BUILD_STRING",
        "kind": "fstring_segment",
        "confidence": "unique",
    }
