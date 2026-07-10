"""Regression guard for the DETERMINISTIC DISPATCH (`run_deterministic_prepass`).

The per-operator tests (test_leaf_operator.py, test_structural_operator.py, ...) each
prove one operator FIRES on its canonical matching input. What they cannot catch is a
*silent miss at the dispatch layer*: an operator that is implemented and imported into
`utils/reattach_source_code_object.py` but never wired into the prepass dispatch tuples,
so a matching object is never routed to it. That operator would look healthy in isolation
yet never run in production.

These tests assert every deterministic operator imported into the reattach engine is
actually invoked inside `run_deterministic_prepass`, that the three regime branches
(is_leaf / is_struct / is_diffbc) are intact, and that no typo'd operator name is called
that was never imported. If someone adds a new `*_candidate` operator + import but forgets
to wire it into the dispatch, `test_all_imported_operators_are_wired` fails.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REATTACH = Path(__file__).resolve().parent.parent / "utils" / "reattach_source_code_object.py"

# `structural_repair_directive` is imported but is an LLM PROMPT SIGNAL, not a prepass
# operator — it does not end in `_candidate` and must not appear in the dispatch. Listed
# here so the intentional exclusion is explicit and documented.
PROMPT_ONLY_SIGNALS = {"structural_repair_directive"}


def _source() -> str:
    return REATTACH.read_text(encoding="utf-8")


def _imported_operators(src: str) -> set[str]:
    """Every `*_candidate` symbol imported from utils.semantic_operators.*"""
    names: set[str] = set()
    for line in src.splitlines():
        if line.startswith("from utils.semantic_operators."):
            names.update(re.findall(r"\b([a-z_]+_candidate)\b", line))
    return names


def _prepass_region(src: str) -> str:
    """Body of run_deterministic_prepass: from its `def` line up to its call site.

    Uses a raw character substring (not splitlines) so the boundary is exact.
    """
    start = src.index("def run_deterministic_prepass")
    # the invocation is a 4-space-indented bare call; the `def` line has `def ` before
    # the name, so this anchor cannot match the definition itself.
    end = src.index("\n    run_deterministic_prepass()", start)
    return src[start:end]


def _dispatched_operators(region: str) -> set[str]:
    """Every `*_candidate` symbol REFERENCED in the region.

    Operators are dispatched two ways: the leaf/offset ones are called directly
    (`leaf_value_candidate(...)`), while the control-flow and different-bytecode ones
    are listed as bare references inside `(fn, op, strat)` tuples and invoked via
    `_fn(...)`. Match the bare name so both forms count as wired.
    """
    return set(re.findall(r"\b([a-z_]+_candidate)\b", region))


def test_imported_operator_set_is_nonempty():
    ops = _imported_operators(_source())
    # sanity: the engine imports a substantial operator suite (>= 15 as of this writing)
    assert len(ops) >= 15, f"expected a full operator suite, found {sorted(ops)}"


def test_all_imported_operators_are_wired_into_prepass():
    """No silent miss: every imported deterministic operator is invoked in the dispatch."""
    src = _source()
    imported = _imported_operators(src)
    called = _dispatched_operators(_prepass_region(src))
    unwired = sorted(imported - called - PROMPT_ONLY_SIGNALS)
    assert not unwired, (
        "these deterministic operators are imported but NEVER called in "
        f"run_deterministic_prepass (matching objects would be silently missed): {unwired}"
    )


def test_no_unimported_operator_is_called():
    """Every operator called in the dispatch is a real imported symbol (catches typos)."""
    src = _source()
    imported = _imported_operators(src)
    called = _dispatched_operators(_prepass_region(src))
    ghost = sorted(called - imported)
    assert not ghost, f"dispatch calls operators that are not imported: {ghost}"


def test_prompt_only_signal_is_not_a_prepass_operator():
    """structural_repair_directive is an LLM prompt signal, not a deterministic operator."""
    region = _prepass_region(_source())
    for signal in PROMPT_ONLY_SIGNALS:
        assert f"{signal}(" not in region, (
            f"{signal} is a prompt-only signal and must not run as a prepass operator"
        )


def test_three_regime_branches_present():
    """The dispatch keeps the leaf / control-flow / different-bytecode routing intact."""
    region = _prepass_region(_source())
    for guard in ("is_leaf", "is_struct", "is_diffbc"):
        assert guard in region, f"regime guard {guard!r} missing from the dispatch"
    # is_leaf must gate on an offset; is_struct on control flow; is_diffbc on Different bytecode
    assert 'tr.message == "Different bytecode" and tr.failed_offset is not None' in region
    assert 'tr.message == "Different control flow"' in region


@pytest.mark.parametrize(
    "operator",
    [
        # leaf / offset regime
        "leaf_value_candidate",
        "collapse_dead_pass_else_candidate",
        # control-flow (is_struct) regime
        "try_except_candidate",
        "exc_regions_candidate",
        "control_flow_candidate",
        "control_flow_renest_candidate",
        # different-bytecode (is_diffbc) regime
        "leaf_value_no_offset_candidate",
        "fstring_segment_candidate",
        "line_too_long_candidate",
        "dropped_assignment_candidate",
        "container_construction_candidate",
        "literal_form_candidate",
        "call_shape_candidate",
        "unary_op_candidate",
        "decorator_default_candidate",
        "unpack_op_candidate",
        "comprehension_op_candidate",
        "ternary_op_candidate",
        "bool_shortcircuit_candidate",
        "dropped_statement_candidate",
    ],
)
def test_each_known_operator_is_dispatched(operator):
    """Each operator we ship is reachable in the prepass (explicit, per-operator lock).

    Match the bare name: leaf/offset operators are called directly while control-flow
    and different-bytecode operators are listed in `(fn, op, strat)` tuples and invoked
    via `_fn(...)`, so a `(`-suffixed check would miss the tuple-listed ones.
    """
    region = _prepass_region(_source())
    assert re.search(rf"\b{operator}\b", region), (
        f"{operator} is not dispatched in run_deterministic_prepass"
    )
