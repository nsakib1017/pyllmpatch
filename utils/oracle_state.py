"""Oracle-state lattice for semantic repair (M2).

The PyLingual equivalence oracle (`compare_pyc` -> per-object results, surfaced by
`run_pylingual_verification`) is the *definition* of a perfect decompilation: a
file is perfect iff every code object has ``success == True``. Today the repair
loop selects targets and accepts candidates off a custom ``combined_distance``,
which provably disagrees with the oracle. This module makes the oracle the
objective by defining a total order on per-object oracle states, so the loop can:

  * tell whether a candidate *strictly improved*, *regressed*, or *held* an
    object's state (not just whether the raw success-count changed), and
  * guarantee a deterministic-operator fixpoint terminates (a monotone increase
    in a well-ordered measure).

Per-object lattice (higher is better)::

    Equal (success)                                   rank 4
    Different bytecode, failed_offset = O             rank 3   (deeper O is better)
    Different bytecode, failed_offset = None          rank 2   (exception-table-only)
    Different control flow                            rank 1
    Missing / Extra bytecode                          rank 0

The ``failed_offset is None`` split is load-bearing: ``compare_bytecode`` checks
``named_exception_table`` equality *before* the per-instruction loop, so an
exception-table-only divergence surfaces as "Different bytecode" with no offset
and must be routed to a structural fix, not a leaf-value one.

This module is intentionally dependency-free (operates on the plain result dicts
produced by ``run_pylingual_verification``) so it imports fast and unit-tests in
isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- regime classification ------------------------------------------------

RANK_EQUAL = 4
RANK_DIFF_BYTECODE_OFFSET = 3
RANK_DIFF_BYTECODE_NO_OFFSET = 2  # exception-table-only
RANK_DIFF_CONTROL_FLOW = 1
RANK_MISSING_EXTRA = 0
RANK_ABSENT = -1  # object not present in this verification's result set at all

MSG_EQUAL = "Equal"
MSG_DIFF_BYTECODE = "Different bytecode"
MSG_DIFF_CONTROL_FLOW = "Different control flow"
MSG_MISSING = "Missing bytecode"
MSG_EXTRA = "Extra bytecode"


def classify_regime(result: dict) -> str:
    """Human-readable regime label for a failing/ passing oracle result."""
    if result.get("success"):
        return "Equal"
    message = result.get("message")
    if message == MSG_DIFF_BYTECODE:
        return "Different bytecode (offset)" if result.get("failed_offset") is not None \
            else "Different bytecode (no-offset / exc-table)"
    return message or "Unknown"


def object_state(result: dict) -> tuple[int, int]:
    """Return the comparable per-object key ``(rank, offset_depth)``.

    A larger key is a *better* (closer-to-equal) state. ``offset_depth`` only
    differentiates within ``RANK_DIFF_BYTECODE_OFFSET`` (deeper = more leading
    instructions matched before divergence); it is 0 for every other rank.
    """
    if result.get("success"):
        return (RANK_EQUAL, 0)
    message = result.get("message")
    if message == MSG_DIFF_BYTECODE:
        offset = result.get("failed_offset")
        if offset is not None:
            return (RANK_DIFF_BYTECODE_OFFSET, int(offset))
        return (RANK_DIFF_BYTECODE_NO_OFFSET, 0)
    if message == MSG_DIFF_CONTROL_FLOW:
        return (RANK_DIFF_CONTROL_FLOW, 0)
    if message in (MSG_MISSING, MSG_EXTRA):
        return (RANK_MISSING_EXTRA, 0)
    # Unknown message: treat as the control-flow regime (no localization).
    return (RANK_DIFF_CONTROL_FLOW, 0)


def _result_name(result: dict) -> str:
    """Stable per-object key. ``names`` is the dotted qualified name."""
    name = result.get("names")
    if name:
        return str(name)
    # Fallbacks for robustness across result shapes.
    return str(result.get("result_key") or result.get("name") or id(result))


def index_by_name(results: list[dict]) -> dict[str, tuple[int, int]]:
    """Map object name -> oracle state. On duplicate names keep the WORST
    (lowest) state, so a regression in any occurrence is visible."""
    out: dict[str, tuple[int, int]] = {}
    for r in results:
        name = _result_name(r)
        state = object_state(r)
        if name not in out or state < out[name]:
            out[name] = state
    return out


def count_successes(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("success"))


def all_equal(results: list[dict]) -> bool:
    return bool(results) and all(r.get("success") for r in results)


@dataclass
class VerificationDelta:
    """Structured comparison of a candidate verification vs the previous one."""
    successes_prev: int
    successes_cand: int
    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)

    @property
    def non_regressing(self) -> bool:
        """No commonly-named object dropped in the lattice, and the candidate
        did not lose any passing objects."""
        return not self.regressed and self.successes_cand >= self.successes_prev

    @property
    def strict_improvement(self) -> bool:
        """At least one object got strictly better and none regressed."""
        return bool(self.improved) and not self.regressed


def compare_verifications(prev_results: list[dict], cand_results: list[dict]) -> VerificationDelta:
    """Compare two whole-file oracle verifications, per object, by the lattice.

    Rules for objects whose presence changes between the two result sets:
      * in both          -> compare states (rank, then offset depth).
      * only in prev      -> regression iff it was passing (we lost a good object).
      * only in candidate -> improvement iff it is now passing (we gained one).
    """
    prev = index_by_name(prev_results)
    cand = index_by_name(cand_results)

    improved: list[str] = []
    regressed: list[str] = []

    for name in set(prev) | set(cand):
        before = prev.get(name)
        after = cand.get(name)
        if before is not None and after is not None:
            if after > before:
                improved.append(name)
            elif after < before:
                regressed.append(name)
        elif after is not None and before is None:
            # newly appearing object
            if after[0] == RANK_EQUAL:
                improved.append(name)
        elif before is not None and after is None:
            # disappeared object
            if before[0] == RANK_EQUAL:
                regressed.append(name)

    return VerificationDelta(
        successes_prev=count_successes(prev_results),
        successes_cand=count_successes(cand_results),
        improved=sorted(improved),
        regressed=sorted(regressed),
    )
