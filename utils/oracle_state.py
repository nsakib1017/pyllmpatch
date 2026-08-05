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
    if result.get("success"):
        return "Equal"
    message = result.get("message")
    if message == MSG_DIFF_BYTECODE:
        return "Different bytecode (offset)" if result.get("failed_offset") is not None \
            else "Different bytecode (no-offset / exc-table)"
    return message or "Unknown"


def object_state(result: dict) -> tuple[int, int]:
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
    name = result.get("names")
    if name:
        return str(name)
    # Fallbacks for robustness across result shapes.
    return str(result.get("result_key") or result.get("name") or id(result))


def index_by_name(results: list[dict]) -> dict[str, tuple[int, int]]:
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
    successes_prev: int
    successes_cand: int
    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)

    @property
    def non_regressing(self) -> bool:
        return not self.regressed and self.successes_cand >= self.successes_prev

    @property
    def strict_improvement(self) -> bool:
        return bool(self.improved) and not self.regressed


def compare_verifications(prev_results: list[dict], cand_results: list[dict]) -> VerificationDelta:
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
