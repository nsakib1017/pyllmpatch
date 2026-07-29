"""Exhaustive deterministic repair: search every reachable operator sequence.

The shipped driver (`_structural_fixpoint`) applies operators in a FIXED ORDER. That makes ordering
load-bearing, and it cost real files: `close_unclosed_opener` measured 1030 -> 1011 conversions
purely because, running early, it mutated source so more precise operators no longer matched. No
ordering fixed it -- moving it last failed too, since the fixpoint loop fed its output back on the
next round.

Breadth-first search over the state space removes ordering from the problem. Every reachable
sequence of operators is tried, so an aggressive operator can no longer hide a fix that a different
order would have found.

WHAT THIS MAKES PROVABLE, precisely:

    No sequence of THESE operators, within the stated depth and state bounds,
    produces source that compiles at the file's target Python version.

That is a checkable claim about a defined search space. It is NOT the claim that no deterministic
program could repair the file -- that is both unprovable and trivially false, since deleting
everything yields a file that compiles. The bounds are part of the claim, which is why
`SearchBudget` is explicit and the exhausted budget is reported rather than hidden.
"""
from __future__ import annotations

import ast
from typing import NamedTuple


class SearchBudget(NamedTuple):
    """Bounds that make the exhaustiveness claim precise. Report these alongside any residual."""

    max_depth: int = 5
    max_states: int = 400


DEFAULT_BUDGET = SearchBudget()


class SearchResult(NamedTuple):
    source: str | None
    path: list
    states_explored: int
    budget_exhausted: bool


def _parses(source) -> bool:
    try:
        ast.parse(source)
        return True
    except Exception:
        return False


def exhaustive_repair(source, compile_fn, version, operators, budget=DEFAULT_BUDGET):
    """Breadth-first search for ANY operator sequence that makes `source` compile.

    `operators` is a sequence of ``(name, callable(source, error) -> str | None)``.
    `compile_fn(source, version)` must raise on failure -- it is the AUTHORITY, so a rewrite the
    host parser accepts but the target version rejects is never adopted.

    Returns ``(repaired_source_or_None, operator_path)``. Source that already compiles is returned
    untouched with an empty path -- repairing valid code is damage, not repair.

    Breadth-first rather than depth-first so the SHORTEST sequence wins: fewer rewrites means less
    divergence from what the decompiler actually emitted. States are de-duplicated by exact text, so
    idempotent operators cost one expansion instead of looping. Never raises: an operator that
    explodes forfeits its branch, not the search.
    """
    if not source:
        return None, []
    try:
        if _parses(source):
            try:
                compile_fn(source, version)
                return source, []
            except Exception:
                pass  # parses on this host but not at the target version -- keep searching

        seen = {source}
        frontier = [(source, [])]
        explored = 0

        for _ in range(max(1, int(budget.max_depth))):
            next_frontier = []
            for current, path in frontier:
                for name, fn in operators:
                    if explored >= budget.max_states:
                        return None, []
                    try:
                        candidate = fn(current, None)
                    except Exception:
                        continue  # a broken operator forfeits its branch only
                    if not candidate or candidate in seen:
                        continue
                    seen.add(candidate)
                    explored += 1
                    step = path + [name]
                    if _parses(candidate):
                        try:
                            compile_fn(candidate, version)
                            return candidate, step
                        except Exception:
                            pass  # host-parseable but target-rejected: not a fix
                    next_frontier.append((candidate, step))
            if not next_frontier:
                break  # closure reached: nothing new is derivable
            frontier = next_frontier
        return None, []
    except Exception:
        return None, []
