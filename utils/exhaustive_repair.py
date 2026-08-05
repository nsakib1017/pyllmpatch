from __future__ import annotations

import ast
from typing import NamedTuple


class SearchBudget(NamedTuple):

    max_depth: int = 5
    max_states: int = 400


DEFAULT_BUDGET = SearchBudget()


def _parses(source) -> bool:
    try:
        ast.parse(source)
        return True
    except Exception:
        return False


def exhaustive_repair(source, compile_fn, version, operators, budget=DEFAULT_BUDGET):
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


class SearchResult(NamedTuple):
    source: str | None
    path: list
    states_explored: int
    budget_exhausted: bool
