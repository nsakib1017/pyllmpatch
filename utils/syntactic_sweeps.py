from __future__ import annotations

import ast
import difflib
from typing import Callable, NamedTuple

from utils._sweep_miners import m2ops, m3_ops, ops4, sweeps, sweeps2, tx3
from utils.syntactic_prepass import (
    SyntaxErrorInfo,
    balance_delimiters,
    dedent_stray_block,
    fix_numeric_literal,
    literal_lhs_rename,
)


class SweepOp(NamedTuple):

    family: str
    name: str
    fn: Callable[[str, SyntaxErrorInfo | None], str | None]
    preserves_code: bool


def parse_error(source):
    try:
        ast.parse(source)
        return None
    except SyntaxError as exc:
        return SyntaxErrorInfo(exc.lineno, exc.offset, (exc.msg or "").strip())
    except Exception as exc:  # ValueError (null bytes), RecursionError, MemoryError
        return SyntaxErrorInfo(None, None, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- calling conventions
# The miners were written independently and take three different signatures. Rather than edit the
# vendored (measured) operator bodies, adapt them at the boundary.


def _from_source(fn):

    def op(source, err):
        return fn(source)

    op.__name__ = fn.__name__
    return op


def _from_source_and_error(fn):

    def op(source, err):
        return fn(source, err)

    op.__name__ = fn.__name__
    return op


def _from_lines(fn):

    def op(source, err):
        lines = source.split("\n")
        had_trailing_newline = bool(lines) and lines[-1] == ""
        if had_trailing_newline:
            lines.pop()
        out = fn(lines, err if err is not None else SyntaxErrorInfo(1, 1, ""))
        if out is None or out == lines:
            return None
        return "\n".join(out) + ("\n" if had_trailing_newline else "")

    op.__name__ = fn.__name__
    return op


def _guarded(op):

    def guard(source, err):
        try:
            out = op(source, err)
        except Exception:
            return None
        if out is None or out == source:
            return None
        return out

    guard.__name__ = op.__name__
    return guard


# ------------------------------------------------------------------------------------ registries
PURE_OPS = (
    SweepOp("lhs-rename", "bad_lhs_rename", _guarded(_from_source(tx3.bad_lhs_rename)), True),
    SweepOp("paren-surplus", "insert_missing_open_paren",
            _guarded(_from_lines(m3_ops.insert_missing_open_paren)), True),
    SweepOp("bracket-kind", "retype_mismatched_closer",
            _guarded(_from_source_and_error(m2ops.retype_mismatched_closer)), True),
    SweepOp("truncation", "complete_truncated_statement",
            _guarded(_from_source(tx3.complete_truncated_statement_nosynth)), True),
    SweepOp("clause-align", "sweep_align_handlers",
            _guarded(_from_source_and_error(sweeps.sweep_align_handlers)), True),
    SweepOp("clause-align", "align_clause_to_opener",
            _guarded(_from_lines(m3_ops.align_clause_to_opener)), True),
    SweepOp("handler-fold", "fold_postinserted_handler_body",
            _guarded(_from_source_and_error(sweeps.sweep_fold_postinserted_handler_body)), True),
    SweepOp("clause-rewrite", "clause_to_if_true",
            _guarded(_from_source(tx3.clause_to_if_true)), True),
    SweepOp("indent", "normalize_indentation",
            _guarded(_from_source(tx3.normalize_indentation_nosynth)), True),
)

SYNTH_OPS = (
    SweepOp("try-scaffold", "complete_all_orphan_try",
            _guarded(_from_source(tx3.complete_all_orphan_try)), False),
    SweepOp("try-scaffold", "synthesize_try_for_orphan_handler",
            _guarded(_from_lines(m3_ops.synthesize_try_for_orphan_handler)), False),
    SweepOp("closer-delete", "drop_surplus_closer",
            _guarded(_from_source_and_error(m2ops.drop_surplus_closer)), False),
    SweepOp("truncation-lossy", "truncate_partial_element",
            _guarded(_from_source_and_error(ops4.truncate_partial_element)), False),
)

# Operators that preserve every line but change what the program DOES. `clause_to_if_true` rewrites
# a dangling `else:` into `if True:`; nothing is deleted, so the repair is genuine under the
# project's definition, but the reconstructed control flow is not the original's. It is the single
# largest contributor in stage 1 (+4.9 points), which is exactly why it must stay countable on its
# own instead of being absorbed into a headline "deterministic fix" number.
CONTROL_FLOW_REWRITES = frozenset({"clause_to_if_true"})

# The shipped error-driven operators, restricted to the ones that add and remove nothing. They run
# as a final sweep inside each round: they are cheap, and they clear the small residue (unbalanced
# delimiters, a stray indent) that the whole-file sweeps expose but do not themselves target.
# `fix_line_continuation` and `complete_orphan_try` are excluded -- both synthesize.
_LEGACY_PURE = (balance_delimiters, dedent_stray_block, fix_numeric_literal, literal_lhs_rename)

_MAX_LEGACY_STEPS = 40
_DEFAULT_MAX_ROUNDS = 12


def _legacy_sweep(source, ops):
    current, fired = source, []
    for _ in range(_MAX_LEGACY_STEPS):
        err = parse_error(current)
        if err is None:
            return current, fired
        progressed = False
        for fn in ops:
            try:
                candidate = fn(current, err)
            except Exception:
                candidate = None
            if not candidate or candidate == current:
                continue
            after = parse_error(candidate)
            if after is not None and (after.lineno, after.offset, after.msg) == (
                err.lineno, err.offset, err.msg
            ):
                continue  # identical error -> no state change -> not progress
            current = candidate
            fired.append("legacy:" + fn.__name__)
            progressed = True
            break
        if not progressed:
            break
    return current, fired


def run_stack(source, ops, legacy=_LEGACY_PURE, max_rounds=_DEFAULT_MAX_ROUNDS):
    current, fired = source, []
    if parse_error(current) is None:
        return current, fired
    for _ in range(max(1, int(max_rounds))):
        changed = False
        for op in ops:
            err = parse_error(current)
            if err is None:
                return current, fired
            candidate = op.fn(current, err)
            if candidate and candidate != current:
                current = candidate
                fired.append(op.name)
                changed = True
                if parse_error(current) is None:
                    return current, fired
        if legacy:
            after_legacy, legacy_fired = _legacy_sweep(current, legacy)
            if after_legacy != current:
                current = after_legacy
                fired.extend(legacy_fired)
                changed = True
                if parse_error(current) is None:
                    return current, fired
        if not changed:
            break
    return current, fired




def _significant_lines(source):
    return [line.strip() for line in source.splitlines() if line.strip()]

def audit_fidelity(before, after):
    """Line-level diff of a rewrite: how much original content vanished, how much appeared.

    Blank lines and indentation are normalized away, so re-indentation and re-flowing count as
    neither deletion nor insertion -- only content that actually disappeared or was invented. This
    is what verifies the stage-1 preservation contract instead of trusting `preserves_code`.
    """
    old, new = _significant_lines(before), _significant_lines(after)
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    deleted = inserted = replaced_old = replaced_new = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            deleted += i2 - i1
        elif tag == "insert":
            inserted += j2 - j1
        elif tag == "replace":
            replaced_old += i2 - i1
            replaced_new += j2 - j1
    return {
        "deleted": deleted,
        "inserted": inserted,
        "replaced_old": replaced_old,
        "replaced_new": replaced_new,
        "lines_before": len(old),
        "lines_after": len(new),
    }
