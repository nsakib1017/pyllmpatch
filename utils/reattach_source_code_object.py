from __future__ import annotations

import hashlib
import argparse
import ast
import inspect
import io
import json
import os
import re
import sys
import textwrap
import time
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYLINGUAL_ROOT = REPO_ROOT / "pylingual"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYLINGUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PYLINGUAL_ROOT))

from pipeline.config import ACCEPTED_CODE_OBJECT_PATH, ACCEPTED_CODE_OBJECT_TELEMETRY_PATH, now_iso
from pipeline.logging_utils import append_log
from utils.generate_bytecode import CompileError, compile_version
from utils.map_source_code_objects import MappingError, map_source_to_pyc
from utils.pyc_code_object_distance import (
    compare_code_object_distances,
    load_editable_bytecode_from_pyc,
    set_exception_table_distance_enabled,
    summarize_results,
    validate_input,
)
from utils.version import PythonVersion
from utils.oracle_state import compare_verifications
from utils.semantic_operators.call_shape import call_shape_candidate
from utils.semantic_operators.comprehension_op import comprehension_op_candidate
from utils.semantic_operators.container import container_construction_candidate
from utils.semantic_operators.control_flow import control_flow_candidate
from utils.semantic_operators.control_flow_renest import control_flow_renest_candidate
from utils.semantic_operators.decorator_default import decorator_default_candidate
from utils.semantic_operators.exc_regions import exc_regions_candidate
from utils.semantic_operators.bool_shortcircuit import bool_shortcircuit_candidate
from utils.semantic_operators.dropped_statement import dropped_statement_candidate
from utils.semantic_operators.literal_form import literal_form_candidate
from utils.semantic_operators.ternary_op import ternary_op_candidate
from utils.semantic_operators.unary_op import unary_op_candidate
from utils.semantic_operators.unpack_op import unpack_op_candidate
from utils.semantic_operators.fstring import fstring_segment_candidate
from utils.semantic_operators.jump import collapse_dead_pass_else_candidate
from utils.semantic_operators.leaf import leaf_value_candidate, leaf_value_no_offset_candidate
from utils.semantic_operators.structural_directive import structural_repair_directive
from utils.semantic_operators.placeholder import dropped_assignment_candidate, line_too_long_candidate
from utils.semantic_operators.structural import try_except_candidate

# Acceptance / target-selection mode (M2). "distance" (default) preserves the
# historical combined-distance behaviour exactly; "oracle" makes the PyLingual
# equivalence oracle primary for BOTH target selection and candidate acceptance.
# Settable per-run via repair_mismatching_code_objects(acceptance_mode=...) or
# process-wide via the SEMANTIC_ACCEPTANCE_MODE environment variable.
ACCEPTANCE_MODE = (os.getenv("SEMANTIC_ACCEPTANCE_MODE", "distance").strip().lower() or "distance")

# M4: run the deterministic operators (leaf, structural) to a FIXPOINT before the
# LLM loop, so proven fixes land without the LLM out-ranking them per step. Gated
# by the same switch that enables the operators at the fixer seam.
# Deterministic prepass is ALWAYS-first by default: it runs every deterministic operator to a
# fixpoint before the LLM ever engages, self-gated on a strict oracle-state improvement (no
# regression), so the LLM only sees what no deterministic move can fix. Kill-switch: set
# SEMANTIC_DETERMINISTIC_OPERATORS=0 to disable.
SEMANTIC_DETERMINISTIC_PREPASS = (
    os.getenv("SEMANTIC_DETERMINISTIC_OPERATORS", "1").strip().lower() in ("1", "true", "yes", "on")
)
# Post-LLM deterministic reconciliation: after EACH LLM iteration, if the file is not yet perfect,
# re-run the oracle-gated deterministic operators on the residual (a structural LLM edit can expose
# a leaf divergence the pre-LLM prepass couldn't reach). Flow: pre-LLM det -> LLM -> if not perfect
# -> post-LLM det -> test perfect -> repeat. Off by default (opt-in for A/B measurement).
SEMANTIC_POST_LLM_DETERMINISTIC = (
    os.getenv("SEMANTIC_POST_LLM_DETERMINISTIC", "0").strip().lower() in ("1", "true", "yes", "on")
)
# Corrective residual feedback (#54): when a fragment LLM candidate is rejected, capture the
# GT-vs-candidate bytecode residual (what the model's own edit still got wrong) and feed it into
# the next retry prompt, instead of only an abstract distance. Prompt-only (never touches the
# oracle gate). Off by default (opt-in for A/B measurement); off ⇒ no capture, prompt unchanged.
SEMANTIC_CORRECTIVE_FEEDBACK = (
    os.getenv("SEMANTIC_CORRECTIVE_FEEDBACK", "0").strip().lower() in ("1", "true", "yes", "on")
)
# Annotation reconciliation (Stage 1, oracle-source backend): when a PEP-649 `__annotate__`
# code object diverges/is missing, repair its ENCLOSING class/func from GT source instead of
# treating the synthetic annotation object itself as an unreattachable target. Off by default
# (opt-in); off => byte-identical no-op branch in the deterministic prepass.
SEMANTIC_ANNOTATION_RECONCILE = (
    os.getenv("SEMANTIC_ANNOTATION_RECONCILE", "0").strip().lower() in ("1", "true", "yes", "on")
)
# Corpus-write kill switch: when TRUE, skip appending accepted steps to the fine-tune
# training-corpus log (`_append_accepted_code_object_dataset`), e.g. to keep a contaminated or
# exploratory run out of the corpus. Does NOT affect `_append_accepted_case_telemetry` (a separate
# log) or any acceptance decision. Off by default (opt-in); off => byte-identical current behavior.
SEMANTIC_DISABLE_CORPUS_WRITE = (
    os.getenv("SEMANTIC_DISABLE_CORPUS_WRITE", "0").strip().lower() in ("1", "true", "yes", "on")
)


def _tail_deadline_seconds() -> float:
    """Lever B (speedup): stop a file whose BEST state has not improved for this many seconds
    (0/unset/garbage = disabled). The hard tail (files grinding to the wall-clock cap without
    converging) is where most run-time is spent; cutting it early is quality-neutral because the
    engine always rolls back to the best state it found."""
    try:
        return float(os.getenv("SEMANTIC_TAIL_DEADLINE", "0") or 0)
    except (TypeError, ValueError):
        return 0.0


def _should_tail_abort(*, now: float, last_improvement: float, tail_deadline: float) -> bool:
    """True iff the tail deadline is enabled (>0) and the best state has been stagnant for at
    least that many seconds. Pure so it is unit-tested without the engine/GPU."""
    return tail_deadline > 0 and (now - last_improvement) >= tail_deadline


class ReattachError(RuntimeError):
    pass


class SemanticRepairTimeoutNoImprovement(TimeoutError):
    pass


class SemanticRepairHardTimeoutNoImprovement(SemanticRepairTimeoutNoImprovement):
    pass


FragmentFixer = Callable[[str, Any, Any, str, dict | None], Any]


@dataclass(frozen=True)
class ReattachmentCandidate:
    kind: str
    replacement_text: str
    updated_source: str
    target_row: dict
    parse_ok: bool
    parse_error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Locate a mapped source code object, optionally replace its source span, "
            "compile with the input .pyc Python version, and optionally compare the resulting .pyc."
        )
    )
    parser.add_argument("source_path", type=Path, help="Path to the source .py file")
    parser.add_argument("pyc_path", type=Path, help="Path to the corresponding .pyc file used for mapping")
    parser.add_argument("qualname", type=str, help="Mapped source qualname to extract or replace")
    parser.add_argument(
        "--replacement-file",
        type=Path,
        default=None,
        help="Path to a file containing replacement source for the mapped span",
    )
    parser.add_argument(
        "--replacement-text",
        type=str,
        default=None,
        help="Literal replacement source for the mapped span",
    )
    parser.add_argument(
        "--output-source",
        type=Path,
        default=None,
        help="Path to write the updated source file. Required when replacing.",
    )
    parser.add_argument(
        "--output-pyc",
        type=Path,
        default=None,
        help="Optional path for the compiled output .pyc. Defaults to __pycache__ next to output source.",
    )
    parser.add_argument(
        "--compare-pyc",
        type=Path,
        default=None,
        help="Optional reference .pyc path to compare against after compilation",
    )
    parser.add_argument(
        "--comparison-json-out",
        type=Path,
        default=None,
        help="Optional path to save the comparison summary as JSON",
    )
    parser.add_argument(
        "--strict-map",
        action="store_true",
        help="Require the mapper to have no unmatched rows while locating the target.",
    )
    return parser


def _load_text(path: Path) -> str:
    return path.expanduser().resolve().read_text(encoding="utf-8")


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    if not text.endswith(("\n", "\r")):
        offsets.append(len(text))
    return offsets


def parenthesize_bare_except(source: str) -> str:
    """Rewrite Python-2-style unparenthesized multi-type ``except`` clauses --
    ``except A, B:`` -- to the parenthesized Python-3 form ``except (A, B):``.

    This is a decompiler artifact that shows up verbatim in some ground-truth sources:
    it is a ``SyntaxError`` in Python 3, so any candidate/fragment that carries it fails to
    (re)compile. The rewrite is semantics-preserving (parenthesized and unparenthesized
    multi-except are equivalent; parenthesized is the only valid Python 3 spelling) and a
    no-op for everything else:

    - ``except X:``            -> unchanged
    - ``except X as e:``       -> unchanged
    - ``except (X, Y):``       -> unchanged (already parenthesized)
    - ``except X, Y:``         -> ``except (X, Y):``
    - ``except X, Y as e:``    -> ``except (X, Y) as e:``
    - ``except a.B, c.D, E:``  -> ``except (a.B, c.D, E):``
    - ``except* X, Y:``        -> ``except* (X, Y):`` (PEP 654 exception groups;
      the inserted ``(`` lands after the ``*``, never between ``except`` and ``*``)

    Deliberately does NOT use ``ast.parse`` -- the input may itself contain the very
    ``SyntaxError`` being fixed. Uses ``tokenize`` instead, which lexes without enforcing
    grammar, so it tolerates an otherwise-invalid ``except`` header. Only a top-level
    (bracket-depth-0) comma between ``except`` and the header-terminating ``:`` (or ``as``)
    triggers a rewrite, so commas in call args, subscripts, comments, and string literals --
    anywhere in the file, including on the same line as an ``except`` -- are never touched.
    Idempotent: an already-parenthesized clause has no top-level comma, so a second pass is
    a no-op. If the source can't be tokenized at all (e.g. an unterminated string), the
    input is returned unchanged rather than raising.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, UnicodeDecodeError):
        return source

    line_offsets = _line_offsets(source)

    def offset(row: int, col: int) -> int:
        return line_offsets[row - 1] + col

    edits: list[tuple[int, int]] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if tok.type == tokenize.NAME and tok.string == "except":
            depth = 0
            has_top_comma = False
            as_offset: int | None = None
            colon_offset: int | None = None
            j = i + 1
            while j < n:
                t = tokens[j]
                if t.type == tokenize.OP:
                    if t.string in "([{":
                        depth += 1
                    elif t.string in ")]}":
                        depth -= 1
                    elif t.string == "," and depth == 0:
                        has_top_comma = True
                    elif t.string == ":" and depth == 0:
                        colon_offset = offset(*t.start)
                        break
                elif t.type == tokenize.NAME and t.string == "as" and depth == 0 and as_offset is None:
                    as_offset = offset(*t.start)
                j += 1
            if has_top_comma and colon_offset is not None:
                close_offset = as_offset if as_offset is not None else colon_offset
                # Trim trailing whitespace so ')' lands right after the last type token,
                # keeping the original spacing before 'as'/':' intact.
                while close_offset > 0 and source[close_offset - 1] in " \t":
                    close_offset -= 1
                k = i + 1
                while k < n and tokens[k].type in (tokenize.NL, tokenize.COMMENT):
                    k += 1
                # PEP 654 exception groups: ``except* X, Y:``. The ``*`` is a separate
                # OP token right after ``except``; skip past it too so the inserted
                # ``(`` lands after ``except*`` (and its following space), not between
                # ``except`` and ``*``.
                if k < n and tokens[k].type == tokenize.OP and tokens[k].string == "*":
                    k += 1
                    while k < n and tokens[k].type in (tokenize.NL, tokenize.COMMENT):
                        k += 1
                if k < n and close_offset > offset(*tokens[k].start):
                    open_offset = offset(*tokens[k].start)
                    edits.append((open_offset, close_offset))
            i = j
        i += 1

    if not edits:
        return source

    # Apply right-to-left so earlier offsets stay valid as we insert characters.
    result = source
    for open_offset, close_offset in sorted(edits, key=lambda e: e[0], reverse=True):
        result = result[:close_offset] + ")" + result[close_offset:]
        result = result[:open_offset] + "(" + result[open_offset:]
    return result


def _span_to_indices(
    text: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    if start_line < 1 or end_line < 1 or start_line > len(lines) or end_line > len(lines):
        raise ReattachError("Mapped span is outside the source file bounds")

    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    start_index = offsets[start_line - 1] + start_col
    end_index = offsets[end_line - 1] + end_col
    return start_index, end_index


_MAPPING_IDENTITY_FIELDS = (
    "source_qualname",
    "source_kind",
    "source_lineno",
    "source_end_lineno",
    "source_col_offset",
    "source_end_col_offset",
    "source_occurrence_index",
    "source_sibling_ordinal",
    "source_ordinal_path",
    "source_immediate_child_count",
    "source_collision_size",
    "pyc_qualname",
    "pyc_firstlineno",
    "pyc_occurrence_index",
    "pyc_sibling_ordinal",
    "pyc_ordinal_path",
    "pyc_immediate_child_count",
    "pyc_collision_size",
    "match_reason",
    "matched",
)


def _optional_int(value: Any, default: int = 10**9) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _ordinal_path_sort_key(raw: Any) -> tuple[int, ...]:
    if not raw:
        return (10**9,)
    try:
        return tuple(int(part) for part in str(raw).split(".") if part != "")
    except Exception:
        return (10**9,)


def _mapping_row_identity(row: dict | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {field: row.get(field) for field in _MAPPING_IDENTITY_FIELDS if field in row}


def _mapping_ambiguity_summary(candidates: list[dict]) -> dict[str, Any]:
    return {
        "candidate_count": len(candidates),
        "source_spans": [
            {
                "source_kind": row.get("source_kind"),
                "source_lineno": row.get("source_lineno"),
                "source_col_offset": row.get("source_col_offset"),
                "source_end_lineno": row.get("source_end_lineno"),
                "source_end_col_offset": row.get("source_end_col_offset"),
                "source_occurrence_index": row.get("source_occurrence_index"),
                "source_ordinal_path": row.get("source_ordinal_path"),
                "source_collision_size": row.get("source_collision_size"),
                "pyc_firstlineno": row.get("pyc_firstlineno"),
                "pyc_occurrence_index": row.get("pyc_occurrence_index"),
                "pyc_ordinal_path": row.get("pyc_ordinal_path"),
                "pyc_collision_size": row.get("pyc_collision_size"),
                "match_reason": row.get("match_reason"),
                "matched": row.get("matched"),
            }
            for row in candidates
        ],
    }


def _mapping_identity_matches(row: dict, target_identity: dict[str, Any]) -> bool:
    for key, expected in target_identity.items():
        if expected is None or key not in _MAPPING_IDENTITY_FIELDS:
            continue
        actual = row.get(key)
        if actual != expected and str(actual) != str(expected):
            return False
    return True


def _mapping_resolution_sort_key(row: dict, qualname: str) -> tuple[Any, ...]:
    matched = row.get("matched")
    source_lineno = _optional_int(row.get("source_lineno"))
    pyc_firstlineno = _optional_int(row.get("pyc_firstlineno"))
    return (
        0 if row.get("pyc_qualname") == qualname else 1,
        0 if matched is True else 1,
        abs(source_lineno - pyc_firstlineno) if pyc_firstlineno != 10**9 else 10**9,
        _optional_int(row.get("source_collision_size")),
        _optional_int(row.get("pyc_collision_size")),
        _ordinal_path_sort_key(row.get("source_ordinal_path")),
        _ordinal_path_sort_key(row.get("pyc_ordinal_path")),
        _optional_int(row.get("source_occurrence_index")),
        _optional_int(row.get("pyc_occurrence_index")),
        source_lineno,
        _optional_int(row.get("source_col_offset")),
        _optional_int(row.get("source_end_lineno")),
        _optional_int(row.get("source_end_col_offset")),
    )


def _find_target_row(
    source_path: Path,
    pyc_path: Path,
    qualname: str,
    strict_map: bool,
    target_identity: dict[str, Any] | None = None,
) -> dict:

    rows = map_source_to_pyc(source_path, pyc_path, strict=strict_map)
    candidates = [
        row
        for row in rows
        if row["row_type"] == "source_to_pyc" and row["source_qualname"] == qualname
    ]
    if not candidates:
        raise ReattachError(f"No mapped source code object found for qualname: {qualname}")
    if target_identity:
        identity_matches = [row for row in candidates if _mapping_identity_matches(row, target_identity)]
        if identity_matches:
            candidates = identity_matches
    if len(candidates) > 1:
        summary = _mapping_ambiguity_summary(candidates)
        if strict_map:
            raise ReattachError(
                f"Qualname is ambiguous across {len(candidates)} rows: {qualname}; "
                f"candidate identity summary: {json.dumps(summary, sort_keys=True, default=str)}"
            )
        candidates = sorted(candidates, key=lambda row: _mapping_resolution_sort_key(row, qualname))
        selected = dict(candidates[0])
        selected["_mapping_resolution"] = {
            "ambiguous": True,
            "resolution": "ranked_by_mapping_identity",
            "candidate_count": len(candidates),
            "candidate_summary": summary,
        }
        return selected
    return candidates[0]


def extract_source_segment(source_text: str, target_row: dict) -> str:
    start_index, end_index = _span_to_indices(
        source_text,
        int(target_row["source_lineno"]),
        int(target_row["source_col_offset"]),
        int(target_row["source_end_lineno"]),
        int(target_row["source_end_col_offset"]),
    )
    return source_text[start_index:end_index]


def replace_source_segment(source_text: str, target_row: dict, replacement_text: str) -> str:
    start_index, end_index = _span_to_indices(
        source_text,
        int(target_row["source_lineno"]),
        int(target_row["source_col_offset"]),
        int(target_row["source_end_lineno"]),
        int(target_row["source_end_col_offset"]),
    )
    return source_text[:start_index] + replacement_text + source_text[end_index:]


def _pass_replacement_for_row(target_row: dict) -> str:
    return " " * int(target_row["source_col_offset"]) + "pass"


def _node_span_to_indices(source_text: str, node: ast.AST) -> tuple[int, int]:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if None in {lineno, col_offset, end_lineno, end_col_offset}:
        raise ReattachError("AST node does not expose a complete source span")
    return _span_to_indices(
        source_text,
        int(lineno),
        int(col_offset),
        int(end_lineno),
        int(end_col_offset),
    )


def _top_level_non_code_statements(source_text: str) -> list[ast.stmt]:
    tree = ast.parse(source_text)
    return [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _extract_node_source(source_text: str, node: ast.AST) -> str:
    start_index, end_index = _node_span_to_indices(source_text, node)
    return source_text[start_index:end_index]


def _is_code_object(value: Any) -> bool:
    return hasattr(value, "co_code") and hasattr(value, "co_name")


def _render_safe_literal(value: Any) -> str:
    if isinstance(value, tuple):
        inner = ", ".join(_render_safe_literal(item) for item in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(_render_safe_literal(item) for item in value) + "]"
    if isinstance(value, set):
        items = sorted((_render_safe_literal(item) for item in value), key=repr)
        return "{" + ", ".join(items) + "}"
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(f"{_render_safe_literal(key)}: {_render_safe_literal(item)}")
        return "{" + ", ".join(parts) + "}"
    return repr(value)


def _render_assignment_text(name: str, value: Any) -> str:
    return f"{name} = {_render_safe_literal(value)}"


def _render_annotation_text(name: str, annotation: str, value: Any) -> str:
    return f"{name}: {annotation} = {_render_safe_literal(value)}"


def _apply_source_edits(source_text: str, edits: list[tuple[int, int, str, int]]) -> str:
    candidate = source_text
    for start_index, end_index, replacement_text, _seq in sorted(edits, key=lambda item: (item[0], item[1], item[3]), reverse=True):
        candidate = candidate[:start_index] + replacement_text + candidate[end_index:]
    return candidate


def _editable_instruction_items(bytecode: Any) -> list[Any]:
    if bytecode is None or not hasattr(bytecode, "instructions"):
        return []
    try:
        return list(bytecode)
    except Exception:
        return []


def _resolve_code310_arg(opname: str, arg: int, code_object: Any) -> Any:
    consts = getattr(code_object, "co_consts", ())
    names = getattr(code_object, "co_names", ())
    varnames = getattr(code_object, "co_varnames", ())
    freevars = getattr(code_object, "co_freevars", ())
    cellvars = getattr(code_object, "co_cellvars", ())
    cell_names = tuple(cellvars) + tuple(freevars)
    if opname == "LOAD_CONST" and arg < len(consts):
        return consts[arg]
    if opname in {
        "STORE_NAME",
        "LOAD_NAME",
        "DELETE_NAME",
        "LOAD_GLOBAL",
        "STORE_GLOBAL",
        "DELETE_GLOBAL",
        "LOAD_ATTR",
        "STORE_ATTR",
        "DELETE_ATTR",
        "LOAD_METHOD",
        "IMPORT_NAME",
        "IMPORT_FROM",
    } and arg < len(names):
        return names[arg]
    if opname in {"LOAD_FAST", "STORE_FAST", "DELETE_FAST"} and arg < len(varnames):
        return varnames[arg]
    if opname in {"LOAD_CLOSURE", "LOAD_DEREF", "STORE_DEREF", "DELETE_DEREF"} and arg < len(cell_names):
        return cell_names[arg]
    return arg


def _decode_code310_instruction_records(code_object: Any) -> list[dict]:
    code = getattr(code_object, "co_code", None)
    if code is None:
        return []
    try:
        from xdis.opcodes import opcode_310
    except Exception:
        return []

    records = []
    extended_arg = 0
    for offset in range(0, len(code), 2):
        op = code[offset]
        raw_arg = code[offset + 1] if offset + 1 < len(code) else 0
        arg = raw_arg | extended_arg
        try:
            opname = opcode_310.opname[op]
        except Exception:
            opname = f"<{op}>"
        argval = _resolve_code310_arg(opname, arg, code_object)
        records.append(
            {
                "index": len(records),
                "offset": offset,
                "starts_line": None,
                "opname": opname,
                "argrepr": "" if argval is None else str(argval),
                "argval": argval,
            }
        )
        if opname == "EXTENDED_ARG":
            extended_arg = arg << 8
        else:
            extended_arg = 0
    return records


def _module_instruction_stream(code_object: Any) -> list[dict]:
    if code_object is None:
        return []
    raw_instructions = _editable_instruction_items(code_object)
    if raw_instructions:
        return [
            {"opname": getattr(ins, "opname", ""), "argval": getattr(ins, "argval", None)}
            for ins in raw_instructions
        ]
    return [
        {"opname": record["opname"], "argval": record["argval"]}
        for record in _decode_code310_instruction_records(code_object)
    ]


def _is_safe_literal(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, int, float, complex, bool)):
        return True
    if isinstance(value, tuple):
        return all(_is_safe_literal(item) for item in value)
    if isinstance(value, list):
        return all(_is_safe_literal(item) for item in value)
    if isinstance(value, set):
        return all(_is_safe_literal(item) for item in value)
    if isinstance(value, dict):
        return all(_is_safe_literal(k) and _is_safe_literal(v) for k, v in value.items())
    return False


def _render_import_text(module_name: str, aliases: list[tuple[str, str | None]]) -> str:
    if len(aliases) == 1 and aliases[0][0] == module_name.split(".")[-1] and aliases[0][1] is None:
        return f"import {module_name}"
    if len(aliases) == 1 and aliases[0][1] is not None:
        alias_name, asname = aliases[0]
        if asname == alias_name:
            return f"import {module_name}"
        return f"import {module_name} as {asname}"
    parts = []
    for alias_name, asname in aliases:
        if asname and asname != alias_name:
            parts.append(f"{alias_name} as {asname}")
        else:
            parts.append(alias_name)
    return f"from {module_name} import {', '.join(parts)}"


def _current_module_statement_records(source_text: str) -> list[dict]:
    tree = ast.parse(source_text)
    records: list[dict] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef) or isinstance(node, ast.ClassDef):
            continue
        start_index, end_index = _node_span_to_indices(source_text, node)
        text = source_text[start_index:end_index]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            records.append({
                "kind": "docstring",
                "key": "docstring",
                "text": text,
                "start": start_index,
                "end": end_index,
                "annotation": None,
            })
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                raise ReattachError("Unsupported top-level assignment expression in module body")
            if not _is_safe_literal(value):
                raise ReattachError("Unsupported top-level assignment literal in module body")
            records.append({
                "kind": "assign",
                "key": f"assign:{node.targets[0].id}",
                "text": text,
                "start": start_index,
                "end": end_index,
                "name": node.targets[0].id,
                "annotation": None,
            })
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                raise ReattachError("Unsupported annotated assignment expression in module body")
            if not _is_safe_literal(value):
                raise ReattachError("Unsupported annotated assignment literal in module body")
            records.append({
                "kind": "assign",
                "key": f"assign:{node.target.id}",
                "text": text,
                "start": start_index,
                "end": end_index,
                "name": node.target.id,
                "annotation": ast.unparse(node.annotation) if hasattr(ast, "unparse") else None,
            })
            continue
        if isinstance(node, ast.Import):
            aliases = [(alias.name, alias.asname) for alias in node.names]
            key = "import:" + ",".join(
                f"{name} as {asname}" if asname else name for name, asname in aliases
            )
            records.append({
                "kind": "import",
                "key": key,
                "text": text,
                "start": start_index,
                "end": end_index,
                "module_name": None,
                "aliases": aliases,
            })
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level:
                raise ReattachError("Relative imports are unsupported in deterministic preprocessing")
            aliases = [(alias.name, alias.asname) for alias in node.names]
            key = f"from:{node.module or ''}:" + ",".join(
                f"{name} as {asname}" if asname else name for name, asname in aliases
            )
            records.append({
                "kind": "from_import",
                "key": key,
                "text": text,
                "start": start_index,
                "end": end_index,
                "module_name": node.module or "",
                "aliases": aliases,
            })
            continue
        raise ReattachError("Unsupported top-level statement in module body")
    return records


def extract_module_body_statements(source_text: str) -> str:
    records = _current_module_statement_records(source_text)
    return "\n\n".join(record["text"] for record in records)


def _decoded_module_statement_records(code_object: Any) -> list[dict]:
    instructions = _module_instruction_stream(code_object)
    if not instructions:
        return []

    records: list[dict] = []
    i = 0
    while i < len(instructions):
        opname = instructions[i]["opname"]
        argval = instructions[i]["argval"]
        if opname in {"RESUME", "RETURN_CONST", "NOP", "CACHE", "SETUP_ANNOTATIONS"}:
            i += 1
            continue

        if opname == "LOAD_BUILD_CLASS":
            j = i + 1
            while j < len(instructions) and instructions[j]["opname"] != "STORE_NAME":
                j += 1
            if j < len(instructions):
                i = j + 1
                continue
            raise ReattachError("Unsupported class definition pattern in deterministic module preprocessing")

        if opname == "LOAD_CONST" and _is_code_object(argval):
            j = i + 1
            while j < len(instructions) and instructions[j]["opname"] != "STORE_NAME":
                j += 1
            if j < len(instructions):
                i = j + 1
                continue
            raise ReattachError("Unsupported function definition pattern in deterministic module preprocessing")

        if opname == "LOAD_CONST" and i + 1 < len(instructions) and instructions[i + 1]["opname"] == "STORE_NAME":
            name = str(instructions[i + 1]["argval"])
            value = argval
            if _is_safe_literal(value):
                records.append({
                    "kind": "docstring" if name == "__doc__" else "assign",
                    "key": "docstring" if name == "__doc__" else f"assign:{name}",
                    "text": repr(value) if name == "__doc__" else _render_assignment_text(name, value),
                    "name": name,
                    "value": value,
                })
                i += 2
                continue

        if (
            opname == "LOAD_CONST"
            and i + 3 < len(instructions)
            and instructions[i + 1]["opname"] == "LOAD_CONST"
            and instructions[i + 2]["opname"] == "IMPORT_NAME"
        ):
            level = instructions[i]["argval"]
            fromlist = instructions[i + 1]["argval"]
            module_name = str(instructions[i + 2]["argval"])
            if fromlist in (None, (), (None,)):
                if i + 3 < len(instructions) and instructions[i + 3]["opname"] == "STORE_NAME":
                    alias = str(instructions[i + 3]["argval"])
                    records.append({
                        "kind": "import",
                        "key": f"import:{module_name} as {alias}" if alias != module_name.split(".")[-1] else f"import:{module_name}",
                        "text": _render_import_text(module_name, [(module_name.split(".")[-1], None if alias == module_name.split(".")[-1] else alias)]),
                        "module_name": module_name,
                        "aliases": [(module_name.split(".")[-1], None if alias == module_name.split(".")[-1] else alias)],
                        "level": level,
                    })
                    i += 4
                    continue
            else:
                aliases = []
                j = i + 3
                while j < len(instructions):
                    if instructions[j]["opname"] == "IMPORT_FROM":
                        imported_name = str(instructions[j]["argval"])
                        if j + 1 < len(instructions) and instructions[j + 1]["opname"] == "STORE_NAME":
                            alias = str(instructions[j + 1]["argval"])
                            aliases.append((imported_name, None if alias == imported_name else alias))
                            j += 2
                            continue
                    if instructions[j]["opname"] == "POP_TOP":
                        break
                    break
                if aliases:
                    records.append({
                        "kind": "from_import",
                        "key": f"from:{module_name}:" + ",".join(
                            f"{name} as {asname}" if asname else name for name, asname in aliases
                        ),
                        "text": _render_import_text(module_name, aliases),
                        "module_name": module_name,
                        "aliases": aliases,
                        "level": level,
                    })
                    i = j + 1
                    continue

        if opname == "LOAD_CONST":
            stack: list[Any] = []
            j = i
            while j < len(instructions):
                cur = instructions[j]
                cur_opname = cur["opname"]
                cur_argval = cur["argval"]
                if cur_opname in {"RESUME", "NOP", "CACHE", "EXTENDED_ARG", "SETUP_ANNOTATIONS"}:
                    j += 1
                    continue
                if cur_opname == "LOAD_CONST":
                    stack.append(cur_argval)
                    j += 1
                    continue
                if cur_opname == "BUILD_TUPLE":
                    n = int(cur_argval)
                    if len(stack) < n:
                        break
                    values = stack[-n:]
                    stack = stack[:-n]
                    stack.append(tuple(values))
                    j += 1
                    continue
                if cur_opname == "BUILD_LIST":
                    n = int(cur_argval)
                    if len(stack) < n:
                        break
                    values = stack[-n:]
                    stack = stack[:-n]
                    stack.append(list(values))
                    j += 1
                    continue
                if cur_opname == "BUILD_SET":
                    n = int(cur_argval)
                    if len(stack) < n:
                        break
                    values = stack[-n:]
                    stack = stack[:-n]
                    stack.append(set(values))
                    j += 1
                    continue
                if cur_opname == "BUILD_CONST_KEY_MAP":
                    n = int(cur_argval)
                    if len(stack) < n + 1:
                        break
                    keys = stack.pop()
                    values = [stack.pop() for _ in range(n)][::-1]
                    if not isinstance(keys, tuple):
                        break
                    stack.append(dict(zip(keys, values)))
                    j += 1
                    continue
                if cur_opname == "BUILD_MAP":
                    n = int(cur_argval)
                    if len(stack) < 2 * n:
                        break
                    items = stack[-2 * n:]
                    stack = stack[:-2 * n]
                    mapping: dict[Any, Any] = {}
                    for pair_index in range(0, len(items), 2):
                        mapping[items[pair_index]] = items[pair_index + 1]
                    stack.append(mapping)
                    j += 1
                    continue
                if cur_opname == "LIST_EXTEND":
                    if len(stack) < 2 or not isinstance(stack[-2], list):
                        break
                    iterable = stack.pop()
                    target = stack.pop()
                    try:
                        target.extend(list(iterable))
                    except TypeError:
                        break
                    stack.append(target)
                    j += 1
                    continue
                if cur_opname == "SET_UPDATE":
                    if len(stack) < 2 or not isinstance(stack[-2], set):
                        break
                    iterable = stack.pop()
                    target = stack.pop()
                    try:
                        target.update(set(iterable))
                    except TypeError:
                        break
                    stack.append(target)
                    j += 1
                    continue
                if cur_opname == "DICT_UPDATE":
                    if len(stack) < 2 or not isinstance(stack[-2], dict):
                        break
                    update_value = stack.pop()
                    target = stack.pop()
                    try:
                        target.update(dict(update_value))
                    except Exception:
                        break
                    stack.append(target)
                    j += 1
                    continue
                if cur_opname == "STORE_NAME" and len(stack) == 1:
                    name = str(cur_argval)
                    value = stack.pop()
                    if not _is_safe_literal(value):
                        break
                    records.append({
                        "kind": "docstring" if name == "__doc__" else "assign",
                        "key": "docstring" if name == "__doc__" else f"assign:{name}",
                        "text": repr(value) if name == "__doc__" else _render_assignment_text(name, value),
                        "name": name,
                        "value": value,
                    })
                    i = j + 1
                    break
                break
            if i != j:
                continue

        raise ReattachError("Unsupported bytecode pattern in deterministic module preprocessing")

    return records


def deterministic_module_body_candidate(
    gt_code_object: Any,
    derived_code_object: Any,
    current_source_text: str,
) -> str:
    gt_records = _decoded_module_statement_records(gt_code_object)
    current_records = _current_module_statement_records(current_source_text)
    if not gt_records or not current_records:
        raise ReattachError("No deterministic module statements available")

    current_keys = [record["key"] for record in current_records]
    target_keys = [record["key"] for record in gt_records]
    sm = __import__("difflib").SequenceMatcher(a=current_keys, b=target_keys)

    edits: list[tuple[int, int, str, int]] = []
    seq = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                current_record = current_records[i1 + offset]
                target_record = gt_records[j1 + offset]
                replacement_text = target_record["text"]
                if current_record["text"].strip() != replacement_text.strip():
                    edits.append((current_record["start"], current_record["end"], replacement_text, seq))
                    seq += 1
        elif tag == "replace":
            paired = min(i2 - i1, j2 - j1)
            for offset in range(paired):
                current_record = current_records[i1 + offset]
                target_record = gt_records[j1 + offset]
                replacement_text = target_record["text"]
                if current_record["text"].strip() != replacement_text.strip():
                    edits.append((current_record["start"], current_record["end"], replacement_text, seq))
                    seq += 1
            for offset in range(paired, i2 - i1):
                record = current_records[i1 + offset]
                edits.append((record["start"], record["end"], "", seq))
                seq += 1
            insert_anchor = current_records[i1]["start"] if i1 < len(current_records) else len(current_source_text)
            for offset in range(paired, j2 - j1):
                replacement_text = gt_records[j1 + offset]["text"]
                edits.append((insert_anchor, insert_anchor, replacement_text, seq))
                seq += 1
        elif tag == "delete":
            for idx in range(i1, i2):
                record = current_records[idx]
                edits.append((record["start"], record["end"], "", seq))
                seq += 1
        elif tag == "insert":
            if i1 < len(current_records):
                anchor = current_records[i1]["start"]
            elif current_records:
                anchor = current_records[-1]["end"]
            else:
                anchor = 0
            for offset in range(j1, j2):
                edits.append((anchor, anchor, gt_records[offset]["text"], seq))
                seq += 1

    if not edits:
        raise ReattachError("No deterministic module statement edits found")

    candidate = _apply_source_edits(current_source_text, edits)
    if candidate == current_source_text:
        raise ReattachError("Deterministic module preprocessing did not change source")
    return candidate


def normalize_semantic_replacement_indentation(
    replacement_text: str,
    target_row: dict,
    original_fragment: str | None = None,
) -> str:
    """Align a semantic repair fragment to the mapped destination span."""
    base_indent = " " * int(target_row["source_col_offset"])
    normalized = textwrap.dedent(replacement_text.strip("\n"))
    lines = normalized.splitlines()
    if not lines:
        return ""

    def _is_suite_header(line: str) -> bool:
        stripped = line.lstrip()
        if not stripped.endswith(":"):
            return False
        head = stripped[:-1].strip()
        return (
            head.startswith("async def ")
            or head.startswith("def ")
            or head.startswith("class ")
            or head.startswith("if ")
            or head.startswith("elif ")
            or head.startswith("for ")
            or head.startswith("while ")
            or head.startswith("try")
            or head.startswith("except ")
            or head.startswith("finally")
            or head.startswith("with ")
            or head.startswith("match ")
            or head.startswith("case ")
            or head == "else"
        )

    def _is_branch_header(line: str) -> bool:
        stripped = line.lstrip()
        head = stripped.split(None, 1)[0].rstrip(":") if stripped else ""
        return head in {"elif", "else", "except", "finally", "case"}

    def _is_parseable_fragment(text: str) -> bool:
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False

    def _normalize_body_lines(*, strict_body_indent: bool) -> str:
        if not _is_suite_header(lines[0]):
            return "\n".join(
                base_indent + line if line.strip() else ""
                for line in lines
            )

        adjusted: list[str] = [lines[0]]
        for line in lines[1:]:
            if not line.strip():
                adjusted.append("")
            elif _is_branch_header(line):
                adjusted.append(line.lstrip())
            elif strict_body_indent:
                adjusted.append(f"    {line.lstrip()}")
            else:
                adjusted.append(line if line.startswith((" ", "\t")) else f"    {line.lstrip()}")
        return "\n".join(adjusted)

    candidate = _normalize_body_lines(strict_body_indent=False)
    if _is_parseable_fragment(candidate):
        return candidate

    if _is_suite_header(lines[0]):
        fallback = _normalize_body_lines(strict_body_indent=True)
        if _is_parseable_fragment(fallback):
            return fallback

    return candidate


def _unique_replacement_candidates(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _source_span_replacement_candidates(
    replacement_text: str,
    source_col_offset: int,
    original_fragment: str | None = None,
) -> list[tuple[str, str]]:
    """Generate candidate fragments for insertion at an exact source span.

    The first replacement line is inserted after the existing source prefix up to
    source_col_offset. Continuation lines must therefore carry file-level
    indentation, not just indentation relative to the first replacement line.
    """
    base_indent = " " * int(source_col_offset)
    stripped = replacement_text.strip("\n")
    dedented = textwrap.dedent(stripped)
    fragment_normalized = normalize_semantic_replacement_indentation(
        replacement_text,
        {"source_col_offset": source_col_offset},
        original_fragment,
    )

    def _span_relative(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return ""
        adjusted = [lines[0].lstrip()]
        adjusted.extend(lines[1:])
        return "\n".join(adjusted)

    def _with_absolute_continuations(text: str) -> str:
        lines = _span_relative(text).splitlines()
        if not lines:
            return ""
        adjusted = [lines[0]]
        for line in lines[1:]:
            adjusted.append(base_indent + line if line.strip() else "")
        return "\n".join(adjusted)

    named_candidates = [
        ("raw", stripped),
        ("dedented", dedented),
        ("fragment_normalized", fragment_normalized),
        ("span_relative_raw", _span_relative(stripped)),
        ("span_relative_dedented", _span_relative(dedented)),
        ("span_relative_fragment_normalized", _span_relative(fragment_normalized)),
        ("absolute_continuations_raw", _with_absolute_continuations(stripped)),
        ("absolute_continuations_dedented", _with_absolute_continuations(dedented)),
        ("absolute_continuations_fragment_normalized", _with_absolute_continuations(fragment_normalized)),
    ]
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for kind, candidate in named_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append((kind, candidate))
    return unique


def _candidate_row_from_slice(
    source_text: str,
    start_index: int,
    end_index: int,
    source_col_offset: int,
) -> dict:
    line_offsets = [0]
    for line in source_text.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    def _line_col(index: int) -> tuple[int, int]:
        line_no = max(1, len(line_offsets) - 1)
        for offset_index in range(1, len(line_offsets)):
            if line_offsets[offset_index] > index:
                line_no = offset_index
                break
        return line_no, index - line_offsets[line_no - 1]

    start_line, start_col = _line_col(start_index)
    end_line, end_col = _line_col(end_index)
    return {
        "source_lineno": start_line,
        "source_col_offset": int(source_col_offset),
        "source_end_lineno": end_line,
        "source_end_col_offset": end_col,
        "_span_start_index": start_index,
        "_span_end_index": end_index,
        "_span_original_col_offset": start_col,
    }


def _row_with_indices(source_text: str, target_row: dict) -> tuple[dict, int, int]:
    row = dict(target_row)
    start_index, end_index = _span_to_indices(
        source_text,
        int(row["source_lineno"]),
        int(row["source_col_offset"]),
        int(row["source_end_lineno"]),
        int(row["source_end_col_offset"]),
    )
    row["_span_start_index"] = start_index
    row["_span_end_index"] = end_index
    return row, start_index, end_index


def _decorated_node_start(node: ast.AST) -> tuple[int, int]:
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        first = decorators[0]
        return int(getattr(first, "lineno")), int(getattr(first, "col_offset"))
    return int(getattr(node, "lineno")), int(getattr(node, "col_offset"))


def _node_contains_row(node: ast.AST, target_row: dict) -> bool:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if lineno is None or end_lineno is None:
        return False
    start_line = int(target_row["source_lineno"])
    end_line = int(target_row["source_end_lineno"])
    node_start_line, _ = _decorated_node_start(node)
    return node_start_line <= start_line and end_line <= int(end_lineno)


def _ast_expanded_replacement_rows(source_text: str, target_row: dict) -> list[dict]:
    """Return mapped row plus enclosing AST definition rows that may be safer."""
    rows: list[dict] = []
    base_row, _, _ = _row_with_indices(source_text, target_row)
    rows.append(base_row)
    source_kind = str(target_row.get("source_kind") or "")
    if source_kind and source_kind not in {"function", "async_function", "class"}:
        return rows
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return rows

    allowed = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, allowed)
        and _node_contains_row(node, target_row)
        and getattr(node, "end_lineno", None) is not None
        and getattr(node, "end_col_offset", None) is not None
    ]
    if source_kind == "class":
        nodes = [node for node in nodes if isinstance(node, ast.ClassDef)]
    elif source_kind == "function":
        nodes = [node for node in nodes if isinstance(node, ast.FunctionDef)]
    elif source_kind == "async_function":
        nodes = [node for node in nodes if isinstance(node, ast.AsyncFunctionDef)]
    if not nodes:
        return rows

    def _span_size(node: ast.AST) -> tuple[int, int]:
        start_line, start_col = _decorated_node_start(node)
        return (int(getattr(node, "end_lineno")) - start_line, int(getattr(node, "end_col_offset")) - start_col)

    node = min(nodes, key=_span_size)
    start_line, start_col = _decorated_node_start(node)
    expanded_row = dict(target_row)
    expanded_row.update(
        {
            "source_lineno": start_line,
            "source_col_offset": start_col,
            "source_end_lineno": int(getattr(node, "end_lineno")),
            "source_end_col_offset": int(getattr(node, "end_col_offset")),
        }
    )
    expanded_row, _, _ = _row_with_indices(source_text, expanded_row)
    if (
        expanded_row["_span_start_index"],
        expanded_row["_span_end_index"],
    ) != (
        base_row["_span_start_index"],
        base_row["_span_end_index"],
    ):
        rows.append(expanded_row)
    return rows


def choose_best_reattachment(
    source_text: str,
    target_row: dict,
    replacement_text: str,
    original_fragment: str | None = None,
) -> ReattachmentCandidate:
    """Select a syntax-valid full-source splice candidate for a mapped row."""
    best_failed: ReattachmentCandidate | None = None
    best_parsed: tuple[tuple[int, int, int, int], ReattachmentCandidate] | None = None
    replacement_head = textwrap.dedent(replacement_text.strip("\n")).lstrip()
    replacement_is_definition = replacement_head.startswith(("def ", "async def ", "class ", "@"))

    def _continuation_escape_penalty(candidate: str, base_indent: str) -> int:
        lines = candidate.splitlines()
        if len(lines) <= 1:
            return 0
        return sum(
            1
            for line in lines[1:]
            if line.strip() and not line.startswith(base_indent)
        )

    for row_index, row in enumerate(_ast_expanded_replacement_rows(source_text, target_row)):
        start_index = int(row["_span_start_index"])
        end_index = int(row["_span_end_index"])
        base_indent = " " * int(row["source_col_offset"])
        for kind, candidate in _source_span_replacement_candidates(
            replacement_text,
            int(row["source_col_offset"]),
            original_fragment,
        ):
            updated_text = source_text[:start_index] + candidate + source_text[end_index:]
            candidate_kind = f"{'mapped' if row_index == 0 else 'ast_expanded'}:{kind}"
            try:
                ast.parse(updated_text)
                parsed = ReattachmentCandidate(
                    kind=candidate_kind,
                    replacement_text=candidate,
                    updated_source=updated_text,
                    target_row=row,
                    parse_ok=True,
                )
                row_preference = int(
                    (replacement_is_definition and row_index == 0)
                    or (not replacement_is_definition and row_index > 0)
                )
                escape_penalty = _continuation_escape_penalty(candidate, base_indent)
                candidate_preference = 0 if kind.startswith("absolute_continuations") else 1
                score = (row_preference, escape_penalty, candidate_preference, row_index)
                if best_parsed is None or score < best_parsed[0]:
                    best_parsed = (score, parsed)
            except SyntaxError as exc:
                best_failed = ReattachmentCandidate(
                    kind=candidate_kind,
                    replacement_text=candidate,
                    updated_source=updated_text,
                    target_row=row,
                    parse_ok=False,
                    parse_error=str(exc),
                )
    if best_parsed is not None:
        return best_parsed[1]
    if best_failed is not None:
        return best_failed
    row, start_index, end_index = _row_with_indices(source_text, target_row)
    updated_text = source_text[:start_index] + source_text[end_index:]
    return ReattachmentCandidate(
        kind="empty",
        replacement_text="",
        updated_source=updated_text,
        target_row=row,
        parse_ok=False,
        parse_error="no replacement candidates generated",
    )


def choose_best_reattachment_for_source_slice(
    source_text: str,
    start_index: int,
    end_index: int,
    source_col_offset: int,
    replacement_text: str,
    original_fragment: str | None = None,
) -> ReattachmentCandidate:
    target_row = _candidate_row_from_slice(source_text, start_index, end_index, source_col_offset)
    return choose_best_reattachment(
        source_text,
        target_row,
        replacement_text,
        original_fragment,
    )


def normalize_semantic_replacement_for_source_slice(
    source_text: str,
    start_index: int,
    end_index: int,
    source_col_offset: int,
    replacement_text: str,
    original_fragment: str | None = None,
) -> str:
    """Choose the first indentation candidate that parses in the full source."""
    candidates = _source_span_replacement_candidates(
        replacement_text,
        source_col_offset,
        original_fragment,
    )
    for _, candidate in candidates:
        updated_text = source_text[:start_index] + candidate + source_text[end_index:]
        try:
            ast.parse(updated_text)
            return candidate
        except SyntaxError:
            continue
    return candidates[-1][1] if candidates else ""


def normalize_semantic_replacement_for_source(
    source_text: str,
    target_row: dict,
    replacement_text: str,
    original_fragment: str | None = None,
) -> str:
    """Normalize a mapped replacement by validating the whole reattached file."""
    candidate = choose_best_reattachment(
        source_text,
        target_row,
        replacement_text,
        original_fragment,
    )
    return candidate.replacement_text


def normalize_semantic_insertion_indentation(replacement_text: str, base_indent: str) -> str:
    """Align a missing code-object fragment for insertion into a parent body."""
    normalized = textwrap.dedent(replacement_text.strip("\n"))
    return "\n".join(
        base_indent + line if line.strip() else ""
        for line in normalized.splitlines()
    )


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _infer_child_indent(lines: list[str], parent_row: dict) -> str:
    parent_indent = " " * int(parent_row["source_col_offset"])
    parent_lineno = int(parent_row["source_lineno"])
    parent_end_lineno = int(parent_row["source_end_lineno"])
    for line in lines[parent_lineno:parent_end_lineno]:
        if line.strip():
            indent = _leading_whitespace(line)
            if len(indent) > len(parent_indent):
                return indent
    return parent_indent + "    "


def _parent_qualname(qualname: str) -> str | None:
    parts = qualname.split(".")
    if len(parts) <= 2:
        return "<module>"
    return ".".join(parts[:-1])


def _find_nearest_existing_parent_row(
    source_path: Path,
    pyc_path: Path,
    qualname: str,
    strict_map: bool,
) -> dict | None:
    parent = _parent_qualname(qualname)
    while parent and parent != "<module>":
        try:
            return _find_target_row(source_path, pyc_path, parent, strict_map=strict_map)
        except ReattachError:
            parent = _parent_qualname(parent)
    return None


def insert_missing_source_segment(
    source_text: str,
    source_path: Path,
    pyc_path: Path,
    qualname: str,
    replacement_text: str,
    *,
    strict_map: bool,
) -> tuple[str, str | None, str]:
    parent_row = _find_nearest_existing_parent_row(source_path, pyc_path, qualname, strict_map)
    lines = source_text.splitlines()
    if parent_row is None:
        base_indent = ""
        insert_line_index = len(lines)
        parent_qualname = "<module>"
    else:
        base_indent = _infer_child_indent(lines, parent_row)
        insert_line_index = int(parent_row["source_end_lineno"])
        parent_qualname = parent_row["source_qualname"]

    insertion = normalize_semantic_insertion_indentation(replacement_text, base_indent)
    updated_lines = lines[:insert_line_index]
    if updated_lines and updated_lines[-1].strip():
        updated_lines.append("")
    updated_lines.extend(insertion.splitlines())
    if insert_line_index < len(lines) and lines[insert_line_index].strip():
        updated_lines.append("")
    updated_lines.extend(lines[insert_line_index:])
    return "\n".join(updated_lines) + "\n", parent_qualname, base_indent


def _coerce_python_version(version: Any) -> PythonVersion:
    if isinstance(version, PythonVersion):
        return version
    if hasattr(version, "as_tuple"):
        version = version.as_tuple()
    return PythonVersion(version)


def _python_version_tag(version: Any) -> str:
    python_version = _coerce_python_version(version)
    return f"{python_version.major}{python_version.minor}"


def _compiled_pyc_path(pyc_dir: Path, source_path: Path, version: Any) -> Path:
    return pyc_dir / f"{source_path.stem}.cpython-{_python_version_tag(version)}.pyc"


def _pyc_python_version(pyc_path: Path) -> PythonVersion:
    bytecode_root = load_editable_bytecode_from_pyc(validate_input(pyc_path))
    return _coerce_python_version(bytecode_root.version)


def infer_semantic_repair_python_version(gt_pyc: Path, derived_pyc: Path) -> PythonVersion:
    gt_version = _pyc_python_version(gt_pyc)
    derived_version = _pyc_python_version(derived_pyc)
    if gt_version != derived_version:
        raise ReattachError(
            "Semantic repair requires matching ground-truth and derived Python bytecode versions: "
            f"gt={gt_version}, derived={derived_version}"
        )
    return gt_version


def compile_source_to_pyc(source_path: Path, output_pyc: Path | None, version: Any = (3, 10)) -> Path:
    python_version = _coerce_python_version(version)
    source_path = source_path.expanduser().resolve()
    if output_pyc is None:
        pycache_dir = source_path.parent / "__pycache__"
        pycache_dir.mkdir(parents=True, exist_ok=True)
        output_pyc = _compiled_pyc_path(pycache_dir, source_path, python_version)
    output_pyc = output_pyc.expanduser().resolve()
    output_pyc.parent.mkdir(parents=True, exist_ok=True)
    prior_uv_cache_dir = os.environ.get("UV_CACHE_DIR")
    os.environ["UV_CACHE_DIR"] = str((Path("/tmp") / "uv-cache").resolve())
    try:
        source_text = source_path.read_text(encoding="utf-8")
        ast.parse(source_text, filename=str(source_path))
        compile_version(source_path, output_pyc, python_version.as_tuple())
    except CompileError as exc:
        raise ReattachError(f"Python {python_version} compilation failed:\n{exc}") from exc
    except SyntaxError as exc:
        raise ReattachError(f"Python source parsing failed:\n{exc}") from exc
    finally:
        if prior_uv_cache_dir is None:
            os.environ.pop("UV_CACHE_DIR", None)
        else:
            os.environ["UV_CACHE_DIR"] = prior_uv_cache_dir
    return output_pyc


def run_comparison(compiled_pyc: Path, compare_pyc: Path) -> dict:
    results = compare_code_object_distances(validate_input(compare_pyc), validate_input(compiled_pyc))
    return summarize_results(results)


def infer_source_from_pyc(pyc_path: Path) -> Path:
    pyc_path = validate_input(pyc_path)
    if pyc_path.parent.name == "__pycache__":
        stem = pyc_path.name.split(".cpython-", 1)[0]
        candidate = pyc_path.parent.parent / f"{stem}.py"
        if candidate.is_file():
            return candidate.resolve()
    name = pyc_path.name
    if ".cpython-" in name:
        stem = name.split(".cpython-", 1)[0]
        candidate = pyc_path.with_name(f"{stem}.py")
        if candidate.is_file():
            return candidate.resolve()
    candidate = pyc_path.with_suffix(".py")
    if candidate.is_file():
        return candidate.resolve()
    raise ReattachError(f"Could not infer source file from pyc path: {pyc_path}")


def _qualname_depth(qualname: str) -> int:
    return qualname.count(".")


def _has_selected_ancestor(qualname: str, selected: set[str]) -> bool:
    parts = qualname.split(".")
    for idx in range(1, len(parts) - 1):
        ancestor = ".".join(parts[: idx + 1])
        if ancestor in selected:
            return True
    return False


def select_repair_targets(distance_rows: list[dict], *, include_module: bool = False) -> list[str]:
    candidates = [
        row["gt_name"]
        for row in distance_rows
        if row["status"] == "matched"
        and row["combined_distance"] > 0
        and row["gt_name"]
        and row["derived_name"]
        and row["gt_name"] == row["derived_name"]
        and (include_module or row["gt_name"] != "<module>")
        and not _is_synthetic_annotation_qualname(row["gt_name"])
    ]
    ordered = sorted(set(candidates), key=lambda name: (_qualname_depth(name), name))
    selected: list[str] = []
    selected_set: set[str] = set()
    for qualname in ordered:
        if _has_selected_ancestor(qualname, selected_set):
            continue
        selected.append(qualname)
        selected_set.add(qualname)
    return selected


def select_unreattachable_missing_targets(distance_rows: list[dict]) -> list[str]:
    return sorted(
        row["gt_name"]
        for row in distance_rows
        if row["status"] == "missing" and row["gt_name"]
    )


def select_missing_repair_targets(distance_rows: list[dict]) -> list[str]:
    return [
        row["gt_name"]
        for row in distance_rows
        if row["status"] == "missing"
        and row["gt_name"]
        and row["gt_name"] != "<module>"
        and not _is_expression_child_qualname(row["gt_name"])
        and not _is_synthetic_annotation_qualname(row["gt_name"])
    ]


EXPRESSION_CHILD_QUALNAME_PARTS = (
    ".<lambda>",
    ".<listcomp>",
    ".<dictcomp>",
    ".<setcomp>",
    ".<genexpr>",
)


def _is_expression_child_qualname(qualname: str | None) -> bool:
    return bool(qualname and any(part in qualname for part in EXPRESSION_CHILD_QUALNAME_PARTS))


SYNTHETIC_ANNOTATION_LEAF = "__annotate__"


def _is_synthetic_annotation_qualname(qualname: str | None) -> bool:
    """PEP-649/749 deferred-annotation code objects (leaf ``__annotate__``) are
    compiler-generated and have no source representation — the decompiled source
    lost the annotations, so the recompiled derived object has no ``__annotate__``
    child and it surfaces as an unreattachable *missing* target. Skip it like an
    expression-level child so one synthetic object can't abort the whole file."""
    if not qualname:
        return False
    return str(qualname).rsplit(".", 1)[-1] == SYNTHETIC_ANNOTATION_LEAF


def _annotate_enclosing_qualname(qualname: str | None) -> str | None:
    """`<module>.A.B.__annotate__` -> `<module>.A.B` (the def whose annotations produced it).
    None when the leaf isn't `__annotate__` or the enclosing scope is `<module>` (no def span)."""
    if not qualname:
        return None
    parts = str(qualname).split(".")
    if parts[-1] != "__annotate__":
        return None
    enclosing = ".".join(parts[:-1])
    return enclosing if enclosing and enclosing != "<module>" else None


def _gt_def_source_by_qualname(gt_source_text: str, qualname: str) -> str | None:
    """Return the exact GT source segment of the class/func at ``qualname``
    (dotted, rooted at ``<module>``), or None on any ambiguity (parse failure,
    qualname not rooted at ``<module>``, or any path segment not found)."""
    import ast as _ast
    parts = str(qualname).split(".")
    if not parts or parts[0] != "<module>":
        return None
    names = parts[1:]
    if not names:
        return None
    try:
        tree = _ast.parse(gt_source_text)
    except SyntaxError:
        return None
    node = tree
    for name in names:
        body = getattr(node, "body", None)
        if body is None:
            return None
        match = next((c for c in body
                      if isinstance(c, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef))
                      and c.name == name), None)
        if match is None:
            return None
        node = match
    seg = _ast.get_source_segment(gt_source_text, node)
    return seg


def _expression_child_parent_qualname(qualname: str | None) -> str | None:
    if not qualname:
        return None
    text = str(qualname)
    split_points = [
        text.index(part)
        for part in EXPRESSION_CHILD_QUALNAME_PARTS
        if part in text
    ]
    if not split_points:
        return None
    return text[: min(split_points)]


def select_missing_expression_child_parent_targets(distance_rows: list[dict]) -> list[str]:
    """Route missing expression-level children to their enclosing code object.

    Lambdas and comprehensions are expression-level source, so standalone
    insertion is usually underconstrained. Repairing the parent gives the model
    the expression that must create the child code object.
    """
    available_names = {
        name
        for row in distance_rows
        for name in (row.get("gt_name"), row.get("derived_name"))
        if name
    }
    parents = []
    for row in distance_rows:
        if row.get("status") != "missing":
            continue
        parent = _expression_child_parent_qualname(row.get("gt_name"))
        if parent and parent in available_names and parent != "<module>":
            parents.append(parent)
    return sorted(set(parents), key=lambda name: (_qualname_depth(name), name))


def select_missing_expression_child_parent_records(distance_rows: list[dict]) -> list[dict[str, Any]]:
    available_names = {
        name
        for row in distance_rows
        for name in (row.get("gt_name"), row.get("derived_name"))
        if name
    }
    records = []
    for row in distance_rows:
        if row.get("status") != "missing":
            continue
        child = row.get("gt_name")
        parent = _expression_child_parent_qualname(child)
        if not parent or parent not in available_names or parent == "<module>":
            continue
        parent_row = _find_distance_row(distance_rows, parent)
        records.append(
            {
                "child_qualname": child,
                "child_kind": _expression_child_kind(child),
                "parent_qualname": parent,
                "parent_score": _score_snapshot(parent_row),
                "child_score": _score_snapshot(row),
            }
        )
    return sorted(
        records,
        key=lambda item: (
            _qualname_depth(str(item.get("parent_qualname"))),
            str(item.get("parent_qualname")),
            str(item.get("child_qualname")),
        ),
    )


def _expression_child_kind(qualname: str | None) -> str | None:
    if not qualname:
        return None
    for part in EXPRESSION_CHILD_QUALNAME_PARTS:
        if part in qualname:
            return part.strip(".<>")
    return None


def select_extra_repair_targets(distance_rows: list[dict]) -> list[str]:
    candidates = [
        row["derived_name"]
        for row in distance_rows
        if row["status"] == "extra"
        and row["derived_name"]
        and row["derived_name"] != "<module>"
    ]
    return sorted(set(candidates), key=lambda name: (_qualname_depth(name), name), reverse=True)


def _distance_row_name(row: dict) -> str | None:
    return row.get("gt_name") or row.get("derived_name")


def _find_distance_row(distance_rows: list[dict], qualname: str) -> dict | None:
    for row in distance_rows:
        if row.get("gt_name") == qualname or row.get("derived_name") == qualname:
            return row
    return None


def _module_needs_repair(distance_rows: list[dict]) -> bool:
    module_row = _find_distance_row(distance_rows, "<module>")
    return bool(module_row and module_row.get("combined_distance", 0) > 0)


def _score_snapshot(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        "code_object": row["code_object"],
        "gt_name": row["gt_name"],
        "derived_name": row["derived_name"],
        "status": row["status"],
        "instruction_distance": row["instruction_distance"],
        "control_flow_distance": row["control_flow_distance"],
        "interaction_penalty": row["interaction_penalty"],
        "unmatched_penalty": row["unmatched_penalty"],
        "combined_distance": row["combined_distance"],
        "normalized_combined_distance": row["normalized_combined_distance"],
    }


def _resolved_score_snapshot(qualname: str) -> dict:
    return {
        "code_object": qualname,
        "gt_name": qualname,
        "derived_name": qualname,
        "status": "resolved",
        "instruction_distance": 0,
        "control_flow_distance": 0,
        "interaction_penalty": 0,
        "unmatched_penalty": 0,
        "combined_distance": 0,
        "normalized_combined_distance": 0,
    }


def _build_code_object_score_changes(initial_rows: list[dict], final_rows: list[dict]) -> list[dict]:
    names = sorted(
        {
            name
            for row in initial_rows + final_rows
            for name in (_distance_row_name(row),)
            if name
        },
        key=lambda name: (_qualname_depth(name), name),
    )
    changes = []
    for name in names:
        initial = _score_snapshot(_find_distance_row(initial_rows, name))
        final = _score_snapshot(_find_distance_row(final_rows, name))
        if initial is not None and final is None:
            final = _resolved_score_snapshot(name)
        initial_distance = initial["combined_distance"] if initial is not None else None
        final_distance = final["combined_distance"] if final is not None else None
        delta = (
            final_distance - initial_distance
            if initial_distance is not None and final_distance is not None
            else None
        )
        changes.append(
            {
                "qualname": name,
                "initial_score": initial,
                "final_score": final,
                "combined_distance_delta": delta,
            }
        )
    return changes


def index_code_objects_by_qualname(pyc_path: Path) -> dict[str, Any]:
    bytecode_root = load_editable_bytecode_from_pyc(validate_input(pyc_path))
    return {bc.name: bc.codeobj for bc in bytecode_root.iter_bytecodes()}


def index_bytecodes_by_qualname(pyc_path: Path) -> dict[str, Any]:
    bytecode_root = load_editable_bytecode_from_pyc(validate_input(pyc_path))
    return {bc.name: bc for bc in bytecode_root.iter_bytecodes()}


def _validate_reattached_code_object_structure(
    *,
    previous_pyc: Path,
    candidate_pyc: Path,
    qualname: str,
) -> tuple[bool, str]:
    previous_names = set(index_code_objects_by_qualname(previous_pyc))
    candidate_names = set(index_code_objects_by_qualname(candidate_pyc))
    if qualname not in candidate_names:
        return False, f"reattached target qualname disappeared: {qualname}"
    removed_siblings = sorted(previous_names - candidate_names - {qualname})
    if removed_siblings:
        preview = ", ".join(removed_siblings[:5])
        suffix = "" if len(removed_siblings) <= 5 else f", ... ({len(removed_siblings)} total)"
        return False, f"reattachment removed unrelated code objects: {preview}{suffix}"
    return True, "reattached code object structure preserved"


def _pylingual_result_name(result: Any) -> str:
    name_a = getattr(getattr(result, "bc_a", None), "name", None)
    name_b = getattr(getattr(result, "bc_b", None), "name", None)
    if name_a == name_b and name_a is not None:
        return str(name_a)
    return f"{name_a or 'None'}, {name_b or 'None'}"


def _pylingual_code_object_metadata(bytecode: Any) -> dict:
    code_object = getattr(bytecode, "codeobj", None)
    return {
        "name": getattr(bytecode, "name", None),
        "co_name": getattr(code_object, "co_name", None),
        "co_qualname": getattr(code_object, "co_qualname", None),
        "co_firstlineno": getattr(code_object, "co_firstlineno", None),
        "instruction_count": len(bytecode) if bytecode is not None else None,
    }


def _prepass_rejection_record(
    *,
    qualname: str,
    phase: str,
    iteration: int,
    operation: str | None,
    strategy: str | None,
    candidate: dict | None,
    reason: str,
    detail: str,
) -> dict:
    """Attribution record for a deterministic-prepass candidate that fired and was rejected.

    ``reason`` is one of ``compile_error`` / ``regressed`` / ``no_improvement``. Acceptance is
    NOT affected by this record: it exists so a null result can be attributed between "the
    operator never fired" and "the operator fired and the gate discarded it".
    """
    candidate = candidate or {}
    return {
        "qualname": qualname,
        "phase": phase,
        "iteration": iteration,
        "repair_operation": operation,
        "strategy": strategy,
        "operator": candidate.get("operator") or candidate.get("detail"),
        "confidence": candidate.get("confidence"),
        "reason": reason,
        "detail": detail,
    }


def run_pylingual_verification(gt_pyc: Path, candidate_pyc: Path) -> dict:
    from pylingual.equivalence_check import compare_pyc

    results = compare_pyc(validate_input(gt_pyc), validate_input(candidate_pyc))
    names = [_pylingual_result_name(result) for result in results]
    duplicate_counts = {name: names.count(name) for name in set(names)}
    occurrence_counts: dict[str, int] = {}
    serialized_results = []
    for result, name in zip(results, names):
        occurrence_index = occurrence_counts.get(name, 0)
        occurrence_counts[name] = occurrence_index + 1
        serialized_results.append(
            {
                "success": result.success,
                "message": result.message,
                "names": name,
                "pylingual_occurrence_index": occurrence_index,
                "duplicate_name_count": duplicate_counts[name],
                "result_key": f"{name}#{occurrence_index}",
                "gt_code_object": _pylingual_code_object_metadata(getattr(result, "bc_a", None)),
                "candidate_code_object": _pylingual_code_object_metadata(getattr(result, "bc_b", None)),
                "failed_line_number": result.failed_line_number,
                "failed_offset": result.failed_offset,
            }
        )
    return {
        "all_equal": all(result["success"] for result in serialized_results),
        "results": serialized_results,
    }


def _count_pylingual_successes(verification: dict | None) -> int | None:
    if verification is None:
        return None
    return sum(1 for result in verification["results"] if result["success"])


def _should_accept_candidate_oracle(
    previous_summary: dict,
    candidate_summary: dict,
    previous_verification: dict,
    candidate_verification: dict,
) -> tuple[bool, str]:
    """Oracle-primary acceptance (M2).

    Accept iff the candidate's per-object oracle state multiset does not regress
    AND either some object strictly improves in the lattice, or the multiset is
    unchanged and the (gradient-only) combined distance strictly drops.
    """
    delta = compare_verifications(
        previous_verification.get("results", []),
        candidate_verification.get("results", []),
    )
    if delta.regressed:
        return False, f"oracle: object regressed ({', '.join(delta.regressed)})"
    if delta.improved:
        return True, f"oracle: object improved ({', '.join(delta.improved)})"
    previous_distance = int(previous_summary["combined_distance"])
    candidate_distance = int(candidate_summary["combined_distance"])
    if candidate_distance < previous_distance:
        return True, "oracle: state held; combined distance improved (gradient)"
    return False, "oracle: no object improved and combined distance did not drop"


def _should_accept_candidate(
    previous_summary: dict,
    candidate_summary: dict,
    previous_verification: dict | None,
    candidate_verification: dict | None,
) -> tuple[bool, str]:
    if (
        ACCEPTANCE_MODE == "oracle"
        and previous_verification is not None
        and candidate_verification is not None
    ):
        return _should_accept_candidate_oracle(
            previous_summary, candidate_summary, previous_verification, candidate_verification
        )
    previous_distance = int(previous_summary["combined_distance"])
    candidate_distance = int(candidate_summary["combined_distance"])
    if candidate_distance < previous_distance:
        previous_successes = _count_pylingual_successes(previous_verification)
        candidate_successes = _count_pylingual_successes(candidate_verification)
        if previous_successes is not None and candidate_successes is not None and candidate_successes < previous_successes:
            return False, "combined distance improved but PyLingual success count regressed"
        return True, "combined distance improved"

    if candidate_verification is not None and previous_verification is not None:
        previous_successes = _count_pylingual_successes(previous_verification)
        candidate_successes = _count_pylingual_successes(candidate_verification)
        if candidate_successes is not None and previous_successes is not None and candidate_successes > previous_successes:
            return True, "PyLingual success count improved"

    return False, "candidate did not improve combined distance or PyLingual success count"


def _summary_combined_distance(summary: dict | None) -> int | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get("combined_distance")
    if not isinstance(value, (int, float)):
        return None
    return int(value)


def _combined_distance_improved(
    initial_summary: dict | None,
    current_summary: dict | None,
    *,
    min_delta: int = 1,
) -> bool:
    initial_distance = _summary_combined_distance(initial_summary)
    current_distance = _summary_combined_distance(current_summary)
    return (
        initial_distance is not None
        and current_distance is not None
        and current_distance <= initial_distance - max(1, int(min_delta))
    )


def _module_failed_line(verification: dict | None, derived_code_object: Any | None = None) -> int | None:
    if verification is None:
        return None
    for result in verification.get("results", []):
        if result.get("names") == "<module>" and not result.get("success"):
            line_number = result.get("failed_line_number")
            if line_number is not None:
                return int(line_number)
            if derived_code_object is not None:
                failed_offset = result.get("failed_offset")
                inferred_line = _infer_line_number_from_instruction_records(
                    _instruction_records(derived_code_object),
                    None if failed_offset is None else int(failed_offset),
                )
                if inferred_line is not None:
                    return inferred_line
    return None


def _infer_line_number_from_instruction_records(records: list[dict], failed_offset: int | None) -> int | None:
    if not records:
        return None

    if failed_offset is None:
        focus_index = 0
    else:
        exact = [record["index"] for record in records if record["offset"] == failed_offset]
        if exact:
            focus_index = exact[0]
        else:
            focus_index = min(
                range(len(records)),
                key=lambda idx: abs(int(records[idx]["offset"]) - int(failed_offset)),
            )

    for idx in range(focus_index, -1, -1):
        starts_line = records[idx].get("starts_line")
        if starts_line is not None:
            return int(starts_line)

    for idx in range(focus_index + 1, len(records)):
        starts_line = records[idx].get("starts_line")
        if starts_line is not None:
            return int(starts_line)

    return None


def _pylingual_target_is_equal(verification: dict | None, qualname: str) -> bool | None:
    if verification is None:
        return None
    matched = [
        result
        for result in verification.get("results", [])
        if result.get("names") == qualname
    ]
    if not matched:
        return None
    return all(result.get("success") for result in matched)


def _has_retryable_pylingual_targets(verification: dict | None, targets: list[str]) -> bool:
    if verification is None:
        return False
    for qualname in targets:
        equal = _pylingual_target_is_equal(verification, qualname)
        if equal is False:
            return True
    return False


def _oracle_failing_qualnames(verification: dict | None) -> list[str]:
    """Distinct qualnames the oracle reports as not-equal (success is False)."""
    if verification is None:
        return []
    failing: list[str] = []
    seen: set[str] = set()
    for result in verification.get("results", []):
        if result.get("success"):
            continue
        name = result.get("names")
        if name and name not in seen:
            seen.add(str(name))
            failing.append(str(name))
    return failing


def _oracle_primary_fragment_targets(
    verification: dict | None,
    distance_rows: list[dict],
    attempt_counts: dict[str, int],
    max_iterations: int,
    *,
    include_module: bool,
) -> list[str]:
    """Oracle-failing objects that the distance-based selector misses (M2).

    These are ``matched`` code objects the oracle rejects even though their
    ``combined_distance`` is 0 (e.g. an exception-table-only divergence). Without
    this, such objects are never selected and the loop declares a false victory.
    ``missing``/``extra`` oracle failures are intentionally left to their own
    selectors.
    """
    if verification is None:
        return []
    matched_names = {
        row["gt_name"]
        for row in distance_rows
        if row.get("status") == "matched" and row.get("gt_name")
    }
    targets: list[str] = []
    for qualname in _oracle_failing_qualnames(verification):
        if qualname not in matched_names:
            continue
        if not include_module and qualname == "<module>":
            continue
        if attempt_counts.get(qualname, 0) >= max_iterations:
            continue
        if _is_synthetic_annotation_qualname(qualname):
            continue
        targets.append(qualname)
    return targets


def _truncated_repr(value: Any, limit: int = 160) -> str:
    if _is_code_object(value):
        return f"<code object {getattr(value, 'co_name', None)} at line {getattr(value, 'co_firstlineno', None)}>"
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _instruction_records(code_object: Any) -> list[dict]:
    instructions = _editable_instruction_items(code_object)
    if not instructions:
        decoded_records = _decode_code310_instruction_records(code_object)
        for record in decoded_records:
            record["argval"] = _truncated_repr(record.get("argval"))
        return decoded_records
    records = []
    for index, inst in enumerate(instructions):
        records.append(
            {
                "index": index,
                "offset": int(getattr(inst, "offset", -1)),
                "starts_line": getattr(inst, "starts_line", None),
                "opname": getattr(inst, "opname", ""),
                "argrepr": getattr(inst, "argrepr", ""),
                "argval": _truncated_repr(getattr(inst, "argval", None)),
            }
        )
    return records


def _instruction_alignment_signature(record: dict) -> tuple[str, str]:
    return (str(record["opname"]), str(record["argval"]))


def _instruction_window(records: list[dict], start: int, end: int, radius: int) -> list[dict]:
    if not records:
        return []
    lo = max(0, start - radius)
    hi = min(len(records), max(end, start + 1) + radius)
    return records[lo:hi]


def _bounded_instruction_window(
    records: list[dict],
    start: int,
    end: int,
    radius: int,
    *,
    max_records: int = 40,
) -> list[dict]:
    window = _instruction_window(records, start, end, radius)
    if len(window) <= max_records:
        return window
    focus_start = max(0, start - max(0, start - window[0]["index"]))
    prefix = max_records // 2
    suffix = max_records - prefix
    return window[:prefix] + window[-suffix:]


def _format_instruction_window(records: list[dict], focus_offsets: set[int]) -> str:
    lines = []
    for record in records:
        marker = "=>" if record["offset"] in focus_offsets else "  "
        line = f"{marker} idx={record['index']} offset={record['offset']} line={record['starts_line']} {record['opname']}"
        if record["argrepr"]:
            line += f" {record['argrepr']}"
        elif record["argval"] != "None":
            line += f" {record['argval']}"
        lines.append(line)
    return "\n".join(lines) if lines else "<instruction window unavailable>"


def _candidate_residual_snapshot(
    gt_bytecode: Any,
    candidate_bytecode: Any,
    *,
    message: str | None,
    failed_offset: int | None,
    radius: int = 4,
) -> dict | None:
    """Compact GT-vs-candidate instruction residual for corrective retry feedback (#54).

    Given the ground-truth and *rejected-candidate* bytecode for one target object, produce a
    short window showing what the candidate's own edit still gets wrong versus GT — so the next
    prompt can say "your last edit still differs here" instead of only a distance number.
    Windowed around ``failed_offset`` when present, else the largest diverging opcode region
    (the dominant "Different bytecode, no single offset" case). Returns a compact
    ``{"message", "failed_offset", "window"}`` dict, or None when bytecode is unavailable.
    Never raises — this only shapes a prompt, so any failure must defer silently."""
    try:
        gt_records = _instruction_records(gt_bytecode)
        cand_records = _instruction_records(candidate_bytecode)
        if not gt_records or not cand_records:
            return None
        if failed_offset is not None:
            focus = {int(failed_offset)}
            cand_idx = next((r["index"] for r in cand_records if r["offset"] == failed_offset), None)
            gt_idx = next((r["index"] for r in gt_records if r["offset"] == failed_offset), cand_idx)
            if cand_idx is None:
                cand_idx = 0
            if gt_idx is None:
                gt_idx = cand_idx
        else:
            focus = set()
            _tag, a1, a2, b1, b2 = _largest_non_equal_opcode(gt_records, cand_records)
            gt_idx, cand_idx = a1, b1
            gt_end, cand_end = max(a2, a1 + 1), max(b2, b1 + 1)
        if failed_offset is not None:
            gt_win = _bounded_instruction_window(gt_records, gt_idx, gt_idx + 1, radius)
            cand_win = _bounded_instruction_window(cand_records, cand_idx, cand_idx + 1, radius)
        else:
            gt_win = _bounded_instruction_window(gt_records, gt_idx, gt_end, radius)
            cand_win = _bounded_instruction_window(cand_records, cand_idx, cand_end, radius)
        window = (
            "ground truth:\n" + _format_instruction_window(gt_win, focus)
            + "\nyour last edit produced:\n" + _format_instruction_window(cand_win, focus)
        )
        return {"message": message, "failed_offset": failed_offset, "window": window}
    except Exception:  # noqa: BLE001 - prompt-shaping only; never crash the repair loop
        return None


def _capture_candidate_residual(
    best_candidate: dict, qualname: str, gt_bytecodes: dict
) -> dict | None:
    """Build the corrective residual (#54) for a rejected fragment candidate: index the
    candidate's compiled bytecode for ``qualname``, pull the failure message/offset from its
    already-computed pylingual verification, and snapshot the GT-vs-candidate divergence.
    Returns None (defer) if the candidate did not compile, the object already matches GT, or
    anything is unavailable. Never raises."""
    try:
        output_pyc = best_candidate.get("output_pyc")
        if not output_pyc:
            return None
        cand_bc = index_bytecodes_by_qualname(validate_input(Path(output_pyc))).get(qualname)
        gt_bc = gt_bytecodes.get(qualname)
        if cand_bc is None or gt_bc is None:
            return None
        message = failed_offset = None
        results = (best_candidate.get("pylingual_verification") or {}).get("results")
        if results:
            target = next((r for r in results if r.get("names") == qualname), None)
            if target is not None:
                if target.get("success"):
                    return None  # target already matches GT; the rejection was for another object
                message = target.get("message")
                failed_offset = target.get("failed_offset")
        return _candidate_residual_snapshot(
            gt_bc, cand_bc, message=message, failed_offset=failed_offset
        )
    except Exception:  # noqa: BLE001 - prompt-shaping only; never crash the repair loop
        return None


def _largest_non_equal_opcode(
    gt_records: list[dict],
    derived_records: list[dict],
) -> tuple[str, int, int, int, int]:
    sm = __import__("difflib").SequenceMatcher(
        a=[_instruction_alignment_signature(record) for record in gt_records],
        b=[_instruction_alignment_signature(record) for record in derived_records],
    )
    opcodes = [opcode for opcode in sm.get_opcodes() if opcode[0] != "equal"]
    if not opcodes:
        return ("equal", 0, min(1, len(gt_records)), 0, min(1, len(derived_records)))
    return max(
        opcodes,
        key=lambda opcode: (
            max(opcode[2] - opcode[1], opcode[4] - opcode[3]),
            (opcode[2] - opcode[1]) + (opcode[4] - opcode[3]),
        ),
    )


def _is_control_flow_opname(opname: str) -> bool:
    """Opcodes that carry a jump-target offset in their arg (branch/loop test)."""
    return (opname.startswith("POP_JUMP") or opname.startswith("JUMP")
            or opname in ("FOR_ITER", "SEND"))


def _instruction_run_repr(records: list[dict], lo: int, hi: int, cap: int = 8) -> str:
    parts = []
    for r in records[lo:hi][:cap]:
        arg = (r.get("argrepr") or "")[:12]
        parts.append(f"{r['opname']}{('(' + arg + ')') if arg else ''}")
    if hi - lo > cap:
        parts.append("…")
    return " ".join(parts)


def _missing_extra_statement_summary(gt_code_object: Any, derived_code_object: Any, max_runs: int = 6) -> str:
    """Statement-level diff for the "Different bytecode, no single offset" bucket (multi-edit:
    a dropped/added/changed STATEMENT, e.g. a local assignment the decompiler omitted and then
    reads as a global). The localized instruction window and the control-flow skeleton do NOT
    cover this — this frames each contiguous divergent run as a missing/spurious statement with
    GT source-line anchors, which is the actionable signal for reconstructing it. GT bytecode
    only (allowed), never GT source."""
    gt = _instruction_records(gt_code_object)
    der = _instruction_records(derived_code_object)
    if not gt or not der:
        return ""
    go = [r["opname"] for r in gt]
    do = [r["opname"] for r in der]
    sm = __import__("difflib").SequenceMatcher(a=go, b=do, autojunk=False)
    lines: list[str] = []
    # No source-line anchor: the fixer only sees the (renumbered) DERIVED fragment, never GT
    # source, so a GT absolute line number would point at a file it can't see and could
    # misdirect it. Distinguish delete/insert/replace so a one-opcode change is not framed as
    # adding/removing a whole statement.
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":  # GT produces instructions the derived fragment lacks
            lines.append(f"- MISSING — add source that produces: {_instruction_run_repr(gt, a1, a2)}")
        elif tag == "insert":  # derived produces instructions GT does not
            lines.append(f"- SPURIOUS — remove the source that produces: {_instruction_run_repr(der, b1, b2)}")
        else:  # replace — the region differs on both sides (may be one opcode or several)
            lines.append(f"- CHANGED — ground truth produces [{_instruction_run_repr(gt, a1, a2)}] "
                         f"where your fragment produces [{_instruction_run_repr(der, b1, b2)}]")
        if len(lines) >= max_runs:
            lines.append("- … (further differences)")
            break
    if not lines:
        return ""
    header = ("Statement-level bytecode difference (make your fragment compile to the ground-truth "
              "instruction stream — add what is MISSING, remove what is SPURIOUS, adjust what is CHANGED):")
    return header + "\n" + "\n".join(lines)


def _control_flow_skeleton_rows(code_object: Any, max_rows: int = 80) -> list[str]:
    """Render the CONTROL-FLOW rows (branches, jump targets, loop back-edges, exception
    ops) of a code object with source-line anchors, from bytecode. Shared by the GT and
    derived skeletons. Returns [] for straight-line code (no control flow to convey)."""
    records = _instruction_records(code_object)
    if not records:
        return []
    targets: set[int] = set()
    for r in records:
        if _is_control_flow_opname(r["opname"]):
            try:
                targets.add(int(str(r.get("argval", "")).strip()))
            except (TypeError, ValueError):
                pass
    has_exc = any(r["opname"].startswith(("RAISE", "RERAISE", "PUSH_EXC", "CHECK_EXC", "POP_EXCEPT", "WITH_EXCEPT"))
                  for r in records)
    if not targets and not has_exc:
        return []
    rows: list[str] = []
    for r in records:
        opn, off = r["opname"], r["offset"]
        is_tgt = off in targets
        interesting = (_is_control_flow_opname(opn) or is_tgt
                       or opn.startswith(("RETURN", "RAISE", "RERAISE", "PUSH_EXC", "CHECK_EXC",
                                          "POP_EXCEPT", "END_FOR", "WITH_EXCEPT")))
        if not interesting:
            continue
        line = r.get("starts_line")
        marker = ">>" if is_tgt else "  "
        tgt = ""
        arg = (r.get("argrepr") or "")[:16]
        if _is_control_flow_opname(opn):
            try:
                tgt = f" -> @{int(str(r.get('argval', '')).strip())}"
                arg = ""  # target rendered explicitly below; drop the redundant "to N" argrepr
            except (TypeError, ValueError):
                tgt = ""
        line_tag = f"L{line}" if line else ""
        rows.append(f"  {marker} @{off:<4} {line_tag:<6} {opn}{(' ' + arg) if arg else ''}{tgt}")
        if len(rows) >= max_rows:
            rows.append("  ... (truncated)")
            break
    return rows


def _derived_control_flow_skeleton(derived_code_object: Any, max_rows: int = 80) -> str:
    """The CURRENT (derived) control-flow skeleton, so the LLM can diff it against GT and
    see exactly which branches/loops/handlers to add, remove, or reshape (report lever #2:
    structured oracle-diff feedback for the control-flow bucket)."""
    rows = _control_flow_skeleton_rows(derived_code_object, max_rows)
    if not rows:
        return ""
    header = ("Current (derived) control-flow skeleton — reshape the fragment's control flow so this "
              "matches the ground-truth skeleton above (same offsets are illustrative, match the STRUCTURE):")
    return header + "\n" + "\n".join(rows)


def _gt_control_flow_skeleton(gt_code_object: Any, max_rows: int = 80) -> str:
    """Serialize the ground-truth code object's CONTROL-FLOW structure (branches, jump
    targets, loop headers, exception ops) with source-line anchors, from GT bytecode.

    Decompilers get operands right but reconstruct loop/branch/try structure poorly, and
    a localized instruction window can't convey whole-object control flow. Handing the LLM
    the GT jump graph (offsets + targets + lines) lets it rewrite the fragment so its
    control flow MATCHES ground truth (the capability lever for "Different control flow").
    Uses only GT bytecode (allowed signal), never GT source. Returns "" for straight-line
    code (no control flow to convey)."""
    rows = _control_flow_skeleton_rows(gt_code_object, max_rows)
    if not rows:
        return ""
    header = ("Ground-truth control-flow skeleton (rewrite the fragment so its control flow matches "
              "this; @N = GT bytecode offset, '>>' = a branch/jump target, '-> @N' = jump destination, "
              "LN = source line):")
    return header + "\n" + "\n".join(rows)


def _brief_jump_target(record: dict) -> int | None:
    """The (GT-internal) target offset carried by a jump/FOR_ITER record, or None."""
    try:
        return int(str(record.get("argval", "")).strip())
    except (TypeError, ValueError):
        return None


def _branch_sense(opname: str) -> str:
    """Human-readable polarity of a conditional jump, derived from the opname family so it
    survives 3.10-3.14 drift (POP_JUMP_IF_FALSE vs POP_JUMP_FORWARD_IF_FALSE, *_IF_NONE)."""
    if "_IF_NOT_NONE" in opname:
        return "if the test is NOT None"
    if "_IF_NONE" in opname:
        return "if the test is None"
    if "_IF_FALSE" in opname:
        return "if the test is FALSE"
    if "_IF_TRUE" in opname:
        return "if the test is TRUE"
    return "if the branch condition holds"


def _gt_control_flow_brief(code_object: Any, max_items: int = 24) -> str:
    """A readable pseudo-CFG reconstructed from GT bytecode ONLY (never GT source).

    For each conditional jump it states the SENSE ("if the test is FALSE, jump to @N; else
    fall through to @M") derived from the opname polarity; loop back-edges (JUMP_BACKWARD or a
    target < the jump's own offset) are labeled as loops with a body extent and exit; FOR_ITER
    is labeled as a for-loop with its exhaustion exit; try regions (via structural._real_try_blocks)
    report DEPTH + the protected dotted callee + the handler shape.

    Offsets are expressed only as GT-INTERNAL anchors (@N) plus source lines (LN) and relative
    direction — never as GT-vs-derived comparable addresses. Direction/polarity use opcode-prefix
    families + offset comparison so they work across 3.10-3.14 opname drift. Returns "" for
    straight-line code with no try region (parity with _gt_control_flow_skeleton). Pure; no I/O."""
    if code_object is None:
        return ""
    records = _instruction_records(code_object)
    if not records:
        return ""
    lines: list[str] = []
    n = len(records)
    # FOR_ITER offsets are for-loop headers; a back-edge that targets one is the for-loop's own
    # loop-back and must NOT also be emitted as a separate while-style loop (avoids double-labeling
    # a single for-loop, which could nudge the model into an extra while alongside the for).
    for_headers = {r["offset"] for r in records if r["opname"].startswith("FOR_ITER")}

    def _line_tag(record: dict) -> str:
        line = record.get("starts_line")
        return f" L{line}" if line else ""

    for idx, record in enumerate(records):
        if len(lines) >= max_items:
            break
        opname = record["opname"]
        off = record["offset"]
        ltag = _line_tag(record)
        fall = records[idx + 1]["offset"] if idx + 1 < n else None
        if opname.startswith("FOR_ITER"):
            exit_off = _brief_jump_target(record)
            if exit_off is not None:
                lines.append(f"- for-loop @{off}{ltag}: FOR_ITER iterates, exits to @{exit_off} when exhausted")
            continue
        is_conditional = opname.startswith("POP_JUMP") or opname.startswith("JUMP_IF")
        if is_conditional:
            tgt = _brief_jump_target(record)
            if tgt is None:
                continue
            if tgt >= off:
                fall_txt = f"; else fall through to @{fall}" if fall is not None else ""
                lines.append(f"- branch @{off}{ltag}: {_branch_sense(opname)}, jump to @{tgt}{fall_txt}")
            elif tgt not in for_headers:
                exit_txt = f"; exits by falling through to @{fall}" if fall is not None else ""
                lines.append(f"- loop @{tgt}..@{off}{ltag}: conditional back-edge {opname} @{off} -> @{tgt}{exit_txt}")
            # else: conditional back-edge into a FOR_ITER header — already a labeled for-loop; skip.
            continue
        if opname.startswith("JUMP"):
            tgt = _brief_jump_target(record)
            if opname.startswith("JUMP_BACKWARD") or (tgt is not None and tgt < off):
                if tgt is not None and tgt not in for_headers:
                    exit_txt = f"; exits to @{fall}" if fall is not None else ""
                    lines.append(f"- loop @{tgt}..@{off}{ltag}: back-edge {opname} @{off} -> @{tgt} (while-style){exit_txt}")
            continue

    from utils.semantic_operators.structural import (
        _handler_skeleton,
        _instructions,
        _protected_call_parts,
        _real_try_blocks,
    )

    insts = _instructions(code_object)
    for (start, end, target, depth) in _real_try_blocks(code_object):
        if len(lines) >= max_items:
            break
        parts = _protected_call_parts(insts, start, end)
        trivial, exc_type = _handler_skeleton(insts, target)
        try_line = None
        for record in records:
            if start is not None and record["offset"] == start:
                try_line = record.get("starts_line")
                break
        ltag = f" L{try_line}" if try_line else ""
        seg = f"- try @{start}..@{end} (depth {depth}){ltag}"
        if parts:
            seg += f": protects call `{'.'.join(parts)}(...)`"
        if exc_type:
            handler = f"except {exc_type}:" + (" pass" if trivial else "")
            seg += f"; handler `{handler}`"
        else:
            # The exception type is not always recoverable from bytecode (e.g. 3.12 terminal
            # handlers end in RETURN_CONST). Do NOT emit a bare `except:` — that would falsely
            # imply a catch-all handler when the source may be typed. Hedge instead.
            tail = " (body is trivial/pass)" if trivial else ""
            seg += f"; handler present (exception type not recoverable from bytecode){tail}"
        lines.append(seg)

    if len(lines) >= max_items:
        lines.append("- ... (further structure omitted)")
    if not lines:
        return ""
    return "\n".join(lines)


def _gt_signature_and_placement_brief(code_object: Any, qualname: str) -> str:
    """Reconstruct a def/class signature and a placement hint from the GT code object's own
    flags/varnames (GT bytecode only). Defaults, annotations, and base classes live in the
    PARENT's MAKE_FUNCTION/CALL bytecode, not in this body, so they are explicitly flagged as
    unknown. Pure; no I/O."""
    co = getattr(code_object, "codeobj", code_object)
    if co is None:
        return ""
    short_name = qualname.rsplit(".", 1)[-1]
    name = getattr(co, "co_name", None) or short_name
    flags = getattr(co, "co_flags", 0)
    parent = qualname.rsplit(".", 1)[0] if "." in qualname else "<module>"

    lines: list[str] = []
    if not (flags & inspect.CO_OPTIMIZED):
        # A class body (or module) code object is not CO_OPTIMIZED.
        lines.append(f"class {name}:  # bases unknown from bytecode")
    else:
        argcount = getattr(co, "co_argcount", 0)
        posonly = getattr(co, "co_posonlyargcount", 0)
        kwonly = getattr(co, "co_kwonlyargcount", 0)
        varnames = list(getattr(co, "co_varnames", ()))
        positional = varnames[:argcount]
        kwonly_names = varnames[argcount:argcount + kwonly]
        cursor = argcount + kwonly
        vararg = None
        kwarg = None
        if flags & inspect.CO_VARARGS and cursor < len(varnames):
            vararg = varnames[cursor]
            cursor += 1
        if flags & inspect.CO_VARKEYWORDS and cursor < len(varnames):
            kwarg = varnames[cursor]
            cursor += 1
        params: list[str] = []
        for i, pname in enumerate(positional):
            params.append(pname)
            if posonly and i == posonly - 1:
                params.append("/")
        if vararg is not None:
            params.append("*" + vararg)
        elif kwonly_names:
            params.append("*")
        params.extend(kwonly_names)
        if kwarg is not None:
            params.append("**" + kwarg)
        prefix = "async def" if (flags & (inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR)) else "def"
        notes = []
        if flags & (inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR):
            notes.append("generator")
        notes.append("defaults/annotations unknown from bytecode")
        lines.append(f"{prefix} {name}({', '.join(params)}):  # " + "; ".join(notes))

    if parent == "<module>":
        lines.append("Placement: insert at module level.")
    else:
        parent_name = parent.rsplit(".", 1)[-1]
        lines.append(
            f"Placement: insert as a member of `{parent_name}` (parent scope {parent}); "
            "keep the leading `self` parameter for a method."
        )
    return "\n".join(lines)


def _gt_operand_inventory(code_object: Any, max_each: int = 8) -> str:
    """Whole-object inventory of the building blocks the LLM needs to rebuild actual
    statements (not just shape): dotted call targets (each CALL's LOAD base + LOAD_ATTR/
    LOAD_METHOD chain), the co_names pool, literal co_consts, and nested child qualnames.
    GT bytecode only. Pure; no I/O."""
    if code_object is None:
        return ""
    from utils.semantic_operators.structural import _clean_name

    records = _instruction_records(code_object)
    attr_ops = ("LOAD_ATTR", "LOAD_METHOD")
    # Callee chains may start from a global/name OR a local/closure base (e.g. `self.log(...)`,
    # a dominant case for the missing-method bucket). Bare local args (`LOAD_FAST x` passed to
    # another call) are excluded below by requiring either a global base or a dotted `.attr`
    # chain, so locals only surface as real method-call targets, never as argument noise.
    base_ops = ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_FAST", "LOAD_DEREF")
    global_base_ops = ("LOAD_GLOBAL", "LOAD_NAME")
    calls: list[str] = []
    seen: set[str] = set()
    n = len(records)
    for i, record in enumerate(records):
        if record["opname"] not in base_ops:
            continue
        is_global_base = record["opname"] in global_base_ops
        base = _clean_name(record.get("argrepr", ""))
        if not base:
            continue
        parts = [base]
        j = i + 1
        while j < n and records[j]["opname"] in attr_ops:
            attr = _clean_name(records[j].get("argrepr", ""))
            if not attr:
                break
            parts.append(attr)
            j += 1
        # Treat this dotted chain as a call target only if a CALL consumes it before the
        # next base-name load begins another potential callee.
        has_call = False
        k = j
        while k < n:
            opk = records[k]["opname"]
            if opk in ("LOAD_GLOBAL", "LOAD_NAME"):
                break
            if opk.startswith("CALL"):
                has_call = True
                break
            k += 1
        # A global/name base is a callee on its own; a local/closure base only counts when it
        # heads a dotted `.attr` chain (a method call like `self.log`), never as a bare arg.
        if has_call and (is_global_base or len(parts) > 1):
            dotted = ".".join(parts)
            if dotted not in seen:
                seen.add(dotted)
                calls.append(dotted)

    names = [str(x) for x in (getattr(code_object, "co_names", ()) or ())]
    consts: list[str] = []
    for value in (getattr(code_object, "co_consts", ()) or ()):
        if _is_code_object(value):
            continue
        if _is_safe_literal(value):
            consts.append(_truncated_repr(value, 40))
    nested: list[str] = []
    for child in (getattr(code_object, "child_bytecodes", ()) or ()):
        child_name = getattr(child, "name", None)
        if child_name:
            nested.append(str(child_name))

    out: list[str] = []
    if calls:
        out.append("calls: " + ", ".join(calls[:max_each]))
    if names:
        out.append("names: " + ", ".join(names[:max_each]))
    if consts:
        out.append("consts: " + ", ".join(consts[:max_each]))
    if nested:
        out.append("nested: " + ", ".join(nested[:max_each]))
    return "\n".join(out)


def _gt_reconstruction_brief(code_object: Any, qualname: str, *, is_missing: bool) -> str:
    """Composer for the GT-bytecode reconstruction brief. Returns "" for a None object.

    For a MISSING target (derived object is None) it bundles the reconstructed signature +
    placement, the control-flow pseudo-CFG (body shape), and the operand inventory, so the
    LLM can synthesize the whole missing object. For a present control-flow target it emits
    just the control-flow pseudo-CFG under a reshaping header. Pure; no I/O."""
    if code_object is None:
        return ""
    control_flow = _gt_control_flow_brief(code_object)
    if is_missing:
        parts = []
        signature = _gt_signature_and_placement_brief(code_object, qualname)
        if signature:
            parts.append(signature)
        if control_flow:
            parts.append(control_flow)
        inventory = _gt_operand_inventory(code_object)
        if inventory:
            parts.append(inventory)
        if not parts:
            return ""
        header = "Reconstruct the missing ground-truth object from its bytecode:"
        return header + "\n" + "\n".join(parts)
    if not control_flow:
        return ""
    header = ("Ground-truth control-flow shape (reconstruct this branch/loop/try structure; "
              "@N = GT offset, LN = source line):")
    return header + "\n" + control_flow


def _localized_instruction_context(
    gt_code_object: Any,
    derived_code_object: Any,
    failed_offset: int | None,
    *,
    radius: int = 6,
) -> dict:
    gt_records = _instruction_records(gt_code_object)
    derived_records = _instruction_records(derived_code_object)
    if not gt_records or not derived_records:
        return {
            "failed_offset": failed_offset,
            "alignment_tag": "unavailable",
            "gt_instruction_window": "<instruction window unavailable>",
            "derived_instruction_window": "<instruction window unavailable>",
        }

    sm = __import__("difflib").SequenceMatcher(
        a=[_instruction_alignment_signature(record) for record in gt_records],
        b=[_instruction_alignment_signature(record) for record in derived_records],
    )
    if failed_offset is None:
        alignment_tag, gt_start, gt_end, derived_start, derived_end = _largest_non_equal_opcode(gt_records, derived_records)
        derived_index = derived_start
    else:
        exact = [record["index"] for record in derived_records if record["offset"] == failed_offset]
        if exact:
            derived_index = exact[0]
        else:
            derived_index = min(
                range(len(derived_records)),
                key=lambda idx: abs(int(derived_records[idx]["offset"]) - int(failed_offset)),
            )
        gt_start = max(0, min(derived_index, len(gt_records) - 1))
        gt_end = gt_start + 1
        derived_start = derived_index
        derived_end = derived_index + 1
        alignment_tag = "nearest_index"
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if b1 <= derived_index < max(b2, b1 + 1):
                alignment_tag = tag
                derived_start, derived_end = b1, b2
                if tag == "equal" and b2 > b1:
                    gt_start = a1 + (derived_index - b1)
                    gt_end = gt_start + 1
                else:
                    gt_start, gt_end = a1, a2
                break

    derived_window = _bounded_instruction_window(derived_records, derived_start, derived_end, radius)
    gt_window = _bounded_instruction_window(gt_records, gt_start, gt_end, radius)
    focus_offsets = {int(failed_offset)} if failed_offset is not None else set()
    return {
        "failed_offset": failed_offset,
        "alignment_tag": alignment_tag,
        "derived_focus_index": derived_index,
        "gt_focus_range": [gt_start, gt_end],
        "derived_focus_range": [derived_start, derived_end],
        "gt_instruction_window": _format_instruction_window(gt_window, set()),
        "derived_instruction_window": _format_instruction_window(derived_window, focus_offsets),
    }


def _module_failed_result(verification: dict | None) -> dict | None:
    if verification is None:
        return None
    for result in verification.get("results", []):
        if result.get("names") == "<module>" and not result.get("success"):
            return result
    return None


def _failed_result_for_target(verification: dict | None, qualname: str) -> dict | None:
    if verification is None:
        return None
    for result in verification.get("results", []):
        if result.get("names") == qualname and not result.get("success"):
            return result
    return None


def _format_instruction_record(record: dict) -> str:
    line = f"idx={record['index']} offset={record['offset']} line={record['starts_line']} {record['opname']}"
    if record["argrepr"]:
        line += f" {record['argrepr']}"
    elif record["argval"] != "None":
        line += f" {record['argval']}"
    return line


def _instruction_diff_context(
    gt_code_object: Any,
    derived_code_object: Any,
    failed_offset: int | None,
    *,
    radius: int = 3,
) -> dict:
    gt_records = _instruction_records(gt_code_object)
    derived_records = _instruction_records(derived_code_object)
    if not gt_records or not derived_records:
        return {
            "failed_offset": failed_offset,
            "alignment_tag": "unavailable",
            "instruction_diff": "<instruction diff unavailable>",
        }

    sm = __import__("difflib").SequenceMatcher(
        a=[_instruction_alignment_signature(record) for record in gt_records],
        b=[_instruction_alignment_signature(record) for record in derived_records],
    )
    if failed_offset is None:
        alignment_tag, gt_start, gt_end, derived_start, derived_end = _largest_non_equal_opcode(gt_records, derived_records)
        derived_index = derived_start
    else:
        exact = [record["index"] for record in derived_records if record["offset"] == failed_offset]
        if exact:
            derived_index = exact[0]
        else:
            derived_index = min(
                range(len(derived_records)),
                key=lambda idx: abs(int(derived_records[idx]["offset"]) - int(failed_offset)),
            )
        gt_start = max(0, min(derived_index, len(gt_records) - 1))
        gt_end = gt_start + 1
        derived_start = derived_index
        derived_end = derived_index + 1
        alignment_tag = "nearest_index"
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if b1 <= derived_index < max(b2, b1 + 1):
                alignment_tag = tag
                derived_start, derived_end = b1, b2
                if tag == "equal" and b2 > b1:
                    gt_start = a1 + (derived_index - b1)
                    gt_end = gt_start + 1
                else:
                    gt_start, gt_end = a1, a2
                break

    gt_window = _bounded_instruction_window(gt_records, gt_start, gt_end, radius)
    derived_window = _bounded_instruction_window(derived_records, derived_start, derived_end, radius)
    focus_offsets = {int(failed_offset)} if failed_offset is not None else set()
    diff_lines: list[str] = []
    max_len = max(len(gt_window), len(derived_window))
    for idx in range(max_len):
        gt_record = gt_window[idx] if idx < len(gt_window) else None
        derived_record = derived_window[idx] if idx < len(derived_window) else None
        marker = "=>" if (
            (gt_record is not None and gt_record["offset"] in focus_offsets)
            or (derived_record is not None and derived_record["offset"] in focus_offsets)
        ) else "  "
        if gt_record is None:
            diff_lines.append(f"{marker} + derived: {_format_instruction_record(derived_record)}")
            continue
        if derived_record is None:
            diff_lines.append(f"{marker} - gt: {_format_instruction_record(gt_record)}")
            continue
        if (
            gt_record["opname"] == derived_record["opname"]
            and gt_record["argval"] == derived_record["argval"]
        ):
            diff_lines.append(f"{marker} = {_format_instruction_record(gt_record)}")
        else:
            diff_lines.append(f"{marker} gt: {_format_instruction_record(gt_record)}")
            diff_lines.append(f"{marker} derived: {_format_instruction_record(derived_record)}")

    summary = (
        f"summary: alignment={alignment_tag}; "
        f"gt_range=[{gt_start}:{gt_end}]; derived_range=[{derived_start}:{derived_end}]; "
        f"failed_offset={failed_offset}"
    )
    bytecode_window_truncated = len(diff_lines) > 40
    if bytecode_window_truncated:
        diff_lines = diff_lines[:20] + ["  ... <instruction diff truncated> ..."] + diff_lines[-20:]

    return {
        "failed_offset": failed_offset,
        "alignment_tag": alignment_tag,
        "gt_instruction_range": [gt_start, gt_end],
        "derived_instruction_range": [derived_start, derived_end],
        "gt_instruction_window": gt_window,
        "derived_instruction_window": derived_window,
        "bytecode_window_radius": radius,
        "bytecode_window_max_records": 40,
        "bytecode_window_truncated": bytecode_window_truncated,
        "instruction_renderer_version": "editable_bytecode_v1",
        "instruction_diff": "\n".join([summary, *diff_lines]) if diff_lines else "<instruction diff unavailable>",
    }


def _selected_action_types_from_repair_context(repair_context: dict | None) -> list[str]:
    if not isinstance(repair_context, dict):
        return []
    return [
        str(action_type)
        for action_type in (repair_context.get("_selected_action_pattern_types") or [])
        if action_type
    ][:3]


def _compact_rejected_attempts_for_prompt(rejected_attempts: list[dict], *, limit: int = 3) -> list[dict]:
    compact: list[dict] = []
    for attempt in rejected_attempts[-limit:]:
        if not isinstance(attempt, dict):
            continue
        compact.append(
            {
                "attempt": attempt.get("attempt"),
                "acceptance_reason": attempt.get("acceptance_reason"),
                "selected_action_types": [
                    str(action_type)
                    for action_type in (attempt.get("selected_action_types") or [])
                    if action_type
                ][:3],
                "target_score_before": attempt.get("target_score_before"),
                "target_score_after": attempt.get("target_score_after"),
                "replacement_hash": attempt.get("replacement_hash"),
                "replacement_structural_hash": attempt.get("replacement_structural_hash"),
                "replacement_fingerprint": attempt.get("replacement_fingerprint"),
                "replacement_delta": attempt.get("replacement_delta"),
                "candidate_residual": attempt.get("candidate_residual"),
            }
        )
    return compact


def _replacement_hash(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = "\n".join(line.rstrip() for line in str(text).strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def _replacement_structural_hash(text: str | None) -> str | None:
    if text is None:
        return None
    source = textwrap.dedent(str(text).strip("\n"))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    normalized = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def _normalize_fragment_candidates(raw_candidates: Any) -> list[dict[str, Any]]:
    if isinstance(raw_candidates, list):
        items = raw_candidates
    else:
        items = [raw_candidates]

    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_structural_hashes: set[str] = set()
    for index, item in enumerate(items):
        metadata: dict[str, Any] = {"candidate_index": index}
        if isinstance(item, dict):
            text = item.get("text") or item.get("candidate") or item.get("replacement_text")
            metadata.update({key: value for key, value in item.items() if key not in {"text", "candidate", "replacement_text"}})
        else:
            text = item
        if text is None:
            continue
        text = str(text)
        candidate_hash = _replacement_hash(text)
        candidate_structural_hash = _replacement_structural_hash(text)
        if (
            (candidate_hash and candidate_hash in seen_hashes)
            or (candidate_structural_hash and candidate_structural_hash in seen_structural_hashes)
        ):
            continue
        if candidate_hash:
            seen_hashes.add(candidate_hash)
        if candidate_structural_hash:
            seen_structural_hashes.add(candidate_structural_hash)
        metadata["text"] = text
        metadata["replacement_hash"] = candidate_hash
        metadata["replacement_structural_hash"] = candidate_structural_hash
        candidates.append(metadata)
    return candidates


def _candidate_result_distance(result: dict[str, Any]) -> int:
    score = result.get("target_score_after") or {}
    if isinstance(score, dict) and isinstance(score.get("combined_distance"), (int, float)):
        return int(score["combined_distance"])
    return 10**9


def _candidate_result_successes(result: dict[str, Any]) -> int:
    successes = _count_pylingual_successes(result.get("pylingual_verification"))
    return -1 if successes is None else int(successes)


def _candidate_result_rank(result: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        0 if result.get("accepted") else 1,
        _candidate_result_distance(result),
        -_candidate_result_successes(result),
        int(result.get("candidate_index") or 0),
    )


def _compact_candidate_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index": result.get("candidate_index"),
        "strategy": result.get("strategy"),
        "accepted": result.get("accepted"),
        "acceptance_reason": result.get("acceptance_reason"),
        "compile_error": result.get("compile_error"),
        "target_score_after": result.get("target_score_after"),
        "reattachment_candidate_kind": result.get("reattachment_candidate_kind"),
        "reattachment_structure_ok": result.get("reattachment_structure_ok"),
        "reattachment_structure_reason": result.get("reattachment_structure_reason"),
        "replacement_hash": result.get("replacement_hash"),
        "replacement_structural_hash": result.get("replacement_structural_hash"),
        "selected_action_types": result.get("selected_action_types"),
        "prompt_path": result.get("prompt_path"),
        "skipped_due_to_token_limit": result.get("skipped_due_to_token_limit"),
        "skip_reason": result.get("skip_reason"),
        "prompt_token_count": result.get("prompt_token_count"),
        "token_threshold_used": result.get("token_threshold_used"),
    }


def _replacement_delta_for_attempt(source_text_before: str | None, replacement_text: str | None) -> dict[str, Any]:
    before = _fragment_feature_snapshot(source_text_before)
    after = _fragment_feature_snapshot(replacement_text)
    delta = _fragment_feature_delta(before, after)
    return {
        key: value
        for key, value in delta.items()
        if value not in (None, [], {}, "False->False", "True->True", 0)
    }


def _remember_rejected_attempt(
    rejected_attempts_by_qualname: dict[str, list[dict]],
    qualname: str,
    *,
    attempt: int,
    source_text_before: str | None,
    replacement_text: str | None,
    acceptance_reason: str | None,
    target_score_before: dict | None,
    target_score_after: dict | None,
    repair_context: dict | None,
    candidate_residual: dict | None = None,
) -> None:
    attempts = rejected_attempts_by_qualname.setdefault(qualname, [])
    attempts.append(
        {
            "attempt": attempt,
            "acceptance_reason": acceptance_reason,
            "selected_action_types": _selected_action_types_from_repair_context(repair_context),
            "target_score_before": target_score_before,
            "target_score_after": target_score_after,
            "replacement_hash": _replacement_hash(replacement_text),
            "replacement_structural_hash": _replacement_structural_hash(replacement_text),
            "replacement_fingerprint": _fragment_feature_snapshot(replacement_text),
            "replacement_delta": _replacement_delta_for_attempt(source_text_before, replacement_text),
            "candidate_residual": candidate_residual,
        }
    )
    del attempts[:-5]


def _exception_table_records(code_object: Any) -> list[dict]:
    """Structured summary of a code object's real try regions (for prompt
    enrichment on control-flow failures). Each record describes what a try block
    protects and its handler shape, read from GT/derived bytecode."""
    if code_object is None:
        return []
    from utils.semantic_operators.structural import (
        _handler_skeleton,
        _instructions,
        _protected_call_parts,
        _real_try_blocks,
    )
    insts = _instructions(code_object)
    records: list[dict] = []
    for (start, end, target, depth) in _real_try_blocks(code_object):
        parts = _protected_call_parts(insts, start, end)
        trivial, exc_type = _handler_skeleton(insts, target)
        protected_ops = [
            i.opname for i in insts
            if start is not None and end is not None and start <= getattr(i, "offset", -1) < end
        ]
        records.append({
            "protected_offsets": [start, end],
            "protected_call": ".".join(parts) if parts else None,
            "protected_ops": protected_ops[:12],
            "handler_exc_type": exc_type,
            "handler_trivial": bool(trivial),
        })
    return records


# --- deterministic diff-chunk diagnosis: the "what diverged" signal for the LLM prompt -------
_DIFF_FAMILIES = (
    ("call arguments", {"CALL", "CALL_KW", "KW_NAMES", "CALL_FUNCTION_EX", "PRECALL"}),
    ("f-string interpolation", {"FORMAT_VALUE", "FORMAT_SIMPLE", "FORMAT_WITH_SPEC", "CONVERT_VALUE", "BUILD_STRING"}),
    ("container literal", {"BUILD_MAP", "BUILD_LIST", "BUILD_SET", "BUILD_TUPLE", "BUILD_CONST_KEY_MAP", "MAP_ADD", "LIST_APPEND", "SET_ADD"}),
    ("exception handling", {"PUSH_EXC_INFO", "CHECK_EXC_MATCH", "SETUP_WITH", "BEFORE_WITH", "WITH_EXCEPT_START", "RERAISE", "POP_EXCEPT"}),
    ("closure variable", {"LOAD_DEREF", "STORE_DEREF", "LOAD_CLASSDEREF", "MAKE_CELL", "COPY_FREE_VARS", "LOAD_CLOSURE"}),
    ("control flow", {"POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_FORWARD_IF_TRUE", "JUMP_FORWARD", "JUMP_BACKWARD", "FOR_ITER", "GET_ITER"}),
    ("import", {"IMPORT_NAME", "IMPORT_FROM"}),
    ("nested function/decorator", {"MAKE_FUNCTION"}),
    ("yield/await", {"YIELD_VALUE", "GET_AWAITABLE", "SEND"}),
)


def _obj_instructions(code_object: Any) -> list:
    fn = getattr(code_object, "get_instructions", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:  # noqa: BLE001
            pass
    try:
        return list(code_object)
    except Exception:  # noqa: BLE001
        return []


def _diff_family(ops: list) -> str:
    s = set(ops)
    for name, fam in _DIFF_FAMILIES:
        if s & fam:
            return name
    if s and s <= {"LOAD_CONST", "RETURN_CONST"}:
        return "constant value"
    if s & {"LOAD_ATTR", "LOAD_METHOD", "STORE_ATTR"}:
        return "attribute access"
    if s & {"LOAD_FAST", "LOAD_GLOBAL", "LOAD_NAME", "STORE_FAST", "STORE_NAME"}:
        return "name/expression"
    return "statement"


def _diff_seg_detail(insts: list) -> str:
    bits = []
    for i in insts[:4]:
        ar = (getattr(i, "argrepr", "") or "").strip()
        op = getattr(i, "opname", "")
        bits.append(f"{op} {ar}".strip() if ar else op)
    return ", ".join(b for b in bits if b)[:120]


def _diff_chunk_diagnosis(gt_code_object: Any, derived_code_object: Any) -> str:
    """Human-readable diagnosis of WHAT diverged, from the GT vs derived opcode-stream diff:
    how many localized change chunks, and for each its kind (dropped / spurious / wrong) and
    construct family (call arguments, f-string, container, closure variable, control flow,
    constant, ...). This is the key per-iteration deterministic signal that tells the LLM
    exactly what to change so the oracle accepts the candidate. Never raises."""
    try:
        if gt_code_object is None or derived_code_object is None:
            return ""
        gi = _obj_instructions(gt_code_object)
        di = _obj_instructions(derived_code_object)
        if not gi or not di:
            return ""
        gops = [getattr(i, "opname", "") for i in gi]
        dops = [getattr(i, "opname", "") for i in di]
        from difflib import SequenceMatcher
        chunks = [(t, i1, i2, j1, j2) for t, i1, i2, j1, j2 in
                  SequenceMatcher(a=gops, b=dops, autojunk=False).get_opcodes() if t != "equal"]
        if not chunks:
            return ""
        parts = []
        for t, i1, i2, j1, j2 in chunks[:3]:
            fam = _diff_family(gops[i1:i2] + dops[j1:j2])
            if t == "delete":
                phrase = f"the ground truth has a {fam} your version dropped"
                detail = _diff_seg_detail(gi[i1:i2])
            elif t == "insert":
                phrase = f"your version has a spurious {fam} the ground truth lacks"
                detail = _diff_seg_detail(di[j1:j2])
            else:
                phrase = f"your {fam} differs from the ground truth"
                detail = _diff_seg_detail(gi[i1:i2])
            parts.append(phrase + (f" [GT bytecode: {detail}]" if detail else ""))
        n = len(chunks)
        head = "1 localized change" if n == 1 else f"{n} localized changes"
        return f"Bytecode divergence diagnosis ({head}): " + "; ".join(parts) + "."
    except Exception:  # noqa: BLE001 - advisory signal; never break the caller.
        return ""


def _build_target_repair_context(
    *,
    qualname: str,
    gt_code_object: Any,
    derived_code_object: Any,
    verification: dict | None,
    line_number: int | None = None,
    rejected_attempts: list[dict],
) -> dict:
    failed_result = _failed_result_for_target(verification, qualname)
    failed_offset = None if failed_result is None else failed_result.get("failed_offset")
    if line_number is None and failed_result is not None:
        line_number = failed_result.get("failed_line_number")
    base_radius = 2 if qualname != "<module>" else 3
    retry_count = len(rejected_attempts or [])
    radius = min(base_radius + retry_count, 5 if qualname != "<module>" else 6)
    instruction_context = _instruction_diff_context(
        gt_code_object,
        derived_code_object,
        None if failed_offset is None else int(failed_offset),
        radius=radius,
    )
    return {
        "target_kind": "module_statement" if qualname == "<module>" else "code_object_fragment",
        "qualname": qualname,
        "localized_line_number": line_number,
        "failed_offset": instruction_context.get("failed_offset"),
        "alignment_tag": instruction_context.get("alignment_tag"),
        "retry_count_for_target": retry_count,
        "pylingual_failed_result": failed_result,
        "gt_control_flow_skeleton": _gt_control_flow_skeleton(gt_code_object),
        # deterministic signals for the LLM prompt (part 2 of the pipeline contract): a short
        # imperative structural directive (bc_to_cft skeleton diff) + a per-object diff-chunk
        # diagnosis naming exactly what diverged. Both guarded (never raise).
        "structural_repair_directive": (
            structural_repair_directive(gt_code_object, derived_code_object)
            if derived_code_object is not None else None),
        "diff_chunk_diagnosis": (
            _diff_chunk_diagnosis(gt_code_object, derived_code_object)
            if derived_code_object is not None else ""),
        "target_is_missing": derived_code_object is None,
        "gt_reconstruction_brief": _gt_reconstruction_brief(
            gt_code_object, qualname, is_missing=(derived_code_object is None)),
        "derived_control_flow_skeleton": _derived_control_flow_skeleton(derived_code_object),
        "missing_extra_statement_summary": (
            _missing_extra_statement_summary(gt_code_object, derived_code_object)
            if failed_offset is None else ""),
        "instruction_diff": instruction_context.get("instruction_diff"),
        "gt_instruction_range": instruction_context.get("gt_instruction_range"),
        "derived_instruction_range": instruction_context.get("derived_instruction_range"),
        "gt_instruction_window": instruction_context.get("gt_instruction_window"),
        "derived_instruction_window": instruction_context.get("derived_instruction_window"),
        "bytecode_window_radius": instruction_context.get("bytecode_window_radius"),
        "bytecode_window_max_records": instruction_context.get("bytecode_window_max_records"),
        "bytecode_window_truncated": instruction_context.get("bytecode_window_truncated"),
        "instruction_renderer_version": instruction_context.get("instruction_renderer_version"),
        "rejected_attempts": _compact_rejected_attempts_for_prompt(rejected_attempts),
        "gt_exception_table": _exception_table_records(gt_code_object),
        "derived_exception_table": _exception_table_records(derived_code_object),
    }


def _compact_semantic_step_record(step_record: dict[str, Any]) -> dict[str, Any]:
    """Keep the normal run log small; heavy prompt data goes to accepted JSONL."""
    compact = dict(step_record)
    repair_context = compact.pop("repair_context", None)
    if isinstance(repair_context, dict):
        compact["repair_context_summary"] = {
            "target_kind": repair_context.get("target_kind"),
            "qualname": repair_context.get("qualname"),
            "localized_line_number": repair_context.get("localized_line_number"),
            "failed_offset": repair_context.get("failed_offset"),
            "alignment_tag": repair_context.get("alignment_tag"),
            "has_instruction_diff": bool(repair_context.get("instruction_diff")),
            "has_llm_prompt_record": bool(repair_context.get("_llm_prompt_record")),
        }
    return compact


def _append_semantic_step_log(log_file: Path | None, run_id: str | None, file_hash: str | None, step_record: dict[str, Any]) -> None:
    del log_file, run_id, file_hash, step_record
    return


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _semantic_repair_code_object_log_path(log_file: Path | None) -> Path:
    del log_file
    path = ACCEPTED_CODE_OBJECT_PATH.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _semantic_repair_telemetry_log_path(log_file: Path | None) -> Path:
    del log_file
    path = ACCEPTED_CODE_OBJECT_TELEMETRY_PATH.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _leading_indent_width(text: str | None) -> int | None:
    if not text:
        return None
    for line in text.splitlines():
        if line.strip():
            return len(line) - len(line.lstrip(" "))
    return None


def _line_count(text: str | None) -> int | None:
    if text is None:
        return None
    return len(text.splitlines())


def _fragment_feature_snapshot(fragment: str | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "parse_ok": False,
        "dedented_parse_ok": False,
        "statement_types": [],
        "control_flow_nodes": [],
        "loaded_names": [],
        "bound_names": [],
        "top_level_statement_count": None,
        "node_count": None,
    }
    if not fragment:
        return snapshot

    snapshot["line_count"] = _line_count(fragment)
    snapshot["char_count"] = len(fragment)
    snapshot["indent_width"] = _leading_indent_width(fragment)

    candidates = [fragment, textwrap.dedent(fragment)]
    for index, candidate in enumerate(candidates):
        try:
            parsed = ast.parse(candidate)
        except SyntaxError:
            continue
        snapshot["parse_ok"] = index == 0
        snapshot["dedented_parse_ok"] = True
        snapshot["top_level_statement_count"] = len(parsed.body)
        snapshot["statement_types"] = sorted({type(node).__name__ for node in parsed.body})
        snapshot["node_count"] = sum(1 for _ in ast.walk(parsed))
        loaded_names = sorted(
            {
                child.id
                for child in ast.walk(parsed)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            }
        )
        bound_names = sorted(
            {
                child.id
                for child in ast.walk(parsed)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            }
        )
        control_flow_nodes = sorted(
            {
                type(child).__name__
                for child in ast.walk(parsed)
                if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
        )
        snapshot["loaded_names"] = loaded_names
        snapshot["bound_names"] = bound_names
        snapshot["control_flow_nodes"] = control_flow_nodes
        return snapshot
    return snapshot


def _fragment_feature_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_names = set(before.get("loaded_names") or []) | set(before.get("bound_names") or [])
    after_names = set(after.get("loaded_names") or []) | set(after.get("bound_names") or [])
    before_controls = set(before.get("control_flow_nodes") or [])
    after_controls = set(after.get("control_flow_nodes") or [])
    before_statements = set(before.get("statement_types") or [])
    after_statements = set(after.get("statement_types") or [])
    return {
        "line_count_delta": _safe_int_delta(before.get("line_count"), after.get("line_count")),
        "char_count_delta": _safe_int_delta(before.get("char_count"), after.get("char_count")),
        "indent_width_delta": _safe_int_delta(before.get("indent_width"), after.get("indent_width")),
        "parse_transition": f"{bool(before.get('parse_ok'))}->{bool(after.get('parse_ok'))}",
        "dedented_parse_transition": f"{bool(before.get('dedented_parse_ok'))}->{bool(after.get('dedented_parse_ok'))}",
        "statement_types_added": sorted(after_statements - before_statements),
        "statement_types_removed": sorted(before_statements - after_statements),
        "control_flow_added": sorted(after_controls - before_controls),
        "control_flow_removed": sorted(before_controls - after_controls),
        "names_added": sorted(after_names - before_names),
        "names_removed": sorted(before_names - after_names),
    }


def _safe_int_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return int(after) - int(before)


def _accepted_case_telemetry_record(
    *,
    run_id: str | None,
    file_hash: str | None,
    step_record: dict[str, Any],
) -> dict[str, Any]:
    repair_context = step_record.get("repair_context") or {}
    pylingual_after = step_record.get("pylingual_verification")
    extracted_before = step_record.get("extracted_before")
    replacement_text = step_record.get("replacement_text")
    before_features = _fragment_feature_snapshot(extracted_before)
    after_features = _fragment_feature_snapshot(replacement_text)
    prompt_record = repair_context.get("_llm_prompt_record")
    source_span = {
        "localized_line_number": repair_context.get("localized_line_number"),
        "source_lineno": step_record.get("source_lineno", repair_context.get("source_lineno")),
        "source_col_offset": step_record.get("source_col_offset", repair_context.get("source_col_offset")),
        "source_end_lineno": step_record.get("source_end_lineno", repair_context.get("source_end_lineno")),
        "source_end_col_offset": step_record.get("source_end_col_offset", repair_context.get("source_end_col_offset")),
        "parent_qualname": step_record.get("parent_qualname"),
        "source_kind": step_record.get("source_kind"),
        "module_body_strategy": step_record.get("module_body_strategy"),
        "target_identity": step_record.get("target_identity"),
        "parent_target_identity": step_record.get("parent_target_identity"),
        "gt_target_identity": step_record.get("gt_target_identity"),
        "target_mapping_resolution": step_record.get("target_mapping_resolution"),
        "parent_mapping_resolution": step_record.get("parent_mapping_resolution"),
        "gt_mapping_resolution": step_record.get("gt_mapping_resolution"),
    }
    case_fingerprint = {
        "source_file_hash": file_hash,
        "qualname": step_record.get("qualname"),
        "repair_operation": step_record.get("repair_operation"),
        "iteration": step_record.get("iteration"),
        "step": step_record.get("step"),
        "localized_line_number": source_span["localized_line_number"],
        "source_lineno": source_span["source_lineno"],
        "source_col_offset": source_span["source_col_offset"],
        "source_end_lineno": source_span["source_end_lineno"],
        "source_end_col_offset": source_span["source_end_col_offset"],
        "source_kind": source_span["source_kind"],
        "target_identity": source_span["target_identity"],
        "parent_target_identity": source_span["parent_target_identity"],
        "gt_target_identity": source_span["gt_target_identity"],
    }
    case_id = hashlib.sha1(json.dumps(case_fingerprint, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return _json_safe({
        "case_id": case_id,
        "run_id": run_id,
        "timestamp": now_iso(),
        "source_file_hash": file_hash,
        "qualname": step_record.get("qualname"),
        "repair_operation": step_record.get("repair_operation"),
        "iteration": step_record.get("iteration"),
        "step": step_record.get("step"),
        "accepted": step_record.get("accepted"),
        "acceptance_reason": step_record.get("acceptance_reason"),
        "target_kind": repair_context.get("target_kind"),
        "source_span": source_span,
        "repair_context_summary": {
            "failed_offset": repair_context.get("failed_offset"),
            "alignment_tag": repair_context.get("alignment_tag"),
            "has_instruction_diff": bool(repair_context.get("instruction_diff")),
            "has_llm_prompt_record": bool(prompt_record),
        },
        "before_fragment": extracted_before,
        "after_fragment": replacement_text,
        "before_features": before_features,
        "after_features": after_features,
        "feature_delta": _fragment_feature_delta(before_features, after_features),
        "semantic_delta": {
            "combined_distance_delta": _step_combined_distance_delta(step_record),
            "target_score_before": step_record.get("target_score_before"),
            "target_score_after": step_record.get("target_score_after"),
        },
        "reattachment": {
            "candidate_kind": step_record.get("reattachment_candidate_kind"),
            "parse_ok": step_record.get("reattachment_parse_ok"),
            "parse_error": step_record.get("reattachment_parse_error"),
            "structure_ok": step_record.get("reattachment_structure_ok"),
            "structure_reason": step_record.get("reattachment_structure_reason"),
        },
        "pylingual_after": {
            "all_equal": None if pylingual_after is None else pylingual_after.get("all_equal"),
            "success_count": _success_count_from_verification(pylingual_after),
            "failed_targets": _failed_targets_from_verification(pylingual_after),
        },
        "prompt": prompt_record,
    })


def _success_count_from_verification(verification: dict | None) -> int | None:
    if verification is None:
        return None
    return _count_pylingual_successes(verification)


def _failed_targets_from_verification(verification: dict | None) -> list[str]:
    if verification is None:
        return []
    return [
        str(result.get("names"))
        for result in verification.get("results", [])
        if result.get("names") and not result.get("success")
    ]


def _accepted_code_object_dataset_record(
    *,
    run_id: str | None,
    file_hash: str | None,
    step_record: dict[str, Any],
) -> dict[str, Any]:
    repair_context = step_record.get("repair_context") or {}
    pylingual_after = step_record.get("pylingual_verification")
    prompt_record = repair_context.get("_llm_prompt_record")
    return _json_safe({
        "run_id": run_id,
        "timestamp": now_iso(),
        "source_file_hash": file_hash,
        "qualname": step_record.get("qualname"),
        "repair_operation": step_record.get("repair_operation"),
        "iteration": step_record.get("iteration"),
        "step": step_record.get("step"),
        "accepted": step_record.get("accepted"),
        "acceptance_reason": step_record.get("acceptance_reason"),
        "target_kind": repair_context.get("target_kind"),
        "source_kind": step_record.get("source_kind"),
        "target_identity": step_record.get("target_identity"),
        "parent_target_identity": step_record.get("parent_target_identity"),
        "gt_target_identity": step_record.get("gt_target_identity"),
        "target_mapping_resolution": step_record.get("target_mapping_resolution"),
        "parent_mapping_resolution": step_record.get("parent_mapping_resolution"),
        "gt_mapping_resolution": step_record.get("gt_mapping_resolution"),
        "extracted_before": step_record.get("extracted_before"),
        "replacement_text": step_record.get("replacement_text"),
        "target_score_before": step_record.get("target_score_before"),
        "target_score_after": step_record.get("target_score_after"),
        "combined_distance_delta": _step_combined_distance_delta(step_record),
        "reattachment": {
            "candidate_kind": step_record.get("reattachment_candidate_kind"),
            "parse_ok": step_record.get("reattachment_parse_ok"),
            "parse_error": step_record.get("reattachment_parse_error"),
            "structure_ok": step_record.get("reattachment_structure_ok"),
            "structure_reason": step_record.get("reattachment_structure_reason"),
        },
        "bytecode_context": {
            "failed_offset": repair_context.get("failed_offset"),
            "alignment_tag": repair_context.get("alignment_tag"),
            "gt_range": repair_context.get("gt_instruction_range"),
            "derived_range": repair_context.get("derived_instruction_range"),
            "gt_instruction_window": repair_context.get("gt_instruction_window"),
            "derived_instruction_window": repair_context.get("derived_instruction_window"),
            "instruction_diff": repair_context.get("instruction_diff"),
            "window_radius": repair_context.get("bytecode_window_radius"),
            "window_max_records": repair_context.get("bytecode_window_max_records"),
            "window_truncated": repair_context.get("bytecode_window_truncated"),
            "renderer_version": repair_context.get("instruction_renderer_version"),
        },
        "pylingual_after": {
            "all_equal": None if pylingual_after is None else pylingual_after.get("all_equal"),
            "success_count": _success_count_from_verification(pylingual_after),
            "failed_targets": _failed_targets_from_verification(pylingual_after),
        },
        "prompt": prompt_record,
    })


def _append_accepted_code_object_dataset(log_file: Path | None, run_id: str | None, file_hash: str | None, step_record: dict[str, Any]) -> None:
    if not step_record.get("accepted"):
        return
    if SEMANTIC_DISABLE_CORPUS_WRITE:
        return
    path = _semantic_repair_code_object_log_path(log_file)
    append_log(
        path,
        _accepted_code_object_dataset_record(
            run_id=run_id,
            file_hash=file_hash,
            step_record=step_record,
        ),
    )


def _append_accepted_case_telemetry(log_file: Path | None, run_id: str | None, file_hash: str | None, step_record: dict[str, Any]) -> None:
    if not step_record.get("accepted"):
        return
    path = _semantic_repair_telemetry_log_path(log_file)
    append_log(
        path,
        _accepted_case_telemetry_record(
            run_id=run_id,
            file_hash=file_hash,
            step_record=step_record,
        ),
    )


def _step_combined_distance_delta(step_record: dict[str, Any]) -> int | None:
    before = step_record.get("target_score_before") or {}
    after = step_record.get("target_score_after") or {}
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    before_distance = before.get("combined_distance")
    after_distance = after.get("combined_distance")
    if before_distance is None or after_distance is None:
        return None
    return int(after_distance) - int(before_distance)


def _semantic_print(message: str, *, indent: int = 0, tagged: bool = True) -> None:
    prefix = "[semantic_repair] " if tagged else ""
    print(f"{'  ' * max(0, indent)}{prefix}{message}", flush=True)


def _store_semantic_step(
    steps: list[dict],
    step_record: dict[str, Any],
    *,
    log_file: Path | None = None,
    run_id: str | None = None,
    file_hash: str | None = None,
) -> None:
    steps.append(step_record)
    repair_operation = step_record.get("repair_operation")
    qualname = step_record.get("qualname")
    iteration = step_record.get("iteration")
    step = step_record.get("step")
    accepted = step_record.get("accepted")
    acceptance_reason = step_record.get("acceptance_reason")
    target_before = step_record.get("target_score_before") or {}
    target_after = step_record.get("target_score_after") or {}
    before_distance = target_before.get("combined_distance") if isinstance(target_before, dict) else None
    after_distance = target_after.get("combined_distance") if isinstance(target_after, dict) else None
    token_limit_skip = bool(step_record.get("skipped_due_to_token_limit"))
    status = "accepted" if accepted else "skipped_token_limit" if token_limit_skip else "rejected"
    _semantic_print(
        f"-> step {step} iter {iteration} {qualname} ({repair_operation}) -> {status} "
        f"(combined_distance {before_distance} -> {after_distance})",
        indent=1,
        tagged=False,
    )
    if token_limit_skip:
        _semantic_print(
            "-> skipped: semantic repair prompt exceeded token threshold; no LLM call was made",
            indent=2,
            tagged=False,
        )
    if acceptance_reason:
        _semantic_print(f"-> reason: {acceptance_reason}", indent=2, tagged=False)
    _append_accepted_code_object_dataset(log_file, run_id, file_hash, step_record)
    _append_accepted_case_telemetry(log_file, run_id, file_hash, step_record)


def _announce_semantic_step(
    *,
    step_index: int,
    iteration: int,
    qualname: str,
    repair_operation: str,
    source_kind: str | None = None,
    file_hash: str | None = None,
) -> None:
    extra = f" source_kind={source_kind}" if source_kind else ""
    _semantic_print(
        f"-> starting step {step_index} iter {iteration} {qualname} ({repair_operation}){extra}",
        indent=1,
        tagged=False,
    )


def _find_top_level_statement_for_line(source_text: str, line_number: int) -> tuple[ast.stmt, int, int]:
    tree = ast.parse(source_text)
    matches: list[ast.stmt] = []
    for node in tree.body:
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if lineno is None or end_lineno is None:
            continue
        if int(lineno) <= line_number <= int(end_lineno):
            matches.append(node)
    if not matches:
        raise ReattachError(f"Could not localize module repair to a top-level statement for line {line_number}")
    node = min(matches, key=lambda item: int(getattr(item, "end_lineno")) - int(getattr(item, "lineno")))
    start_index, end_index = _node_span_to_indices(source_text, node)
    return node, start_index, end_index


def _apply_module_statement_candidate(
    *,
    gt_pyc: Path,
    current_source: Path,
    current_pyc: Path,
    current_summary: dict,
    current_pylingual_verification: dict | None,
    candidate_text: str,
    line_number: int,
    output_dir: Path,
    pyc_dir: Path,
    fragments_dir: Path,
    derived_source_stem: str,
    step_index: int,
    iteration: int,
    target_version: PythonVersion,
    verify_with_pylingual: bool,
    verify_each_step_with_pylingual: bool,
    reject_non_improving_candidates: bool,
) -> tuple[bool, str, Path, Path, dict, dict | None, dict, int]:
    target_score_before = _score_snapshot(_find_distance_row(compare_code_object_distances(gt_pyc, current_pyc), "<module>"))
    current_text = _load_text(current_source)
    node, start_index, end_index = _find_top_level_statement_for_line(current_text, line_number)
    extracted_before = current_text[start_index:end_index]
    reattachment_candidate = choose_best_reattachment_for_source_slice(
        current_text,
        start_index,
        end_index,
        int(getattr(node, "col_offset", 0)),
        candidate_text,
        extracted_before,
    )
    replacement_text = reattachment_candidate.replacement_text
    updated_text = reattachment_candidate.updated_source

    fragment_path = fragments_dir / f"{step_index:02d}_module_line_{line_number}.pyfrag"
    fragment_path.write_text(replacement_text, encoding="utf-8")

    next_source = output_dir / f"step{step_index}_{derived_source_stem}.py"
    next_source.write_text(updated_text, encoding="utf-8")
    next_pyc = _compiled_pyc_path(pyc_dir, next_source, target_version)
    compile_source_to_pyc(next_source, next_pyc, target_version)

    step_rows = compare_code_object_distances(gt_pyc, next_pyc)
    step_summary = summarize_results(step_rows)
    target_score_after = _score_snapshot(_find_distance_row(step_rows, "<module>"))
    step_pylingual_verification = (
        run_pylingual_verification(gt_pyc, next_pyc)
        if verify_with_pylingual and verify_each_step_with_pylingual
        else None
    )
    accepted, acceptance_reason = _should_accept_candidate(
        current_summary,
        step_summary,
        current_pylingual_verification,
        step_pylingual_verification,
    )
    if not reject_non_improving_candidates:
        accepted = True
        acceptance_reason = "candidate retained without acceptance filtering"
    step = {
        "step": step_index,
        "iteration": iteration,
        "qualname": "<module>",
        "repair_operation": "repair_module_statement",
        "module_body_strategy": "localized_module_statement_repair",
        "localized_line_number": line_number,
        "source_lineno": line_number,
        "source_col_offset": int(getattr(node, "col_offset", 0)),
        "source_end_lineno": int(getattr(node, "end_lineno", line_number)),
        "source_end_col_offset": int(getattr(node, "end_col_offset", 0)),
        "fragment_path": str(fragment_path),
        "output_source": str(next_source),
        "output_pyc": str(next_pyc),
        "extracted_before": extracted_before,
        "replacement_text": replacement_text,
        "reattachment_candidate_kind": reattachment_candidate.kind,
        "reattachment_parse_ok": reattachment_candidate.parse_ok,
        "reattachment_parse_error": reattachment_candidate.parse_error,
        "target_score_before": target_score_before,
        "target_score_after": target_score_after,
        "summary": step_summary,
        "pylingual_verification": step_pylingual_verification,
        "accepted": accepted,
        "acceptance_reason": acceptance_reason,
    }
    return (
        accepted,
        acceptance_reason,
        next_source,
        next_pyc,
        step_summary,
        step_pylingual_verification,
        step,
        step_index,
    )

def repair_mismatching_code_objects(
    gt_pyc: Path,
    derived_pyc: Path,
    derived_source: Path,
    *,
    gt_source: Path | None = None,
    output_dir: Path | None = None,
    log_file: Path | None = None,
    run_id: str | None = None,
    file_hash: str | None = None,
    fragment_fixer: FragmentFixer | None = None,
    strict_map: bool = False,
    verify_with_pylingual: bool = True,
    verify_each_step_with_pylingual: bool = True,
    reject_non_improving_candidates: bool = True,
    max_iterations: int = 1,
    sample_timeout_deadline: float | None = None,
    sample_hard_timeout_deadline: float | None = None,
    sample_timeout_seconds: int | None = None,
    sample_hard_timeout_seconds: int | None = None,
    sample_timeout_min_improvement_delta: int = 1,
    acceptance_mode: str | None = None,
) -> dict:
    global ACCEPTANCE_MODE
    if acceptance_mode is not None:
        ACCEPTANCE_MODE = acceptance_mode.strip().lower() or "distance"
    # Keep the distance module's exception-table gradient in sync with the mode.
    set_exception_table_distance_enabled(ACCEPTANCE_MODE == "oracle")
    gt_pyc = validate_input(gt_pyc)
    derived_pyc = validate_input(derived_pyc)
    target_version = infer_semantic_repair_python_version(gt_pyc, derived_pyc)
    derived_source = derived_source.expanduser().resolve()
    gt_source = gt_source.expanduser().resolve() if gt_source is not None else None
    gt_source_text: str | None = None

    def load_gt_source_text() -> tuple[Path, str]:
        nonlocal gt_source, gt_source_text
        if gt_source is None:
            gt_source = infer_source_from_pyc(gt_pyc)
        if gt_source_text is None:
            gt_source_text = parenthesize_bare_except(_load_text(gt_source))
        return gt_source, gt_source_text

    if output_dir is None:
        output_dir = derived_source.parent / f"{derived_source.stem}_repair_pipeline"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fragments_dir = output_dir / "fragments"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    pyc_dir = output_dir / "__pycache__"
    pyc_dir.mkdir(parents=True, exist_ok=True)

    max_iterations = max(1, int(max_iterations))
    sample_timeout_min_improvement_delta = max(1, int(sample_timeout_min_improvement_delta))
    initial_rows = compare_code_object_distances(gt_pyc, derived_pyc)
    initial_summary = summarize_results(initial_rows)
    initial_pylingual_verification = run_pylingual_verification(gt_pyc, derived_pyc) if verify_with_pylingual else None
    repair_targets = select_repair_targets(initial_rows, include_module=fragment_fixer is not None)
    initial_missing_targets = select_unreattachable_missing_targets(initial_rows)
    initial_extra_targets = select_extra_repair_targets(initial_rows)
    gt_code_objects = index_code_objects_by_qualname(gt_pyc)
    gt_bytecodes = index_bytecodes_by_qualname(gt_pyc)

    current_source = derived_source
    current_pyc = derived_pyc
    current_summary = initial_summary
    current_pylingual_verification = initial_pylingual_verification
    best_source = current_source
    best_pyc = current_pyc
    best_summary = current_summary
    best_pylingual_verification = current_pylingual_verification
    best_improvement_reason: str | None = None
    timeout_checkpoint_summary = current_summary
    timeout_checkpoint_count = 0
    steps: list[dict] = []
    post_llm_det_edits_total = 0  # attribution: total edits accepted by the post-LLM det passes

    all_repair_targets = list(repair_targets)
    all_extra_targets = list(initial_extra_targets)
    step_index = 0
    unsupported_missing_targets: set[str] = set()
    skipped_expression_child_targets: set[str] = set()
    unsupported_extra_targets: set[str] = set()
    unsupported_module_body_repair = False
    module_rejected_attempts: list[dict] = []
    rejected_attempts_by_qualname: dict[str, list[dict]] = {}
    # Deterministic-prepass candidates that FIRED and then lost. `_store_semantic_step` runs only
    # once the gate has passed, so without this both rejection paths vanish and "did the operator
    # fire?" is unanswerable: a builder that deferred and a builder that was correct-but-rejected
    # look identical from accepted-step counts alone.
    prepass_rejected_attempts: list[dict] = []
    sample_timed_out = False
    sample_timeout_reached = False
    sample_hard_timeout_reached = False
    sample_timeout_reason: str | None = None
    sample_timeout_action: str | None = None
    # Lever B (speedup): stop a file whose BEST state has been stagnant for this long — an earlier
    # hard-timeout, so it reuses the exact restore-best / raise-no-improvement semantics below.
    # 0 (default) => disabled => byte-identical to today. last_improvement clock starts at sample
    # start and is reset on every best-state promotion (see record_best_state_if_improved).
    sample_tail_deadline_seconds = _tail_deadline_seconds()
    last_improvement_monotonic = time.monotonic()

    def record_best_state_if_improved(reason: str) -> None:
        nonlocal best_source, best_pyc, best_summary, best_pylingual_verification
        nonlocal best_improvement_reason, last_improvement_monotonic
        promote = _combined_distance_improved(
            best_summary,
            current_summary,
            min_delta=sample_timeout_min_improvement_delta,
        )
        if (
            not promote
            and SEMANTIC_POST_LLM_DETERMINISTIC
            and ACCEPTANCE_MODE == "oracle"
            and best_pylingual_verification is not None
            and current_pylingual_verification is not None
        ):
            # A post-LLM deterministic accept can be an oracle-lattice win with combined distance
            # flat (e.g. an exception-table / regime flip). Promote it so it survives timeout
            # rollback and is visible to the timeout accountant. Gated on the flag + oracle mode,
            # so the default distance path (and the 680-test suite) is unaffected.
            _bd = compare_verifications(
                best_pylingual_verification["results"], current_pylingual_verification["results"]
            )
            promote = _bd.improved and not _bd.regressed
        if not promote:
            return
        best_source = current_source
        best_pyc = current_pyc
        best_summary = current_summary
        best_pylingual_verification = current_pylingual_verification
        best_improvement_reason = reason
        last_improvement_monotonic = time.monotonic()  # reset the Lever-B stagnation clock

    def restore_best_state_if_needed() -> None:
        nonlocal current_source, current_pyc, current_summary, current_pylingual_verification
        if best_source == current_source and best_pyc == current_pyc:
            return
        current_source = best_source
        current_pyc = best_pyc
        current_summary = best_summary
        current_pylingual_verification = best_pylingual_verification

    def handle_sample_timeout_if_needed(*, location: str) -> bool:
        nonlocal sample_timeout_deadline, sample_timeout_reached, sample_timeout_reason, sample_timeout_action
        nonlocal sample_timed_out
        nonlocal sample_hard_timeout_reached
        nonlocal timeout_checkpoint_summary, timeout_checkpoint_count
        if sample_hard_timeout_reached:
            return True
        # Lever B (speedup): tail early-abort. If enabled and the BEST state has been stagnant for
        # the tail deadline, stop now — an EARLIER hard timeout, reusing the exact restore-best /
        # raise-no-improvement semantics of the hard-timeout branch below (so it is quality-neutral
        # and recorded identically). Disabled by default (deadline 0) => this is a no-op.
        if sample_tail_deadline_seconds > 0 and _should_tail_abort(
            now=time.monotonic(),
            last_improvement=last_improvement_monotonic,
            tail_deadline=sample_tail_deadline_seconds,
        ):
            initial_distance = _summary_combined_distance(initial_summary)
            current_distance = _summary_combined_distance(current_summary)
            best_distance = _summary_combined_distance(best_summary)
            tail_label = f"{sample_tail_deadline_seconds:.0f}s tail deadline"
            if not _combined_distance_improved(
                initial_summary,
                best_summary,
                min_delta=sample_timeout_min_improvement_delta,
            ):
                raise SemanticRepairHardTimeoutNoImprovement(
                    f"sample stalled past {tail_label} at {location} without combined distance improvement "
                    f"({initial_distance} -> {current_distance})"
                )
            restore_best_state_if_needed()
            sample_timed_out = True
            sample_timeout_reached = True
            sample_hard_timeout_reached = True
            sample_timeout_action = "stopped_at_tail_deadline_kept_best_result"
            sample_timeout_reason = (
                f"sample stalled past {tail_label} at {location}; keeping best result "
                f"({initial_distance} -> {best_distance})"
            )
            if best_improvement_reason:
                sample_timeout_reason += f"; best improvement: {best_improvement_reason}"
            _semantic_print(f"-> tail deadline reached: {sample_timeout_reason}", indent=1)
            return True
        if sample_hard_timeout_deadline is not None and time.monotonic() >= sample_hard_timeout_deadline:
            initial_distance = _summary_combined_distance(initial_summary)
            current_distance = _summary_combined_distance(current_summary)
            best_distance = _summary_combined_distance(best_summary)
            hard_label = (
                f"{sample_hard_timeout_seconds} seconds"
                if sample_hard_timeout_seconds is not None and sample_hard_timeout_seconds > 0
                else "configured hard timeout"
            )
            if not _combined_distance_improved(
                initial_summary,
                best_summary,
                min_delta=sample_timeout_min_improvement_delta,
            ):
                raise SemanticRepairHardTimeoutNoImprovement(
                    f"sample exceeded hard timeout {hard_label} at {location} without combined distance improvement "
                    f"({initial_distance} -> {current_distance})"
                )
            restore_best_state_if_needed()
            sample_timed_out = True
            sample_timeout_reached = True
            sample_hard_timeout_reached = True
            sample_timeout_action = "stopped_at_hard_timeout_kept_best_result"
            sample_timeout_reason = (
                f"sample exceeded hard timeout {hard_label} at {location}; keeping best result "
                f"({initial_distance} -> {best_distance})"
            )
            if best_improvement_reason:
                sample_timeout_reason += f"; best improvement: {best_improvement_reason}"
            _semantic_print(f"-> hard timeout reached: {sample_timeout_reason}", indent=1)
            return True
        if sample_timeout_deadline is None or time.monotonic() < sample_timeout_deadline:
            return False
        checkpoint_distance = _summary_combined_distance(timeout_checkpoint_summary)
        current_distance = _summary_combined_distance(current_summary)
        best_distance = _summary_combined_distance(best_summary)
        elapsed_label = (
            f"{sample_timeout_seconds} seconds"
            if sample_timeout_seconds is not None and sample_timeout_seconds > 0
            else "configured timeout"
        )
        if not _combined_distance_improved(
            timeout_checkpoint_summary,
            best_summary,
            min_delta=sample_timeout_min_improvement_delta,
        ):
            raise SemanticRepairTimeoutNoImprovement(
                f"sample exceeded {elapsed_label} at {location} without combined distance improvement "
                f"since the previous checkpoint ({checkpoint_distance} -> {current_distance})"
            )
        timeout_checkpoint_count += 1
        sample_timeout_reached = True
        sample_timeout_action = "continued_after_checkpoint_improvement"
        sample_timeout_reason = (
            f"sample exceeded {elapsed_label} at {location}; continuing because combined distance improved since "
            f"the previous checkpoint ({checkpoint_distance} -> {best_distance})"
        )
        if best_improvement_reason:
            sample_timeout_reason += f"; best improvement: {best_improvement_reason}"
        _semantic_print(f"-> timeout checkpoint reached: {sample_timeout_reason}", indent=1)
        timeout_checkpoint_summary = best_summary
        if sample_timeout_seconds is not None and sample_timeout_seconds > 0:
            sample_timeout_deadline = time.monotonic() + sample_timeout_seconds
        else:
            sample_timeout_deadline = None
        return False

    def run_deterministic_prepass(*, phase: str = "pre_llm", iteration_label: int = 0) -> int:
        """Apply deterministic operators to a FIXPOINT and return the number of accepted edits.

        Every edit reads the correct value from GT bytecode, is spliced into the source,
        recompiled, and accepted only if the oracle-state lattice strictly improves with no
        regression. Runs no LLM and does not consume the per-target attempt cap.

        phase="pre_llm" (default): the pre-LLM leg. phase="post_llm": re-invoked after an LLM
        iteration on the residual (a structural LLM edit can expose a leaf the pre-LLM pass could
        not reach). The post phase is hard-gated on SEMANTIC_POST_LLM_DETERMINISTIC AND oracle
        acceptance mode so both legs stay lattice-non-regressing (never undo each other).
        """
        nonlocal current_source, current_pyc, current_summary
        nonlocal current_pylingual_verification, step_index
        # Each candidate is accepted only on a strict oracle-state improvement with no regression
        # (compare_verifications below), so the deterministic layer never weakens any gate.
        enabled = SEMANTIC_DETERMINISTIC_PREPASS if phase == "pre_llm" else (
            SEMANTIC_POST_LLM_DETERMINISTIC and ACCEPTANCE_MODE == "oracle"
        )
        if not enabled:
            return 0
        if current_pylingual_verification is None:
            return 0
        from pylingual.equivalence_check import compare_pyc

        if phase == "post_llm":
            # Refresh the baseline against the LLM-advanced current_pyc so non-regression is judged
            # vs the true current state (closes the stale-baseline hole when per-step verify is off).
            current_pylingual_verification = run_pylingual_verification(gt_pyc, current_pyc)
            if current_pylingual_verification is None:
                return 0

        prepass_dir = output_dir / "prepass"
        prepass_pyc_dir = prepass_dir / "__pycache__"
        prepass_pyc_dir.mkdir(parents=True, exist_ok=True)

        text = _load_text(current_source)
        # Stable base stem: filenames are prepass{step_index}_{base_stem}.py. Using
        # current_source.stem would compound each accept (prepass2_prepass1_...) and eventually
        # blow past NAME_MAX; derived_source is the stable clean stem, so re-invocation across the
        # post-LLM phase never nests the stem.
        base_stem = derived_source.stem
        max_prepass_edits = 200
        applied = 0
        while applied < max_prepass_edits:
            if sample_hard_timeout_deadline is not None and time.monotonic() > sample_hard_timeout_deadline:
                return applied
            results = compare_pyc(validate_input(gt_pyc), validate_input(current_pyc))
            progressed = False
            for tr in results:
                if tr.success:
                    continue
                is_leaf = tr.message == "Different bytecode" and tr.failed_offset is not None
                is_struct = tr.message == "Different control flow"
                # "Different bytecode" with no single offset (e.g. a PyLingual "line too long"
                # placeholder statement) is also deterministically recoverable in some cases.
                is_diffbc = tr.message == "Different bytecode"
                # A code object entirely ABSENT from derived.pyc (bc_b is None) -- e.g. PyLingual
                # never emitted a synthetic __annotate__ object for a class/def at all, not just a
                # diverged one. tr.names() glues "name_a, name_b" whenever the two sides differ,
                # and name_b falls back to the literal string "None" here, so read bc_a.name
                # directly rather than feeding that glued string into the qualname map below.
                is_missing = tr.message == "Missing bytecode"
                raw_qualname = tr.name_a if is_missing else tr.names()
                # Annotation reconciliation (Stage 1, oracle-source backend, flag-gated): a
                # diverged OR entirely missing synthetic `__annotate__` object has no source
                # representation of its own, so remap the target to the ENCLOSING def/class whose
                # annotations produced it and repair that instead, using the verbatim GT source
                # segment as the deterministic candidate. Falls through to the SAME recompile +
                # oracle gate below; never fires when the flag is off, there is no GT source, or
                # the qualname isn't a `__annotate__` leaf (defer, never a guessed annotation).
                # Computed BEFORE (and folded into) the is_leaf/is_struct/is_diffbc gate below so a
                # missing __annotate__ object -- which is none of those three message kinds -- still
                # reaches this branch instead of being filtered out untouched.
                annotate_enclosing = (
                    _annotate_enclosing_qualname(raw_qualname)
                    if (SEMANTIC_ANNOTATION_RECONCILE and gt_source is not None)
                    else None
                )
                if not (is_leaf or is_struct or is_diffbc or annotate_enclosing is not None):
                    continue
                qualname = annotate_enclosing if annotate_enclosing is not None else raw_qualname
                # The <module> object has no extractable code-object span (its row span is
                # degenerate/empty), so feed the WHOLE source as its fragment and write the
                # operator's full output back directly. This lets the deterministic operators
                # repair module-level statements (where most constant globals live). Every
                # candidate still passes the recompile + oracle-distance gate below.
                is_module_target = qualname == "<module>"
                if is_module_target:
                    target_row = None
                    fragment = text
                else:
                    try:
                        target_row = _find_target_row(current_source, current_pyc, qualname, False)
                        fragment = extract_source_segment(text, target_row)
                    except ReattachError:
                        continue
                repair_context = {
                    "pylingual_failed_result": {"message": tr.message},
                    "failed_offset": tr.failed_offset,
                }
                candidate = None
                operation = strategy = None
                if annotate_enclosing is not None:
                    annotation_candidate_text = _gt_def_source_by_qualname(
                        parenthesize_bare_except(_load_text(gt_source)), qualname
                    )
                    if annotation_candidate_text is not None and annotation_candidate_text != fragment:
                        candidate = {
                            "text": annotation_candidate_text,
                            "operator": "annotation_reconcile",
                            "confidence": "high",
                        }
                        operation, strategy = "deterministic_prepass_annotation", "deterministic_annotation"
                    if candidate is None:
                        continue
                elif is_leaf:
                    candidate = leaf_value_candidate(tr.bc_a, tr.bc_b, fragment, repair_context)
                    if candidate is not None:
                        operation, strategy = "deterministic_prepass_leaf", "deterministic_leaf"
                    else:
                        # jump-sense inversion: collapse `if C: pass else: BODY` -> `if C: BODY`
                        candidate = collapse_dead_pass_else_candidate(tr.bc_a, tr.bc_b, fragment, repair_context)
                        if candidate is not None:
                            operation, strategy = "deterministic_prepass_jump", "deterministic_jump"
                elif is_struct:
                    # ordered deterministic control-flow reconstructions, each reading verbatim
                    # from GT bytecode metadata (exception table, bc_to_cft skeleton, co_positions
                    # columns) and deferring on ambiguity/content-loss: narrow try/except -> the
                    # generalized exception-region (N-stmt try/except/finally/with) -> dropped
                    # simple-if -> co_positions-driven re-nesting (re-parenting). Content-missing
                    # cases (wiped bodies, real handler bodies) defer to the LLM.
                    for _fn, _op, _strat in (
                        (try_except_candidate, "deterministic_prepass_try_except", "deterministic_try_except"),
                        (exc_regions_candidate, "deterministic_prepass_exc_regions", "deterministic_exc_regions"),
                        (control_flow_candidate, "deterministic_prepass_control_flow", "deterministic_control_flow"),
                        (control_flow_renest_candidate, "deterministic_prepass_control_flow_renest", "deterministic_control_flow_renest"),
                        # dropped `assert` and value-producing short-circuit booleans surface as
                        # "Different control flow" too — both guard their own regime internally.
                        (dropped_statement_candidate, "deterministic_prepass_dropped_statement", "deterministic_dropped_statement"),
                        (bool_shortcircuit_candidate, "deterministic_prepass_bool_shortcircuit", "deterministic_bool_shortcircuit"),
                        (ternary_op_candidate, "deterministic_prepass_ternary", "deterministic_ternary"),
                    ):
                        candidate = _fn(tr.bc_a, tr.bc_b, fragment, repair_context)
                        if candidate is not None:
                            operation, strategy = _op, _strat
                            break
                if candidate is None and is_diffbc:
                    # ordered deterministic reconstructions for "Different bytecode" (no single
                    # operand offset): spurious f-string segment -> line-too-long placeholder ->
                    # dropped scalar-literal assignment -> mangled all-constant container literal.
                    # Each reads verbatim from GT, defers on ambiguity.
                    for _fn, _op, _strat in (
                        # offset-independent leaf operand swap: PyLingual reports "Different
                        # bytecode" with no single failed_offset (~84% of residual objects),
                        # so the offset-based leaf path above never fires. Infer the lone
                        # diverging operand from the GT/derived stream diff. Oracle-gated.
                        (leaf_value_no_offset_candidate, "deterministic_prepass_leaf_no_offset", "deterministic_leaf_no_offset"),
                        (fstring_segment_candidate, "deterministic_prepass_fstring", "deterministic_fstring"),
                        (line_too_long_candidate, "deterministic_prepass_placeholder", "deterministic_placeholder"),
                        (dropped_assignment_candidate, "deterministic_prepass_dropped_assign", "deterministic_dropped_assign"),
                        (container_construction_candidate, "deterministic_prepass_container", "deterministic_container"),
                        (literal_form_candidate, "deterministic_prepass_literal_form", "deterministic_literal_form"),
                        (call_shape_candidate, "deterministic_prepass_call_shape", "deterministic_call_shape"),
                        (unary_op_candidate, "deterministic_prepass_unary", "deterministic_unary"),
                        (decorator_default_candidate, "deterministic_prepass_decorator_default", "deterministic_decorator_default"),
                        (unpack_op_candidate, "deterministic_prepass_unpack", "deterministic_unpack"),
                        (dropped_statement_candidate, "deterministic_prepass_dropped_statement", "deterministic_dropped_statement"),
                        (bool_shortcircuit_candidate, "deterministic_prepass_bool_shortcircuit", "deterministic_bool_shortcircuit"),
                        (ternary_op_candidate, "deterministic_prepass_ternary", "deterministic_ternary"),
                        (comprehension_op_candidate, "deterministic_prepass_comprehension", "deterministic_comprehension"),
                    ):
                        candidate = _fn(tr.bc_a, tr.bc_b, fragment, repair_context)
                        if candidate is not None:
                            operation, strategy = _op, _strat
                            break
                if candidate is None:
                    continue
                new_text = candidate["text"] if is_module_target else replace_source_segment(
                    text, target_row, candidate["text"]
                )
                if new_text == text:
                    continue
                step_index += 1
                next_source = prepass_dir / f"prepass{step_index}_{base_stem}.py"
                try:
                    next_source.write_text(new_text, encoding="utf-8")
                    next_pyc = compile_source_to_pyc(
                        next_source, prepass_pyc_dir / f"prepass{step_index}.pyc", target_version
                    )
                except (ReattachError, CompileError, OSError) as exc:
                    step_index -= 1
                    prepass_rejected_attempts.append(_prepass_rejection_record(
                        qualname=qualname, phase=phase, iteration=iteration_label,
                        operation=operation, strategy=strategy, candidate=candidate,
                        reason="compile_error", detail=f"{type(exc).__name__}: {exc}",
                    ))
                    continue
                candidate_verification = run_pylingual_verification(gt_pyc, next_pyc)
                delta = compare_verifications(
                    current_pylingual_verification["results"], candidate_verification["results"]
                )
                if delta.regressed or not delta.improved:
                    step_index -= 1
                    prepass_rejected_attempts.append(_prepass_rejection_record(
                        qualname=qualname, phase=phase, iteration=iteration_label,
                        operation=operation, strategy=strategy, candidate=candidate,
                        reason="regressed" if delta.regressed else "no_improvement",
                        detail=(f"regressed: {', '.join(delta.regressed)}" if delta.regressed
                                else "no object improved in the oracle lattice"),
                    ))
                    continue
                candidate_rows = compare_code_object_distances(gt_pyc, next_pyc)
                candidate_summary = summarize_results(candidate_rows)
                target_score_after = _score_snapshot(_find_distance_row(candidate_rows, qualname))
                _store_semantic_step(
                    steps,
                    {
                        "step": step_index,
                        "iteration": iteration_label,
                        "phase": phase,
                        "qualname": qualname,
                        "repair_operation": operation,
                        "selected_candidate_strategy": strategy,
                        "operator": candidate.get("operator") or candidate.get("detail"),
                        "confidence": candidate.get("confidence"),
                        "output_source": str(next_source),
                        "output_pyc": str(next_pyc),
                        "target_score_after": target_score_after,
                        "summary": candidate_summary,
                        "pylingual_verification": candidate_verification,
                        "accepted": True,
                        "acceptance_reason": f"deterministic prepass improved {', '.join(delta.improved)}",
                    },
                    log_file=log_file,
                    run_id=run_id,
                    file_hash=file_hash,
                )
                text = new_text
                current_source = next_source
                current_pyc = next_pyc
                current_summary = candidate_summary
                current_pylingual_verification = candidate_verification
                record_best_state_if_improved("deterministic prepass leaf fix")
                applied += 1
                progressed = True
                break
            if not progressed:
                break
        return applied

    preprocessing_rows = compare_code_object_distances(gt_pyc, current_pyc)
    if _module_needs_repair(preprocessing_rows) and "<module>" not in all_repair_targets:
        all_repair_targets.append("<module>")

    run_deterministic_prepass()

    target_attempt_counts: dict[str, int] = {}
    iteration = 0
    while True:
        iteration += 1
        if handle_sample_timeout_if_needed(location=f"iteration {iteration} start"):
            break
        iteration_rows = compare_code_object_distances(gt_pyc, current_pyc)
        iteration_targets = [
            qualname
            for qualname in select_repair_targets(iteration_rows, include_module=fragment_fixer is not None)
            if target_attempt_counts.get(qualname, 0) < max_iterations
            and _pylingual_target_is_equal(current_pylingual_verification, qualname) is not True
        ]
        expression_child_parent_targets = []
        skipped_expression_child_records = []
        seen_expression_child_parents: set[str] = set()
        for record in select_missing_expression_child_parent_records(iteration_rows):
            child_qualname = str(record.get("child_qualname"))
            if child_qualname in skipped_expression_child_targets:
                continue
            qualname = str(record["parent_qualname"])
            parent_score = record.get("parent_score") or {}
            parent_distance = parent_score.get("combined_distance")
            parent_pylingual_equal = _pylingual_target_is_equal(current_pylingual_verification, qualname)
            parent_pylingual_failed = parent_pylingual_equal is False
            parent_distance_nonzero = _optional_int(parent_distance, default=0) > 0
            if (
                target_attempt_counts.get(qualname, 0) < max_iterations
                and parent_pylingual_equal is not True
                and (parent_distance_nonzero or parent_pylingual_failed)
            ):
                if qualname not in seen_expression_child_parents:
                    expression_child_parent_targets.append(qualname)
                    seen_expression_child_parents.add(qualname)
                continue
            skipped_expression_child_records.append(
                {
                    **record,
                    "parent_pylingual_equal": parent_pylingual_equal,
                    "skip_reason": (
                        "parent distance is zero and parent is not explicitly PyLingual-failed"
                        if not parent_distance_nonzero and not parent_pylingual_failed
                        else "parent already exhausted or PyLingual-accepted"
                    ),
                }
            )
        oracle_failing_targets = (
            _oracle_primary_fragment_targets(
                current_pylingual_verification,
                iteration_rows,
                target_attempt_counts,
                max_iterations,
                include_module=fragment_fixer is not None,
            )
            if ACCEPTANCE_MODE == "oracle"
            else []
        )
        iteration_targets = sorted(
            set(iteration_targets + expression_child_parent_targets + oracle_failing_targets),
            key=lambda name: (_qualname_depth(name), name),
        )
        iteration_missing_targets = [
            qualname
            for qualname in select_missing_repair_targets(iteration_rows)
            if target_attempt_counts.get(qualname, 0) < max_iterations
            and _pylingual_target_is_equal(current_pylingual_verification, qualname) is not True
        ]
        iteration_extra_targets = [
            qualname
            for qualname in select_extra_repair_targets(iteration_rows)
            if target_attempt_counts.get(qualname, 0) < max_iterations
            and _pylingual_target_is_equal(current_pylingual_verification, qualname) is not True
        ]
        all_repair_targets.extend(
            qualname for qualname in iteration_targets if qualname not in all_repair_targets
        )
        all_repair_targets.extend(
            qualname for qualname in iteration_missing_targets if qualname not in all_repair_targets
        )
        all_repair_targets.extend(
            qualname for qualname in iteration_extra_targets if qualname not in all_repair_targets
        )
        all_extra_targets.extend(
            qualname for qualname in iteration_extra_targets if qualname not in all_extra_targets
        )
        if skipped_expression_child_records:
            for record in skipped_expression_child_records:
                child_qualname = str(record.get("child_qualname"))
                skipped_expression_child_targets.add(child_qualname)
                step_index += 1
                _store_semantic_step(
                    steps,
                    {
                        "step": step_index,
                        "iteration": iteration,
                        "qualname": child_qualname,
                        "repair_operation": "skip_missing_expression_child",
                        "parent_qualname": record.get("parent_qualname"),
                        "child_kind": record.get("child_kind"),
                        "target_score_before": record.get("child_score"),
                        "parent_score_before": record.get("parent_score"),
                        "parent_pylingual_equal": record.get("parent_pylingual_equal"),
                        "target_score_after": None,
                        "accepted": False,
                        "acceptance_reason": record.get("skip_reason"),
                    },
                    log_file=log_file,
                    run_id=run_id,
                    file_hash=file_hash,
                )
        if not iteration_targets and not iteration_missing_targets and not iteration_extra_targets:
            break
        accepted_this_iteration = 0
        attempted_pylingual_targets: list[str] = []
        for qualname in iteration_targets:
            if handle_sample_timeout_if_needed(location=f"iteration {iteration} before {qualname}"):
                break
            target_attempt_counts[qualname] = target_attempt_counts.get(qualname, 0) + 1
            step_index += 1
            step_before_rows = compare_code_object_distances(gt_pyc, current_pyc)
            target_score_before = _score_snapshot(_find_distance_row(step_before_rows, qualname))
            attempted_pylingual_targets.append(qualname)
            if qualname == "<module>":
                current_text = _load_text(current_source)
                current_code_objects = index_code_objects_by_qualname(current_pyc)
                current_bytecodes = index_bytecodes_by_qualname(current_pyc)
                gt_code_object = gt_code_objects.get(qualname)
                derived_code_object = current_code_objects.get(qualname)
                gt_bytecode = gt_bytecodes.get(qualname)
                derived_bytecode = current_bytecodes.get(qualname)
                if gt_code_object is None:
                    raise ReattachError(f"No ground-truth code object found for qualname: {qualname}")
                if derived_code_object is None:
                    raise ReattachError(f"No derived code object found for qualname: {qualname}")
                if gt_bytecode is None:
                    raise ReattachError(f"No ground-truth bytecode found for qualname: {qualname}")
                if derived_bytecode is None:
                    raise ReattachError(f"No derived bytecode found for qualname: {qualname}")
                if fragment_fixer is None:
                    _store_semantic_step(
                        steps,
                        {
                            "step": step_index,
                            "iteration": iteration,
                            "qualname": qualname,
                            "repair_operation": "repair_module_statement",
                            "module_body_strategy": "localized_module_statement_repair",
                            "target_score_before": target_score_before,
                            "target_score_after": None,
                            "accepted": False,
                            "acceptance_reason": "module repair requires a fragment fixer",
                        },
                        log_file=log_file,
                        run_id=run_id,
                        file_hash=file_hash,
                    )
                    continue
                try:
                    line_number = _module_failed_line(current_pylingual_verification, derived_bytecode)
                    if line_number is None:
                        raise ReattachError("module repair requires a PyLingual failed line; full-file module repair is disabled")
                    _, start_index, end_index = _find_top_level_statement_for_line(current_text, line_number)
                    extracted_before = current_text[start_index:end_index]
                    repair_context = _build_target_repair_context(
                        qualname=qualname,
                        gt_code_object=gt_bytecode,
                        derived_code_object=derived_bytecode,
                        verification=current_pylingual_verification,
                        line_number=line_number,
                        rejected_attempts=module_rejected_attempts,
                    )
                    _announce_semantic_step(
                        step_index=step_index,
                        iteration=iteration,
                        qualname=qualname,
                        repair_operation="repair_module_statement",
                        file_hash=file_hash,
                    )
                    raw_module_candidates = fragment_fixer(
                        qualname,
                        gt_bytecode,
                        derived_bytecode,
                        extracted_before,
                        repair_context,
                    )
                    # The fixer may hand back a bare string (the historical module contract) or a
                    # list of candidates (the fragment contract). `_normalize_fragment_candidates`
                    # accepts both -- a string normalizes to a single-candidate list -- so this is
                    # backward compatible while letting module repair be best-of-N. Single-shot is
                    # why 27% of module rejections on the module-only cohort are purely mechanical:
                    # one unparseable generation used to lose the whole file.
                    module_candidate_inputs = _normalize_fragment_candidates(raw_module_candidates)
                    replacement_text = str(
                        (module_candidate_inputs[0].get("text") if module_candidate_inputs else "") or ""
                    )
                    prompt_record = repair_context.get("_llm_prompt_record") if isinstance(repair_context, dict) else {}
                    if isinstance(prompt_record, dict) and prompt_record.get("skipped_due_to_token_limit"):
                        acceptance_reason = prompt_record.get("skip_reason") or "semantic repair prompt exceeded token limit"
                        skipped_attempt = target_attempt_counts[qualname]
                        module_rejected_attempts.append(
                            {
                                "attempt": skipped_attempt,
                                "localized_line_number": line_number,
                                "replacement_text": replacement_text,
                                "acceptance_reason": acceptance_reason,
                            }
                        )
                        _store_semantic_step(
                            steps,
                            {
                                "step": step_index,
                                "iteration": iteration,
                                "qualname": qualname,
                                "repair_operation": "repair_module_statement",
                                "module_body_strategy": "localized_module_statement_repair",
                                "localized_line_number": line_number,
                                "source_lineno": line_number,
                                "fragment_path": None,
                                "output_source": None,
                                "output_pyc": None,
                                "extracted_before": extracted_before,
                                "replacement_text": replacement_text,
                                "target_score_before": target_score_before,
                                "target_score_after": None,
                                "repair_context": repair_context,
                                "skipped_due_to_token_limit": True,
                                "skip_reason": acceptance_reason,
                                "prompt_token_count": prompt_record.get("prompt_token_count"),
                                "token_threshold_used": prompt_record.get("token_threshold_used"),
                                "accepted": False,
                                "acceptance_reason": acceptance_reason,
                            },
                            log_file=log_file,
                            run_id=run_id,
                            file_hash=file_hash,
                        )
                        target_attempt_counts[qualname] = max_iterations
                        continue
                    # Best-of-N: give every candidate a turn and stop at the first the ORACLE
                    # accepts. A candidate that cannot parse or compile now costs one candidate
                    # instead of the file. Acceptance itself is unchanged -- the same gate decides.
                    accepted = False
                    module_candidate_results: list[dict[str, Any]] = []
                    for module_candidate in module_candidate_inputs:
                        candidate_text = str(module_candidate.get("text") or "")
                        if not candidate_text:
                            continue
                        replacement_text = candidate_text
                        try:
                            (
                                accepted, _, next_source, next_pyc, step_summary,
                                step_pylingual_verification, step, _,
                            ) = _apply_module_statement_candidate(
                                gt_pyc=gt_pyc,
                                current_source=current_source,
                                current_pyc=current_pyc,
                                current_summary=current_summary,
                                current_pylingual_verification=current_pylingual_verification,
                                candidate_text=candidate_text,
                                line_number=line_number,
                                output_dir=output_dir,
                                pyc_dir=pyc_dir,
                                fragments_dir=fragments_dir,
                                derived_source_stem=derived_source.stem,
                                step_index=step_index,
                                iteration=iteration,
                                target_version=target_version,
                                verify_with_pylingual=verify_with_pylingual,
                                verify_each_step_with_pylingual=verify_each_step_with_pylingual,
                                reject_non_improving_candidates=reject_non_improving_candidates,
                            )
                        except (ReattachError, CompileError) as candidate_exc:
                            # A single unparseable/uncompilable candidate must not abort the
                            # remaining ones -- that is the mechanical loss we are here to remove.
                            module_candidate_results.append({
                                "candidate_index": module_candidate.get("candidate_index"),
                                "strategy": module_candidate.get("strategy"),
                                "accepted": False,
                                "acceptance_reason": f"module repair candidate unavailable: {candidate_exc}",
                            })
                            continue
                        module_candidate_results.append({
                            "candidate_index": module_candidate.get("candidate_index"),
                            "strategy": module_candidate.get("strategy"),
                            "accepted": bool(accepted),
                            "acceptance_reason": step.get("acceptance_reason"),
                        })
                        step["repair_context"] = repair_context
                        step["candidate_results"] = list(module_candidate_results)
                        step["candidate_count"] = len(module_candidate_inputs)
                        step["selected_candidate_index"] = module_candidate.get("candidate_index")
                        step["selected_candidate_strategy"] = module_candidate.get("strategy")
                        _store_semantic_step(steps, step, log_file=log_file, run_id=run_id, file_hash=file_hash)
                        if accepted:
                            accepted_this_iteration += 1
                            current_source = next_source
                            current_pyc = next_pyc
                            current_summary = step_summary
                            if step_pylingual_verification is not None:
                                current_pylingual_verification = step_pylingual_verification
                            record_best_state_if_improved(f"accepted {qualname} at step {step_index}")
                            break
                        module_rejected_attempts.append(
                            {
                                "attempt": target_attempt_counts[qualname],
                                "localized_line_number": line_number,
                                "replacement_text": candidate_text,
                                "acceptance_reason": step["acceptance_reason"],
                                "selected_action_types": _selected_action_types_from_repair_context(repair_context),
                                "target_score_before": step.get("target_score_before"),
                                "target_score_after": step.get("target_score_after"),
                                "replacement_hash": _replacement_hash(candidate_text),
                                "replacement_structural_hash": _replacement_structural_hash(candidate_text),
                                "replacement_fingerprint": _fragment_feature_snapshot(candidate_text),
                                "replacement_delta": _replacement_delta_for_attempt(extracted_before, candidate_text),
                            }
                        )
                    if not accepted and not module_candidate_results:
                        # Every candidate was empty/unusable: keep the historical ReattachError
                        # surface so the outer handler records it exactly as before.
                        raise ReattachError("module repair produced no usable candidate")
                    if not accepted and all(
                        str(r.get("acceptance_reason") or "").startswith("module repair candidate unavailable")
                        for r in module_candidate_results
                    ):
                        module_rejected_attempts.append(
                            {
                                "attempt": target_attempt_counts[qualname],
                                "localized_line_number": line_number,
                                "acceptance_reason": module_candidate_results[0]["acceptance_reason"],
                            }
                        )
                        unsupported_module_body_repair = True
                except ReattachError as exc:
                    unsupported_module_body_repair = True
                    module_rejected_attempts.append(
                        {
                            "attempt": target_attempt_counts[qualname],
                            "acceptance_reason": f"module repair candidate unavailable: {exc}",
                        }
                    )
                    _store_semantic_step(
                        steps,
                        {
                            "step": step_index,
                            "iteration": iteration,
                            "qualname": qualname,
                            "repair_operation": "repair_module_statement",
                            "module_body_strategy": "localized_module_statement_repair",
                            "target_score_before": target_score_before,
                            "target_score_after": None,
                            "accepted": False,
                            "acceptance_reason": f"module repair candidate unavailable: {exc}",
                        },
                        log_file=log_file,
                        run_id=run_id,
                        file_hash=file_hash,
                    )
                continue
            target_row = _find_target_row(current_source, current_pyc, qualname, strict_map=strict_map)
            current_text = _load_text(current_source)
            extracted_before = extract_source_segment(current_text, target_row)
            current_code_objects = index_code_objects_by_qualname(current_pyc)
            current_bytecodes = index_bytecodes_by_qualname(current_pyc)
            gt_code_object = gt_code_objects.get(qualname)
            derived_code_object = current_code_objects.get(qualname)
            gt_bytecode = gt_bytecodes.get(qualname)
            derived_bytecode = current_bytecodes.get(qualname)
            if gt_code_object is None:
                raise ReattachError(f"No ground-truth code object found for qualname: {qualname}")
            if derived_code_object is None:
                raise ReattachError(f"No derived code object found for qualname: {qualname}")
            repair_context = _build_target_repair_context(
                qualname=qualname,
                gt_code_object=gt_bytecode,
                derived_code_object=derived_bytecode,
                verification=current_pylingual_verification,
                rejected_attempts=rejected_attempts_by_qualname.get(qualname, []),
            )
            repair_context["_target_identity"] = _mapping_row_identity(target_row)
            repair_context["_target_mapping_resolution"] = target_row.get("_mapping_resolution")
            gt_target_identity = None
            if fragment_fixer is None:
                source_path, source_text = load_gt_source_text()
                gt_row = _find_target_row(source_path, gt_pyc, qualname, strict_map=strict_map)
                gt_target_identity = _mapping_row_identity(gt_row)
                gt_mapping_resolution = gt_row.get("_mapping_resolution")
                replacement_text = extract_source_segment(source_text, gt_row)
                raw_candidates = replacement_text
            else:
                gt_mapping_resolution = None
                _announce_semantic_step(
                    step_index=step_index,
                    iteration=iteration,
                    qualname=qualname,
                    repair_operation="repair_source_fragment",
                    source_kind=target_row.get("source_kind"),
                    file_hash=file_hash,
                )
                raw_candidates = fragment_fixer(
                    qualname,
                    gt_bytecode,
                    derived_bytecode,
                    extracted_before,
                    repair_context,
                )
            candidate_inputs = _normalize_fragment_candidates(raw_candidates)
            candidate_results: list[dict[str, Any]] = []
            safe_qualname = qualname.replace("<", "").replace(">", "").replace(".", "_")
            for candidate_input in candidate_inputs:
                candidate_index = int(candidate_input.get("candidate_index") or 0)
                candidate_text = str(candidate_input.get("text") or "")
                candidate_result: dict[str, Any] = {
                    "candidate_index": candidate_index,
                    "strategy": candidate_input.get("strategy"),
                    "selected_action_types": candidate_input.get("selected_action_types"),
                    "prompt_path": candidate_input.get("prompt_path"),
                    "prompt_record": candidate_input.get("prompt_record"),
                    "replacement_hash": candidate_input.get("replacement_hash"),
                    "replacement_structural_hash": candidate_input.get("replacement_structural_hash"),
                    "skipped_due_to_token_limit": candidate_input.get("skipped_due_to_token_limit"),
                    "skip_reason": candidate_input.get("skip_reason"),
                    "prompt_token_count": candidate_input.get("prompt_token_count"),
                    "token_threshold_used": candidate_input.get("token_threshold_used"),
                    "accepted": False,
                }
                if candidate_input.get("skipped_due_to_token_limit"):
                    _semantic_print(
                        (
                            f"-> skipped candidate {candidate_index} for {qualname}: "
                            f"prompt token count {candidate_input.get('prompt_token_count')} exceeds "
                            f"threshold {candidate_input.get('token_threshold_used')}; no LLM call was made"
                        ),
                        indent=2,
                        tagged=False,
                    )
                    candidate_result.update(
                        {
                            "acceptance_reason": candidate_input.get("skip_reason")
                            or "semantic repair prompt exceeded token limit",
                            "replacement_text": candidate_text,
                            "replacement_hash": _replacement_hash(candidate_text),
                            "replacement_structural_hash": _replacement_structural_hash(candidate_text),
                        }
                    )
                    candidate_results.append(candidate_result)
                    continue
                try:
                    candidate_reattachment = choose_best_reattachment(
                        current_text,
                        target_row,
                        candidate_text,
                        extracted_before,
                    )
                    candidate_replacement_text = candidate_reattachment.replacement_text
                    candidate_updated_text = candidate_reattachment.updated_source
                    candidate_source = output_dir / f"step{step_index}_cand{candidate_index}_{derived_source.stem}.py"
                    candidate_source.write_text(candidate_updated_text, encoding="utf-8")
                    candidate_pyc = _compiled_pyc_path(pyc_dir, candidate_source, target_version)
                    compile_source_to_pyc(candidate_source, candidate_pyc, target_version)
                    structure_ok, structure_reason = _validate_reattached_code_object_structure(
                        previous_pyc=current_pyc,
                        candidate_pyc=candidate_pyc,
                        qualname=qualname,
                    )

                    step_rows = compare_code_object_distances(gt_pyc, candidate_pyc)
                    candidate_summary = summarize_results(step_rows)
                    candidate_target_score_after = _score_snapshot(_find_distance_row(step_rows, qualname))
                    candidate_pylingual_verification = (
                        run_pylingual_verification(gt_pyc, candidate_pyc)
                        if verify_with_pylingual and verify_each_step_with_pylingual
                        else None
                    )
                    candidate_accepted, candidate_acceptance_reason = _should_accept_candidate(
                        current_summary,
                        candidate_summary,
                        current_pylingual_verification,
                        candidate_pylingual_verification,
                    )
                    if not reject_non_improving_candidates:
                        candidate_accepted = True
                        candidate_acceptance_reason = "candidate retained without acceptance filtering"
                    if not structure_ok:
                        candidate_accepted = False
                        candidate_acceptance_reason = structure_reason
                    candidate_result.update(
                        {
                            "accepted": candidate_accepted,
                            "acceptance_reason": candidate_acceptance_reason,
                            "replacement_text": candidate_replacement_text,
                            "replacement_hash": _replacement_hash(candidate_replacement_text),
                            "replacement_structural_hash": _replacement_structural_hash(candidate_replacement_text),
                            "updated_text": candidate_updated_text,
                            "output_source": candidate_source,
                            "output_pyc": candidate_pyc,
                            "reattachment_candidate_kind": candidate_reattachment.kind,
                            "reattachment_parse_ok": candidate_reattachment.parse_ok,
                            "reattachment_parse_error": candidate_reattachment.parse_error,
                            "reattachment_structure_ok": structure_ok,
                            "reattachment_structure_reason": structure_reason,
                            "target_score_after": candidate_target_score_after,
                            "summary": candidate_summary,
                            "pylingual_verification": candidate_pylingual_verification,
                        }
                    )
                except Exception as exc:
                    candidate_result.update(
                        {
                            "accepted": False,
                            "acceptance_reason": f"candidate could not be evaluated: {type(exc).__name__}: {exc}",
                            "compile_error": f"{type(exc).__name__}: {exc}",
                            "replacement_text": candidate_text,
                            "replacement_hash": _replacement_hash(candidate_text),
                            "replacement_structural_hash": _replacement_structural_hash(candidate_text),
                        }
                    )
                candidate_results.append(candidate_result)

            if not candidate_results:
                candidate_results.append(
                    {
                        "candidate_index": 0,
                        "accepted": False,
                        "acceptance_reason": "fragment fixer returned no candidates",
                        "replacement_text": extracted_before,
                    }
                )
            best_candidate = min(candidate_results, key=_candidate_result_rank)
            replacement_text = str(best_candidate.get("replacement_text") or extracted_before)
            fragment_path = fragments_dir / f"{step_index:02d}_{safe_qualname}.pyfrag"
            fragment_path.write_text(replacement_text, encoding="utf-8")
            next_source = best_candidate.get("output_source")
            next_pyc = best_candidate.get("output_pyc")
            step_summary = best_candidate.get("summary")
            target_score_after = best_candidate.get("target_score_after")
            step_pylingual_verification = best_candidate.get("pylingual_verification")
            accepted = bool(best_candidate.get("accepted"))
            acceptance_reason = str(best_candidate.get("acceptance_reason") or "candidate rejected")
            reattachment_candidate_kind = best_candidate.get("reattachment_candidate_kind")
            reattachment_parse_ok = best_candidate.get("reattachment_parse_ok")
            reattachment_parse_error = best_candidate.get("reattachment_parse_error")
            structure_ok = bool(best_candidate.get("reattachment_structure_ok"))
            structure_reason = best_candidate.get("reattachment_structure_reason")
            if isinstance(repair_context, dict):
                repair_context["_selected_action_pattern_types"] = best_candidate.get("selected_action_types") or []
                if isinstance(best_candidate.get("prompt_record"), dict):
                    repair_context["_llm_prompt_record"] = best_candidate["prompt_record"]
            _store_semantic_step(
                steps,
                {
                    "step": step_index,
                    "iteration": iteration,
                    "qualname": qualname,
                    "repair_operation": "repair_source_fragment",
                    "source_kind": target_row.get("source_kind"),
                    "source_lineno": int(target_row["source_lineno"]),
                    "source_col_offset": int(target_row["source_col_offset"]),
                    "source_end_lineno": int(target_row["source_end_lineno"]),
                    "source_end_col_offset": int(target_row["source_end_col_offset"]),
                    "target_identity": _mapping_row_identity(target_row),
                    "gt_target_identity": gt_target_identity,
                    "target_mapping_resolution": target_row.get("_mapping_resolution"),
                    "gt_mapping_resolution": gt_mapping_resolution,
                    "fragment_path": str(fragment_path),
                    "output_source": str(next_source) if next_source is not None else None,
                    "output_pyc": str(next_pyc) if next_pyc is not None else None,
                    "gt_code_object_name": getattr(gt_code_object, "co_name", None),
                    "derived_code_object_name": getattr(derived_code_object, "co_name", None),
                    "extracted_before": extracted_before,
                    "replacement_text": replacement_text,
                    "candidate_count": len(candidate_results),
                    "selected_candidate_index": best_candidate.get("candidate_index"),
                    "selected_candidate_strategy": best_candidate.get("strategy"),
                    "candidate_results": [_compact_candidate_result(result) for result in candidate_results],
                    "reattachment_candidate_kind": reattachment_candidate_kind,
                    "reattachment_parse_ok": reattachment_parse_ok,
                    "reattachment_parse_error": reattachment_parse_error,
                    "reattachment_structure_ok": structure_ok,
                    "reattachment_structure_reason": structure_reason,
                    "target_score_before": target_score_before,
                    "target_score_after": target_score_after,
                    "summary": step_summary,
                    "pylingual_verification": step_pylingual_verification,
                    "repair_context": repair_context,
                    "skipped_due_to_token_limit": best_candidate.get("skipped_due_to_token_limit"),
                    "skip_reason": best_candidate.get("skip_reason"),
                    "prompt_token_count": best_candidate.get("prompt_token_count"),
                    "token_threshold_used": best_candidate.get("token_threshold_used"),
                    "accepted": accepted,
                    "acceptance_reason": acceptance_reason,
                },
                log_file=log_file,
                run_id=run_id,
                file_hash=file_hash,
            )
            if accepted:
                accepted_this_iteration += 1
                if next_source is None or next_pyc is None or step_summary is None:
                    raise ReattachError("accepted candidate is missing output artifacts")
                current_source = next_source
                current_pyc = next_pyc
                current_summary = step_summary
                if step_pylingual_verification is not None:
                    current_pylingual_verification = step_pylingual_verification
                record_best_state_if_improved(f"accepted {qualname} at step {step_index}")
                if _pylingual_target_is_equal(current_pylingual_verification, qualname) is True:
                    continue
            else:
                if isinstance(repair_context, dict):
                    repair_context["_selected_action_pattern_types"] = best_candidate.get("selected_action_types") or []
                skipped_due_to_token_limit = bool(best_candidate.get("skipped_due_to_token_limit"))
                rejected_attempt = target_attempt_counts[qualname]
                candidate_residual = (
                    _capture_candidate_residual(best_candidate, qualname, gt_bytecodes)
                    if SEMANTIC_CORRECTIVE_FEEDBACK else None
                )
                _remember_rejected_attempt(
                    rejected_attempts_by_qualname,
                    qualname,
                    attempt=rejected_attempt,
                    source_text_before=extracted_before,
                    replacement_text=replacement_text,
                    acceptance_reason=acceptance_reason,
                    target_score_before=target_score_before,
                    target_score_after=target_score_after,
                    repair_context=repair_context,
                    candidate_residual=candidate_residual,
                )
                if skipped_due_to_token_limit:
                    target_attempt_counts[qualname] = max_iterations

        extra_targets = select_extra_repair_targets(compare_code_object_distances(gt_pyc, current_pyc))
        all_repair_targets.extend(
            qualname for qualname in extra_targets if qualname not in all_repair_targets
        )
        all_extra_targets.extend(
            qualname for qualname in extra_targets if qualname not in all_extra_targets
        )
        for qualname in extra_targets:
            if handle_sample_timeout_if_needed(location=f"iteration {iteration} before extra target {qualname}"):
                break
            if qualname in unsupported_extra_targets:
                continue
            if target_attempt_counts.get(qualname, 0) >= max_iterations:
                continue
            target_attempt_counts[qualname] = target_attempt_counts.get(qualname, 0) + 1
            step_index += 1
            step_before_rows = compare_code_object_distances(gt_pyc, current_pyc)
            target_score_before = _score_snapshot(_find_distance_row(step_before_rows, qualname))
            current_text = _load_text(current_source)
            try:
                target_row = _find_target_row(current_source, current_pyc, qualname, strict_map=strict_map)
            except ReattachError as exc:
                unsupported_extra_targets.add(qualname)
                _store_semantic_step(
                    steps,
                    {
                        "step": step_index,
                        "iteration": iteration,
                        "qualname": qualname,
                        "repair_operation": "delete_extra",
                        "target_score_before": target_score_before,
                        "target_score_after": None,
                        "accepted": False,
                        "acceptance_reason": f"extra source could not be mapped: {exc}",
                    },
                    log_file=log_file,
                    run_id=run_id,
                    file_hash=file_hash,
                )
                continue

            if target_row["source_kind"] not in {"function", "async_function", "class"}:
                unsupported_extra_targets.add(qualname)
                _store_semantic_step(
                    steps,
                    {
                        "step": step_index,
                        "iteration": iteration,
                        "qualname": qualname,
                        "repair_operation": "delete_extra",
                        "target_score_before": target_score_before,
                        "target_score_after": None,
                        "target_identity": _mapping_row_identity(target_row),
                        "target_mapping_resolution": target_row.get("_mapping_resolution"),
                        "accepted": False,
                        "acceptance_reason": f"extra source kind is not safely statement-deletable: {target_row['source_kind']}",
                    },
                    log_file=log_file,
                    run_id=run_id,
                    file_hash=file_hash,
                )
                continue

            extracted_before = extract_source_segment(current_text, target_row)
            fragment_path = fragments_dir / f"{step_index:02d}_{qualname.replace('<', '').replace('>', '').replace('.', '_')}_delete.pyfrag"
            fragment_path.write_text("", encoding="utf-8")

            next_source = output_dir / f"step{step_index}_{derived_source.stem}.py"
            next_pyc = _compiled_pyc_path(pyc_dir, next_source, target_version)
            compile_error: str | None = None
            deletion_strategy = "remove_source_span"
            replacement_text = ""
            updated_text = replace_source_segment(current_text, target_row, replacement_text)
            next_source.write_text(updated_text, encoding="utf-8")
            try:
                compile_source_to_pyc(next_source, next_pyc, target_version)
            except ReattachError as exc:
                compile_error = str(exc)
                deletion_strategy = "replace_source_span_with_pass"
                replacement_text = _pass_replacement_for_row(target_row)
                updated_text = replace_source_segment(current_text, target_row, replacement_text)
                next_source.write_text(updated_text, encoding="utf-8")
                try:
                    compile_source_to_pyc(next_source, next_pyc, target_version)
                    compile_error = None
                    fragment_path.write_text(replacement_text, encoding="utf-8")
                except ReattachError as fallback_exc:
                    unsupported_extra_targets.add(qualname)
                    _store_semantic_step(
                        steps,
                        {
                            "step": step_index,
                            "iteration": iteration,
                            "qualname": qualname,
                            "repair_operation": "delete_extra",
                            "deletion_strategy": deletion_strategy,
                            "fragment_path": str(fragment_path),
                            "output_source": str(next_source),
                            "output_pyc": None,
                            "extracted_before": extracted_before,
                            "replacement_text": replacement_text,
                            "target_identity": _mapping_row_identity(target_row),
                            "target_mapping_resolution": target_row.get("_mapping_resolution"),
                            "target_score_before": target_score_before,
                            "target_score_after": None,
                            "accepted": False,
                            "acceptance_reason": f"deletion candidate did not compile: {fallback_exc}",
                            "initial_compile_error": compile_error,
                        },
                        log_file=log_file,
                        run_id=run_id,
                        file_hash=file_hash,
                    )
                    continue

            step_rows = compare_code_object_distances(gt_pyc, next_pyc)
            step_summary = summarize_results(step_rows)
            target_score_after = _score_snapshot(_find_distance_row(step_rows, qualname))
            if target_score_after is None:
                target_score_after = _resolved_score_snapshot(qualname)
            step_pylingual_verification = (
                run_pylingual_verification(gt_pyc, next_pyc)
                if verify_with_pylingual and verify_each_step_with_pylingual
                else None
            )
            accepted, acceptance_reason = _should_accept_candidate(
                current_summary,
                step_summary,
                current_pylingual_verification,
                step_pylingual_verification,
            )
            if not reject_non_improving_candidates:
                accepted = True
                acceptance_reason = "candidate retained without acceptance filtering"
            _store_semantic_step(
                steps,
                {
                    "step": step_index,
                    "iteration": iteration,
                    "qualname": qualname,
                    "repair_operation": "delete_extra",
                    "deletion_strategy": deletion_strategy,
                    "source_kind": target_row.get("source_kind"),
                    "source_lineno": int(target_row["source_lineno"]),
                    "source_col_offset": int(target_row["source_col_offset"]),
                    "source_end_lineno": int(target_row["source_end_lineno"]),
                    "source_end_col_offset": int(target_row["source_end_col_offset"]),
                    "target_identity": _mapping_row_identity(target_row),
                    "target_mapping_resolution": target_row.get("_mapping_resolution"),
                    "fragment_path": str(fragment_path),
                    "output_source": str(next_source),
                    "output_pyc": str(next_pyc),
                    "extracted_before": extracted_before,
                    "replacement_text": replacement_text,
                    "target_score_before": target_score_before,
                    "target_score_after": target_score_after,
                    "summary": step_summary,
                    "pylingual_verification": step_pylingual_verification,
                    "accepted": accepted,
                    "acceptance_reason": acceptance_reason,
                },
                log_file=log_file,
                run_id=run_id,
                file_hash=file_hash,
            )
            if accepted:
                accepted_this_iteration += 1
                current_source = next_source
                current_pyc = next_pyc
                current_summary = step_summary
                if step_pylingual_verification is not None:
                    current_pylingual_verification = step_pylingual_verification
                record_best_state_if_improved(f"accepted {qualname} at step {step_index}")
        missing_targets = select_missing_repair_targets(compare_code_object_distances(gt_pyc, current_pyc))
        all_repair_targets.extend(
            qualname for qualname in missing_targets if qualname not in all_repair_targets
        )
        for qualname in missing_targets:
            if handle_sample_timeout_if_needed(location=f"iteration {iteration} before missing target {qualname}"):
                break
            if qualname in unsupported_missing_targets:
                continue
            if target_attempt_counts.get(qualname, 0) >= max_iterations:
                continue
            target_attempt_counts[qualname] = target_attempt_counts.get(qualname, 0) + 1
            step_index += 1
            step_before_rows = compare_code_object_distances(gt_pyc, current_pyc)
            target_score_before = _score_snapshot(_find_distance_row(step_before_rows, qualname))
            current_text = _load_text(current_source)
            current_code_objects = index_code_objects_by_qualname(current_pyc)
            current_bytecodes = index_bytecodes_by_qualname(current_pyc)
            gt_code_object = gt_code_objects.get(qualname)
            gt_bytecode = gt_bytecodes.get(qualname)
            if gt_code_object is None:
                raise ReattachError(f"No ground-truth code object found for missing qualname: {qualname}")

            parent_row = _find_nearest_existing_parent_row(current_source, current_pyc, qualname, strict_map)
            insertion_context = current_text
            if parent_row is not None:
                insertion_context = extract_source_segment(current_text, parent_row)
            if fragment_fixer is None:
                source_path, source_text = load_gt_source_text()
                gt_row = _find_target_row(source_path, gt_pyc, qualname, strict_map=strict_map)
                gt_target_identity = _mapping_row_identity(gt_row)
                gt_mapping_resolution = gt_row.get("_mapping_resolution")
                if gt_row["source_kind"] not in {"function", "async_function", "class"}:
                    unsupported_missing_targets.add(qualname)
                    _store_semantic_step(
                        steps,
                        {
                            "step": step_index,
                            "iteration": iteration,
                            "qualname": qualname,
                            "repair_operation": "insert_missing",
                            "source_kind": gt_row["source_kind"],
                            "source_lineno": int(gt_row["source_lineno"]),
                            "source_col_offset": int(gt_row["source_col_offset"]),
                            "source_end_lineno": int(gt_row["source_end_lineno"]),
                            "source_end_col_offset": int(gt_row["source_end_col_offset"]),
                            "gt_target_identity": gt_target_identity,
                            "gt_mapping_resolution": gt_mapping_resolution,
                            "accepted": False,
                            "acceptance_reason": f"missing source kind is not statement-insertable: {gt_row['source_kind']}",
                        },
                        log_file=log_file,
                        run_id=run_id,
                        file_hash=file_hash,
                    )
                    continue
                gt_fragment = extract_source_segment(source_text, gt_row)
                replacement_text = gt_fragment
            else:
                gt_target_identity = None
                gt_mapping_resolution = None
                repair_context = _build_target_repair_context(
                    qualname=qualname,
                    gt_code_object=gt_bytecode,
                    derived_code_object=current_bytecodes.get(qualname),
                    verification=current_pylingual_verification,
                    rejected_attempts=rejected_attempts_by_qualname.get(qualname, []),
                )
                repair_context["_parent_target_identity"] = _mapping_row_identity(parent_row)
                repair_context["_parent_mapping_resolution"] = (
                    parent_row.get("_mapping_resolution") if parent_row is not None else None
                )
                _announce_semantic_step(
                    step_index=step_index,
                    iteration=iteration,
                    qualname=qualname,
                    repair_operation="insert_missing",
                    source_kind=parent_row.get("source_kind") if parent_row is not None else None,
                    file_hash=file_hash,
                )
                replacement_text = fragment_fixer(
                    qualname,
                    gt_bytecode,
                    current_bytecodes.get(qualname),
                    insertion_context,
                    repair_context,
                )
                prompt_record = repair_context.get("_llm_prompt_record") if isinstance(repair_context, dict) else {}
                if isinstance(prompt_record, dict) and prompt_record.get("skipped_due_to_token_limit"):
                    acceptance_reason = prompt_record.get("skip_reason") or "semantic repair prompt exceeded token limit"
                    unsupported_missing_targets.add(qualname)
                    _store_semantic_step(
                        steps,
                        {
                            "step": step_index,
                            "iteration": iteration,
                            "qualname": qualname,
                            "repair_operation": "insert_missing",
                            "parent_qualname": parent_row.get("source_qualname") if parent_row is not None else "<module>",
                            "source_kind": parent_row.get("source_kind") if parent_row is not None else None,
                            "source_lineno": int(parent_row["source_lineno"]) if parent_row is not None else None,
                            "source_col_offset": int(parent_row["source_col_offset"]) if parent_row is not None else None,
                            "source_end_lineno": int(parent_row["source_end_lineno"]) if parent_row is not None else None,
                            "source_end_col_offset": int(parent_row["source_end_col_offset"]) if parent_row is not None else None,
                            "parent_target_identity": _mapping_row_identity(parent_row),
                            "gt_target_identity": gt_target_identity,
                            "parent_mapping_resolution": parent_row.get("_mapping_resolution") if parent_row is not None else None,
                            "gt_mapping_resolution": gt_mapping_resolution,
                            "fragment_path": None,
                            "output_source": None,
                            "output_pyc": None,
                            "gt_code_object_name": getattr(gt_code_object, "co_name", None),
                            "derived_code_object_name": None,
                            "extracted_before": insertion_context,
                            "replacement_text": replacement_text,
                            "target_score_before": target_score_before,
                            "target_score_after": None,
                            "repair_context": repair_context,
                            "skipped_due_to_token_limit": True,
                            "skip_reason": acceptance_reason,
                            "prompt_token_count": prompt_record.get("prompt_token_count"),
                            "token_threshold_used": prompt_record.get("token_threshold_used"),
                            "accepted": False,
                            "acceptance_reason": acceptance_reason,
                        },
                        log_file=log_file,
                        run_id=run_id,
                        file_hash=file_hash,
                    )
                    _remember_rejected_attempt(
                        rejected_attempts_by_qualname,
                        qualname,
                        attempt=target_attempt_counts[qualname],
                        source_text_before=insertion_context,
                        replacement_text=replacement_text,
                        acceptance_reason=acceptance_reason,
                        target_score_before=target_score_before,
                        target_score_after=None,
                        repair_context=repair_context,
                    )
                    continue

            updated_text, parent_qualname, insertion_indent = insert_missing_source_segment(
                current_text,
                current_source,
                current_pyc,
                qualname,
                replacement_text,
                strict_map=strict_map,
            )
            fragment_path = fragments_dir / f"{step_index:02d}_{qualname.replace('<', '').replace('>', '').replace('.', '_')}_missing.pyfrag"
            fragment_path.write_text(
                normalize_semantic_insertion_indentation(replacement_text, insertion_indent),
                encoding="utf-8",
            )

            next_source = output_dir / f"step{step_index}_{derived_source.stem}.py"
            next_source.write_text(updated_text, encoding="utf-8")
            next_pyc = _compiled_pyc_path(pyc_dir, next_source, target_version)
            compile_source_to_pyc(next_source, next_pyc, target_version)

            step_rows = compare_code_object_distances(gt_pyc, next_pyc)
            step_summary = summarize_results(step_rows)
            target_score_after = _score_snapshot(_find_distance_row(step_rows, qualname))
            step_pylingual_verification = (
                run_pylingual_verification(gt_pyc, next_pyc)
                if verify_with_pylingual and verify_each_step_with_pylingual
                else None
            )
            accepted, acceptance_reason = _should_accept_candidate(
                current_summary,
                step_summary,
                current_pylingual_verification,
                step_pylingual_verification,
            )
            if not reject_non_improving_candidates:
                accepted = True
                acceptance_reason = "candidate retained without acceptance filtering"
            _store_semantic_step(
                steps,
                {
                    "step": step_index,
                    "iteration": iteration,
                    "qualname": qualname,
                    "repair_operation": "insert_missing",
                    "parent_qualname": parent_qualname,
                    "source_kind": parent_row.get("source_kind") if parent_row is not None else None,
                    "source_lineno": int(parent_row["source_lineno"]) if parent_row is not None else None,
                    "source_col_offset": int(parent_row["source_col_offset"]) if parent_row is not None else None,
                    "source_end_lineno": int(parent_row["source_end_lineno"]) if parent_row is not None else None,
                    "source_end_col_offset": int(parent_row["source_end_col_offset"]) if parent_row is not None else None,
                    "parent_target_identity": _mapping_row_identity(parent_row),
                    "gt_target_identity": gt_target_identity,
                    "parent_mapping_resolution": parent_row.get("_mapping_resolution") if parent_row is not None else None,
                    "gt_mapping_resolution": gt_mapping_resolution,
                    "fragment_path": str(fragment_path),
                    "output_source": str(next_source),
                    "output_pyc": str(next_pyc),
                    "gt_code_object_name": getattr(gt_code_object, "co_name", None),
                    "derived_code_object_name": None,
                    "extracted_before": insertion_context,
                    "replacement_text": replacement_text,
                    "target_score_before": target_score_before,
                    "target_score_after": target_score_after,
                    "summary": step_summary,
                    "pylingual_verification": step_pylingual_verification,
                    "repair_context": repair_context if fragment_fixer is not None else None,
                    "accepted": accepted,
                    "acceptance_reason": acceptance_reason,
                },
                log_file=log_file,
                run_id=run_id,
                file_hash=file_hash,
            )
            if accepted:
                accepted_this_iteration += 1
                current_source = next_source
                current_pyc = next_pyc
                current_summary = step_summary
                if step_pylingual_verification is not None:
                    current_pylingual_verification = step_pylingual_verification
                record_best_state_if_improved(f"accepted {qualname} at step {step_index}")
            else:
                _remember_rejected_attempt(
                    rejected_attempts_by_qualname,
                    qualname,
                    attempt=target_attempt_counts[qualname],
                    source_text_before=insertion_context,
                    replacement_text=replacement_text,
                    acceptance_reason=acceptance_reason,
                    target_score_before=target_score_before,
                    target_score_after=target_score_after,
                    repair_context=repair_context if fragment_fixer is not None else None,
                )

        # POST-LLM DETERMINISTIC RECONCILIATION (the "if not perfect -> post-llm det -> test perfect
        # -> repeat" leg). After this iteration's LLM accepts, a structural edit may have exposed a
        # leaf divergence the pre-LLM prepass could not reach; re-run the oracle-gated operators on
        # the residual. Fires only when something was accepted (else the residual is byte-identical
        # to what the last pass already drove to fixpoint) and the file is not yet perfect. Wrapped
        # so an error on exotic LLM-restructured bytecode can never discard the LLM's accepted state.
        if (
            SEMANTIC_POST_LLM_DETERMINISTIC
            and ACCEPTANCE_MODE == "oracle"
            and accepted_this_iteration > 0
            and current_pylingual_verification is not None
            and not current_pylingual_verification.get("all_equal")
        ):
            try:
                _post_applied = run_deterministic_prepass(phase="post_llm", iteration_label=iteration)
            except Exception as exc:  # keep the LLM's already-accepted state on any post-pass error
                _semantic_print(f"-> post-LLM deterministic pass errored; keeping LLM state: {exc}", indent=1)
                _post_applied = 0
            post_llm_det_edits_total += _post_applied
            accepted_this_iteration += _post_applied
            if current_pylingual_verification is not None and current_pylingual_verification.get("all_equal"):
                break

        if handle_sample_timeout_if_needed(location=f"iteration {iteration} end"):
            break

        if accepted_this_iteration == 0 and not _has_retryable_pylingual_targets(current_pylingual_verification, attempted_pylingual_targets):
            break

    handle_sample_timeout_if_needed(location="finalization")

    final_rows = compare_code_object_distances(gt_pyc, current_pyc)
    final_summary = summarize_results(final_rows)
    final_missing_targets = select_unreattachable_missing_targets(final_rows)
    final_extra_targets = select_extra_repair_targets(final_rows)
    code_object_score_changes = _build_code_object_score_changes(initial_rows, final_rows)
    pylingual_verification = current_pylingual_verification
    if verify_with_pylingual and pylingual_verification is None:
        pylingual_verification = run_pylingual_verification(gt_pyc, current_pyc)

    return {
        "gt_source": None if gt_source is None else str(gt_source),
        "gt_pyc": str(gt_pyc),
        "derived_source": str(derived_source),
        "derived_pyc": str(derived_pyc),
        "target_python_version": target_version.as_str(),
        "repair_targets": all_repair_targets,
        "initial_missing_targets": initial_missing_targets,
        "final_missing_targets": final_missing_targets,
        "initial_extra_targets": initial_extra_targets,
        "final_extra_targets": final_extra_targets,
        "extra_deletion_targets": all_extra_targets,
        "unreattachable_missing_targets": initial_missing_targets,
        "final_unreattachable_missing_targets": final_missing_targets,
        "unsupported_extra_targets": sorted(unsupported_extra_targets),
        "unsupported_module_body_repair": unsupported_module_body_repair,
        "prepass_rejected_attempts": prepass_rejected_attempts,
        "max_iterations": max_iterations,
        "sample_timed_out": sample_timed_out,
        "sample_timeout_reached": sample_timeout_reached,
        "sample_hard_timeout_reached": sample_hard_timeout_reached,
        "sample_timeout_checkpoint_count": timeout_checkpoint_count,
        "sample_timeout_seconds": sample_timeout_seconds,
        "sample_hard_timeout_seconds": sample_hard_timeout_seconds,
        "sample_timeout_min_improvement_delta": sample_timeout_min_improvement_delta,
        "sample_timeout_action": sample_timeout_action,
        "sample_timeout_reason": sample_timeout_reason,
        "sample_timeout_best_improvement_reason": best_improvement_reason,
        "sample_timeout_best_combined_distance": _summary_combined_distance(best_summary),
        "sample_timeout_combined_distance_improved": (
            _combined_distance_improved(
                initial_summary,
                final_summary,
                min_delta=sample_timeout_min_improvement_delta,
            )
            if sample_timeout_reached
            else None
        ),
        "target_attempt_counts": target_attempt_counts,
        "post_llm_deterministic_enabled": SEMANTIC_POST_LLM_DETERMINISTIC and ACCEPTANCE_MODE == "oracle",
        "post_llm_deterministic_edits": post_llm_det_edits_total,
        "module_rejected_attempts": module_rejected_attempts,
        "initial_summary": initial_summary,
        "initial_pylingual_verification": initial_pylingual_verification,
        "final_source": str(current_source),
        "final_pyc": str(current_pyc),
        "final_summary": final_summary,
        "code_object_score_changes": code_object_score_changes,
        "pylingual_verification": pylingual_verification,
        "steps": steps,
    }


def main() -> int:
    args = build_parser().parse_args()

    source_path = args.source_path.expanduser().resolve()
    pyc_path = validate_input(args.pyc_path)
    target_row = _find_target_row(source_path, pyc_path, args.qualname, strict_map=args.strict_map)
    source_text = _load_text(source_path)
    extracted = extract_source_segment(source_text, target_row)

    print("Matched row:")
    print(json.dumps(target_row, indent=2))
    print("\nExtracted source:")
    print(extracted)

    replacement_text = None
    if args.replacement_file is not None and args.replacement_text is not None:
        raise ReattachError("Use only one of --replacement-file or --replacement-text")
    if args.replacement_file is not None:
        replacement_text = _load_text(args.replacement_file)
    elif args.replacement_text is not None:
        replacement_text = args.replacement_text

    if replacement_text is None:
        return 0

    if args.output_source is None:
        raise ReattachError("--output-source is required when replacing source")

    output_source = args.output_source.expanduser().resolve()
    output_source.parent.mkdir(parents=True, exist_ok=True)
    updated_text = replace_source_segment(source_text, target_row, replacement_text)
    output_source.write_text(updated_text, encoding="utf-8")
    print(f"\nUpdated source written to: {output_source}")

    target_version = _pyc_python_version(pyc_path)
    compiled_pyc = compile_source_to_pyc(output_source, args.output_pyc, target_version)
    print(f"Compiled Python {target_version} .pyc: {compiled_pyc}")

    if args.compare_pyc is not None:
        summary = run_comparison(compiled_pyc, args.compare_pyc.expanduser().resolve())
        print("\nComparison summary:")
        print(json.dumps(summary, indent=2))
        if args.comparison_json_out is not None:
            out_path = args.comparison_json_out.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Comparison JSON written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
