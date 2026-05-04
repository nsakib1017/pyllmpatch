from __future__ import annotations

import hashlib
import argparse
import ast
import json
import os
import re
import sys
import textwrap
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

from pipeline.config import now_iso
from pipeline.logging_utils import append_log
from utils.generate_bytecode import CompileError, compile_version
from utils.map_source_code_objects import MappingError, map_source_to_pyc
from utils.pyc_code_object_distance import (
    compare_code_object_distances,
    load_editable_bytecode_from_pyc,
    summarize_results,
    validate_input,
)

class ReattachError(RuntimeError):
    pass


FragmentFixer = Callable[[str, Any, Any, str, dict | None], str]


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
            "compile with Python 3.10, and optionally compare the resulting .pyc."
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


def compile_source_to_pyc(source_path: Path, output_pyc: Path | None) -> Path:
    source_path = source_path.expanduser().resolve()
    if output_pyc is None:
        pycache_dir = source_path.parent / "__pycache__"
        pycache_dir.mkdir(parents=True, exist_ok=True)
        output_pyc = pycache_dir / f"{source_path.stem}.cpython-310.pyc"
    output_pyc = output_pyc.expanduser().resolve()
    output_pyc.parent.mkdir(parents=True, exist_ok=True)
    prior_uv_cache_dir = os.environ.get("UV_CACHE_DIR")
    os.environ["UV_CACHE_DIR"] = str((Path("/tmp") / "uv-cache").resolve())
    try:
        source_text = source_path.read_text(encoding="utf-8")
        ast.parse(source_text, filename=str(source_path))
        compile_version(source_path, output_pyc, (3, 10))
    except CompileError as exc:
        raise ReattachError(f"Python 3.10 compilation failed:\n{exc}") from exc
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


def _should_accept_candidate(
    previous_summary: dict,
    candidate_summary: dict,
    previous_verification: dict | None,
    candidate_verification: dict | None,
) -> tuple[bool, str]:
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
    instruction_context = _instruction_diff_context(
        gt_code_object,
        derived_code_object,
        None if failed_offset is None else int(failed_offset),
        radius=2 if qualname != "<module>" else 3,
    )
    return {
        "target_kind": "module_statement" if qualname == "<module>" else "code_object_fragment",
        "qualname": qualname,
        "localized_line_number": line_number,
        "failed_offset": instruction_context.get("failed_offset"),
        "alignment_tag": instruction_context.get("alignment_tag"),
        "pylingual_failed_result": failed_result,
        "instruction_diff": instruction_context.get("instruction_diff"),
        "gt_instruction_range": instruction_context.get("gt_instruction_range"),
        "derived_instruction_range": instruction_context.get("derived_instruction_range"),
        "gt_instruction_window": instruction_context.get("gt_instruction_window"),
        "derived_instruction_window": instruction_context.get("derived_instruction_window"),
        "bytecode_window_radius": instruction_context.get("bytecode_window_radius"),
        "bytecode_window_max_records": instruction_context.get("bytecode_window_max_records"),
        "bytecode_window_truncated": instruction_context.get("bytecode_window_truncated"),
        "instruction_renderer_version": instruction_context.get("instruction_renderer_version"),
        "rejected_attempts": rejected_attempts[-1:],
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


def _accepted_code_object_dataset_path(log_file: Path | None) -> Path | None:
    if log_file is None:
        return None
    return log_file.expanduser().resolve().parent / "semantic_repair_accepted_code_objects.jsonl"


def _accepted_case_telemetry_path(log_file: Path | None) -> Path | None:
    if log_file is None:
        return None
    return log_file.expanduser().resolve().parent / "semantic_repair_accepted_case_telemetry.jsonl"


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
    path = _accepted_code_object_dataset_path(log_file)
    if path is None:
        return
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
    path = _accepted_case_telemetry_path(log_file)
    if path is None:
        return
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
    status = "accepted" if accepted else "rejected"
    _semantic_print(
        f"-> step {step} iter {iteration} {qualname} ({repair_operation}) -> {status} "
        f"(combined_distance {before_distance} -> {after_distance})",
        indent=1,
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
    next_pyc = pyc_dir / f"{next_source.stem}.cpython-310.pyc"
    compile_source_to_pyc(next_source, next_pyc)

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
) -> dict:
    gt_pyc = validate_input(gt_pyc)
    derived_pyc = validate_input(derived_pyc)
    derived_source = derived_source.expanduser().resolve()
    gt_source = gt_source.expanduser().resolve() if gt_source is not None else None
    gt_source_text: str | None = None

    def load_gt_source_text() -> tuple[Path, str]:
        nonlocal gt_source, gt_source_text
        if gt_source is None:
            gt_source = infer_source_from_pyc(gt_pyc)
        if gt_source_text is None:
            gt_source_text = _load_text(gt_source)
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
    steps: list[dict] = []

    all_repair_targets = list(repair_targets)
    all_extra_targets = list(initial_extra_targets)
    step_index = 0
    unsupported_missing_targets: set[str] = set()
    skipped_expression_child_targets: set[str] = set()
    unsupported_extra_targets: set[str] = set()
    unsupported_module_body_repair = False
    module_rejected_attempts: list[dict] = []

    preprocessing_rows = compare_code_object_distances(gt_pyc, current_pyc)
    if _module_needs_repair(preprocessing_rows) and "<module>" not in all_repair_targets:
        all_repair_targets.append("<module>")

    target_attempt_counts: dict[str, int] = {}
    iteration = 0
    while True:
        iteration += 1
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
        iteration_targets = sorted(
            set(iteration_targets + expression_child_parent_targets),
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
                    replacement_text = fragment_fixer(
                        qualname,
                        gt_bytecode,
                        derived_bytecode,
                        extracted_before,
                        repair_context,
                    )
                    accepted, _, next_source, next_pyc, step_summary, step_pylingual_verification, step, _ = _apply_module_statement_candidate(
                        gt_pyc=gt_pyc,
                        current_source=current_source,
                        current_pyc=current_pyc,
                        current_summary=current_summary,
                        current_pylingual_verification=current_pylingual_verification,
                        candidate_text=replacement_text,
                        line_number=line_number,
                        output_dir=output_dir,
                        pyc_dir=pyc_dir,
                        fragments_dir=fragments_dir,
                        derived_source_stem=derived_source.stem,
                        step_index=step_index,
                        iteration=iteration,
                        verify_with_pylingual=verify_with_pylingual,
                        verify_each_step_with_pylingual=verify_each_step_with_pylingual,
                        reject_non_improving_candidates=reject_non_improving_candidates,
                    )
                    step["repair_context"] = repair_context
                    _store_semantic_step(steps, step, log_file=log_file, run_id=run_id, file_hash=file_hash)
                    if accepted:
                        accepted_this_iteration += 1
                        current_source = next_source
                        current_pyc = next_pyc
                        current_summary = step_summary
                        if step_pylingual_verification is not None:
                            current_pylingual_verification = step_pylingual_verification
                    else:
                        module_rejected_attempts.append(
                            {
                                "attempt": target_attempt_counts[qualname],
                                "localized_line_number": line_number,
                                "replacement_text": replacement_text,
                                "acceptance_reason": step["acceptance_reason"],
                                "target_score_before": step.get("target_score_before"),
                                "target_score_after": step.get("target_score_after"),
                            }
                        )
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
                rejected_attempts=[],
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
                replacement_text = fragment_fixer(
                    qualname,
                    gt_bytecode,
                    derived_bytecode,
                    extracted_before,
                    repair_context,
                )
            reattachment_candidate = choose_best_reattachment(
                current_text,
                target_row,
                replacement_text,
                extracted_before,
            )
            replacement_text = reattachment_candidate.replacement_text
            fragment_path = fragments_dir / f"{step_index:02d}_{qualname.replace('<', '').replace('>', '').replace('.', '_')}.pyfrag"
            fragment_path.write_text(replacement_text, encoding="utf-8")
            updated_text = reattachment_candidate.updated_source

            next_source = output_dir / f"step{step_index}_{derived_source.stem}.py"
            next_source.write_text(updated_text, encoding="utf-8")
            next_pyc = pyc_dir / f"{next_source.stem}.cpython-310.pyc"
            compile_source_to_pyc(next_source, next_pyc)
            structure_ok, structure_reason = _validate_reattached_code_object_structure(
                previous_pyc=current_pyc,
                candidate_pyc=next_pyc,
                qualname=qualname,
            )

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
            if not structure_ok:
                accepted = False
                acceptance_reason = structure_reason
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
                    "output_source": str(next_source),
                    "output_pyc": str(next_pyc),
                    "gt_code_object_name": getattr(gt_code_object, "co_name", None),
                    "derived_code_object_name": getattr(derived_code_object, "co_name", None),
                    "extracted_before": extracted_before,
                    "replacement_text": replacement_text,
                    "reattachment_candidate_kind": reattachment_candidate.kind,
                    "reattachment_parse_ok": reattachment_candidate.parse_ok,
                    "reattachment_parse_error": reattachment_candidate.parse_error,
                    "reattachment_structure_ok": structure_ok,
                    "reattachment_structure_reason": structure_reason,
                    "target_score_before": target_score_before,
                    "target_score_after": target_score_after,
                    "summary": step_summary,
                    "pylingual_verification": step_pylingual_verification,
                    "repair_context": repair_context,
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
                if _pylingual_target_is_equal(current_pylingual_verification, qualname) is True:
                    continue

        extra_targets = select_extra_repair_targets(compare_code_object_distances(gt_pyc, current_pyc))
        all_repair_targets.extend(
            qualname for qualname in extra_targets if qualname not in all_repair_targets
        )
        all_extra_targets.extend(
            qualname for qualname in extra_targets if qualname not in all_extra_targets
        )
        for qualname in extra_targets:
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
            next_pyc = pyc_dir / f"{next_source.stem}.cpython-310.pyc"
            compile_error: str | None = None
            deletion_strategy = "remove_source_span"
            replacement_text = ""
            updated_text = replace_source_segment(current_text, target_row, replacement_text)
            next_source.write_text(updated_text, encoding="utf-8")
            try:
                compile_source_to_pyc(next_source, next_pyc)
            except ReattachError as exc:
                compile_error = str(exc)
                deletion_strategy = "replace_source_span_with_pass"
                replacement_text = _pass_replacement_for_row(target_row)
                updated_text = replace_source_segment(current_text, target_row, replacement_text)
                next_source.write_text(updated_text, encoding="utf-8")
                try:
                    compile_source_to_pyc(next_source, next_pyc)
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
        missing_targets = select_missing_repair_targets(compare_code_object_distances(gt_pyc, current_pyc))
        all_repair_targets.extend(
            qualname for qualname in missing_targets if qualname not in all_repair_targets
        )
        for qualname in missing_targets:
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
                    rejected_attempts=[],
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
            next_pyc = pyc_dir / f"{next_source.stem}.cpython-310.pyc"
            compile_source_to_pyc(next_source, next_pyc)

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

        if accepted_this_iteration == 0 and not _has_retryable_pylingual_targets(current_pylingual_verification, attempted_pylingual_targets):
            break

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
        "max_iterations": max_iterations,
        "target_attempt_counts": target_attempt_counts,
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

    compiled_pyc = compile_source_to_pyc(output_source, args.output_pyc)
    print(f"Compiled Python 3.10 .pyc: {compiled_pyc}")

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
