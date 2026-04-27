from __future__ import annotations

import argparse
import ast
import dis
import json
import os
import sys
import textwrap
from collections.abc import Callable
from typing import Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYLINGUAL_ROOT = REPO_ROOT / "pylingual"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYLINGUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PYLINGUAL_ROOT))

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


def _find_target_row(source_path: Path, pyc_path: Path, qualname: str, strict_map: bool) -> dict:
    rows = map_source_to_pyc(source_path, pyc_path, strict=strict_map)
    candidates = [
        row
        for row in rows
        if row["row_type"] == "source_to_pyc" and row["source_qualname"] == qualname
    ]
    if not candidates:
        raise ReattachError(f"No mapped source code object found for qualname: {qualname}")
    if len(candidates) > 1:
        raise ReattachError(f"Qualname is ambiguous across {len(candidates)} rows: {qualname}")
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


def _module_instruction_stream(code_object: Any) -> list[dict]:
    if code_object is None:
        return []
    try:
        raw_instructions = list(dis.get_instructions(code_object))
        return [
            {"opname": ins.opname, "argval": ins.argval}
            for ins in raw_instructions
        ]
    except (AttributeError, TypeError):
        from xdis.opcodes import opcode_310

        instructions = []
        extended_arg = 0
        code = getattr(code_object, "co_code", b"")
        consts = getattr(code_object, "co_consts", ())
        names = getattr(code_object, "co_names", ())
        for index in range(0, len(code), 2):
            op = code[index]
            arg = code[index + 1] | extended_arg if index + 1 < len(code) else extended_arg
            opname = opcode_310.opname[op]
            if opname == "EXTENDED_ARG":
                extended_arg = arg << 8
                continue
            extended_arg = 0
            argval = arg
            if opname == "LOAD_CONST" and arg < len(consts):
                argval = consts[arg]
            elif opname == "STORE_NAME" and arg < len(names):
                argval = names[arg]
            elif opname == "IMPORT_NAME" and arg < len(names):
                argval = names[arg]
            elif opname == "IMPORT_FROM" and arg < len(names):
                argval = names[arg]
            instructions.append({"opname": opname, "argval": argval})
        return instructions


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
    if original_fragment is not None:
        for line in original_fragment.splitlines():
            if line.strip():
                base_indent = line[: len(line) - len(line.lstrip())]
                break
    normalized = textwrap.dedent(replacement_text.strip("\n"))
    return "\n".join(
        base_indent + line if line.strip() else ""
        for line in normalized.splitlines()
    )


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
        compile_version(source_path, output_pyc, (3, 10))
    except CompileError as exc:
        raise ReattachError(f"Python 3.10 compilation failed:\n{exc}") from exc
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
    ]


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


def _module_failed_line(verification: dict | None) -> int | None:
    if verification is None:
        return None
    for result in verification.get("results", []):
        if result.get("names") == "<module>" and not result.get("success"):
            line_number = result.get("failed_line_number")
            if line_number is not None:
                return int(line_number)
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
    try:
        instructions = list(code_object) if hasattr(code_object, "instructions") else list(dis.get_instructions(code_object))
    except Exception:
        return []
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

    if failed_offset is None:
        derived_index = 0
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
    sm = __import__("difflib").SequenceMatcher(
        a=[_instruction_alignment_signature(record) for record in gt_records],
        b=[_instruction_alignment_signature(record) for record in derived_records],
    )
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

    derived_window = _instruction_window(derived_records, derived_start, derived_end, radius)
    gt_window = _instruction_window(gt_records, gt_start, gt_end, radius)
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


def _build_module_repair_context(
    *,
    gt_code_object: Any,
    derived_code_object: Any,
    verification: dict | None,
    line_number: int,
    rejected_attempts: list[dict],
) -> dict:
    failed_result = _module_failed_result(verification)
    failed_offset = None if failed_result is None else failed_result.get("failed_offset")
    instruction_context = _localized_instruction_context(
        gt_code_object,
        derived_code_object,
        None if failed_offset is None else int(failed_offset),
    )
    return {
        "target_kind": "module_statement",
        "localized_line_number": line_number,
        "pylingual_failed_result": failed_result,
        "localized_instruction_context": instruction_context,
        "rejected_attempts": rejected_attempts[-3:],
    }


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
    replacement_text = normalize_semantic_replacement_indentation(
        candidate_text,
        {"source_col_offset": int(getattr(node, "col_offset", 0))},
        extracted_before,
    )
    updated_text = current_text[:start_index] + replacement_text + current_text[end_index:]

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
    output_dir: Path | None = None,
    fragment_fixer: FragmentFixer | None = None,
    strict_map: bool = True,
    verify_with_pylingual: bool = True,
    verify_each_step_with_pylingual: bool = True,
    reject_non_improving_candidates: bool = True,
    max_iterations: int = 1,
) -> dict:
    gt_pyc = validate_input(gt_pyc)
    derived_pyc = validate_input(derived_pyc)
    derived_source = derived_source.expanduser().resolve()
    gt_source: Path | None = None
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
                    steps.append(
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
                        }
                    )
                    continue
                try:
                    line_number = _module_failed_line(current_pylingual_verification)
                    if line_number is None:
                        raise ReattachError("module repair requires a PyLingual failed line; full-file module repair is disabled")
                    _, start_index, end_index = _find_top_level_statement_for_line(current_text, line_number)
                    extracted_before = current_text[start_index:end_index]
                    repair_context = _build_module_repair_context(
                        gt_code_object=gt_bytecode,
                        derived_code_object=derived_bytecode,
                        verification=current_pylingual_verification,
                        line_number=line_number,
                        rejected_attempts=module_rejected_attempts,
                    )
                    replacement_text = fragment_fixer(
                        qualname,
                        gt_code_object,
                        derived_code_object,
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
                    steps.append(step)
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
                    steps.append(
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
                        }
                    )
                continue
            target_row = _find_target_row(current_source, current_pyc, qualname, strict_map=strict_map)
            current_text = _load_text(current_source)
            extracted_before = extract_source_segment(current_text, target_row)
            current_code_objects = index_code_objects_by_qualname(current_pyc)
            gt_code_object = gt_code_objects.get(qualname)
            derived_code_object = current_code_objects.get(qualname)
            if gt_code_object is None:
                raise ReattachError(f"No ground-truth code object found for qualname: {qualname}")
            if derived_code_object is None:
                raise ReattachError(f"No derived code object found for qualname: {qualname}")
            if fragment_fixer is None:
                source_path, source_text = load_gt_source_text()
                gt_row = _find_target_row(source_path, gt_pyc, qualname, strict_map=strict_map)
                replacement_text = extract_source_segment(source_text, gt_row)
            else:
                replacement_text = fragment_fixer(
                    qualname,
                    gt_code_object,
                    derived_code_object,
                    extracted_before,
                    None,
                )
            replacement_text = normalize_semantic_replacement_indentation(
                replacement_text,
                target_row,
                extracted_before,
            )
            fragment_path = fragments_dir / f"{step_index:02d}_{qualname.replace('<', '').replace('>', '').replace('.', '_')}.pyfrag"
            fragment_path.write_text(replacement_text, encoding="utf-8")
            updated_text = replace_source_segment(current_text, target_row, replacement_text)

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
            steps.append(
                {
                    "step": step_index,
                    "iteration": iteration,
                    "qualname": qualname,
                    "fragment_path": str(fragment_path),
                    "output_source": str(next_source),
                    "output_pyc": str(next_pyc),
                    "gt_code_object_name": getattr(gt_code_object, "co_name", None),
                    "derived_code_object_name": getattr(derived_code_object, "co_name", None),
                    "extracted_before": extracted_before,
                    "replacement_text": replacement_text,
                    "target_score_before": target_score_before,
                    "target_score_after": target_score_after,
                    "summary": step_summary,
                    "pylingual_verification": step_pylingual_verification,
                    "accepted": accepted,
                    "acceptance_reason": acceptance_reason,
                }
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
                steps.append(
                    {
                        "step": step_index,
                        "iteration": iteration,
                        "qualname": qualname,
                        "repair_operation": "delete_extra",
                        "target_score_before": target_score_before,
                        "target_score_after": None,
                        "accepted": False,
                        "acceptance_reason": f"extra source could not be mapped: {exc}",
                    }
                )
                continue

            if target_row["source_kind"] not in {"function", "async_function", "class"}:
                unsupported_extra_targets.add(qualname)
                steps.append(
                    {
                        "step": step_index,
                        "iteration": iteration,
                        "qualname": qualname,
                        "repair_operation": "delete_extra",
                        "target_score_before": target_score_before,
                        "target_score_after": None,
                        "accepted": False,
                        "acceptance_reason": f"extra source kind is not safely statement-deletable: {target_row['source_kind']}",
                    }
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
                    steps.append(
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
                            "target_score_before": target_score_before,
                            "target_score_after": None,
                            "accepted": False,
                            "acceptance_reason": f"deletion candidate did not compile: {fallback_exc}",
                            "initial_compile_error": compile_error,
                        }
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
            steps.append(
                {
                    "step": step_index,
                    "iteration": iteration,
                    "qualname": qualname,
                    "repair_operation": "delete_extra",
                    "deletion_strategy": deletion_strategy,
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
                }
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
            gt_code_object = gt_code_objects.get(qualname)
            if gt_code_object is None:
                raise ReattachError(f"No ground-truth code object found for missing qualname: {qualname}")

            parent_row = _find_nearest_existing_parent_row(current_source, current_pyc, qualname, strict_map)
            insertion_context = current_text
            if parent_row is not None:
                insertion_context = extract_source_segment(current_text, parent_row)
            if fragment_fixer is None:
                source_path, source_text = load_gt_source_text()
                gt_row = _find_target_row(source_path, gt_pyc, qualname, strict_map=strict_map)
                if gt_row["source_kind"] not in {"function", "async_function", "class"}:
                    unsupported_missing_targets.add(qualname)
                    steps.append(
                        {
                            "step": step_index,
                            "iteration": iteration,
                            "qualname": qualname,
                            "repair_operation": "insert_missing",
                            "accepted": False,
                            "acceptance_reason": f"missing source kind is not statement-insertable: {gt_row['source_kind']}",
                        }
                    )
                    continue
                gt_fragment = extract_source_segment(source_text, gt_row)
                replacement_text = gt_fragment
            else:
                replacement_text = fragment_fixer(
                    qualname,
                    gt_code_object,
                    current_code_objects.get(qualname),
                    insertion_context,
                    None,
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
            steps.append(
                {
                    "step": step_index,
                    "iteration": iteration,
                    "qualname": qualname,
                    "repair_operation": "insert_missing",
                    "parent_qualname": parent_qualname,
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
                    "accepted": accepted,
                    "acceptance_reason": acceptance_reason,
                }
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
