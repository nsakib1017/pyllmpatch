from __future__ import annotations

import argparse
import ast
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pyc_code_object_distance import load_editable_bytecode_from_pyc, validate_input


CODE_OBJECT_BASE_NAMES = {
    ast.FunctionDef: "function",
    ast.AsyncFunctionDef: "async_function",
    ast.ClassDef: "class",
    ast.Lambda: "lambda",
    ast.ListComp: "listcomp",
    ast.SetComp: "setcomp",
    ast.DictComp: "dictcomp",
    ast.GeneratorExp: "genexpr",
}


@dataclass
class SourceCodeObject:
    qualname: str
    base_name: str
    kind: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int
    scope_path: tuple[str, ...]
    occurrence_index: int
    sibling_ordinal: int
    ordinal_path: tuple[int, ...]
    immediate_child_count: int
    source_context: str


@dataclass
class PycCodeObject:
    qualname: str
    base_name: str
    firstlineno: int
    scope_path: tuple[str, ...]
    occurrence_index: int
    sibling_ordinal: int
    ordinal_path: tuple[int, ...]
    immediate_child_count: int


class MappingError(RuntimeError):
    pass


class SourceCodeObjectCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.records: list[SourceCodeObject] = []
        self.scope_stack: list[str] = []
        self.context_stack: list[str] = ["module"]
        self._occurrence_counts: dict[tuple[tuple[str, ...], str, int], int] = {}
        self._ordinal_counters: dict[tuple[str, ...], int] = {}
        self.records.append(
            SourceCodeObject(
                qualname="<module>",
                base_name="<module>",
                kind="module",
                lineno=1,
                end_lineno=1,
                col_offset=0,
                end_col_offset=0,
                scope_path=(),
                occurrence_index=0,
                sibling_ordinal=0,
                ordinal_path=(0,),
                immediate_child_count=0,
                source_context="module",
            )
        )

    def _node_start_line(self, node: ast.AST) -> int:
        decorator_lines = [decorator.lineno for decorator in getattr(node, "decorator_list", []) if hasattr(decorator, "lineno")]
        candidate_lines = decorator_lines + [getattr(node, "lineno", 1)]
        return min(candidate_lines)

    def _node_end_line(self, node: ast.AST) -> int:
        return getattr(node, "end_lineno", self._node_start_line(node))

    def _node_end_col(self, node: ast.AST) -> int:
        return getattr(node, "end_col_offset", getattr(node, "col_offset", 0))

    def _with_context(self, context: str, nodes: Iterable[ast.AST]) -> None:
        self.context_stack.append(context)
        try:
            for node in nodes:
                self.visit(node)
        finally:
            self.context_stack.pop()

    def _push_record(self, node: ast.AST, base_name: str, kind: str) -> None:
        lineno = self._node_start_line(node)
        key = (tuple(self.scope_stack), base_name, lineno)
        occurrence_index = self._occurrence_counts.get(key, 0)
        self._occurrence_counts[key] = occurrence_index + 1
        parent_scope = tuple(self.scope_stack)
        sibling_ordinal = self._ordinal_counters.get(parent_scope, 0)
        self._ordinal_counters[parent_scope] = sibling_ordinal + 1
        qualname = ".".join(("<module>", *self.scope_stack, base_name)) if self.scope_stack else f"<module>.{base_name}"
        parent_path = self._current_ordinal_path()
        self.records.append(
            SourceCodeObject(
                qualname=qualname,
                base_name=base_name,
                kind=kind,
                lineno=lineno,
                end_lineno=self._node_end_line(node),
                col_offset=getattr(node, "col_offset", 0),
                end_col_offset=self._node_end_col(node),
                scope_path=tuple(self.scope_stack),
                occurrence_index=occurrence_index,
                sibling_ordinal=sibling_ordinal,
                ordinal_path=(*parent_path, sibling_ordinal),
                immediate_child_count=0,
                source_context=self.context_stack[-1],
            )
        )

    def _current_ordinal_path(self) -> tuple[int, ...]:
        if not self.scope_stack:
            return (0,)
        current_qualname = ".".join(("<module>", *self.scope_stack))
        for record in reversed(self.records):
            if record.qualname == current_qualname:
                return record.ordinal_path
        return (0,)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push_record(node, node.name, "function")
        self._with_context("decorator", node.decorator_list)
        self._with_context("default_or_annotation", [node.args])
        if node.returns is not None:
            self._with_context("annotation", [node.returns])
        self.scope_stack.append(node.name)
        try:
            self._with_context("body", node.body)
        finally:
            self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push_record(node, node.name, "async_function")
        self._with_context("decorator", node.decorator_list)
        self._with_context("default_or_annotation", [node.args])
        if node.returns is not None:
            self._with_context("annotation", [node.returns])
        self.scope_stack.append(node.name)
        try:
            self._with_context("body", node.body)
        finally:
            self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push_record(node, node.name, "class")
        self._with_context("decorator", node.decorator_list)
        self._with_context("class_base", node.bases)
        self._with_context("class_keyword", node.keywords)
        self.scope_stack.append(node.name)
        try:
            self._with_context("body", node.body)
        finally:
            self.scope_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._push_record(node, "<lambda>", "lambda")
        self.scope_stack.append("<lambda>")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._push_record(node, "<listcomp>", "listcomp")
        self.scope_stack.append("<listcomp>")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._push_record(node, "<setcomp>", "setcomp")
        self.scope_stack.append("<setcomp>")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._push_record(node, "<dictcomp>", "dictcomp")
        self.scope_stack.append("<dictcomp>")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._push_record(node, "<genexpr>", "genexpr")
        self.scope_stack.append("<genexpr>")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_arguments(self, node: ast.arguments) -> None:
        for arg in (*node.posonlyargs, *node.args, *node.kwonlyargs):
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if node.vararg and node.vararg.annotation is not None:
            self.visit(node.vararg.annotation)
        if node.kwarg and node.kwarg.annotation is not None:
            self.visit(node.kwarg.annotation)
        for default in node.defaults:
            self.visit(default)
        for default in node.kw_defaults:
            if default is not None:
                self.visit(default)

    def visit_match_case(self, node: ast.match_case) -> None:
        pattern_nodes: list[ast.AST] = [node.pattern]
        if node.guard is not None:
            pattern_nodes.append(node.guard)
        self._with_context("match_case", pattern_nodes)
        self._with_context("body", node.body)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self._with_context("except_handler", [node.type])
        if node.name is not None and isinstance(node.name, ast.AST):
            self._with_context("except_handler", [node.name])
        self._with_context("body", node.body)

    def visit_If(self, node: ast.If) -> None:
        self._with_context("if_test", [node.test])
        self._with_context("if_body", node.body)
        self._with_context("else_body", node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self._with_context("for_header", [node.target, node.iter])
        self._with_context("for_body", node.body)
        self._with_context("else_body", node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._with_context("for_header", [node.target, node.iter])
        self._with_context("for_body", node.body)
        self._with_context("else_body", node.orelse)

    def visit_While(self, node: ast.While) -> None:
        self._with_context("while_test", [node.test])
        self._with_context("while_body", node.body)
        self._with_context("else_body", node.orelse)

    def visit_Try(self, node: ast.Try) -> None:
        self._with_context("try_body", node.body)
        for handler in node.handlers:
            self.visit(handler)
        self._with_context("else_body", node.orelse)
        self._with_context("finally_body", node.finalbody)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self.visit_Try(node)

    def visit_With(self, node: ast.With) -> None:
        items: list[ast.AST] = []
        for item in node.items:
            items.append(item.context_expr)
            if item.optional_vars is not None:
                items.append(item.optional_vars)
        self._with_context("with_header", items)
        self._with_context("with_body", node.body)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        items: list[ast.AST] = []
        for item in node.items:
            items.append(item.context_expr)
            if item.optional_vars is not None:
                items.append(item.optional_vars)
        self._with_context("with_header", items)
        self._with_context("with_body", node.body)


def collect_source_code_objects(source_path: Path) -> list[SourceCodeObject]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    collector = SourceCodeObjectCollector()
    collector.visit(tree)
    return _populate_immediate_child_counts(collector.records)


def collect_pyc_code_objects(pyc_path: Path) -> list[PycCodeObject]:
    bytecode_root = load_editable_bytecode_from_pyc(validate_input(pyc_path))
    records: list[PycCodeObject] = []
    occurrence_counts: dict[tuple[tuple[str, ...], str, int], int] = {}
    ordinal_counters: dict[tuple[str, ...], int] = {}
    ordinal_paths: dict[str, tuple[int, ...]] = {"<module>": (0,)}
    for bc in bytecode_root.iter_bytecodes():
        name_parts = tuple(bc.name.split("."))
        qualname = bc.name
        base_name = name_parts[-1]
        scope_path = name_parts[1:-1] if len(name_parts) > 1 else ()
        firstlineno = getattr(bc.codeobj, "co_firstlineno", 1)
        key = (scope_path, base_name, firstlineno)
        occurrence_index = occurrence_counts.get(key, 0)
        occurrence_counts[key] = occurrence_index + 1
        parent_qualname = ".".join(("<module>", *scope_path)) if scope_path else "<module>"
        sibling_ordinal = ordinal_counters.get(scope_path, 0)
        ordinal_counters[scope_path] = sibling_ordinal + 1
        parent_ordinal_path = ordinal_paths.get(parent_qualname, (0,))
        ordinal_path = (*parent_ordinal_path, sibling_ordinal)
        ordinal_paths[qualname] = ordinal_path
        records.append(
            PycCodeObject(
                qualname=qualname,
                base_name=base_name,
                firstlineno=firstlineno,
                scope_path=scope_path,
                occurrence_index=occurrence_index,
                sibling_ordinal=sibling_ordinal,
                ordinal_path=ordinal_path,
                immediate_child_count=0,
            )
        )
    return _populate_immediate_child_counts(records)


def _populate_immediate_child_counts(records: list[SourceCodeObject] | list[PycCodeObject]):
    child_counts: dict[tuple[int, ...], int] = {}
    for record in records:
        if len(record.ordinal_path) <= 1:
            continue
        parent_path = record.ordinal_path[:-1]
        child_counts[parent_path] = child_counts.get(parent_path, 0) + 1

    updated_records = []
    for record in records:
        updated_records.append(
            record.__class__(**{
                **record.__dict__,
                "immediate_child_count": child_counts.get(record.ordinal_path, 0),
            })
        )
    return updated_records


def _line_distance(a: int | None, b: int | None) -> int:
    if a is None or b is None:
        return 10**9
    return abs(a - b)


def _choose_fallback_candidate(
    source_record: SourceCodeObject,
    unmatched_pyc_records: list[PycCodeObject],
) -> tuple[PycCodeObject | None, str]:
    same_scope_name_and_ordinal = [
        record
        for record in unmatched_pyc_records
        if record.scope_path == source_record.scope_path
        and record.base_name == source_record.base_name
        and record.ordinal_path == source_record.ordinal_path
    ]
    if len(same_scope_name_and_ordinal) == 1:
        return same_scope_name_and_ordinal[0], "fallback(scope,name,ordinal_path)"

    same_scope_and_name = [
        record
        for record in unmatched_pyc_records
        if record.scope_path == source_record.scope_path and record.base_name == source_record.base_name
    ]
    if not same_scope_and_name:
        return None, "unmatched_source"

    same_occurrence = [
        record
        for record in same_scope_and_name
        if record.occurrence_index == source_record.occurrence_index
    ]
    if len(same_occurrence) == 1:
        return same_occurrence[0], "fallback(scope,name,occurrence)"

    same_child_shape = [
        record
        for record in same_scope_and_name
        if record.immediate_child_count == source_record.immediate_child_count
    ]
    if len(same_child_shape) == 1:
        return same_child_shape[0], "fallback(scope,name,child_count)"

    nearby_candidates = sorted(
        same_scope_and_name,
        key=lambda record: (_line_distance(record.firstlineno, source_record.lineno), record.occurrence_index),
    )
    if len(nearby_candidates) == 1:
        return nearby_candidates[0], "fallback(scope,name,nearest_line)"

    if len(nearby_candidates) >= 2:
        first_distance = _line_distance(nearby_candidates[0].firstlineno, source_record.lineno)
        second_distance = _line_distance(nearby_candidates[1].firstlineno, source_record.lineno)
        if first_distance < second_distance and first_distance <= 2:
            return nearby_candidates[0], "fallback(scope,name,nearest_line)"

    return None, "ambiguous_source"


def map_source_to_pyc(source_path: Path, pyc_path: Path, strict: bool = False) -> list[dict]:
    source_records = collect_source_code_objects(source_path)
    pyc_records = collect_pyc_code_objects(pyc_path)

    pyc_index: dict[tuple[tuple[str, ...], str, int, int], PycCodeObject] = {}
    pyc_collision_sizes: dict[tuple[tuple[str, ...], str, int], int] = {}
    for record in pyc_records:
        key = (record.scope_path, record.base_name, record.firstlineno, record.occurrence_index)
        pyc_index[key] = record
        collision_key = (record.scope_path, record.base_name, record.firstlineno)
        pyc_collision_sizes[collision_key] = pyc_collision_sizes.get(collision_key, 0) + 1

    source_collision_sizes: dict[tuple[tuple[str, ...], str, int], int] = {}
    for record in source_records:
        collision_key = (record.scope_path, record.base_name, record.lineno)
        source_collision_sizes[collision_key] = source_collision_sizes.get(collision_key, 0) + 1

    mapped_rows: list[dict] = []
    matched_pyc_keys: set[tuple[tuple[str, ...], str, int, int]] = set()
    for source_record in source_records:
        key = (
            source_record.scope_path,
            source_record.base_name,
            source_record.lineno,
            source_record.occurrence_index,
        )
        matched = pyc_index.get(key)
        match_reason = "exact(scope,name,line,occurrence)" if matched else "unmatched_source"
        if matched is None:
            unmatched_pyc_records = [
                record
                for pyc_key, record in pyc_index.items()
                if pyc_key not in matched_pyc_keys
            ]
            matched, match_reason = _choose_fallback_candidate(source_record, unmatched_pyc_records)
        if matched is not None:
            matched_key = (
                matched.scope_path,
                matched.base_name,
                matched.firstlineno,
                matched.occurrence_index,
            )
            matched_pyc_keys.add(matched_key)
        source_collision_key = (source_record.scope_path, source_record.base_name, source_record.lineno)
        pyc_collision_key = (
            matched.scope_path,
            matched.base_name,
            matched.firstlineno,
        ) if matched is not None else None
        mapped_rows.append(
            {
                "row_type": "source_to_pyc",
                "source_path": str(source_path),
                "pyc_path": str(pyc_path),
                "source_qualname": source_record.qualname,
                "source_kind": source_record.kind,
                "source_context": source_record.source_context,
                "source_lineno": source_record.lineno,
                "source_end_lineno": source_record.end_lineno,
                "source_col_offset": source_record.col_offset,
                "source_end_col_offset": source_record.end_col_offset,
                "source_occurrence_index": source_record.occurrence_index,
                "source_sibling_ordinal": source_record.sibling_ordinal,
                "source_ordinal_path": ".".join(map(str, source_record.ordinal_path)),
                "source_immediate_child_count": source_record.immediate_child_count,
                "source_collision_size": source_collision_sizes.get(source_collision_key, 1),
                "pyc_qualname": matched.qualname if matched else None,
                "pyc_firstlineno": matched.firstlineno if matched else None,
                "pyc_occurrence_index": matched.occurrence_index if matched else None,
                "pyc_sibling_ordinal": matched.sibling_ordinal if matched else None,
                "pyc_ordinal_path": ".".join(map(str, matched.ordinal_path)) if matched else None,
                "pyc_immediate_child_count": matched.immediate_child_count if matched else None,
                "pyc_collision_size": pyc_collision_sizes.get(pyc_collision_key, 1) if pyc_collision_key is not None else None,
                "match_reason": match_reason,
                "matched": matched is not None,
            }
        )

    for pyc_record in pyc_records:
        key = (
            pyc_record.scope_path,
            pyc_record.base_name,
            pyc_record.firstlineno,
            pyc_record.occurrence_index,
        )
        if key in matched_pyc_keys:
            continue
        mapped_rows.append(
            {
                "row_type": "pyc_only",
                "source_path": str(source_path),
                "pyc_path": str(pyc_path),
                "source_qualname": None,
                "source_kind": None,
                "source_context": None,
                "source_lineno": None,
                "source_end_lineno": None,
                "source_col_offset": None,
                "source_end_col_offset": None,
                "source_occurrence_index": None,
                "source_sibling_ordinal": None,
                "source_ordinal_path": None,
                "source_immediate_child_count": None,
                "source_collision_size": None,
                "pyc_qualname": pyc_record.qualname,
                "pyc_firstlineno": pyc_record.firstlineno,
                "pyc_occurrence_index": pyc_record.occurrence_index,
                "pyc_sibling_ordinal": pyc_record.sibling_ordinal,
                "pyc_ordinal_path": ".".join(map(str, pyc_record.ordinal_path)),
                "pyc_immediate_child_count": pyc_record.immediate_child_count,
                "pyc_collision_size": pyc_collision_sizes.get(
                    (pyc_record.scope_path, pyc_record.base_name, pyc_record.firstlineno),
                    1,
                ),
                "match_reason": "unmatched_pyc",
                "matched": False,
            }
        )

    if strict:
        unmatched_rows = [row for row in mapped_rows if row["matched"] is not True]
        if unmatched_rows:
            raise MappingError(
                f"Strict mode failed with {len(unmatched_rows)} unmatched row(s); "
                f"first unmatched reason={unmatched_rows[0]['match_reason']}"
            )

    return mapped_rows


def save_mapping_csv(rows: list[dict], csv_out: Path) -> None:
    csv_out = csv_out.expanduser().resolve()
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_type",
                "source_path",
                "pyc_path",
                "source_qualname",
                "source_kind",
                "source_context",
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map source-defined code objects to PyLingual editable bytecode code objects."
    )
    parser.add_argument("source_path", type=Path, help="Path to the Python source file")
    parser.add_argument("pyc_path", type=Path, help="Path to the compiled .pyc file")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any source code object is unmatched or if any unmatched pyc-only code object remains.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = map_source_to_pyc(
        args.source_path.expanduser().resolve(),
        args.pyc_path.expanduser().resolve(),
        strict=args.strict,
    )

    for row in rows:
        print(
            "\t".join(
                [
                    row["row_type"],
                    row["source_qualname"] or "None",
                    row["source_kind"] or "None",
                    str(row["source_lineno"]) if row["source_lineno"] is not None else "None",
                    str(row["source_occurrence_index"]) if row["source_occurrence_index"] is not None else "None",
                    str(row["source_collision_size"]) if row["source_collision_size"] is not None else "None",
                    row["pyc_qualname"] or "None",
                    str(row["pyc_firstlineno"]) if row["pyc_firstlineno"] is not None else "None",
                    str(row["pyc_occurrence_index"]) if row["pyc_occurrence_index"] is not None else "None",
                    str(row["pyc_collision_size"]) if row["pyc_collision_size"] is not None else "None",
                    row["match_reason"],
                    str(row["matched"]),
                ]
            )
        )

    if args.csv_out is not None:
        save_mapping_csv(rows, args.csv_out)
        print(f"\nSaved CSV to: {args.csv_out.expanduser().resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
