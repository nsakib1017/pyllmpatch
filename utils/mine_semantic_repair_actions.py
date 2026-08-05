from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/semantic_repair_action_patterns")
DEFAULT_CANDIDATE_INPUTS = [
    Path("dataset/semantic_repair_accepted_case_telemetry.jsonl"),
    Path("dataset/accepted_codeobject_mining_telemetry.jsonl"),
]


@dataclass(frozen=True)
class FragmentFeatures:
    parse_ok: bool
    statement_types: tuple[str, ...]
    control_flow_nodes: tuple[str, ...]
    loaded_names: tuple[str, ...]
    bound_names: tuple[str, ...]
    constants: tuple[str, ...]
    return_value_kinds: tuple[str, ...]
    assign_value_kinds: tuple[str, ...]
    expr_value_kinds: tuple[str, ...]
    call_func_kinds: tuple[str, ...]
    call_count: int
    call_arg_count: int
    call_keyword_count: int
    attribute_count: int
    compare_count: int
    boolop_count: int
    binop_count: int
    unaryop_count: int
    subscript_count: int
    assign_count: int
    return_count: int
    return_value_count: int
    yield_count: int
    raise_count: int
    boundary_kinds: tuple[str, ...]
    line_count: int
    char_count: int


def resolve_input_path(path: Path | None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    for candidate in DEFAULT_CANDIDATE_INPUTS:
        if candidate.exists():
            return candidate.resolve()
    candidates = ", ".join(str(path) for path in DEFAULT_CANDIDATE_INPUTS)
    raise FileNotFoundError(f"No telemetry JSONL found. Tried: {candidates}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def parse_fragment_features(fragment: str | None) -> FragmentFeatures:
    text = fragment or ""
    parsed = None
    for candidate in (text, _dedent_like_python(text)):
        try:
            parsed = ast.parse(candidate)
            break
        except SyntaxError:
            continue
    if parsed is None:
        return FragmentFeatures(
            parse_ok=False,
            statement_types=(),
            control_flow_nodes=(),
            loaded_names=(),
            bound_names=(),
            constants=(),
            return_value_kinds=(),
            assign_value_kinds=(),
            expr_value_kinds=(),
            call_func_kinds=(),
            call_count=0,
            call_arg_count=0,
            call_keyword_count=0,
            attribute_count=0,
            compare_count=0,
            boolop_count=0,
            binop_count=0,
            unaryop_count=0,
            subscript_count=0,
            assign_count=0,
            return_count=0,
            return_value_count=0,
            yield_count=0,
            raise_count=0,
            boundary_kinds=(),
            line_count=len(text.splitlines()),
            char_count=len(text),
        )

    loaded_names: set[str] = set()
    bound_names: set[str] = set()
    constants: set[str] = set()
    return_value_kinds: set[str] = set()
    assign_value_kinds: set[str] = set()
    expr_value_kinds: set[str] = set()
    call_func_kinds: set[str] = set()
    control_flow: set[str] = set()
    boundary_kinds: set[str] = set()
    call_count = 0
    call_arg_count = 0
    call_keyword_count = 0
    attribute_count = 0
    compare_count = 0
    boolop_count = 0
    binop_count = 0
    unaryop_count = 0
    subscript_count = 0
    assign_count = 0
    return_count = 0
    return_value_count = 0
    yield_count = 0
    raise_count = 0

    for node in ast.walk(parsed):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded_names.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                bound_names.add(node.id)
        elif isinstance(node, ast.Constant):
            constants.add(type(node.value).__name__)
        elif isinstance(node, ast.Call):
            call_count += 1
            call_arg_count += len(node.args)
            call_keyword_count += len(node.keywords)
            call_func_kinds.add(expr_kind(node.func))
        elif isinstance(node, ast.Attribute):
            attribute_count += 1
        elif isinstance(node, ast.Compare):
            compare_count += 1
        elif isinstance(node, ast.BoolOp):
            boolop_count += 1
        elif isinstance(node, ast.BinOp):
            binop_count += 1
        elif isinstance(node, ast.UnaryOp):
            unaryop_count += 1
        elif isinstance(node, ast.Subscript):
            subscript_count += 1
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            assign_count += 1
            assign_value = getattr(node, "value", None)
            if assign_value is not None:
                assign_value_kinds.add(expr_kind(assign_value))
        elif isinstance(node, ast.Return):
            return_count += 1
            if node.value is not None:
                return_value_count += 1
                return_value_kinds.add(expr_kind(node.value))
        elif isinstance(node, ast.Expr):
            expr_value_kinds.add(expr_kind(node.value))
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            yield_count += 1
        elif isinstance(node, ast.Raise):
            raise_count += 1
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)):
            control_flow.add(type(node).__name__)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            boundary_kinds.add(type(node).__name__)

    return FragmentFeatures(
        parse_ok=True,
        statement_types=tuple(sorted({type(node).__name__ for node in parsed.body})),
        control_flow_nodes=tuple(sorted(control_flow)),
        loaded_names=tuple(sorted(loaded_names)),
        bound_names=tuple(sorted(bound_names)),
        constants=tuple(sorted(constants)),
        return_value_kinds=tuple(sorted(return_value_kinds)),
        assign_value_kinds=tuple(sorted(assign_value_kinds)),
        expr_value_kinds=tuple(sorted(expr_value_kinds)),
        call_func_kinds=tuple(sorted(call_func_kinds)),
        call_count=call_count,
        call_arg_count=call_arg_count,
        call_keyword_count=call_keyword_count,
        attribute_count=attribute_count,
        compare_count=compare_count,
        boolop_count=boolop_count,
        binop_count=binop_count,
        unaryop_count=unaryop_count,
        subscript_count=subscript_count,
        assign_count=assign_count,
        return_count=return_count,
        return_value_count=return_value_count,
        yield_count=yield_count,
        raise_count=raise_count,
        boundary_kinds=tuple(sorted(boundary_kinds)),
        line_count=len(text.splitlines()),
        char_count=len(text),
    )


def _dedent_like_python(text: str) -> str:
    lines = text.splitlines()
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    if not indents:
        return text
    margin = min(indents)
    return "\n".join(line[margin:] if len(line) >= margin else line for line in lines)


def expr_kind(node: ast.AST | None) -> str:
    if node is None:
        return "None"
    if isinstance(node, ast.Call):
        return "Call"
    if isinstance(node, ast.Attribute):
        return "Attribute"
    if isinstance(node, ast.Name):
        return "Name"
    if isinstance(node, ast.Constant):
        return "Constant"
    if isinstance(node, ast.Compare):
        return "Compare"
    if isinstance(node, ast.BoolOp):
        return "BoolOp"
    if isinstance(node, ast.BinOp):
        return "BinOp"
    if isinstance(node, ast.UnaryOp):
        return "UnaryOp"
    if isinstance(node, ast.Subscript):
        return "Subscript"
    if isinstance(node, ast.IfExp):
        return "IfExp"
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return type(node).__name__
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return "Comprehension"
    if isinstance(node, ast.Await):
        return "Await"
    if isinstance(node, ast.Lambda):
        return "Lambda"
    return type(node).__name__


def list_delta(before: tuple[str, ...], after: tuple[str, ...]) -> tuple[list[str], list[str]]:
    before_set = set(before)
    after_set = set(after)
    return sorted(after_set - before_set), sorted(before_set - after_set)


def metadata_delta(record: dict[str, Any]) -> dict[str, Any]:
    prompt_inputs = ((record.get("prompt") or {}).get("prompt_reconstruction_inputs") or {})
    metadata_context = prompt_inputs.get("metadata_context") or {}
    gt = metadata_context.get("gt_code_object_metadata") or {}
    derived = metadata_context.get("derived_code_object_metadata") or {}

    gt_names = {str(value) for value in gt.get("co_names") or []}
    derived_names = {str(value) for value in derived.get("co_names") or []}
    gt_varnames = {str(value) for value in gt.get("co_varnames") or []}
    derived_varnames = {str(value) for value in derived.get("co_varnames") or []}
    gt_consts = {repr(value) for value in gt.get("co_consts") or []}
    derived_consts = {repr(value) for value in derived.get("co_consts") or []}

    return {
        "gt_only_names_count": len(gt_names - derived_names),
        "derived_only_names_count": len(derived_names - gt_names),
        "gt_only_consts_count": len(gt_consts - derived_consts),
        "derived_only_consts_count": len(derived_consts - gt_consts),
        "gt_only_varnames_count": len(gt_varnames - derived_varnames),
        "derived_only_varnames_count": len(derived_varnames - gt_varnames),
        "arg_shape_changed": any(
            gt.get(field) != derived.get(field)
            for field in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount")
        ),
        "flags_changed": gt.get("co_flags") != derived.get("co_flags"),
    }


def instruction_delta(record: dict[str, Any]) -> dict[str, bool]:
    prompt_inputs = ((record.get("prompt") or {}).get("prompt_reconstruction_inputs") or {})
    bytecode_context = prompt_inputs.get("bytecode_context") or {}
    text = "\n".join(
        str(value or "")
        for value in (
            bytecode_context.get("instruction_diff"),
            bytecode_context.get("gt_instruction_window"),
            bytecode_context.get("derived_instruction_window"),
        )
    ).upper()
    return {
        "instruction_has_call_or_attr": any(token in text for token in ("LOAD_METHOD", "CALL", "LOAD_ATTR", "STORE_ATTR")),
        "instruction_has_jump": "JUMP" in text or "FOR_ITER" in text,
        "instruction_has_return": "RETURN_VALUE" in text,
        "instruction_has_load_const": "LOAD_CONST" in text,
        "instruction_has_store": "STORE_" in text,
        "instruction_has_import": "IMPORT_" in text,
        "instruction_has_compare": "COMPARE_OP" in text or "IS_OP" in text or "CONTAINS_OP" in text,
        "instruction_has_binary_op": "BINARY_" in text or "BINARY_OP" in text,
        "instruction_has_subscript": "SUBSCR" in text,
        "instruction_has_kw_names": "KW_NAMES" in text or "CALL_KW" in text,
        "instruction_has_build_collection": "BUILD_LIST" in text or "BUILD_TUPLE" in text or "BUILD_MAP" in text or "BUILD_SET" in text,
    }


def source_delta(before: FragmentFeatures, after: FragmentFeatures) -> dict[str, Any]:
    names_added, names_removed = list_delta(
        tuple(sorted(set(before.loaded_names) | set(before.bound_names))),
        tuple(sorted(set(after.loaded_names) | set(after.bound_names))),
    )
    constants_added, constants_removed = list_delta(before.constants, after.constants)
    return_value_kinds_added, return_value_kinds_removed = list_delta(before.return_value_kinds, after.return_value_kinds)
    assign_value_kinds_added, assign_value_kinds_removed = list_delta(before.assign_value_kinds, after.assign_value_kinds)
    expr_value_kinds_added, expr_value_kinds_removed = list_delta(before.expr_value_kinds, after.expr_value_kinds)
    call_func_kinds_added, call_func_kinds_removed = list_delta(before.call_func_kinds, after.call_func_kinds)
    statements_added, statements_removed = list_delta(before.statement_types, after.statement_types)
    control_flow_added, control_flow_removed = list_delta(before.control_flow_nodes, after.control_flow_nodes)
    boundary_added, boundary_removed = list_delta(before.boundary_kinds, after.boundary_kinds)
    return {
        "names_added_count": len(names_added),
        "names_removed_count": len(names_removed),
        "constants_added_count": len(constants_added),
        "constants_removed_count": len(constants_removed),
        "return_value_kinds_before": ",".join(before.return_value_kinds),
        "return_value_kinds_after": ",".join(after.return_value_kinds),
        "return_value_kinds_added": return_value_kinds_added,
        "return_value_kinds_removed": return_value_kinds_removed,
        "assign_value_kinds_before": ",".join(before.assign_value_kinds),
        "assign_value_kinds_after": ",".join(after.assign_value_kinds),
        "assign_value_kinds_added": assign_value_kinds_added,
        "assign_value_kinds_removed": assign_value_kinds_removed,
        "expr_value_kinds_before": ",".join(before.expr_value_kinds),
        "expr_value_kinds_after": ",".join(after.expr_value_kinds),
        "expr_value_kinds_added": expr_value_kinds_added,
        "expr_value_kinds_removed": expr_value_kinds_removed,
        "call_func_kinds_before": ",".join(before.call_func_kinds),
        "call_func_kinds_after": ",".join(after.call_func_kinds),
        "call_func_kinds_added": call_func_kinds_added,
        "call_func_kinds_removed": call_func_kinds_removed,
        "statements_added": statements_added,
        "statements_removed": statements_removed,
        "control_flow_added": control_flow_added,
        "control_flow_removed": control_flow_removed,
        "boundary_changed": bool(boundary_added or boundary_removed),
        "call_count_delta": after.call_count - before.call_count,
        "call_count_before": before.call_count,
        "call_count_after": after.call_count,
        "call_arg_count_delta": after.call_arg_count - before.call_arg_count,
        "call_arg_count_before": before.call_arg_count,
        "call_arg_count_after": after.call_arg_count,
        "call_keyword_count_delta": after.call_keyword_count - before.call_keyword_count,
        "call_keyword_count_before": before.call_keyword_count,
        "call_keyword_count_after": after.call_keyword_count,
        "attribute_count_delta": after.attribute_count - before.attribute_count,
        "attribute_count_before": before.attribute_count,
        "attribute_count_after": after.attribute_count,
        "compare_count_delta": after.compare_count - before.compare_count,
        "compare_count_before": before.compare_count,
        "compare_count_after": after.compare_count,
        "boolop_count_delta": after.boolop_count - before.boolop_count,
        "boolop_count_before": before.boolop_count,
        "boolop_count_after": after.boolop_count,
        "binop_count_delta": after.binop_count - before.binop_count,
        "binop_count_before": before.binop_count,
        "binop_count_after": after.binop_count,
        "unaryop_count_delta": after.unaryop_count - before.unaryop_count,
        "unaryop_count_before": before.unaryop_count,
        "unaryop_count_after": after.unaryop_count,
        "subscript_count_delta": after.subscript_count - before.subscript_count,
        "subscript_count_before": before.subscript_count,
        "subscript_count_after": after.subscript_count,
        "assign_count_delta": after.assign_count - before.assign_count,
        "assign_count_before": before.assign_count,
        "assign_count_after": after.assign_count,
        "return_count_delta": after.return_count - before.return_count,
        "return_count_before": before.return_count,
        "return_count_after": after.return_count,
        "return_value_count_delta": after.return_value_count - before.return_value_count,
        "return_value_count_before": before.return_value_count,
        "return_value_count_after": after.return_value_count,
        "yield_count_delta": after.yield_count - before.yield_count,
        "raise_count_delta": after.raise_count - before.raise_count,
        "source_has_return_before": before.return_count > 0,
        "source_has_return_after": after.return_count > 0,
        "source_has_assignment_before": before.assign_count > 0,
        "source_has_assignment_after": after.assign_count > 0,
        "source_has_call_or_attribute_before": before.call_count > 0 or before.attribute_count > 0,
        "source_has_call_or_attribute_after": after.call_count > 0 or after.attribute_count > 0,
        "source_has_compare_before": before.compare_count > 0,
        "source_has_compare_after": after.compare_count > 0,
        "source_has_boolop_before": before.boolop_count > 0,
        "source_has_boolop_after": after.boolop_count > 0,
        "source_has_arithmetic_or_unary_before": before.binop_count > 0 or before.unaryop_count > 0,
        "source_has_arithmetic_or_unary_after": after.binop_count > 0 or after.unaryop_count > 0,
        "source_has_subscript_before": before.subscript_count > 0,
        "source_has_subscript_after": after.subscript_count > 0,
        "has_call_or_attribute": before.call_count > 0 or after.call_count > 0 or before.attribute_count > 0 or after.attribute_count > 0,
        "has_compare": before.compare_count > 0 or after.compare_count > 0,
        "has_boolop": before.boolop_count > 0 or after.boolop_count > 0,
        "has_arithmetic_or_unary": before.binop_count > 0 or after.binop_count > 0 or before.unaryop_count > 0 or after.unaryop_count > 0,
        "has_subscript": before.subscript_count > 0 or after.subscript_count > 0,
        "has_assignment": before.assign_count > 0 or after.assign_count > 0,
        "has_return_value": before.return_value_count > 0 or after.return_value_count > 0,
        "line_count_delta": after.line_count - before.line_count,
        "char_count_delta": after.char_count - before.char_count,
        "source_shape_before": ",".join(before.statement_types),
        "source_shape_after": ",".join(after.statement_types),
        "source_control_flow_before": ",".join(before.control_flow_nodes),
        "source_control_flow_after": ",".join(after.control_flow_nodes),
    }


def alignment_category(record: dict[str, Any]) -> str:
    tag = str((record.get("repair_context_summary") or {}).get("alignment_tag") or "").lower()
    if "insert" in tag or record.get("repair_operation") == "insert_missing":
        return "insert"
    if "delete" in tag:
        return "delete"
    if "replace" in tag:
        return "replace"
    if record.get("repair_operation") == "repair_module_statement":
        return "module_statement"
    return tag or "unknown"


def classify_action(record: dict[str, Any], md: dict[str, Any], inst: dict[str, bool], sd: dict[str, Any]) -> str:
    operation = record.get("repair_operation")
    if operation == "insert_missing":
        return "insert_missing_definition_fragment"
    if operation == "repair_module_statement":
        return "repair_top_level_module_statement"
    if sd["control_flow_added"]:
        return "add_missing_control_flow"
    if sd["control_flow_removed"]:
        return "remove_extra_control_flow"
    if sd["return_count_delta"] > 0 or sd["yield_count_delta"] > 0:
        return "add_missing_return_or_yield"
    if sd["return_count_delta"] < 0:
        return "simplify_extra_return_or_branch"
    if sd["constants_added_count"] > 0 and md["gt_only_consts_count"] > 0:
        return "restore_missing_literal_or_constant"
    if sd["constants_removed_count"] > 0 and md["derived_only_consts_count"] > 0:
        return "remove_extra_literal_or_constant"
    if sd["names_added_count"] > 0 and (md["gt_only_names_count"] > 0 or inst["instruction_has_call_or_attr"]):
        return "restore_missing_expression_reference"
    if sd["names_removed_count"] > 0 and md["derived_only_names_count"] > 0:
        return "remove_extra_expression_reference"
    if "Return" in sd["source_shape_before"] or "Return" in sd["source_shape_after"] or sd["has_return_value"]:
        return_kinds = set(str(sd.get("return_value_kinds_after") or "").split(",")) | set(str(sd.get("return_value_kinds_before") or "").split(","))
        if "Call" in return_kinds:
            return "adjust_return_call_expression"
        if "Attribute" in return_kinds:
            return "adjust_return_attribute_expression"
        if "Constant" in return_kinds:
            return "adjust_return_literal_expression"
        if return_kinds & {"BoolOp", "Compare", "IfExp"}:
            return "adjust_return_condition_expression"
        if "Subscript" in return_kinds:
            return "adjust_return_subscript_expression"
        return "adjust_return_expression"
    if sd["call_count_delta"] != 0 or sd["attribute_count_delta"] != 0:
        if sd["call_arg_count_delta"] != 0 or sd["call_keyword_count_delta"] != 0 or inst.get("instruction_has_kw_names"):
            return "adjust_call_arguments"
        if sd["call_count_delta"] != 0 and sd["attribute_count_delta"] != 0:
            return "adjust_method_vs_attribute_access"
        if sd["attribute_count_delta"] != 0:
            return "adjust_attribute_access"
        return "adjust_call_or_attribute_expression"
    if sd["compare_count_delta"] != 0:
        return "adjust_comparison_expression"
    if sd["boolop_count_delta"] != 0:
        return "adjust_boolean_expression"
    if sd["binop_count_delta"] != 0 or sd["unaryop_count_delta"] != 0:
        return "adjust_arithmetic_or_unary_expression"
    if sd["subscript_count_delta"] != 0:
        return "adjust_subscript_or_index_expression"
    if sd["assign_count_delta"] != 0 or "Assign" in sd["source_shape_before"] or "Assign" in sd["source_shape_after"]:
        assign_kinds = set(str(sd.get("assign_value_kinds_after") or "").split(",")) | set(str(sd.get("assign_value_kinds_before") or "").split(","))
        if "Call" in assign_kinds:
            return "adjust_assignment_call_value"
        if "Constant" in assign_kinds:
            return "adjust_assignment_literal_value"
        return "adjust_assignment_value_or_target"
    if sd["has_call_or_attribute"] or inst["instruction_has_call_or_attr"]:
        if sd["call_arg_count_delta"] != 0 or sd["call_keyword_count_delta"] != 0 or inst.get("instruction_has_kw_names"):
            return "adjust_call_arguments"
        if sd["attribute_count_delta"] != 0:
            return "adjust_attribute_access"
        return "adjust_call_or_attribute_expression"
    if sd["has_compare"]:
        return "adjust_comparison_expression"
    if sd["has_boolop"]:
        return "adjust_boolean_expression"
    if sd["has_arithmetic_or_unary"]:
        return "adjust_arithmetic_or_unary_expression"
    if sd["has_subscript"]:
        return "adjust_subscript_or_index_expression"
    if sd["has_assignment"]:
        return "adjust_assignment_value_or_target"
    if not sd["boundary_changed"]:
        return "preserve_structure_expression_edit"
    return "other_structural_edit"


def action_summary(action_type: str, operation: str, md: dict[str, Any], inst: dict[str, bool], sd: dict[str, Any]) -> str:
    situation_parts: list[str] = []
    if md["gt_only_names_count"] or md["gt_only_consts_count"]:
        situation_parts.append("ground-truth metadata contained names or constants absent from the derived code object")
    if md["derived_only_names_count"] or md["derived_only_consts_count"]:
        situation_parts.append("derived metadata contained extra names or constants not present in ground truth")
    if inst["instruction_has_jump"]:
        situation_parts.append("the instruction evidence involved control-flow jumps or blocks")
    if inst["instruction_has_call_or_attr"]:
        situation_parts.append("the instruction evidence involved call or attribute access operations")
    if not situation_parts:
        situation_parts.append("the derived fragment had semantic drift from the target code object")

    action_text = {
        "insert_missing_definition_fragment": "inserted only the missing definition or statement fragment without repeating parent context",
        "repair_top_level_module_statement": "localized the edit to the mismatching top-level module statement",
        "add_missing_control_flow": "added the missing local control-flow structure indicated by bytecode evidence",
        "remove_extra_control_flow": "removed or simplified extra local control-flow structure unsupported by the target bytecode",
        "add_missing_return_or_yield": "restored a missing return or yield shape while preserving the surrounding fragment",
        "simplify_extra_return_or_branch": "simplified an extra return or branch shape unsupported by the target bytecode",
        "restore_missing_literal_or_constant": "restored a missing literal or constant use with a minimal local edit",
        "remove_extra_literal_or_constant": "removed or replaced an extra literal or constant use from the derived fragment",
        "restore_missing_expression_reference": "restored a missing expression-level name, call, or attribute reference",
        "remove_extra_expression_reference": "removed or replaced an extra expression-level reference from the derived fragment",
        "adjust_return_call_expression": "adjusted a returned call expression while preserving surrounding structure",
        "adjust_return_attribute_expression": "adjusted a returned attribute expression while preserving surrounding structure",
        "adjust_return_literal_expression": "adjusted a returned literal/constant expression while preserving surrounding structure",
        "adjust_return_condition_expression": "adjusted a returned conditional/boolean expression while preserving surrounding structure",
        "adjust_return_subscript_expression": "adjusted a returned subscript/indexing expression while preserving surrounding structure",
        "adjust_call_or_attribute_expression": "adjusted call or attribute expression shape without rewriting the whole fragment",
        "adjust_call_arguments": "adjusted call argument or keyword shape without rewriting the whole fragment",
        "adjust_method_vs_attribute_access": "changed method-call versus attribute-access expression shape without rewriting the whole fragment",
        "adjust_attribute_access": "adjusted attribute-access expression shape without rewriting the whole fragment",
        "adjust_comparison_expression": "adjusted comparison expression shape while preserving surrounding structure",
        "adjust_boolean_expression": "adjusted boolean expression shape while preserving surrounding structure",
        "adjust_arithmetic_or_unary_expression": "adjusted arithmetic or unary expression shape while preserving surrounding structure",
        "adjust_subscript_or_index_expression": "adjusted subscript or indexing expression shape while preserving surrounding structure",
        "adjust_assignment_value_or_target": "adjusted an assignment target or value while preserving surrounding structure",
        "adjust_assignment_call_value": "adjusted a call expression used as an assignment value while preserving surrounding structure",
        "adjust_assignment_literal_value": "adjusted a literal/constant used as an assignment value while preserving surrounding structure",
        "adjust_return_expression": "adjusted the returned expression while preserving surrounding structure",
        "preserve_structure_expression_edit": "made an expression-level edit while preserving surrounding statement and boundary structure",
        "other_structural_edit": "changed local structure to better match the target bytecode",
    }.get(action_type, "made a local source edit that reduced semantic drift")

    constraint = "preserved function/class boundary" if not sd["boundary_changed"] else "changed a function/class boundary"
    if operation == "insert_missing":
        constraint = "returned only the missing insertable fragment"
    elif operation == "repair_module_statement":
        constraint = "kept the edit inside the selected module statement"

    return (
        f"Situation: {'; '.join(situation_parts)}. "
        f"Action that reduced drift: {action_text}. "
        f"Constraint observed: {constraint}."
    )


def prompt_action_summary(action_type: str, operation: str, md: dict[str, Any], inst: dict[str, bool], sd: dict[str, Any]) -> str:
    outer_context = "the current outer statement"
    if "Return" in sd["source_shape_before"] or "Return" in sd["source_shape_after"]:
        outer_context = "the surrounding return statement"
    elif "Assign" in sd["source_shape_before"] or "Assign" in sd["source_shape_after"]:
        outer_context = "the surrounding assignment"
    elif "Expr" in sd["source_shape_before"] or "Expr" in sd["source_shape_after"]:
        outer_context = "the surrounding expression statement"

    missing_target_data = bool(md["gt_only_names_count"] or md["gt_only_consts_count"] or md["gt_only_varnames_count"])
    extra_derived_data = bool(md["derived_only_names_count"] or md["derived_only_consts_count"] or md["derived_only_varnames_count"])
    preserve_clause = "preserve the function/class boundary"
    if operation == "insert_missing":
        preserve_clause = "return only the missing insertable fragment"
    elif operation == "repair_module_statement":
        preserve_clause = "keep the edit inside the selected top-level statement"

    if action_type in {
        "adjust_return_expression",
        "adjust_return_call_expression",
        "adjust_return_attribute_expression",
        "adjust_return_literal_expression",
        "adjust_return_condition_expression",
        "adjust_return_subscript_expression",
    }:
        detail = {
            "adjust_return_call_expression": "returned call expression",
            "adjust_return_attribute_expression": "returned attribute expression",
            "adjust_return_literal_expression": "returned literal/constant expression",
            "adjust_return_condition_expression": "returned conditional/boolean expression",
            "adjust_return_subscript_expression": "returned subscript/indexing expression",
        }.get(action_type, "returned expression")
        if missing_target_data:
            return (
                f"Keep the return statement and change only the {detail} so it accounts for missing "
                f"target-side references/constants; {preserve_clause}."
            )
        if extra_derived_data:
            return (
                f"Keep the return statement and simplify or replace only the {detail} to remove unsupported "
                f"derived-side artifacts; {preserve_clause}."
            )
        return f"Keep the return statement and adjust only its {detail} to match the bytecode; {preserve_clause}."

    if action_type in {"adjust_call_or_attribute_expression", "adjust_call_arguments", "adjust_method_vs_attribute_access", "adjust_attribute_access"}:
        if missing_target_data:
            direction = "restore a missing call, method, or attribute access"
        elif extra_derived_data:
            direction = "remove or replace unsupported call, method, or attribute access"
        else:
            direction = "match the call or attribute bytecode pattern"
        if action_type == "adjust_call_arguments":
            direction = "match the call argument or keyword-argument bytecode pattern"
        elif action_type == "adjust_method_vs_attribute_access":
            direction = "match whether the bytecode expects a method call or attribute access"
        elif action_type == "adjust_attribute_access":
            direction = "match the expected attribute-access chain"
        return f"Within {outer_context}, change only the call/attribute expression shape to {direction}; {preserve_clause}."

    if action_type == "restore_missing_expression_reference":
        return (
            f"Within {outer_context}, add the smallest expression-level reference needed to account for target-side metadata or bytecode; "
            f"avoid adding unrelated statements and {preserve_clause}."
        )

    if action_type == "remove_extra_expression_reference":
        return (
            f"Within {outer_context}, remove or replace the unsupported derived-side expression reference; "
            f"avoid restructuring surrounding code and {preserve_clause}."
        )

    if action_type == "restore_missing_literal_or_constant":
        return f"Within {outer_context}, restore the missing literal/constant use indicated by target metadata or LOAD_CONST evidence; {preserve_clause}."

    if action_type == "remove_extra_literal_or_constant":
        return f"Within {outer_context}, remove or replace the extra derived-side literal/constant unsupported by target metadata or bytecode; {preserve_clause}."

    if action_type in {"adjust_assignment_value_or_target", "adjust_assignment_call_value", "adjust_assignment_literal_value"}:
        target = {
            "adjust_assignment_call_value": "call expression used as the assignment value",
            "adjust_assignment_literal_value": "literal/constant used as the assignment value",
        }.get(action_type, "target or value")
        if missing_target_data:
            return f"Keep the assignment structure and adjust its {target} to account for missing target-side metadata; {preserve_clause}."
        if extra_derived_data:
            return f"Keep the assignment structure and adjust its {target} to remove unsupported derived-side metadata; {preserve_clause}."
        return f"Keep the assignment statement and change only the {target} needed by the bytecode; {preserve_clause}."

    if action_type == "adjust_comparison_expression":
        return f"Keep the surrounding statement and adjust only the comparison expression or comparator operands indicated by bytecode; {preserve_clause}."

    if action_type == "adjust_boolean_expression":
        return f"Keep the surrounding statement and adjust only the boolean condition/expression shape indicated by jump or truth-test bytecode; {preserve_clause}."

    if action_type == "adjust_arithmetic_or_unary_expression":
        return f"Keep the surrounding statement and adjust only the arithmetic or unary expression shape indicated by operand bytecode; {preserve_clause}."

    if action_type == "adjust_subscript_or_index_expression":
        return f"Keep the surrounding statement and adjust only the subscript/indexing expression indicated by bytecode; {preserve_clause}."

    if action_type == "add_missing_control_flow":
        return (
            "Add the smallest missing local branch/loop/block only when the target instruction window shows missing jump/control-flow behavior; "
            f"{preserve_clause}."
        )

    if action_type == "remove_extra_control_flow":
        return (
            "Remove or simplify extra local branch/loop/block structure only when the derived instruction window has unsupported jumps/control flow; "
            f"{preserve_clause}."
        )

    if action_type == "add_missing_return_or_yield":
        return f"Restore the missing return/yield shape indicated by target bytecode while keeping surrounding structure minimal; {preserve_clause}."

    if action_type == "simplify_extra_return_or_branch":
        return f"Simplify an unsupported extra return or branch path instead of rewriting the whole fragment; {preserve_clause}."

    if action_type == "insert_missing_definition_fragment":
        return "Return only the missing definition or statement fragment that should be inserted; do not repeat parent context."

    if action_type == "repair_top_level_module_statement":
        return "Repair only the selected top-level module statement; do not rewrite neighboring imports, functions, or classes."

    return f"Make the smallest expression-level edit that explains the current metadata and instruction mismatch; {preserve_clause}."


def observed_reduction_summary(
    action_type: str,
    before: FragmentFeatures,
    after: FragmentFeatures,
    md: dict[str, Any],
    inst: dict[str, bool],
    sd: dict[str, Any],
    semantic_delta: dict[str, Any],
) -> str:
    changes: list[str] = []

    if before.statement_types == after.statement_types and before.statement_types:
        changes.append("preserved the outer statement shape")
    elif sd["statements_added"] or sd["statements_removed"]:
        if sd["statements_added"]:
            changes.append("added a local statement form")
        if sd["statements_removed"]:
            changes.append("removed a local statement form")

    if before.boundary_kinds == after.boundary_kinds:
        changes.append("kept the function/class boundary unchanged")
    elif sd["boundary_changed"]:
        changes.append("changed the function/class boundary")

    expression_changes: list[str] = []
    if action_type in {
        "adjust_return_expression",
        "adjust_return_call_expression",
        "adjust_return_attribute_expression",
        "adjust_return_literal_expression",
        "adjust_return_condition_expression",
        "adjust_return_subscript_expression",
    } or sd["return_value_count_delta"] != 0:
        detail = {
            "adjust_return_call_expression": "changed the returned call expression",
            "adjust_return_attribute_expression": "changed the returned attribute expression",
            "adjust_return_literal_expression": "changed the returned literal/constant expression",
            "adjust_return_condition_expression": "changed the returned conditional/boolean expression",
            "adjust_return_subscript_expression": "changed the returned subscript/indexing expression",
        }.get(action_type, "changed the returned expression")
        expression_changes.append(detail)
    if action_type == "adjust_call_or_attribute_expression" or sd["call_count_delta"] or sd["attribute_count_delta"]:
        if sd["call_count_delta"] > 0:
            expression_changes.append("added call structure")
        elif sd["call_count_delta"] < 0:
            expression_changes.append("removed call structure")
        elif sd["attribute_count_delta"] > 0:
            expression_changes.append("added attribute-access structure")
        elif sd["attribute_count_delta"] < 0:
            expression_changes.append("removed attribute-access structure")
        else:
            expression_changes.append("changed call/attribute expression structure")
    if action_type == "adjust_assignment_value_or_target" or sd["assign_count_delta"] != 0:
        expression_changes.append("changed assignment target/value structure")
    if action_type == "adjust_comparison_expression" or sd["compare_count_delta"] != 0:
        expression_changes.append("changed comparison structure")
    if action_type == "adjust_boolean_expression" or sd["boolop_count_delta"] != 0:
        expression_changes.append("changed boolean expression structure")
    if action_type == "adjust_arithmetic_or_unary_expression" or sd["binop_count_delta"] != 0 or sd["unaryop_count_delta"] != 0:
        expression_changes.append("changed arithmetic/unary expression structure")
    if action_type == "adjust_subscript_or_index_expression" or sd["subscript_count_delta"] != 0:
        expression_changes.append("changed subscript/indexing structure")
    if action_type == "add_missing_control_flow" or sd["control_flow_added"]:
        expression_changes.append("added local control-flow structure")
    if action_type == "remove_extra_control_flow" or sd["control_flow_removed"]:
        expression_changes.append("removed local control-flow structure")
    if action_type == "add_missing_return_or_yield":
        expression_changes.append("added missing return/yield shape")
    if action_type == "simplify_extra_return_or_branch":
        expression_changes.append("simplified an extra return/branch shape")

    if sd["names_added_count"]:
        expression_changes.append("introduced expression-level references")
    if sd["names_removed_count"]:
        expression_changes.append("removed expression-level references")
    if sd["constants_added_count"]:
        expression_changes.append("introduced literal/constant usage")
    if sd["constants_removed_count"]:
        expression_changes.append("removed literal/constant usage")

    if expression_changes:
        changes.append("; ".join(dict.fromkeys(expression_changes)))

    evidence: list[str] = []
    if md["gt_only_names_count"] or md["gt_only_consts_count"] or md["gt_only_varnames_count"]:
        evidence.append("target-side metadata had missing names/constants/locals")
    if md["derived_only_names_count"] or md["derived_only_consts_count"] or md["derived_only_varnames_count"]:
        evidence.append("derived-side metadata had extra names/constants/locals")
    if inst["instruction_has_call_or_attr"]:
        evidence.append("instruction evidence involved calls or attribute access")
    if inst["instruction_has_jump"]:
        evidence.append("instruction evidence involved jumps/control flow")
    if inst["instruction_has_return"]:
        evidence.append("instruction evidence involved return behavior")
    if inst["instruction_has_load_const"]:
        evidence.append("instruction evidence involved constants")

    before_distance = (semantic_delta.get("target_score_before") or {}).get("combined_distance")
    after_distance = (semantic_delta.get("target_score_after") or {}).get("combined_distance")
    distance_delta = semantic_delta.get("combined_distance_delta")
    if after_distance == 0:
        result = "made the target code-object distance equivalent to zero"
    elif distance_delta is not None:
        result = f"reduced target combined distance by {abs(int(distance_delta))}" if int(distance_delta) < 0 else f"changed target combined distance by {distance_delta}"
    elif before_distance is not None and after_distance is not None:
        result = f"changed target combined distance from {before_distance} to {after_distance}"
    else:
        result = "was accepted as a drift-reducing repair"

    change_text = "; ".join(changes) if changes else "made a local source transformation"
    evidence_text = "; ".join(evidence) if evidence else "the accepted candidate aligned better with target bytecode/metadata"
    return f"Observed accepted change: {change_text}. Why it likely helped: {evidence_text}. Result: {result}."


def prompt_snippet(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Prior accepted repair pattern:",
            f"- Situation: {row['situation_agnostic']}",
            f"- Action that reduced drift: {row['action_agnostic']}",
            f"- Prompt guidance: {row['action_summary_prompt']}",
            f"- Constraint observed: {row['constraint_agnostic']}",
            f"- Repair variant: {row['repair_operation']}",
        ]
    )


def split_summary(summary: str) -> tuple[str, str, str]:
    situation = ""
    action = ""
    constraint = ""
    for part in summary.split(". "):
        if part.startswith("Situation: "):
            situation = part.removeprefix("Situation: ").rstrip(".")
        elif part.startswith("Action that reduced drift: "):
            action = part.removeprefix("Action that reduced drift: ").rstrip(".")
        elif part.startswith("Constraint observed: "):
            constraint = part.removeprefix("Constraint observed: ").rstrip(".")
    return situation, action, constraint


def build_row(record: dict[str, Any]) -> dict[str, Any]:
    before = parse_fragment_features(record.get("before_fragment"))
    after = parse_fragment_features(record.get("after_fragment") or (record.get("prompt") or {}).get("returned_text"))
    md = metadata_delta(record)
    inst = instruction_delta(record)
    sd = source_delta(before, after)
    action_type = classify_action(record, md, inst, sd)
    summary = action_summary(str(action_type), str(record.get("repair_operation") or ""), md, inst, sd)
    prompt_summary = prompt_action_summary(str(action_type), str(record.get("repair_operation") or ""), md, inst, sd)
    situation, action, constraint = split_summary(summary)
    semantic_delta = record.get("semantic_delta") or {}
    observed_summary = observed_reduction_summary(
        str(action_type),
        before,
        after,
        md,
        inst,
        sd,
        semantic_delta,
    )
    reattachment = record.get("reattachment") or {}
    pylingual_after = record.get("pylingual_after") or {}
    row: dict[str, Any] = {
        "case_id": record.get("case_id"),
        "source_file_hash": record.get("source_file_hash"),
        "qualname": record.get("qualname"),
        "repair_operation": record.get("repair_operation"),
        "target_kind": record.get("target_kind"),
        "alignment_category": alignment_category(record),
        "iteration": record.get("iteration"),
        "step": record.get("step"),
        "accepted": record.get("accepted"),
        "acceptance_reason": record.get("acceptance_reason"),
        "combined_distance_delta": semantic_delta.get("combined_distance_delta"),
        "before_combined_distance": (semantic_delta.get("target_score_before") or {}).get("combined_distance"),
        "after_combined_distance": (semantic_delta.get("target_score_after") or {}).get("combined_distance"),
        "reattachment_candidate_kind": reattachment.get("candidate_kind"),
        "reattachment_structure_reason": reattachment.get("structure_reason"),
        "pylingual_all_equal": pylingual_after.get("all_equal"),
        "pylingual_success_count": pylingual_after.get("success_count"),
        "action_type": action_type,
        "action_summary_agnostic": summary,
        "action_summary_prompt": prompt_summary,
        "observed_reduction_summary": observed_summary,
        "situation_agnostic": situation,
        "action_agnostic": action,
        "constraint_agnostic": constraint,
        "before_parse_ok": before.parse_ok,
        "after_parse_ok": after.parse_ok,
        **md,
        **inst,
        **sd,
    }
    row["prompt_snippet_agnostic"] = prompt_snippet(row)
    return row


def flatten_for_output(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_excel(path: Path, rows: list[dict[str, Any]]) -> None:
    accepted_df = pd.DataFrame([{key: flatten_for_output(value) for key, value in row.items()} for row in rows])
    if accepted_df.empty:
        accepted_df = pd.DataFrame(columns=["case_id", "action_type", "action_summary_agnostic"])

    summary_records: list[dict[str, Any]] = []
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[str(row.get("action_type"))].append(row)
    for action_type, action_rows in sorted(by_action.items()):
        deltas = [
            row.get("combined_distance_delta")
            for row in action_rows
            if isinstance(row.get("combined_distance_delta"), (int, float))
        ]
        summary_records.append(
            {
                "action_type": action_type,
                "count": len(action_rows),
                "mean_combined_distance_delta": sum(deltas) / len(deltas) if deltas else None,
                "best_combined_distance_delta": min(deltas) if deltas else None,
                "example_summary": action_rows[0].get("action_summary_agnostic"),
                "example_prompt_guidance": action_rows[0].get("action_summary_prompt"),
            }
        )
    summary_df = pd.DataFrame(summary_records)

    best_rows = sorted(
        rows,
        key=lambda row: row.get("combined_distance_delta")
        if isinstance(row.get("combined_distance_delta"), (int, float))
        else 10**9,
    )[:200]
    best_df = pd.DataFrame([{key: flatten_for_output(value) for key, value in row.items()} for row in best_rows])

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        accepted_df.to_excel(writer, sheet_name="accepted_actions", index=False)
        summary_df.to_excel(writer, sheet_name="action_summary", index=False)
        best_df.to_excel(writer, sheet_name="best_drift_reductions", index=False)
        workbook = writer.book
        wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
        sheet_columns = {
            "accepted_actions": max(0, len(accepted_df.columns) - 1),
            "action_summary": max(0, len(summary_df.columns) - 1),
            "best_drift_reductions": max(0, len(best_df.columns) - 1),
        }
        for sheet_name in ("accepted_actions", "action_summary", "best_drift_reductions"):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, 0, sheet_columns[sheet_name])
            worksheet.set_column(0, 12, 18)
            worksheet.set_column(13, 30, 24)
            worksheet.set_column(31, 80, 34, wrap)


def write_action_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("action_type")) for row in rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action_type", "count"])
        writer.writeheader()
        for action_type, count in counts.most_common():
            writer.writerow({"action_type": action_type, "count": count})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine accepted semantic-repair telemetry into agnostic action patterns.")
    parser.add_argument("--input", type=Path, default=None, help="Accepted telemetry JSONL path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional max accepted records to process.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = resolve_input_path(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [record for record in read_jsonl(input_path) if record.get("accepted")]
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]
    rows = [build_row(record) for record in records]

    xlsx_path = output_dir / "semantic_repair_action_patterns.xlsx"
    jsonl_path = output_dir / "semantic_repair_action_patterns.jsonl"
    csv_path = output_dir / "semantic_repair_action_type_counts.csv"

    write_excel(xlsx_path, rows)
    write_jsonl(jsonl_path, rows)
    write_action_summary_csv(csv_path, rows)

    print(f"Input telemetry: {input_path}")
    print(f"Accepted records processed: {len(rows)}")
    print(f"Wrote Excel: {xlsx_path}")
    print(f"Wrote JSONL: {jsonl_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
