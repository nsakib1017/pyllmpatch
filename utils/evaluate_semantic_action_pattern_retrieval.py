from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INDEX_PATH = Path("results/semantic_repair_action_patterns/semantic_repair_action_patterns.jsonl")
DEFAULT_OUTPUT_PATH = Path("results/semantic_repair_action_patterns/retrieval_eval_summary.json")
DEFAULT_MISSES_PATH = Path("results/semantic_repair_action_patterns/retrieval_eval_misses.csv")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return bool(value)


def row_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "repair_operation": row.get("repair_operation"),
        "target_kind": row.get("target_kind"),
        "alignment_category": row.get("alignment_category"),
        "gt_only_names": row_bool(row, "gt_only_names_count"),
        "derived_only_names": row_bool(row, "derived_only_names_count"),
        "gt_only_consts": row_bool(row, "gt_only_consts_count"),
        "derived_only_consts": row_bool(row, "derived_only_consts_count"),
        "gt_only_varnames": row_bool(row, "gt_only_varnames_count"),
        "derived_only_varnames": row_bool(row, "derived_only_varnames_count"),
        "arg_shape_changed": row_bool(row, "arg_shape_changed"),
        "instruction_has_call_or_attr": row_bool(row, "instruction_has_call_or_attr"),
        "instruction_has_jump": row_bool(row, "instruction_has_jump"),
        "instruction_has_return": row_bool(row, "instruction_has_return"),
        "instruction_has_load_const": row_bool(row, "instruction_has_load_const"),
        "instruction_has_store": row_bool(row, "instruction_has_store"),
        "instruction_has_import": row_bool(row, "instruction_has_import"),
        "instruction_has_compare": row_bool(row, "instruction_has_compare"),
        "instruction_has_binary_op": row_bool(row, "instruction_has_binary_op"),
        "instruction_has_subscript": row_bool(row, "instruction_has_subscript"),
        "instruction_has_kw_names": row_bool(row, "instruction_has_kw_names"),
        "instruction_has_build_collection": row_bool(row, "instruction_has_build_collection"),
        "source_statement_types": set(str(row.get("source_shape_before") or "").split(",")) - {""},
        "has_return": row_bool(row, "source_has_return_before"),
        "has_assignment": row_bool(row, "source_has_assignment_before"),
        "has_call_or_attribute": row_bool(row, "source_has_call_or_attribute_before"),
        "has_compare": row_bool(row, "source_has_compare_before"),
        "has_boolop": row_bool(row, "source_has_boolop_before"),
        "has_subscript": row_bool(row, "source_has_subscript_before"),
        "return_value_kinds": set(str(row.get("return_value_kinds_before") or "").split(",")) - {""},
        "assign_value_kinds": set(str(row.get("assign_value_kinds_before") or "").split(",")) - {""},
        "expr_value_kinds": set(str(row.get("expr_value_kinds_before") or "").split(",")) - {""},
        "call_func_kinds": set(str(row.get("call_func_kinds_before") or "").split(",")) - {""},
        "call_arg_count": row.get("call_arg_count_before") or 0,
        "call_keyword_count": row.get("call_keyword_count_before") or 0,
    }


def score_pattern(current: dict[str, Any], pattern: dict[str, Any]) -> int | None:
    if pattern.get("repair_operation") != current.get("repair_operation"):
        return None
    if pattern.get("target_kind") and current.get("target_kind") and pattern.get("target_kind") != current.get("target_kind"):
        return None
    if (
        pattern.get("alignment_category") not in (None, "", "unknown")
        and current.get("alignment_category") not in (None, "", "unknown")
        and pattern.get("alignment_category") != current.get("alignment_category")
    ):
        return None

    score = 6
    if pattern.get("target_kind") == current.get("target_kind"):
        score += 2
    if pattern.get("alignment_category") == current.get("alignment_category"):
        score += 2

    for current_key, pattern_key in (
        ("gt_only_names", "gt_only_names_count"),
        ("derived_only_names", "derived_only_names_count"),
        ("gt_only_consts", "gt_only_consts_count"),
        ("derived_only_consts", "derived_only_consts_count"),
        ("gt_only_varnames", "gt_only_varnames_count"),
        ("derived_only_varnames", "derived_only_varnames_count"),
        ("arg_shape_changed", "arg_shape_changed"),
    ):
        current_value = bool(current.get(current_key))
        pattern_value = row_bool(pattern, pattern_key)
        if current_value and pattern_value:
            score += 3
        elif current_value != pattern_value:
            score -= 1

    for key in (
        "instruction_has_call_or_attr",
        "instruction_has_jump",
        "instruction_has_return",
        "instruction_has_load_const",
        "instruction_has_store",
        "instruction_has_import",
        "instruction_has_compare",
        "instruction_has_binary_op",
        "instruction_has_subscript",
        "instruction_has_kw_names",
        "instruction_has_build_collection",
    ):
        current_value = bool(current.get(key))
        pattern_value = row_bool(pattern, key)
        if current_value and pattern_value:
            score += 3
        elif pattern_value and not current_value:
            score -= 1

    for current_key, pattern_key in (
        ("has_return", "source_has_return_before"),
        ("has_assignment", "source_has_assignment_before"),
        ("has_call_or_attribute", "source_has_call_or_attribute_before"),
        ("has_compare", "source_has_compare_before"),
        ("has_boolop", "source_has_boolop_before"),
        ("has_subscript", "source_has_subscript_before"),
    ):
        current_value = bool(current.get(current_key))
        pattern_value = row_bool(pattern, pattern_key)
        if current_value and pattern_value:
            score += 3
        elif pattern_value and not current_value:
            score -= 2

    current_types = set(current.get("source_statement_types") or set())
    pattern_types = set(str(pattern.get("source_shape_before") or "").split(",")) - {""}
    if current_types and pattern_types:
        if current_types & pattern_types:
            score += min(3, len(current_types & pattern_types))
        else:
            score -= 2

    for current_key, pattern_key in (
        ("return_value_kinds", "return_value_kinds_before"),
        ("assign_value_kinds", "assign_value_kinds_before"),
        ("expr_value_kinds", "expr_value_kinds_before"),
        ("call_func_kinds", "call_func_kinds_before"),
    ):
        current_values = set(current.get(current_key) or set())
        pattern_values = set(str(pattern.get(pattern_key) or "").split(",")) - {""}
        if current_values and pattern_values:
            if current_values & pattern_values:
                score += min(3, len(current_values & pattern_values))
            else:
                score -= 1

    pattern_call_arg_delta = int(pattern.get("call_arg_count_delta") or 0)
    pattern_call_kw_delta = int(pattern.get("call_keyword_count_delta") or 0)
    if pattern_call_arg_delta or pattern_call_kw_delta:
        if current.get("call_arg_count") or current.get("call_keyword_count") or current.get("instruction_has_kw_names"):
            score += 2
        else:
            score -= 3

    action_type = str(pattern.get("action_type") or "")
    if action_type == "add_missing_control_flow" and not current.get("instruction_has_jump"):
        score -= 5
    if action_type == "remove_extra_control_flow" and not current.get("instruction_has_jump"):
        score -= 5
    if action_type == "restore_missing_expression_reference" and not (current.get("gt_only_names") or current.get("instruction_has_call_or_attr")):
        score -= 4
    if action_type == "remove_extra_expression_reference" and not current.get("derived_only_names"):
        score -= 4
    if action_type.startswith("adjust_return_") and not (current.get("has_return") or current.get("instruction_has_return")):
        score -= 3
    if action_type in {"adjust_call_or_attribute_expression", "adjust_call_arguments", "adjust_method_vs_attribute_access", "adjust_attribute_access"} and not (
        current.get("has_call_or_attribute") or current.get("instruction_has_call_or_attr") or current.get("call_func_kinds")
    ):
        score -= 3
    if action_type == "adjust_call_arguments" and not (current.get("call_arg_count") or current.get("call_keyword_count") or current.get("instruction_has_kw_names")):
        score -= 4
    if action_type == "adjust_comparison_expression" and not (current.get("instruction_has_compare") or "Compare" in current.get("return_value_kinds", set())):
        score -= 3
    if action_type == "adjust_boolean_expression" and not (current.get("instruction_has_jump") or "BoolOp" in current.get("return_value_kinds", set())):
        score -= 3
    if action_type == "adjust_arithmetic_or_unary_expression" and not current.get("instruction_has_binary_op"):
        score -= 2
    if action_type == "adjust_subscript_or_index_expression" and not (current.get("instruction_has_subscript") or "Subscript" in current.get("return_value_kinds", set())):
        score -= 3
    return score


def retrieve(rows: list[dict[str, Any]], target_index: int, *, top_k: int, min_score: int) -> list[tuple[int, dict[str, Any]]]:
    current = row_signature(rows[target_index])
    by_action_type: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if index == target_index:
            continue
        score = score_pattern(current, row)
        if score is not None and score >= min_score:
            action_type = str(row.get("action_type") or "")
            if action_type:
                by_action_type[action_type].append((score, row))

    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for matches in by_action_type.values():
        matches.sort(key=lambda item: item[0], reverse=True)
        best_score, best_row = matches[0]
        consensus_bonus = sum(max(0, score - min_score) for score, _ in matches[:5]) // 5
        ranked.append((best_score + consensus_bonus, best_score, best_row))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(score, row) for score, _, row in ranked[:top_k]]


def evaluate(rows: list[dict[str, Any]], *, top_k: int, min_score: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated = 0
    no_retrieval = 0
    top1_hits = 0
    topk_hits = 0
    misses: list[dict[str, Any]] = []
    by_action = Counter()
    hit_by_action = Counter()

    for index, row in enumerate(rows):
        expected = row.get("action_type")
        if not expected:
            continue
        evaluated += 1
        by_action[str(expected)] += 1
        retrieved = retrieve(rows, index, top_k=top_k, min_score=min_score)
        if not retrieved:
            no_retrieval += 1
            misses.append({"case_id": row.get("case_id"), "expected": expected, "retrieved": "", "reason": "no_retrieval"})
            continue
        retrieved_actions = [item[1].get("action_type") for item in retrieved]
        if retrieved_actions[0] == expected:
            top1_hits += 1
            hit_by_action[str(expected)] += 1
        elif expected in retrieved_actions:
            topk_hits += 1
            hit_by_action[str(expected)] += 1
        else:
            misses.append(
                {
                    "case_id": row.get("case_id"),
                    "expected": expected,
                    "retrieved": ";".join(str(action) for action in retrieved_actions),
                    "reason": "wrong_action",
                }
            )

    summary = {
        "evaluated": evaluated,
        "top_k": top_k,
        "min_score": min_score,
        "no_retrieval": no_retrieval,
        "top1_accuracy": top1_hits / evaluated if evaluated else 0,
        "topk_accuracy": (top1_hits + topk_hits) / evaluated if evaluated else 0,
        "action_counts": dict(by_action),
        "action_hit_counts": dict(hit_by_action),
    }
    return summary, misses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation for semantic repair action-pattern retrieval.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--misses", type=Path, default=DEFAULT_MISSES_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.input.expanduser().resolve())
    summary, misses = evaluate(rows, top_k=args.top_k, min_score=args.min_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with args.misses.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "expected", "retrieved", "reason"])
        writer.writeheader()
        writer.writerows(misses)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote summary: {args.output}")
    print(f"Wrote misses: {args.misses}")


if __name__ == "__main__":
    main()
