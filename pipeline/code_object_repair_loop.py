from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
import textwrap
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import (
    ACCEPTED_CODE_OBJECT_TELEMETRY_PATH,
    BASE_DATASET_PATH,
    build_run_paths,
    current_run_timestamp,
    now_iso,
)
from pipeline.logging_utils import append_log
from utils.file_helpers import fetch_pyllmpatch_repair_paths, fetch_pyllmpatch_source_path, strip_code_fences
from utils.providers import find_llm_config, make_llm_call_from_config
from utils.token_helpers import count_tokens_safe
from utils.reattach_source_code_object import (
    SemanticRepairHardTimeoutNoImprovement,
    SemanticRepairTimeoutNoImprovement,
    _find_target_row,
    compare_code_object_distances,
    extract_source_segment,
    infer_source_from_pyc,
    repair_mismatching_code_objects,
    select_extra_repair_targets,
    select_missing_expression_child_parent_records,
    select_repair_targets,
    select_unreattachable_missing_targets,
    summarize_results,
)

PYLINGUAL_ROOT = REPO_ROOT / "pylingual"
if str(PYLINGUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PYLINGUAL_ROOT))

SEMANTIC_REPAIR_SYSTEM_PROMPT = """You are a Python bytecode-aware semantic repair specialist.

You will receive:
- the target ground-truth Python code object metadata
- the current derived Python code object metadata
- a localized instruction diff derived from the ground-truth and derived bytecode
- the editable source fragment from the derived source file

Edit only the provided source fragment. Preserve the same public names, function/class boundary, decorators, parameters, and return shape unless the bytecode evidence clearly requires a small change. Make the minimal source changes that best align the derived code object with the ground-truth code object. Keep relative indentation valid. Return only the repaired Python source fragment, with no markdown fences and no explanation.
"""

SEMANTIC_PROMPT_VARIANT = "semantic_v4_indent_contract_line_numbers_compact_metadata"


def _bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


SEMANTIC_REPAIR_CANDIDATE_COUNT = _bounded_int_env("SEMANTIC_REPAIR_CANDIDATE_COUNT", 3, minimum=1, maximum=5)
SEMANTIC_REPAIR_TOKEN_HEADROOM = _bounded_int_env("SEMANTIC_REPAIR_TOKEN_HEADROOM", 5000, minimum=0, maximum=200000)
SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS = _bounded_int_env(
    "SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS",
    60 * 60,
    minimum=0,
    maximum=30 * 24 * 60 * 60,
)
SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA = _bounded_int_env(
    "SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA",
    1,
    minimum=1,
    maximum=200000,
)
SEMANTIC_REPAIR_SAMPLE_HARD_TIMEOUT_SECONDS = _bounded_int_env(
    "SEMANTIC_REPAIR_SAMPLE_HARD_TIMEOUT_SECONDS",
    3 * 60 * 60,
    minimum=0,
    maximum=30 * 24 * 60 * 60,
)
SEMANTIC_REPAIR_PREFLIGHT_MAX_REPAIR_TARGETS = _bounded_int_env(
    "SEMANTIC_REPAIR_PREFLIGHT_MAX_REPAIR_TARGETS",
    2,
    minimum=0,
    maximum=100000,
)
SEMANTIC_REPAIR_PREFLIGHT_MAX_INITIAL_COMBINED_DISTANCE = _bounded_int_env(
    "SEMANTIC_REPAIR_PREFLIGHT_MAX_INITIAL_COMBINED_DISTANCE",
    200,
    minimum=0,
    maximum=1000000,
)


INDENTATION_CONTRACT = """Indentation contract:
- Preserve the first line's relative indentation exactly as shown.
- If the fragment starts with `def`, `async def`, or `class`, do not dedent or reindent that header.
- Only adjust indentation inside the fragment when needed to make the repaired Python valid.
- Return a complete replacement for the provided fragment, not only the changed lines.
- Do not include source line-number prefixes such as `12|` in the returned fragment."""

BYTECODE_PROMPT_WINDOW_RADIUS = 12
BYTECODE_PROMPT_MAX_LINES_PER_SIDE = 30
BYTECODE_PROMPT_FULL_LISTING_MAX_INSTRUCTIONS = 80
SEMANTIC_ACTION_PATTERN_INDEX_PATH = REPO_ROOT / "results" / "semantic_repair_action_patterns" / "semantic_repair_action_patterns.jsonl"


def _message_text_for_token_count(messages: list[dict]) -> str:
    return "\n\n".join(
        f"{message.get('role', '')}:\n{message.get('content', '')}"
        for message in messages
        if isinstance(message, dict)
    )


def _semantic_prompt_token_limit(llm_config: dict[str, Any]) -> int | None:
    try:
        token_limit = int(llm_config.get("token_for_completion") or 0)
    except (TypeError, ValueError):
        return None
    return token_limit if token_limit > 0 else None


def _semantic_prompt_token_threshold(llm_config: dict[str, Any]) -> int | None:
    token_limit = _semantic_prompt_token_limit(llm_config)
    if token_limit is None:
        return None
    return max(1, token_limit - SEMANTIC_REPAIR_TOKEN_HEADROOM)


def _semantic_prompt_token_count(messages: list[dict], llm_config: dict[str, Any]) -> int:
    return count_tokens_safe(
        _message_text_for_token_count(messages),
        llm_config.get("provider"),
        llm_config.get("name"),
    )


def _fallback_semantic_candidate(
    *,
    qualname: str,
    derived_code_object: Any,
    derived_source_fragment: str,
) -> str:
    if qualname != "<module>" and derived_code_object is None:
        return ""
    return derived_source_fragment


def _safe_instruction_argrepr(opname: str, arg: int | None, code_object: Any) -> str:
    if arg is None:
        return ""
    consts = getattr(code_object, "co_consts", ())
    names = getattr(code_object, "co_names", ())
    varnames = getattr(code_object, "co_varnames", ())
    freevars = getattr(code_object, "co_freevars", ())
    cellvars = getattr(code_object, "co_cellvars", ())
    closure_vars = tuple(cellvars) + tuple(freevars)
    if opname in {"LOAD_CONST"} and 0 <= arg < len(consts):
        value = consts[arg]
        if hasattr(value, "co_name"):
            return f"{arg} (<code object {getattr(value, 'co_name', None)}>)"
        return f"{arg} ({value!r})"
    if opname in {
        "LOAD_NAME",
        "STORE_NAME",
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
    } and 0 <= arg < len(names):
        return f"{arg} ({names[arg]})"
    if opname in {"LOAD_FAST", "STORE_FAST", "DELETE_FAST"} and 0 <= arg < len(varnames):
        return f"{arg} ({varnames[arg]})"
    if opname in {"LOAD_DEREF", "STORE_DEREF", "DELETE_DEREF", "LOAD_CLASSDEREF"} and 0 <= arg < len(closure_vars):
        return f"{arg} ({closure_vars[arg]})"
    return str(arg)


def _format_editable_bytecode_instructions(bytecode: Any) -> str:
    try:
        instructions = list(bytecode)
    except Exception as exc:
        return f"<editable bytecode instruction listing unavailable: {type(exc).__name__}: {exc}>"
    lines = []
    for index, inst in enumerate(instructions):
        opcode = getattr(inst, "opcode", None)
        optype = getattr(inst, "optype", None)
        starts_line = getattr(inst, "starts_line", None)
        is_jump_target = getattr(inst, "is_jump_target", False)
        has_extended_arg = getattr(inst, "has_extended_arg", False)
        markers = []
        if is_jump_target:
            markers.append("target")
        if has_extended_arg:
            markers.append("extended")
        try:
            if getattr(inst, "is_jump", False):
                target = getattr(inst, "target", None)
                target_offset = getattr(target, "offset", None)
                markers.append(f"jump->{target_offset}")
        except Exception:
            markers.append("jump")
        marker_text = f" [{' '.join(markers)}]" if markers else ""
        try:
            dis_view = inst.get_dis_view()
        except Exception:
            dis_view = repr(inst)
        line = f"{index:4} line={starts_line} {dis_view}"
        if opcode is not None:
            line += f" opcode={opcode}"
        if optype:
            line += f" optype={optype}"
        line += marker_text
        lines.append(line)
    return "\n".join(lines) if lines else "<editable bytecode instruction listing unavailable: empty>"


def _editable_bytecode_instruction_lines(bytecode: Any) -> list[str]:
    listing = _format_editable_bytecode_instructions(bytecode)
    if listing.startswith("<editable bytecode instruction listing unavailable:"):
        return []
    return listing.splitlines()


def _format_editable_bytecode_window(
    bytecode: Any,
    *,
    failed_offset: int | None = None,
    radius: int = BYTECODE_PROMPT_WINDOW_RADIUS,
    max_lines: int = BYTECODE_PROMPT_MAX_LINES_PER_SIDE,
) -> str:
    try:
        instructions = list(bytecode)
    except Exception as exc:
        return f"<editable bytecode instruction window unavailable: {type(exc).__name__}: {exc}>"
    if not instructions:
        return "<editable bytecode instruction window unavailable: empty>"

    if failed_offset is None:
        focus_index = 0
    else:
        exact = [
            index
            for index, inst in enumerate(instructions)
            if getattr(inst, "offset", None) == failed_offset
        ]
        if exact:
            focus_index = exact[0]
        else:
            focus_index = min(
                range(len(instructions)),
                key=lambda index: abs(int(getattr(instructions[index], "offset", 0)) - int(failed_offset)),
            )
    start = max(0, focus_index - radius)
    end = min(len(instructions), focus_index + radius + 1)
    if end - start > max_lines:
        overflow = end - start - max_lines
        trim_left = overflow // 2
        trim_right = overflow - trim_left
        start += trim_left
        end -= trim_right
    window_bytecode = type(
        "EditableBytecodeWindow",
        (),
        {"__iter__": lambda self: iter(instructions[start:end])},
    )()
    lines = _format_editable_bytecode_instructions(window_bytecode).splitlines()
    header = f"<EditableBytecode instruction window: focus_index={focus_index}, rows={start}:{end}, total={len(instructions)}>"
    return "\n".join([header, *lines])


def _code_object_from_prompt_object(code_object_or_bytecode: Any) -> Any:
    return getattr(code_object_or_bytecode, "codeobj", code_object_or_bytecode)


CODE_OBJECT_PROMPT_FIELDS = [
    "co_name",
    "co_qualname",
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_flags",
    "co_varnames",
    "co_names",
    "co_freevars",
    "co_cellvars",
    "co_consts",
]


def _code_object_prompt_values(code_object: Any) -> dict[str, Any]:
    if code_object is None:
        return {}
    code_object = _code_object_from_prompt_object(code_object)

    def serialize_value(value: Any) -> Any:
        if hasattr(value, "co_name") and hasattr(value, "co_consts"):
            return {
                "co_name": getattr(value, "co_name", None),
                "co_qualname": getattr(value, "co_qualname", None),
                "co_firstlineno": getattr(value, "co_firstlineno", None),
            }
        if isinstance(value, tuple):
            return [serialize_value(item) for item in value]
        if isinstance(value, list):
            return [serialize_value(item) for item in value]
        if isinstance(value, set):
            return [serialize_value(item) for item in sorted(value, key=repr)]
        if isinstance(value, dict):
            return {str(key): serialize_value(item) for key, item in value.items()}
        return value

    return {
        field: serialize_value(getattr(code_object, field, None if field != "co_consts" else ()))
        for field in CODE_OBJECT_PROMPT_FIELDS
    }


def _format_code_object_metadata_for_prompt(code_object: Any) -> str:
    if code_object is None:
        return "<missing>"
    values = _code_object_prompt_values(code_object)
    return "\n".join(f"{field}: {values[field]}" for field in CODE_OBJECT_PROMPT_FIELDS)


def _format_line_numbered_source_fragment(source: str, *, start_line: int = 1) -> str:
    lines = source.splitlines()
    if not lines:
        return ""
    end_line = start_line + len(lines) - 1
    width = max(3, len(str(end_line)))
    return "\n".join(f"{line_no:0{width}d}| {line}" for line_no, line in enumerate(lines, start_line))


def strip_prompt_line_numbers(text: str) -> str:
    """Remove copied `NNN|` source-display prefixes from an LLM response."""
    lines = text.splitlines()
    if not lines:
        return text
    nonblank = [line for line in lines if line.strip()]
    if not nonblank:
        return text
    prefix_pattern = re.compile(r"^\s*\d+\|\s?")
    if not all(prefix_pattern.match(line) for line in nonblank):
        return text
    stripped = [prefix_pattern.sub("", line, count=1) if line.strip() else "" for line in lines]
    return "\n".join(stripped)


def _format_code_object_for_prompt(code_object: Any, bytecode: Any | None = None) -> str:
    """Legacy formatter: metadata plus EditableBytecode listing when available."""
    if code_object is None:
        return "<missing>"
    if bytecode is None and hasattr(code_object, "codeobj"):
        bytecode = code_object
    fields = _format_code_object_metadata_for_prompt(code_object)
    if bytecode is not None:
        disassembly = "<using EditableBytecode instruction listing>\n" + _format_editable_bytecode_instructions(bytecode)
    else:
        disassembly = "<instruction listing unavailable: EditableBytecode required for Python 3.10 prompt rendering>"
    return fields + "\n\nDisassembly:\n" + disassembly


def _format_pair_metadata_for_prompt(gt_code_object: Any, derived_code_object: Any) -> str:
    if gt_code_object is None and derived_code_object is None:
        return "Ground-truth: <missing>\nDerived: <missing>"
    if gt_code_object is None:
        return "Ground-truth: <missing>\n\nDerived metadata:\n" + _format_code_object_metadata_for_prompt(derived_code_object)
    if derived_code_object is None:
        return "Ground-truth metadata:\n" + _format_code_object_metadata_for_prompt(gt_code_object) + "\n\nDerived: <missing>"

    gt_values = _code_object_prompt_values(gt_code_object)
    derived_values = _code_object_prompt_values(derived_code_object)
    same_fields = [field for field in CODE_OBJECT_PROMPT_FIELDS if gt_values.get(field) == derived_values.get(field)]
    different_fields = [field for field in CODE_OBJECT_PROMPT_FIELDS if gt_values.get(field) != derived_values.get(field)]
    lines = []
    if same_fields:
        lines.append("Identical fields: " + ", ".join(same_fields))
        lines.extend(f"{field}: {gt_values[field]}" for field in same_fields)
    if different_fields:
        lines.append("\nDifferent fields:")
        for field in different_fields:
            lines.append(f"gt.{field}: {gt_values.get(field)}")
            lines.append(f"derived.{field}: {derived_values.get(field)}")
    return "\n".join(lines)


def _instruction_diff_available(repair_context: dict | None) -> bool:
    if not repair_context:
        return False
    diff = str(repair_context.get("instruction_diff") or "")
    return bool(diff.strip()) and "unavailable" not in diff.lower()


def _format_bytecode_evidence_for_prompt(
    *,
    gt_bytecode: Any | None,
    derived_bytecode: Any | None,
    repair_context: dict | None,
) -> str:
    if gt_bytecode is None and derived_bytecode is None:
        return ""
    failed_offset = None if not repair_context else repair_context.get("failed_offset")
    failed_offset = None if failed_offset is None else int(failed_offset)
    diff_available = _instruction_diff_available(repair_context)
    if diff_available:
        return ""
    blocks = ["Bytecode evidence fallback:"]
    for label, bytecode in (("Ground-truth", gt_bytecode), ("Derived", derived_bytecode)):
        if bytecode is None:
            blocks.append(f"{label}: <missing>")
            continue
        try:
            instruction_count = len(list(bytecode))
        except Exception:
            instruction_count = 0
        if 0 < instruction_count <= BYTECODE_PROMPT_FULL_LISTING_MAX_INSTRUCTIONS:
            blocks.append(f"{label} full instruction listing ({instruction_count} instructions):")
            blocks.append(_format_editable_bytecode_instructions(bytecode))
        else:
            blocks.append(f"{label} localized instruction window:")
            blocks.append(_format_editable_bytecode_window(bytecode, failed_offset=failed_offset))
    return "\n".join(blocks)


def _format_code_object_summary_for_prompt(code_object: Any) -> str:
    if code_object is None:
        return "<missing>"
    code_object = _code_object_from_prompt_object(code_object)
    fields = [
        f"co_name: {getattr(code_object, 'co_name', None)}",
        f"co_qualname: {getattr(code_object, 'co_qualname', None)}",
        f"co_firstlineno: {getattr(code_object, 'co_firstlineno', None)}",
        f"co_argcount: {getattr(code_object, 'co_argcount', None)}",
        f"co_kwonlyargcount: {getattr(code_object, 'co_kwonlyargcount', None)}",
        f"co_nlocals: {getattr(code_object, 'co_nlocals', None)}",
        f"co_stacksize: {getattr(code_object, 'co_stacksize', None)}",
        f"co_flags: {getattr(code_object, 'co_flags', None)}",
        f"co_varnames: {getattr(code_object, 'co_varnames', ())}",
        f"co_names: {getattr(code_object, 'co_names', ())}",
    ]
    return "\n".join(fields)


def _format_module_code_object_for_prompt(code_object: Any) -> str:
    if code_object is None:
        return "<missing>"
    code_object = _code_object_from_prompt_object(code_object)
    fields = [
        f"co_name: {getattr(code_object, 'co_name', None)}",
        f"co_flags: {getattr(code_object, 'co_flags', None)}",
    ]
    return "\n".join(fields)


def _format_repair_context(repair_context: dict | None, *, module_mode: bool = False) -> str:
    del module_mode
    if not repair_context:
        return ""
    instruction_diff = str(repair_context.get("instruction_diff") or "").strip()
    if not instruction_diff:
        instruction_diff = "<unavailable>"
    return f"""Instruction diff:
```text
{instruction_diff}
```
"""


def _prompt_feature_flags(user_prompt: str, repair_context: dict | None) -> dict[str, bool]:
    return {
        "indentation_contract": "Indentation contract:" in user_prompt,
        "line_numbered_source": bool(re.search(r"(?m)^\d{3,}\| ", user_prompt)),
        "compact_metadata": "Identical fields:" in user_prompt or "co_firstlineno:" not in user_prompt,
        "semantic_repair_summaries": "Current repair guidance:" in user_prompt,
        "localized_instruction_diff": _instruction_diff_available(repair_context),
        "jump_target_diagnosis": "Jump target diagnosis:" in user_prompt,
        "bytecode_evidence_fallback": "Bytecode evidence fallback:" in user_prompt,
        "rejected_attempt_feedback": bool(repair_context and repair_context.get("rejected_attempts")),
        "duplicate_retry_feedback": "Duplicate retry constraint:" in user_prompt,
        "candidate_diversity_feedback": "Candidate diversity:" in user_prompt,
        "gt_instruction_window": bool(repair_context and repair_context.get("gt_instruction_window")),
        "derived_instruction_window": bool(repair_context and repair_context.get("derived_instruction_window")),
        "telemetry_guidance": "Action guidance from prior accepted repairs:" in user_prompt,
        "action_pattern_guidance": "Relevant prior repair patterns:" in user_prompt,
    }


def _public_repair_context(repair_context: dict | None) -> dict | None:
    if repair_context is None:
        return None
    return {key: value for key, value in repair_context.items() if not str(key).startswith("_")}


def _nested_get(record: dict, path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _top_values(values: list[Any], *, limit: int = 3) -> list[str]:
    counts = Counter(str(value) for value in values if value not in (None, "", []))
    return [value for value, _ in counts.most_common(limit)]


def _top_list_values(records: list[dict], path: tuple[str, ...], *, limit: int = 5) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        values = _nested_get(record, path)
        if isinstance(values, list):
            counts.update(str(value) for value in values if value not in (None, ""))
    return [value for value, _ in counts.most_common(limit)]


def _semantic_repair_operation_for_prompt(qualname: str, derived_code_object: Any | None) -> str:
    if qualname == "<module>":
        return "repair_module_statement"
    if derived_code_object is None:
        return "insert_missing"
    return "repair_source_fragment"


def _telemetry_prompt_variant(record: dict) -> str | None:
    prompt_record = record.get("prompt") or {}
    return prompt_record.get("prompt_variant") or _nested_get(prompt_record, ("prompt_reconstruction_inputs", "prompt_variant"))


def _telemetry_similarity(current_case: dict, past_case: dict) -> int:
    score = 0
    if current_case.get("prompt_variant") == _telemetry_prompt_variant(past_case):
        score += 4
    if current_case.get("repair_operation") == past_case.get("repair_operation"):
        score += 5
    if current_case.get("target_kind") == past_case.get("target_kind"):
        score += 3
    if current_case.get("alignment_tag") == _nested_get(past_case, ("repair_context_summary", "alignment_tag")):
        score += 2
    if current_case.get("qualname") == past_case.get("qualname"):
        score += 1
    if past_case.get("accepted"):
        score += 1
    return score


def _format_semantic_repair_guidance(current_case: dict, similar_cases: list[dict]) -> str:
    if not similar_cases:
        return ""

    names_added = _top_list_values(similar_cases, ("feature_delta", "names_added"))
    names_removed = _top_list_values(similar_cases, ("feature_delta", "names_removed"))
    statements_added = _top_list_values(similar_cases, ("feature_delta", "statement_types_added"))
    statements_removed = _top_list_values(similar_cases, ("feature_delta", "statement_types_removed"))
    control_flow_added = _top_list_values(similar_cases, ("feature_delta", "control_flow_added"))
    control_flow_removed = _top_list_values(similar_cases, ("feature_delta", "control_flow_removed"))

    action_lines: list[str] = []
    if names_added:
        action_lines.append(
            f"- Prior accepted edits restored missing references to: {', '.join(names_added)}. "
            "If these names appear in the ground-truth metadata or instruction diff, add the smallest expression-level use that accounts for them."
        )
    if names_removed:
        action_lines.append(
            f"- Prior accepted edits removed or replaced extra references to: {', '.join(names_removed)}. "
            "If these names appear only in the derived side, remove the local use instead of restructuring the fragment."
        )
    if statements_added:
        action_lines.append(
            f"- Prior accepted edits added these statement forms: {', '.join(statements_added)}. "
            "Only add the same form when the current bytecode shows the corresponding missing branch, loop, definition, or return."
        )
    if statements_removed:
        action_lines.append(
            f"- Prior accepted edits removed these statement forms: {', '.join(statements_removed)}. "
            "If the derived bytecode has extra structure, simplify that local statement rather than adding new code."
        )
    if control_flow_added:
        action_lines.append(
            f"- Prior accepted edits introduced control flow: {', '.join(control_flow_added)}. "
            "Use this only when the ground-truth instruction window has jumps or blocks missing from the derived fragment."
        )
    if control_flow_removed:
        action_lines.append(
            f"- Prior accepted edits removed control flow: {', '.join(control_flow_removed)}. "
            "Use this only when the derived instruction window has extra jumps or blocks."
        )

    operation = current_case.get("repair_operation")
    if operation == "insert_missing":
        action_lines.append("- Return only the missing fragment; do not repeat the parent context around the insertion point.")
    elif operation == "repair_module_statement":
        action_lines.append("- Keep the edit inside the shown top-level statement; do not rewrite neighboring module code.")

    if not action_lines:
        return ""
    return "\n".join(
        [
            "Action guidance from prior accepted repairs:",
            *action_lines,
            "- Apply these actions only when they explain the current source, metadata, and bytecode mismatch.",
        ]
    )


def generate_semantic_repair_guidance(
    *,
    qualname: str,
    derived_code_object: Any | None,
    repair_context: dict | None,
    telemetry_records: list[dict] | None,
    prompt_variant: str = SEMANTIC_PROMPT_VARIANT,
    max_cases: int = 3,
) -> str:
    """Generate compact prompt guidance from accepted telemetry matching this semantic repair variant."""
    if not telemetry_records:
        return ""
    current_case = {
        "prompt_variant": prompt_variant,
        "repair_operation": _semantic_repair_operation_for_prompt(qualname, derived_code_object),
        "target_kind": None if not repair_context else repair_context.get("target_kind"),
        "alignment_tag": None if not repair_context else repair_context.get("alignment_tag"),
        "qualname": qualname,
    }
    ranked: list[tuple[int, dict]] = []
    for record in telemetry_records:
        if not isinstance(record, dict) or not record.get("accepted"):
            continue
        record_variant = _telemetry_prompt_variant(record)
        if record_variant and record_variant != prompt_variant:
            continue
        if record.get("repair_operation") != current_case["repair_operation"]:
            continue
        score = _telemetry_similarity(current_case, record)
        if score > 0:
            ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return _format_semantic_repair_guidance(current_case, [record for _, record in ranked[:max_cases]])


def _load_semantic_action_patterns(path: Path = SEMANTIC_ACTION_PATTERN_INDEX_PATH) -> list[dict]:
    try:
        pattern_path = path.expanduser().resolve()
    except Exception:
        return []
    if not pattern_path.exists():
        return []
    records: list[dict] = []
    try:
        with pattern_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return []
    return records


def _alignment_category_from_context(repair_context: dict | None, repair_operation: str) -> str:
    if repair_operation == "insert_missing":
        return "insert"
    if repair_operation == "repair_module_statement":
        return "module_statement"
    tag = str((repair_context or {}).get("alignment_tag") or "").lower()
    if "insert" in tag:
        return "insert"
    if "delete" in tag:
        return "delete"
    if "replace" in tag:
        return "replace"
    return tag or "unknown"


def _metadata_delta_signature(gt_code_object: Any, derived_code_object: Any) -> dict[str, bool]:
    gt = _code_object_prompt_values(gt_code_object)
    derived = _code_object_prompt_values(derived_code_object)
    gt_names = {str(value) for value in gt.get("co_names") or []}
    derived_names = {str(value) for value in derived.get("co_names") or []}
    gt_varnames = {str(value) for value in gt.get("co_varnames") or []}
    derived_varnames = {str(value) for value in derived.get("co_varnames") or []}
    gt_consts = {repr(value) for value in gt.get("co_consts") or []}
    derived_consts = {repr(value) for value in derived.get("co_consts") or []}
    return {
        "gt_only_names": bool(gt_names - derived_names),
        "derived_only_names": bool(derived_names - gt_names),
        "gt_only_consts": bool(gt_consts - derived_consts),
        "derived_only_consts": bool(derived_consts - gt_consts),
        "gt_only_varnames": bool(gt_varnames - derived_varnames),
        "derived_only_varnames": bool(derived_varnames - gt_varnames),
        "arg_shape_changed": any(
            gt.get(field) != derived.get(field)
            for field in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount")
        ),
    }


def _instruction_delta_signature(repair_context: dict | None) -> dict[str, bool]:
    if not repair_context:
        text = ""
    else:
        text = "\n".join(
            str(value or "")
            for value in (
                repair_context.get("instruction_diff"),
                repair_context.get("gt_instruction_window"),
                repair_context.get("derived_instruction_window"),
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
        "instruction_has_build_collection": any(token in text for token in ("BUILD_LIST", "BUILD_TUPLE", "BUILD_MAP", "BUILD_SET")),
    }


def _expr_kind_for_prompt(node: ast.AST | None) -> str:
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


def _source_shape_signature(fragment: str) -> dict[str, Any]:
    text = fragment or ""
    parsed = None
    for candidate in (text, _dedent_like_python_for_prompt(text)):
        try:
            parsed = ast.parse(candidate)
            break
        except SyntaxError:
            continue
    if parsed is None:
        return {
            "source_statement_types": set(),
            "has_control_flow": False,
            "has_return": False,
            "has_assignment": False,
            "has_call_or_attribute": False,
            "has_compare": False,
            "has_boolop": False,
            "has_subscript": False,
            "return_value_kinds": set(),
            "assign_value_kinds": set(),
            "expr_value_kinds": set(),
            "call_func_kinds": set(),
            "call_arg_count": 0,
            "call_keyword_count": 0,
        }
    statement_types = {type(node).__name__ for node in parsed.body}
    nodes = list(ast.walk(parsed))
    return_value_kinds: set[str] = set()
    assign_value_kinds: set[str] = set()
    expr_value_kinds: set[str] = set()
    call_func_kinds: set[str] = set()
    call_arg_count = 0
    call_keyword_count = 0
    for node in nodes:
        if isinstance(node, ast.Return) and node.value is not None:
            return_value_kinds.add(_expr_kind_for_prompt(node.value))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = getattr(node, "value", None)
            if value is not None:
                assign_value_kinds.add(_expr_kind_for_prompt(value))
        elif isinstance(node, ast.Expr):
            expr_value_kinds.add(_expr_kind_for_prompt(node.value))
        elif isinstance(node, ast.Call):
            call_func_kinds.add(_expr_kind_for_prompt(node.func))
            call_arg_count += len(node.args)
            call_keyword_count += len(node.keywords)
    return {
        "source_statement_types": statement_types,
        "has_control_flow": any(isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)) for node in nodes),
        "has_return": any(isinstance(node, ast.Return) for node in nodes),
        "has_assignment": any(isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for node in nodes),
        "has_call_or_attribute": any(isinstance(node, (ast.Call, ast.Attribute)) for node in nodes),
        "has_compare": any(isinstance(node, ast.Compare) for node in nodes),
        "has_boolop": any(isinstance(node, ast.BoolOp) for node in nodes),
        "has_subscript": any(isinstance(node, ast.Subscript) for node in nodes),
        "return_value_kinds": return_value_kinds,
        "assign_value_kinds": assign_value_kinds,
        "expr_value_kinds": expr_value_kinds,
        "call_func_kinds": call_func_kinds,
        "call_arg_count": call_arg_count,
        "call_keyword_count": call_keyword_count,
    }


def _dedent_like_python_for_prompt(text: str) -> str:
    lines = text.splitlines()
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    if not indents:
        return text
    margin = min(indents)
    return "\n".join(line[margin:] if len(line) >= margin else line for line in lines)


def _current_action_pattern_signature(
    *,
    qualname: str,
    gt_code_object: Any,
    derived_code_object: Any,
    derived_source_fragment: str,
    repair_context: dict | None,
) -> dict[str, Any]:
    repair_operation = _semantic_repair_operation_for_prompt(qualname, derived_code_object)
    signature = {
        "repair_operation": repair_operation,
        "target_kind": None if not repair_context else repair_context.get("target_kind"),
        "alignment_category": _alignment_category_from_context(repair_context, repair_operation),
    }
    signature.update(_metadata_delta_signature(gt_code_object, derived_code_object))
    signature.update(_instruction_delta_signature(repair_context))
    signature.update(_source_shape_signature(derived_source_fragment))
    return signature


def _pattern_bool(record: dict, key: str) -> bool:
    value = record.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return bool(value)


def _score_action_pattern(current: dict[str, Any], pattern: dict[str, Any]) -> int | None:
    if pattern.get("repair_operation") != current.get("repair_operation"):
        return None
    target_kind = pattern.get("target_kind")
    if target_kind and current.get("target_kind") and target_kind != current.get("target_kind"):
        return None
    pattern_alignment = pattern.get("alignment_category")
    current_alignment = current.get("alignment_category")
    if pattern_alignment not in (None, "", "unknown") and current_alignment not in (None, "", "unknown") and pattern_alignment != current_alignment:
        return None

    score = 6
    if target_kind and target_kind == current.get("target_kind"):
        score += 2
    if pattern_alignment and pattern_alignment == current_alignment:
        score += 2

    metadata_pairs = [
        ("gt_only_names", "gt_only_names_count"),
        ("derived_only_names", "derived_only_names_count"),
        ("gt_only_consts", "gt_only_consts_count"),
        ("derived_only_consts", "derived_only_consts_count"),
        ("gt_only_varnames", "gt_only_varnames_count"),
        ("derived_only_varnames", "derived_only_varnames_count"),
        ("arg_shape_changed", "arg_shape_changed"),
    ]
    for current_key, pattern_key in metadata_pairs:
        current_value = bool(current.get(current_key))
        pattern_value = _pattern_bool(pattern, pattern_key)
        if current_value and pattern_value:
            score += 3
        elif current_value != pattern_value:
            score -= 1

    instruction_keys = [
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
    ]
    for key in instruction_keys:
        current_value = bool(current.get(key))
        pattern_value = _pattern_bool(pattern, key)
        if current_value and pattern_value:
            score += 3
        elif pattern_value and not current_value:
            score -= 1

    source_evidence_pairs = [
        ("has_return", "source_has_return_before"),
        ("has_assignment", "source_has_assignment_before"),
        ("has_call_or_attribute", "source_has_call_or_attribute_before"),
        ("has_compare", "source_has_compare_before"),
        ("has_boolop", "source_has_boolop_before"),
        ("has_subscript", "source_has_subscript_before"),
    ]
    for current_key, pattern_key in source_evidence_pairs:
        current_value = bool(current.get(current_key))
        pattern_value = _pattern_bool(pattern, pattern_key)
        if current_value and pattern_value:
            score += 3
        elif pattern_value and not current_value:
            score -= 2

    source_types = set(current.get("source_statement_types") or set())
    pattern_types = set(str(pattern.get("source_shape_before") or "").split(",")) - {""}
    if source_types and pattern_types:
        overlap = source_types & pattern_types
        if overlap:
            score += min(3, len(overlap))
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
    if action_type == "adjust_return_expression" and not (current.get("has_return") or current.get("instruction_has_return")):
        score -= 3
    if action_type.startswith("adjust_return_") and not (current.get("has_return") or current.get("instruction_has_return")):
        score -= 3
    if action_type in {"adjust_call_or_attribute_expression", "adjust_call_arguments", "adjust_method_vs_attribute_access", "adjust_attribute_access"} and not (
        current.get("has_call_or_attribute") or current.get("instruction_has_call_or_attr")
    ):
        score -= 3
    if action_type == "adjust_call_arguments" and not (current.get("call_arg_count") or current.get("call_keyword_count") or current.get("instruction_has_kw_names")):
        score -= 4
    if action_type == "adjust_comparison_expression" and not (current.get("has_compare") or current.get("instruction_has_compare")):
        score -= 3
    if action_type == "adjust_boolean_expression" and not (current.get("has_boolop") or current.get("instruction_has_jump")):
        score -= 3
    if action_type == "adjust_arithmetic_or_unary_expression" and not current.get("instruction_has_binary_op"):
        score -= 2
    if action_type == "adjust_subscript_or_index_expression" and not (current.get("has_subscript") or current.get("instruction_has_subscript")):
        score -= 3
    return score


def _rejected_action_type_penalties(repair_context: dict | None) -> dict[str, int]:
    if not repair_context:
        return {}
    rejected_attempts = repair_context.get("rejected_attempts") or []
    penalties: dict[str, int] = {}
    for reverse_index, attempt in enumerate(reversed(rejected_attempts[-3:])):
        if not isinstance(attempt, dict):
            continue
        selected_action_types = [
            str(action_type)
            for action_type in (attempt.get("selected_action_types") or [])
            if action_type
        ]
        if not selected_action_types:
            continue
        top_penalty = 18 if reverse_index == 0 else 8
        secondary_penalty = 8 if reverse_index == 0 else 4
        for index, action_type in enumerate(selected_action_types[:3]):
            penalty = top_penalty if index == 0 else secondary_penalty
            penalties[action_type] = max(penalties.get(action_type, 0), penalty)
    return penalties


def _rank_action_patterns_by_action_type(
    current: dict[str, Any],
    action_patterns: list[dict],
    *,
    min_score: int,
    action_type_penalties: dict[str, int] | None = None,
) -> list[tuple[int, int, dict]]:
    by_action_type: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for pattern in action_patterns:
        score = _score_action_pattern(current, pattern)
        if score is None or score < min_score:
            continue
        action_type = str(pattern.get("action_type") or "")
        if not action_type:
            continue
        by_action_type[action_type].append((score, pattern))

    ranked: list[tuple[int, int, dict]] = []
    for matches in by_action_type.values():
        matches.sort(key=lambda item: item[0], reverse=True)
        best_score, best_pattern = matches[0]
        action_type = str(best_pattern.get("action_type") or "")
        consensus_bonus = sum(max(0, score - min_score) for score, _ in matches[:5]) // 5
        penalty = 0 if action_type_penalties is None else int(action_type_penalties.get(action_type, 0))
        ranked.append((best_score + consensus_bonus - penalty, best_score, best_pattern))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked


def generate_action_pattern_guidance(
    *,
    qualname: str,
    gt_code_object: Any,
    derived_code_object: Any,
    derived_source_fragment: str,
    repair_context: dict | None,
    action_patterns: list[dict] | None,
    max_patterns: int = 3,
    min_score: int = 12,
) -> str:
    if not action_patterns:
        return ""
    current = _current_action_pattern_signature(
        qualname=qualname,
        gt_code_object=gt_code_object,
        derived_code_object=derived_code_object,
        derived_source_fragment=derived_source_fragment,
        repair_context=repair_context,
    )
    action_type_penalties = _rejected_action_type_penalties(repair_context)
    selected = [
        pattern
        for _, _, pattern in _rank_action_patterns_by_action_type(
            current,
            action_patterns,
            min_score=min_score,
            action_type_penalties=action_type_penalties,
        )[:max_patterns]
    ]
    if repair_context is not None:
        repair_context["_selected_action_pattern_types"] = [
            str(pattern.get("action_type") or "")
            for pattern in selected
            if pattern.get("action_type")
        ]
        repair_context["_rejected_action_type_penalties"] = action_type_penalties
    if not selected:
        return ""

    lines = ["Relevant prior repair patterns:"]
    for index, pattern in enumerate(selected, start=1):
        guidance = pattern.get("action_summary_prompt") or pattern.get("action_agnostic") or pattern.get("prompt_snippet_agnostic")
        situation = pattern.get("situation_agnostic")
        lines.append(f"{index}. Situation: {situation}")
        lines.append(f"   Action that reduced drift: {guidance}")
    lines.append("Use these only if they explain the current metadata and instruction diff.")
    return "\n".join(lines)


def _format_variant_constraint(repair_operation: str) -> str:
    if repair_operation == "insert_missing":
        return "\n".join(
            [
                "Variant constraint:",
                "- Return only the missing source fragment to insert.",
                "- Do not repeat the surrounding parent context.",
            ]
        )
    if repair_operation == "repair_module_statement":
        return "\n".join(
            [
                "Variant constraint:",
                "- Repair only the shown top-level module statement.",
                "- Do not rewrite neighboring imports, functions, classes, or unrelated module code.",
            ]
        )
    return "\n".join(
        [
            "Variant constraint:",
            "- Edit only the existing source fragment.",
            "- Preserve function/class boundaries unless the current bytecode evidence requires a small local change.",
        ]
    )


def _format_target_context_summary(repair_context: dict | None) -> str:
    if not repair_context:
        return ""
    fields = [
        ("target_kind", repair_context.get("target_kind")),
        ("localized_line_number", repair_context.get("localized_line_number")),
        ("failed_offset", repair_context.get("failed_offset")),
        ("alignment_tag", repair_context.get("alignment_tag")),
    ]
    lines = [f"- {name}: {value}" for name, value in fields if value is not None]
    retry_count = int(repair_context.get("retry_count_for_target") or 0)
    if retry_count:
        lines.append(f"- retry_count_for_target: {retry_count}")
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message"):
        lines.append(f"- pylingual_message: {failed.get('message')}")
    if not lines:
        return ""
    return "\n".join(["Target context:", *lines])


def _format_current_mismatch_summary(repair_context: dict | None) -> str:
    if not repair_context:
        return ""
    lines = ["Mismatch focus:"]
    if _instruction_diff_available(repair_context):
        lines.append("- Use the first differing instruction block as the main source-edit target.")
    elif repair_context.get("gt_instruction_window") or repair_context.get("derived_instruction_window"):
        lines.append("- Instruction windows are available; compare missing or extra names, constants, jumps, and calls before editing.")
    return "\n".join(lines) if len(lines) > 1 else ""


def _jump_target_mismatch(repair_context: dict | None) -> dict[str, Any] | None:
    if not repair_context:
        return None
    instruction_diff = str(repair_context.get("instruction_diff") or "")
    gt_match = re.search(r"=>\s*gt:.*?\b(?P<op>[A-Z_]*JUMP[A-Z_]*)\b.*?\bto\s+(?P<target>\d+)", instruction_diff)
    derived_match = re.search(r"=>\s*derived:.*?\b(?P<op>[A-Z_]*JUMP[A-Z_]*)\b.*?\bto\s+(?P<target>\d+)", instruction_diff)
    if not gt_match or not derived_match:
        return None
    if gt_match.group("op") != derived_match.group("op"):
        return None
    gt_target = int(gt_match.group("target"))
    derived_target = int(derived_match.group("target"))
    if gt_target == derived_target:
        return None
    return {
        "opname": gt_match.group("op"),
        "gt_target": gt_target,
        "derived_target": derived_target,
        "direction": "later" if derived_target > gt_target else "earlier",
    }


def _format_jump_target_diagnosis(repair_context: dict | None) -> str:
    mismatch = _jump_target_mismatch(repair_context)
    if not mismatch:
        return ""
    lines = [
        "Jump target diagnosis:",
        f"- {mismatch['opname']} target differs: ground-truth jumps to {mismatch['gt_target']}, derived jumps to {mismatch['derived_target']}.",
    ]
    if mismatch["direction"] == "later":
        lines.append("- The derived branch/loop body is too large; move one existing statement out of the current block or remove an unsupported nested statement.")
    else:
        lines.append("- The derived branch/loop body is too small; move one existing statement into the current block or restore a missing nested statement.")
    lines.append("- Prefer block-placement changes over call, argument, receiver, or indentation-only edits.")
    return "\n".join(lines)


def _limited_list(values: list[str], *, limit: int = 8) -> list[str]:
    return values[:limit]


def _format_code_shape_summary(*, gt_code_object: Any, derived_code_object: Any) -> str:
    gt = _code_object_prompt_values(gt_code_object)
    derived = _code_object_prompt_values(derived_code_object)
    if not gt and not derived:
        return ""

    lines = ["Code shape summary:"]
    for field in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_flags"):
        if gt.get(field) != derived.get(field):
            lines.append(f"- {field}: ground-truth={gt.get(field)!r} derived={derived.get(field)!r}")

    gt_names = {str(name) for name in gt.get("co_names") or []}
    derived_names = {str(name) for name in derived.get("co_names") or []}
    gt_varnames = {str(name) for name in gt.get("co_varnames") or []}
    derived_varnames = {str(name) for name in derived.get("co_varnames") or []}
    gt_consts = {repr(value) for value in gt.get("co_consts") or []}
    derived_consts = {repr(value) for value in derived.get("co_consts") or []}

    names_gt_only = sorted(gt_names - derived_names)
    names_derived_only = sorted(derived_names - gt_names)
    varnames_gt_only = sorted(gt_varnames - derived_varnames)
    varnames_derived_only = sorted(derived_varnames - gt_varnames)
    consts_gt_only = sorted(gt_consts - derived_consts)
    consts_derived_only = sorted(derived_consts - gt_consts)

    if names_gt_only:
        lines.append(f"- names only in ground-truth metadata: {names_gt_only[:8]}")
    if names_derived_only:
        lines.append(f"- names only in derived metadata: {names_derived_only[:8]}")
    if varnames_gt_only:
        lines.append(f"- local variables only in ground-truth metadata: {varnames_gt_only[:8]}")
    if varnames_derived_only:
        lines.append(f"- local variables only in derived metadata: {varnames_derived_only[:8]}")
    if consts_gt_only:
        lines.append(f"- constants only in ground-truth metadata: {_limited_list(consts_gt_only)}")
    if consts_derived_only:
        lines.append(f"- constants only in derived metadata: {_limited_list(consts_derived_only)}")

    if len(lines) == 1:
        return ""
    lines.append("- Prefer edits that account for ground-truth-only names/constants and remove derived-only artifacts when bytecode agrees.")
    return "\n".join(lines)


def _format_rejected_attempt_summary(repair_context: dict | None) -> str:
    rejected_attempts = [] if not repair_context else repair_context.get("rejected_attempts") or []
    rejected_attempts = [item for item in rejected_attempts[-3:] if isinstance(item, dict)]
    if not rejected_attempts:
        return ""
    last_attempt = rejected_attempts[-1]
    last_reason = last_attempt.get("acceptance_reason")
    last_action_types = [str(value) for value in (last_attempt.get("selected_action_types") or []) if value]
    previous_action_types: list[str] = []
    for attempt in rejected_attempts[:-1]:
        for action_type in attempt.get("selected_action_types") or []:
            action_type = str(action_type)
            if action_type and action_type not in previous_action_types:
                previous_action_types.append(action_type)

    lines = ["Retry feedback:"]
    if last_reason:
        lines.append(f"- last rejection reason: {last_reason}")
    if last_action_types:
        lines.append(f"- last rejected action guidance: {', '.join(last_action_types[:3])}")
    if previous_action_types:
        lines.append(f"- earlier rejected action guidance: {', '.join(previous_action_types[:3])}")
    last_delta = _format_rejected_replacement_delta(last_attempt.get("replacement_delta"))
    if last_delta:
        lines.append(f"- last failed edit shape: {last_delta}")
    recent_after_scores = [
        _score_combined_distance(attempt.get("target_score_after"))
        for attempt in rejected_attempts
    ]
    recent_after_scores = [score for score in recent_after_scores if score is not None]
    if recent_after_scores:
        lines.append(f"- recent rejected after-distances: {', '.join(str(score) for score in recent_after_scores[-3:])}")
    lines.append("- Prefer a different minimal edit unless the current bytecode evidence strongly supports the rejected action.")
    return "\n".join(lines)


def _format_duplicate_retry_summary(repair_context: dict | None) -> str:
    if not repair_context or not repair_context.get("_force_distinct_retry"):
        return ""
    source = str(repair_context.get("_force_distinct_retry_source") or "duplicate")
    source_label = "prior same-round candidate" if source == "same_round_candidate" else "rejected replacement"
    lines = [
        "Duplicate retry constraint:",
        f"- The immediately previous generated candidate duplicated a {source_label}.",
        "- Do not return that same replacement again; choose a different local edit shape.",
        "- Whitespace-only or indentation-only variants of the same AST repair still count as duplicates.",
    ]
    duplicate_excerpt = _truncate_rejected_replacement_excerpt(repair_context.get("_duplicate_rejected_output_excerpt"))
    if duplicate_excerpt:
        lines.append("Duplicate output excerpt:")
        lines.append("```python")
        lines.append(duplicate_excerpt)
        lines.append("```")
    return "\n".join(lines)


def _format_candidate_strategy_summary(repair_context: dict | None) -> str:
    if not repair_context or not repair_context.get("_candidate_strategy_prompt"):
        return ""
    return "\n".join(
        [
            "Candidate strategy:",
            f"- {repair_context.get('_candidate_strategy_prompt')}",
        ]
    )


def _format_prior_candidate_outputs_summary(repair_context: dict | None) -> str:
    prior_candidates = [] if not repair_context else repair_context.get("_prior_candidate_outputs") or []
    prior_candidates = [item for item in prior_candidates[-2:] if isinstance(item, dict)]
    if not prior_candidates:
        return ""

    lines = [
        "Candidate diversity:",
        "- Do not return any prior candidate below.",
        "- Produce a structurally different repair while still following the bytecode and metadata evidence.",
        "- Whitespace-only or indentation-only variants of prior candidates do not count as different repairs.",
    ]
    for index, candidate in enumerate(prior_candidates, start=1):
        strategy = str(candidate.get("strategy") or "unknown")
        candidate_hash = str(candidate.get("replacement_hash") or "")
        structural_hash = str(candidate.get("replacement_structural_hash") or "")
        if structural_hash:
            hash_suffix = f", ast={structural_hash[:12]}"
        elif candidate_hash:
            hash_suffix = f", hash={candidate_hash[:12]}"
        else:
            hash_suffix = ""
        excerpt = _truncate_rejected_replacement_excerpt(candidate.get("text"), max_chars=700)
        if not excerpt:
            continue
        lines.append(f"Prior same-round candidate {index} ({strategy}{hash_suffix}):")
        lines.append("```python")
        lines.append(excerpt)
        lines.append("```")
    return "\n".join(lines)


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


def _candidate_repeats_rejected_attempt(candidate: str, repair_context: dict | None) -> bool:
    if not repair_context:
        return False
    candidate_hash = _replacement_hash(candidate)
    candidate_structural_hash = _replacement_structural_hash(candidate)
    if not candidate_hash and not candidate_structural_hash:
        return False
    for attempt in repair_context.get("rejected_attempts") or []:
        if not isinstance(attempt, dict):
            continue
        if candidate_hash and attempt.get("replacement_hash") == candidate_hash:
            return True
        if candidate_structural_hash and attempt.get("replacement_structural_hash") == candidate_structural_hash:
            return True
    return False


def _truncate_rejected_replacement_excerpt(text: Any, *, max_chars: int = 900) -> str:
    if not text:
        return ""
    value = str(text).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n# ... truncated ..."


def _score_combined_distance(score: Any) -> int | None:
    if not isinstance(score, dict):
        return None
    value = score.get("combined_distance")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _format_rejected_replacement_delta(delta: Any) -> str:
    if not isinstance(delta, dict) or not delta:
        return ""
    parts: list[str] = []
    for key, label in (
        ("statement_types_added", "statements added"),
        ("statement_types_removed", "statements removed"),
        ("control_flow_added", "control flow added"),
        ("control_flow_removed", "control flow removed"),
        ("names_added", "names added"),
        ("names_removed", "names removed"),
    ):
        values = delta.get(key)
        if isinstance(values, list) and values:
            parts.append(f"{label}={values[:5]}")
    for key, label in (
        ("line_count_delta", "line delta"),
        ("char_count_delta", "char delta"),
        ("indent_width_delta", "indent delta"),
    ):
        value = delta.get(key)
        if isinstance(value, (int, float)) and value:
            parts.append(f"{label}={int(value)}")
    for key, label in (
        ("parse_transition", "parse"),
        ("dedented_parse_transition", "dedented parse"),
    ):
        value = delta.get(key)
        if value and value != "True->True":
            parts.append(f"{label}={value}")
    return "; ".join(parts[:6])


def build_semantic_prompt_summaries(
    *,
    qualname: str,
    gt_code_object: Any,
    derived_code_object: Any,
    repair_context: dict | None,
) -> str:
    repair_operation = _semantic_repair_operation_for_prompt(qualname, derived_code_object)
    sections = [
        _format_variant_constraint(repair_operation),
        _format_target_context_summary(repair_context),
        _format_current_mismatch_summary(repair_context),
        _format_jump_target_diagnosis(repair_context),
        _format_code_shape_summary(gt_code_object=gt_code_object, derived_code_object=derived_code_object),
        _format_rejected_attempt_summary(repair_context),
        _format_duplicate_retry_summary(repair_context),
        _format_candidate_strategy_summary(repair_context),
        _format_prior_candidate_outputs_summary(repair_context),
    ]
    content = "\n\n".join(section for section in sections if section)
    if not content:
        return ""
    return "Current repair guidance:\n" + content


def build_semantic_repair_prompt_payload(
    *,
    qualname: str,
    gt_code_object: Any,
    derived_code_object: Any,
    derived_source_fragment: str,
    repair_context: dict | None = None,
    gt_bytecode: Any | None = None,
    derived_bytecode: Any | None = None,
    telemetry_guidance: str | None = None,
    action_pattern_guidance: str | None = None,
) -> dict:
    if qualname == "<module>":
        task_text = "Edit only the localized top-level module source statement. Preserve valid Python syntax and make the smallest source change that best aligns the module bytecode with the ground-truth module bytecode."
        source_label = "Current derived top-level module statement"
    elif derived_code_object is None:
        task_text = "The target code object is missing from the derived bytecode. Synthesize only the missing Python source fragment that should be inserted into the derived source."
        source_label = "Derived insertion context"
    else:
        task_text = "Edit only the current derived source fragment."
        source_label = "Current derived source fragment"
    metadata_text = (
        _format_pair_metadata_for_prompt(gt_code_object, derived_code_object)
        if qualname != "<module>"
        else "Ground-truth module metadata:\n"
        + _format_module_code_object_for_prompt(gt_code_object)
        + "\n\nDerived module metadata:\n"
        + _format_module_code_object_for_prompt(derived_code_object)
    )
    bytecode_evidence = _format_bytecode_evidence_for_prompt(
        gt_bytecode=gt_bytecode,
        derived_bytecode=derived_bytecode,
        repair_context=repair_context,
    )
    semantic_summaries = build_semantic_prompt_summaries(
        qualname=qualname,
        gt_code_object=gt_code_object,
        derived_code_object=derived_code_object,
        repair_context=repair_context,
    )
    source_start_line = 1
    if repair_context and repair_context.get("source_lineno") is not None:
        source_start_line = int(repair_context["source_lineno"])
    line_numbered_source_fragment = _format_line_numbered_source_fragment(derived_source_fragment, start_line=source_start_line)
    public_context = _public_repair_context(repair_context)
    return {
        "prompt_variant": SEMANTIC_PROMPT_VARIANT,
        "task_text": task_text,
        "target_qualname": qualname,
        "source_label": source_label,
        "source_fragment": derived_source_fragment,
        "line_numbered_source_fragment": line_numbered_source_fragment,
        "source_start_line": source_start_line,
        "indentation_contract": INDENTATION_CONTRACT,
        "repair_context": public_context,
        "bytecode_context": None if not repair_context else {
            "gt_instruction_range": repair_context.get("gt_instruction_range"),
            "derived_instruction_range": repair_context.get("derived_instruction_range"),
            "gt_instruction_window": repair_context.get("gt_instruction_window"),
            "derived_instruction_window": repair_context.get("derived_instruction_window"),
            "instruction_diff": repair_context.get("instruction_diff"),
            "instruction_renderer_version": repair_context.get("instruction_renderer_version"),
            "bytecode_window_radius": repair_context.get("bytecode_window_radius"),
            "bytecode_window_max_records": repair_context.get("bytecode_window_max_records"),
            "bytecode_window_truncated": repair_context.get("bytecode_window_truncated"),
        },
        "metadata_context": {
            "metadata_text": metadata_text,
            "gt_code_object_metadata": _code_object_prompt_values(gt_code_object),
            "derived_code_object_metadata": _code_object_prompt_values(derived_code_object),
        },
        "bytecode_evidence_fallback": bytecode_evidence or None,
        "semantic_summaries": semantic_summaries or None,
        "telemetry_guidance": telemetry_guidance or None,
        "action_pattern_guidance": action_pattern_guidance or None,
        "return_contract": "Return only the repaired source fragment.",
    }


def build_semantic_repair_messages(
    *,
    qualname: str,
    gt_code_object: Any,
    derived_code_object: Any,
    derived_source_fragment: str,
    repair_context: dict | None = None,
    gt_bytecode: Any | None = None,
    derived_bytecode: Any | None = None,
    telemetry_guidance: str | None = None,
    action_pattern_guidance: str | None = None,
) -> list[dict]:
    payload = build_semantic_repair_prompt_payload(
        qualname=qualname,
        gt_code_object=gt_code_object,
        derived_code_object=derived_code_object,
        derived_source_fragment=derived_source_fragment,
        repair_context=repair_context,
        gt_bytecode=gt_bytecode,
        derived_bytecode=derived_bytecode,
        telemetry_guidance=telemetry_guidance,
        action_pattern_guidance=action_pattern_guidance,
    )
    task_text = payload["task_text"]
    source_label = payload["source_label"]
    metadata_text = payload["metadata_context"]["metadata_text"]
    bytecode_evidence = payload["bytecode_evidence_fallback"]
    bytecode_section = f"\n\n{bytecode_evidence}" if bytecode_evidence else ""
    semantic_summaries = payload["semantic_summaries"]
    summary_section = semantic_summaries or ""
    telemetry_guidance = payload["telemetry_guidance"]
    telemetry_section = f"\n\n{telemetry_guidance}" if telemetry_guidance else ""
    action_pattern_guidance = payload["action_pattern_guidance"]
    action_pattern_section = f"\n\n{action_pattern_guidance}" if action_pattern_guidance else ""
    numbered_fragment = payload["line_numbered_source_fragment"]
    repair_context_section = _format_repair_context(repair_context, module_mode=qualname == "<module>")

    user_prompt = f"""Task: {task_text}

Target qualname: {qualname}

{INDENTATION_CONTRACT}

{source_label}:
```python
{numbered_fragment}
```

{summary_section}

{repair_context_section}

Code object metadata:
{metadata_text}{bytecode_section}{action_pattern_section}{telemetry_section}

Return only the repaired source fragment."""
    return [
        {"role": "system", "content": SEMANTIC_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class FragmentFixer(ABC):
    @abstractmethod
    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_candidates(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> list[Any]:
        return [
            self.generate_candidate(
                qualname=qualname,
                gt_code_object=gt_code_object,
                derived_code_object=derived_code_object,
                derived_source_fragment=derived_source_fragment,
                repair_context=repair_context,
            )
        ]


class OracleFragmentFixer(FragmentFixer):
    def __init__(self, gt_pyc: Path, gt_source: Path | None = None):
        self.gt_pyc = gt_pyc.expanduser().resolve()
        self.gt_source = gt_source.expanduser().resolve() if gt_source is not None else infer_source_from_pyc(self.gt_pyc)
        self.gt_source_text = self.gt_source.read_text(encoding="utf-8")

    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        del gt_code_object
        del derived_code_object
        del derived_source_fragment
        del repair_context
        row = _find_target_row(self.gt_source, self.gt_pyc, qualname, strict_map=False)
        return extract_source_segment(self.gt_source_text, row)


def _load_semantic_repair_telemetry_records(path: Path = ACCEPTED_CODE_OBJECT_TELEMETRY_PATH) -> list[dict]:
    records: list[dict] = []
    try:
        telemetry_path = path.expanduser().resolve()
    except Exception:
        return records
    if not telemetry_path.exists():
        return records
    try:
        with telemetry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return []
    return records


SEMANTIC_CANDIDATE_STRATEGIES = [
    (
        "retrieval_guided",
        "Follow the highest-ranked retrieved repair pattern when it matches the current instruction diff.",
    ),
    (
        "control_flow_residual",
        "Keep call, method, attribute, receiver, and argument expressions unchanged unless impossible. Change only jump, loop, branch, else/finally, break/continue, or block-placement structure.",
    ),
    (
        "call_attribute_residual",
        "Keep the existing branch/loop/else layout unchanged. Change only call, method, attribute, receiver, or argument expression shape.",
    ),
    (
        "metadata_literal_name_residual",
        "Keep control-flow and call structure unchanged. Change only constants, names, local variables, or literal/reference expressions shown by metadata.",
    ),
    (
        "conservative_minimal_delta",
        "Make the smallest local edit that changes the current bytecode evidence. Do not repeat any rejected or prior same-round candidate edit shape.",
    ),
]


def _candidate_strategy_specs(count: int, repair_context: dict | None) -> list[tuple[str, str]]:
    mismatch = _jump_target_mismatch(repair_context)
    if mismatch:
        if mismatch["direction"] == "later":
            block_prompt = (
                "The visible mismatch is a jump target that lands too late. Shrink the current branch/loop body by moving one existing statement out of it; do not change calls, arguments, receivers, or indentation only."
            )
        else:
            block_prompt = (
                "The visible mismatch is a jump target that lands too early. Expand the current branch/loop body by moving one existing statement into it; do not change calls, arguments, receivers, or indentation only."
            )
        strategies = [
            ("jump_target_block_boundary", block_prompt),
            (
                "control_flow_residual",
                "Change only block membership, loop/if/else placement, break/continue placement, or statement indentation that changes AST block structure; preserve expression text.",
            ),
            (
                "conservative_minimal_delta",
                "Make the smallest AST-structural edit that changes the jump target. Whitespace-only indentation changes do not count as a distinct repair.",
            ),
            (
                "metadata_literal_name_residual",
                "Only if block placement cannot explain the jump target, change the smallest literal/name reference shown by metadata.",
            ),
        ]
        return strategies[: max(1, min(count, len(strategies)))]
    return SEMANTIC_CANDIDATE_STRATEGIES[: max(1, min(count, len(SEMANTIC_CANDIDATE_STRATEGIES)))]


class LLMFragmentFixer(FragmentFixer):
    def __init__(self, *, provider: str = "Google", model: str = "gemini-2.5-flash-lite"):
        self.llm_config = find_llm_config(provider, model)
        self.calls: list[dict] = []
        self.prompt_output_dir: Path | None = None
        self._prompt_call_index = 0
        self.telemetry_records = _load_semantic_repair_telemetry_records()
        self.action_patterns = _load_semantic_action_patterns()

    def set_prompt_output_dir(self, prompt_output_dir: Path | None) -> None:
        self.prompt_output_dir = None if prompt_output_dir is None else prompt_output_dir.expanduser().resolve()
        self._prompt_call_index = 0

    def _record_prompt(self, *, prompt_record: dict[str, Any], qualname: str, repair_context: dict | None) -> None:
        if self.prompt_output_dir is not None:
            self.prompt_output_dir.mkdir(parents=True, exist_ok=True)
            self._prompt_call_index += 1
            safe_qualname = (
                qualname.replace("<", "").replace(">", "").replace(".", "_").replace("/", "_").replace("\\", "_")
            )
            prompt_path = self.prompt_output_dir / f"{self._prompt_call_index:04d}_{safe_qualname}.json"
            prompt_path.write_text(json.dumps(prompt_record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            prompt_record["prompt_path"] = str(prompt_path)
        self.calls.append(prompt_record)
        if repair_context is not None:
            repair_context["_llm_prompt_record"] = prompt_record

    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        fallback_candidate = _fallback_semantic_candidate(
            qualname=qualname,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
        )
        final_candidate = fallback_candidate
        for duplicate_retry_index in range(2):
            suppress_retrieval_guidance = bool(
                repair_context and repair_context.get("_suppress_action_pattern_guidance")
            )
            if suppress_retrieval_guidance:
                telemetry_guidance = ""
                action_pattern_guidance = ""
                if repair_context is not None:
                    repair_context["_selected_action_pattern_types"] = []
            else:
                telemetry_guidance = generate_semantic_repair_guidance(
                    qualname=qualname,
                    derived_code_object=derived_code_object,
                    repair_context=repair_context,
                    telemetry_records=self.telemetry_records,
                )
                action_pattern_guidance = generate_action_pattern_guidance(
                    qualname=qualname,
                    gt_code_object=gt_code_object,
                    derived_code_object=derived_code_object,
                    derived_source_fragment=derived_source_fragment,
                    repair_context=repair_context,
                    action_patterns=self.action_patterns,
                )
            if action_pattern_guidance:
                telemetry_guidance = ""
            prompt_reconstruction_inputs = build_semantic_repair_prompt_payload(
                qualname=qualname,
                gt_code_object=gt_code_object,
                derived_code_object=derived_code_object,
                derived_source_fragment=derived_source_fragment,
                repair_context=repair_context,
                telemetry_guidance=telemetry_guidance,
                action_pattern_guidance=action_pattern_guidance,
            )
            messages = build_semantic_repair_messages(
                qualname=qualname,
                gt_code_object=gt_code_object,
                derived_code_object=derived_code_object,
                derived_source_fragment=derived_source_fragment,
                repair_context=repair_context,
                telemetry_guidance=telemetry_guidance,
                action_pattern_guidance=action_pattern_guidance,
            )
            prompt_record = {
                "provider": self.llm_config["provider"],
                "model": self.llm_config["name"],
                "qualname": qualname,
                "prompt_variant": SEMANTIC_PROMPT_VARIANT,
                "candidate_index": None if not repair_context else repair_context.get("_candidate_index"),
                "candidate_strategy": None if not repair_context else repair_context.get("_candidate_strategy"),
                "candidate_strategy_prompt": None
                if not repair_context
                else repair_context.get("_candidate_strategy_prompt"),
                "suppressed_action_pattern_guidance": suppress_retrieval_guidance,
                "prior_same_round_candidate_count": 0
                if not repair_context
                else len(repair_context.get("_prior_candidate_outputs") or []),
                "duplicate_retry_index": duplicate_retry_index,
                "prompt_features": _prompt_feature_flags(messages[1]["content"] if len(messages) > 1 else "", repair_context),
                "prompt_reconstruction_inputs": prompt_reconstruction_inputs,
                "messages": messages,
                "system_prompt": messages[0]["content"] if messages else None,
                "user_prompt": messages[1]["content"] if len(messages) > 1 else None,
            }
            prompt_token_count = _semantic_prompt_token_count(messages, self.llm_config)
            token_limit = _semantic_prompt_token_limit(self.llm_config)
            token_threshold = _semantic_prompt_token_threshold(self.llm_config)
            skipped_due_to_token_limit = (
                token_threshold is not None and prompt_token_count > token_threshold
            )
            prompt_record.update(
                {
                    "prompt_token_count": prompt_token_count,
                    "token_limit_for_completion": token_limit,
                    "token_threshold_used": token_threshold,
                    "semantic_repair_token_headroom": SEMANTIC_REPAIR_TOKEN_HEADROOM,
                    "skipped_due_to_token_limit": skipped_due_to_token_limit,
                }
            )
            if skipped_due_to_token_limit:
                candidate = fallback_candidate
                candidate_label = (
                    f"candidate {prompt_record.get('candidate_index')}"
                    if prompt_record.get("candidate_index") is not None
                    else "single candidate"
                )
                strategy_label = prompt_record.get("candidate_strategy") or "default"
                print(
                    "[semantic_repair]   -> skipping LLM call for "
                    f"{qualname} {candidate_label} strategy={strategy_label}: "
                    f"prompt token count {prompt_token_count} exceeds threshold {token_threshold}",
                    flush=True,
                )
                prompt_record.update(
                    {
                        "latency_ms": 0,
                        "usage": None,
                        "response_text": "",
                        "line_number_stripped_text": "",
                        "returned_text": candidate,
                        "replacement_hash": _replacement_hash(candidate),
                        "replacement_structural_hash": _replacement_structural_hash(candidate),
                        "duplicate_rejected_candidate": False,
                        "duplicate_same_round_candidate": False,
                        "skip_reason": (
                            f"semantic repair prompt token count {prompt_token_count} exceeds "
                            f"threshold {token_threshold}"
                        ),
                    }
                )
                self._record_prompt(prompt_record=prompt_record, qualname=qualname, repair_context=repair_context)
                final_candidate = candidate
                break
            started = time.perf_counter()
            try:
                response = make_llm_call_from_config(messages, self.llm_config)
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                error_message = f"{type(exc).__name__}: {exc}"
                prompt_record.update(
                    {
                        "latency_ms": elapsed_ms,
                        "usage": None,
                        "response_text": "",
                        "line_number_stripped_text": "",
                        "returned_text": fallback_candidate,
                        "replacement_hash": _replacement_hash(fallback_candidate),
                        "replacement_structural_hash": _replacement_structural_hash(fallback_candidate),
                        "duplicate_rejected_candidate": False,
                        "duplicate_same_round_candidate": False,
                        "llm_call_error": error_message,
                    }
                )
                self._record_prompt(prompt_record=prompt_record, qualname=qualname, repair_context=repair_context)
                print(
                    f"[semantic_repair]   -> LLM call failed for {qualname}: {error_message}",
                    flush=True,
                )
                raise
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            response_text = strip_code_fences(response)
            cleaned_response_text = strip_prompt_line_numbers(response_text)
            content = cleaned_response_text.strip()
            candidate = content if content else fallback_candidate
            candidate_hash = _replacement_hash(candidate)
            candidate_structural_hash = _replacement_structural_hash(candidate)
            duplicate_rejected_candidate = _candidate_repeats_rejected_attempt(candidate, repair_context)
            same_round_seen_hashes = set()
            same_round_seen_structural_hashes = set()
            if repair_context is not None:
                same_round_seen_hashes = set(repair_context.get("_same_round_seen_hashes") or [])
                same_round_seen_structural_hashes = set(repair_context.get("_same_round_seen_structural_hashes") or [])
            duplicate_same_round_candidate = bool(
                (candidate_hash and candidate_hash in same_round_seen_hashes)
                or (candidate_structural_hash and candidate_structural_hash in same_round_seen_structural_hashes)
            )
            prompt_record.update(
                {
                    "latency_ms": elapsed_ms,
                    "usage": None if response is None else str(response.get("usage")),
                    "response_text": response_text,
                    "line_number_stripped_text": cleaned_response_text,
                    "returned_text": candidate,
                    "replacement_hash": candidate_hash,
                    "replacement_structural_hash": candidate_structural_hash,
                    "duplicate_rejected_candidate": duplicate_rejected_candidate,
                    "duplicate_same_round_candidate": duplicate_same_round_candidate,
                }
            )
            self._record_prompt(prompt_record=prompt_record, qualname=qualname, repair_context=repair_context)
            final_candidate = candidate
            if duplicate_rejected_candidate and duplicate_retry_index == 0 and repair_context is not None:
                repair_context["_force_distinct_retry"] = True
                repair_context["_force_distinct_retry_source"] = "rejected_attempt"
                repair_context["_duplicate_rejected_output_excerpt"] = _truncate_rejected_replacement_excerpt(candidate)
                continue
            if duplicate_same_round_candidate and duplicate_retry_index == 0 and repair_context is not None:
                repair_context["_force_distinct_retry"] = True
                repair_context["_force_distinct_retry_source"] = "same_round_candidate"
                repair_context["_duplicate_rejected_output_excerpt"] = _truncate_rejected_replacement_excerpt(candidate)
                continue
            break
        if repair_context is not None:
            repair_context.pop("_force_distinct_retry", None)
            repair_context.pop("_force_distinct_retry_source", None)
            repair_context.pop("_duplicate_rejected_output_excerpt", None)
        return final_candidate

    def generate_candidates(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> list[dict[str, Any]]:
        strategies = _candidate_strategy_specs(SEMANTIC_REPAIR_CANDIDATE_COUNT, repair_context)
        candidates: list[dict[str, Any]] = []
        prior_candidate_outputs: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        seen_structural_hashes: set[str] = set()
        for candidate_index, (strategy_name, strategy_prompt) in enumerate(strategies):
            candidate_context = dict(repair_context or {})
            candidate_context.pop("_llm_prompt_record", None)
            candidate_context["_candidate_index"] = candidate_index
            if len(strategies) > 1:
                candidate_context["_candidate_strategy"] = strategy_name
                candidate_context["_candidate_strategy_prompt"] = strategy_prompt
            if prior_candidate_outputs:
                candidate_context["_prior_candidate_outputs"] = prior_candidate_outputs[-2:]
            if seen_hashes:
                candidate_context["_same_round_seen_hashes"] = set(seen_hashes)
            if seen_structural_hashes:
                candidate_context["_same_round_seen_structural_hashes"] = set(seen_structural_hashes)
            if strategy_name != "retrieval_guided":
                candidate_context["_suppress_action_pattern_guidance"] = True
            text = self.generate_candidate(
                qualname=qualname,
                gt_code_object=gt_code_object,
                derived_code_object=derived_code_object,
                derived_source_fragment=derived_source_fragment,
                repair_context=candidate_context,
            )
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
            prior_candidate_outputs.append(
                {
                    "strategy": strategy_name,
                    "replacement_hash": candidate_hash,
                    "replacement_structural_hash": candidate_structural_hash,
                    "text": text,
                }
            )
            prompt_record = candidate_context.get("_llm_prompt_record") or {}
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "strategy": strategy_name,
                    "text": text,
                    "replacement_hash": candidate_hash,
                    "replacement_structural_hash": candidate_structural_hash,
                    "selected_action_types": candidate_context.get("_selected_action_pattern_types") or [],
                    "prompt_path": prompt_record.get("prompt_path") if isinstance(prompt_record, dict) else None,
                    "prompt_record": prompt_record if isinstance(prompt_record, dict) else None,
                    "duplicate_rejected_candidate": prompt_record.get("duplicate_rejected_candidate") if isinstance(prompt_record, dict) else None,
                    "skipped_due_to_token_limit": prompt_record.get("skipped_due_to_token_limit") if isinstance(prompt_record, dict) else None,
                    "skip_reason": prompt_record.get("skip_reason") if isinstance(prompt_record, dict) else None,
                    "prompt_token_count": prompt_record.get("prompt_token_count") if isinstance(prompt_record, dict) else None,
                    "token_threshold_used": prompt_record.get("token_threshold_used") if isinstance(prompt_record, dict) else None,
                }
            )
        return candidates or [{"candidate_index": 0, "strategy": "fallback", "text": derived_source_fragment}]


class CodeObjectRepairLoop:
    def __init__(self, fixer: FragmentFixer):
        self.fixer = fixer

    def _set_prompt_output_dir(self, output_dir: Path | None, derived_source: Path) -> None:
        prompt_dir: Path | None
        if output_dir is None:
            prompt_dir = derived_source.parent / f"{derived_source.stem}_repair_pipeline" / "prompts"
        else:
            prompt_dir = output_dir / "prompts"
        if hasattr(self.fixer, "set_prompt_output_dir"):
            getattr(self.fixer, "set_prompt_output_dir")(prompt_dir)

    def run(
        self,
        *,
        gt_pyc: Path,
        derived_pyc: Path,
        derived_source: Path,
        output_dir: Path | None = None,
        log_file: Path | None = None,
        run_id: str | None = None,
        file_hash: str | None = None,
        verify_with_pylingual: bool = True,
        verify_each_step_with_pylingual: bool = True,
        reject_non_improving_candidates: bool = True,
        max_iterations: int = 1,
        sample_timeout_deadline: float | None = None,
        sample_hard_timeout_deadline: float | None = None,
        sample_timeout_seconds: int | None = None,
        sample_hard_timeout_seconds: int | None = None,
        sample_timeout_min_improvement_delta: int = SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA,
    ) -> dict:
        self._set_prompt_output_dir(output_dir, derived_source)
        result = repair_mismatching_code_objects(
            gt_pyc=gt_pyc,
            derived_pyc=derived_pyc,
            derived_source=derived_source,
            output_dir=output_dir,
            log_file=log_file,
            run_id=run_id,
            file_hash=file_hash,
            fragment_fixer=self._fix_fragment,
            strict_map=False,
            verify_with_pylingual=verify_with_pylingual,
            verify_each_step_with_pylingual=verify_each_step_with_pylingual,
            reject_non_improving_candidates=reject_non_improving_candidates,
            max_iterations=max_iterations,
            sample_timeout_deadline=sample_timeout_deadline,
            sample_hard_timeout_deadline=sample_hard_timeout_deadline,
            sample_timeout_seconds=sample_timeout_seconds,
            sample_hard_timeout_seconds=sample_hard_timeout_seconds,
            sample_timeout_min_improvement_delta=sample_timeout_min_improvement_delta,
        )
        if hasattr(self.fixer, "calls"):
            result["llm_calls"] = getattr(self.fixer, "calls")
        return result

    def _fix_fragment(
        self,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> Any:
        if qualname != "<module>" and derived_code_object is not None and hasattr(self.fixer, "generate_candidates"):
            return getattr(self.fixer, "generate_candidates")(
                qualname=qualname,
                gt_code_object=gt_code_object,
                derived_code_object=derived_code_object,
                derived_source_fragment=derived_source_fragment,
                repair_context=repair_context,
            )
        return self.fixer.generate_candidate(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
            repair_context=repair_context,
        )


def _dataset_fieldnames() -> list[str]:
    return [
        "file_hash",
        "source",
        "error_type",
        "perfect_decompilation",
        "gt_pyc",
        "derived_pyc",
        "derived_source",
        "initial_combined_distance",
        "final_combined_distance",
        "initial_gt_code_object_count",
        "initial_derived_code_object_count",
        "final_gt_code_object_count",
        "final_derived_code_object_count",
        "repair_target_count",
        "accepted_step_count",
        "pylingual_all_equal",
        "error_message",
        "result_json",
    ]


def _dataset_deferred_fieldnames() -> list[str]:
    return [
        "file_hash",
        "source",
        "error_type",
        "defer_stage",
        "defer_reason",
        "gt_pyc",
        "derived_pyc",
        "derived_source",
        "initial_combined_distance",
        "initial_gt_code_object_count",
        "initial_derived_code_object_count",
        "repair_target_count",
        "missing_target_count",
        "extra_target_count",
        "expression_child_parent_target_count",
        "sample_timeout_seconds",
        "sample_hard_timeout_seconds",
        "sample_timeout_min_improvement_delta",
        "error_message",
    ]


def _dataset_result_row(row, result: dict, result_json_path: Path) -> dict:
    accepted_steps = sum(1 for step in result["steps"] if step["accepted"])
    verification = result.get("pylingual_verification")
    if verification is None:
        perfect_decompilation = None
    else:
        perfect_decompilation = bool(verification.get("all_equal") is True)
    return {
        "file_hash": row.file_hash,
        "source": row.source,
        "error_type": row.error_type,
        "perfect_decompilation": perfect_decompilation,
        "gt_pyc": result["gt_pyc"],
        "derived_pyc": result["derived_pyc"],
        "derived_source": result["derived_source"],
        "initial_combined_distance": result["initial_summary"]["combined_distance"],
        "final_combined_distance": result["final_summary"]["combined_distance"],
        "initial_gt_code_object_count": result["initial_summary"]["gt_code_object_count"],
        "initial_derived_code_object_count": result["initial_summary"]["derived_code_object_count"],
        "final_gt_code_object_count": result["final_summary"]["gt_code_object_count"],
        "final_derived_code_object_count": result["final_summary"]["derived_code_object_count"],
        "repair_target_count": len(result["repair_targets"]),
        "accepted_step_count": accepted_steps,
        "pylingual_all_equal": None if verification is None else verification["all_equal"],
        "error_message": None,
        "result_json": str(result_json_path),
    }


def _dataset_error_row(row, gt_pyc: Path | None, derived_pyc: Path | None, derived_source: Path | None, message: str) -> dict:
    return {
        "file_hash": row.file_hash,
        "source": row.source,
        "error_type": row.error_type,
        "perfect_decompilation": None,
        "gt_pyc": str(gt_pyc) if gt_pyc else None,
        "derived_pyc": str(derived_pyc) if derived_pyc else None,
        "derived_source": str(derived_source) if derived_source else None,
        "initial_combined_distance": None,
        "final_combined_distance": None,
        "initial_gt_code_object_count": None,
        "initial_derived_code_object_count": None,
        "final_gt_code_object_count": None,
        "final_derived_code_object_count": None,
        "repair_target_count": None,
        "accepted_step_count": None,
        "pylingual_all_equal": None,
        "error_message": message,
        "result_json": None,
    }


def _dataset_deferred_row(
    row,
    gt_pyc: Path | None,
    derived_pyc: Path | None,
    derived_source: Path | None,
    *,
    defer_stage: str,
    defer_reason: str,
    preflight: dict[str, Any] | None = None,
    error_message: str | None = None,
    sample_timeout_seconds: int | None = None,
    sample_hard_timeout_seconds: int | None = None,
    sample_timeout_min_improvement_delta: int | None = None,
) -> dict[str, Any]:
    preflight = preflight or {}
    return {
        "file_hash": row.file_hash,
        "source": row.source,
        "error_type": row.error_type,
        "defer_stage": defer_stage,
        "defer_reason": defer_reason,
        "gt_pyc": str(gt_pyc) if gt_pyc else None,
        "derived_pyc": str(derived_pyc) if derived_pyc else None,
        "derived_source": str(derived_source) if derived_source else None,
        "initial_combined_distance": preflight.get("initial_combined_distance"),
        "initial_gt_code_object_count": preflight.get("initial_gt_code_object_count"),
        "initial_derived_code_object_count": preflight.get("initial_derived_code_object_count"),
        "repair_target_count": preflight.get("repair_target_count"),
        "missing_target_count": preflight.get("missing_target_count"),
        "extra_target_count": preflight.get("extra_target_count"),
        "expression_child_parent_target_count": preflight.get("expression_child_parent_target_count"),
        "sample_timeout_seconds": sample_timeout_seconds,
        "sample_hard_timeout_seconds": sample_hard_timeout_seconds,
        "sample_timeout_min_improvement_delta": sample_timeout_min_improvement_delta,
        "error_message": error_message,
    }


def _semantic_repair_preflight(gt_pyc: Path, derived_pyc: Path) -> dict[str, Any]:
    distance_rows = compare_code_object_distances(gt_pyc, derived_pyc)
    summary = summarize_results(distance_rows)
    repair_targets = select_repair_targets(distance_rows, include_module=True)
    missing_targets = select_unreattachable_missing_targets(distance_rows)
    extra_targets = select_extra_repair_targets(distance_rows)
    expression_child_parent_records = select_missing_expression_child_parent_records(distance_rows)
    return {
        "initial_combined_distance": summary.get("combined_distance"),
        "initial_gt_code_object_count": summary.get("gt_code_object_count"),
        "initial_derived_code_object_count": summary.get("derived_code_object_count"),
        "repair_target_count": len(repair_targets),
        "missing_target_count": len(missing_targets),
        "extra_target_count": len(extra_targets),
        "expression_child_parent_target_count": len(expression_child_parent_records),
        "repair_targets": repair_targets,
        "missing_targets": missing_targets,
        "extra_targets": extra_targets,
        "expression_child_parent_targets": [
            record.get("parent_qualname") for record in expression_child_parent_records
        ],
    }


def _preflight_defer_reason(
    preflight: dict[str, Any],
    *,
    max_repair_targets: int | None,
    max_initial_combined_distance: int | None,
    allow_missing_targets: bool,
    allow_extra_targets: bool,
) -> str | None:
    reasons: list[str] = []
    initial_distance = preflight.get("initial_combined_distance")
    repair_target_count = preflight.get("repair_target_count")
    missing_target_count = preflight.get("missing_target_count")
    extra_target_count = preflight.get("extra_target_count")
    if (
        max_initial_combined_distance is not None
        and isinstance(initial_distance, (int, float))
        and initial_distance > max_initial_combined_distance
    ):
        reasons.append(
            f"initial_combined_distance {initial_distance} exceeds {max_initial_combined_distance}"
        )
    if (
        max_repair_targets is not None
        and isinstance(repair_target_count, int)
        and repair_target_count > max_repair_targets
    ):
        reasons.append(f"repair_target_count {repair_target_count} exceeds {max_repair_targets}")
    if not allow_missing_targets and missing_target_count:
        reasons.append(f"missing_target_count {missing_target_count} is nonzero")
    if not allow_extra_targets and extra_target_count:
        reasons.append(f"extra_target_count {extra_target_count} is nonzero")
    return "; ".join(reasons) if reasons else None


def _dataset_row_key(row) -> tuple[str, str]:
    return str(row.source), str(row.file_hash)


def _dataset_cell(row, name: str) -> str | None:
    if not hasattr(row, name):
        return None
    value = getattr(row, name)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _dataset_path_cell(row, name: str, base_dir: Path) -> Path | None:
    value = _dataset_cell(row, name)
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_dataset_repair_paths(row, dataset_base_dir: Path) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    gt_pyc = _dataset_path_cell(row, "gt_pyc", dataset_base_dir)
    derived_pyc = _dataset_path_cell(row, "derived_pyc", dataset_base_dir)
    derived_source = _dataset_path_cell(row, "derived_source", dataset_base_dir)
    if gt_pyc is not None and derived_pyc is not None and derived_source is not None:
        gt_source = _dataset_path_cell(row, "gt_source", dataset_base_dir)
        if gt_source is None:
            gt_source = fetch_pyllmpatch_source_path(row.file_hash, row.source)
        return gt_source, gt_pyc, derived_pyc, derived_source

    gt_source = fetch_pyllmpatch_source_path(row.file_hash, row.source)
    gt_pyc, derived_pyc, derived_source = fetch_pyllmpatch_repair_paths(row.file_hash, row.source)
    return gt_source, gt_pyc, derived_pyc, derived_source


def _preflight_easy_sort_key(preflight: dict[str, Any] | None) -> tuple[int, int, int, int]:
    if not preflight:
        return (1, 1_000_000, 1_000_000_000, 1_000_000)
    topology_penalty = int(bool(preflight.get("missing_target_count") or preflight.get("extra_target_count")))
    repair_target_count = preflight.get("repair_target_count")
    initial_distance = preflight.get("initial_combined_distance")
    expression_parent_count = preflight.get("expression_child_parent_target_count")
    return (
        topology_penalty,
        int(repair_target_count) if isinstance(repair_target_count, int) else 1_000_000,
        int(initial_distance) if isinstance(initial_distance, (int, float)) else 1_000_000_000,
        int(expression_parent_count) if isinstance(expression_parent_count, int) else 1_000_000,
    )


def _filter_semantic_dataset_rows(
    df: pd.DataFrame,
    *,
    source: str | None = None,
    file_hash: str | None = None,
    row_range: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    semantic_df = df[df["error_type"] == "semantic_error"].copy()
    if source is not None:
        semantic_df = semantic_df[semantic_df["source"].astype(str).str.lower() == str(source).lower()]
    if file_hash is not None:
        semantic_df = semantic_df[semantic_df["file_hash"].astype(str) == str(file_hash)]
    if row_range is not None:
        semantic_df = semantic_df.iloc[_parse_row_range(row_range)]
    if limit is not None:
        semantic_df = semantic_df.head(limit)
    return semantic_df


def _parse_row_range(row_range: str) -> slice:
    value = str(row_range).strip()
    if not value:
        raise ValueError("--row-range cannot be empty")

    def parse_bound(raw: str) -> int | None:
        raw = raw.strip()
        if raw == "":
            return None
        bound = int(raw)
        if bound < 0:
            raise ValueError("--row-range only accepts non-negative indexes")
        return bound

    if ":" not in value:
        index = parse_bound(value)
        if index is None:
            raise ValueError("--row-range must be an index or START:END")
        return slice(index, index + 1)

    start_raw, end_raw = value.split(":", 1)
    start = parse_bound(start_raw)
    end = parse_bound(end_raw)
    if start is not None and end is not None and start > end:
        raise ValueError("--row-range START must be <= END")
    return slice(start, end)


def run_dataset_repair_loop(
    *,
    fixer_name: str,
    dataset_path: Path = BASE_DATASET_PATH,
    output_dir: Path | None = None,
    limit: int | None = None,
    file_hash: str | None = None,
    source: str | None = None,
    row_range: str | None = None,
    verify_with_pylingual: bool = True,
    verify_each_step_with_pylingual: bool = True,
    reject_non_improving_candidates: bool = True,
    max_iterations: int = 1,
    llm_provider: str = "Google",
    llm_model: str = "gemini-2.5-flash-lite",
    sample_timeout_seconds: int = SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS,
    sample_hard_timeout_seconds: int = SEMANTIC_REPAIR_SAMPLE_HARD_TIMEOUT_SECONDS,
    sample_timeout_min_improvement_delta: int = SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA,
    defer_preflight_risky_samples: bool = False,
    defer_timeout_no_improvement: bool = False,
    process_easy_cases_first: bool = False,
    preflight_max_repair_targets: int | None = SEMANTIC_REPAIR_PREFLIGHT_MAX_REPAIR_TARGETS,
    preflight_max_initial_combined_distance: int | None = SEMANTIC_REPAIR_PREFLIGHT_MAX_INITIAL_COMBINED_DISTANCE,
    preflight_allow_missing_targets: bool = False,
    preflight_allow_extra_targets: bool = False,
) -> dict:
    dataset_path = dataset_path.expanduser().resolve()
    sample_timeout_seconds = max(0, int(sample_timeout_seconds))
    sample_hard_timeout_seconds = max(0, int(sample_hard_timeout_seconds))
    sample_timeout_min_improvement_delta = max(1, int(sample_timeout_min_improvement_delta))
    preflight_max_repair_targets = (
        None
        if preflight_max_repair_targets is None
        else max(0, int(preflight_max_repair_targets))
    )
    preflight_max_initial_combined_distance = (
        None
        if preflight_max_initial_combined_distance is None
        else max(0, int(preflight_max_initial_combined_distance))
    )
    if output_dir is None:
        run_id, log_base, log_file = build_run_paths(current_run_timestamp())
    else:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = output_dir.name
        log_base = output_dir
        log_file = output_dir / f"run_log_{run_id}_{dataset_path.stem}.jsonl"
    results_csv = log_base / f"semantic_repair_results_{dataset_path.stem}.csv"
    deferred_csv = log_base / f"semantic_repair_deferred_{dataset_path.stem}.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    append_log(
        log_file,
        {
            "run_id": run_id,
            "timestamp": now_iso(),
            "mode": "semantic_repair",
            "stage": "start",
            "dataset_path": str(dataset_path),
            "output_dir": str(log_base),
            "sample_timeout_seconds": sample_timeout_seconds,
            "sample_hard_timeout_seconds": sample_hard_timeout_seconds,
            "sample_timeout_min_improvement_delta": sample_timeout_min_improvement_delta,
            "defer_preflight_risky_samples": defer_preflight_risky_samples,
            "defer_timeout_no_improvement": defer_timeout_no_improvement,
            "process_easy_cases_first": process_easy_cases_first,
            "preflight_max_repair_targets": preflight_max_repair_targets,
            "preflight_max_initial_combined_distance": preflight_max_initial_combined_distance,
            "preflight_allow_missing_targets": preflight_allow_missing_targets,
            "preflight_allow_extra_targets": preflight_allow_extra_targets,
        },
    )

    df = pd.read_csv(dataset_path)
    semantic_df = _filter_semantic_dataset_rows(df, source=source, file_hash=file_hash, row_range=row_range, limit=limit)

    processed = 0
    repaired = 0
    failed = 0
    timed_out = 0
    hard_timed_out = 0
    deferred = 0
    timeout_label = f"{sample_timeout_seconds} seconds" if sample_timeout_seconds > 0 else "disabled"
    hard_timeout_label = (
        f"{sample_hard_timeout_seconds} seconds" if sample_hard_timeout_seconds > 0 else "disabled"
    )
    print(
        f"[semantic_repair] per-sample timeout: checkpoint={timeout_label}; "
        f"hard_cap={hard_timeout_label}; min_delta={sample_timeout_min_improvement_delta}",
        flush=True,
    )
    row_items = list(semantic_df.itertuples(index=False))
    preflight_plan: dict[tuple[str, str], dict[str, Any]] = {}
    if process_easy_cases_first:
        scored_rows = []
        preflight_total = len(row_items)
        preflight_progress_interval = (
            1 if preflight_total <= 20 else max(10, min(100, preflight_total // 20))
        )
        print(
            f"[semantic_repair] preflighting {preflight_total} rows for easy-first ordering",
            flush=True,
        )
        for preflight_index, row in enumerate(row_items, start=1):
            plan: dict[str, Any] = {}
            try:
                planned_gt_source, planned_gt_pyc, planned_derived_pyc, planned_derived_source = (
                    _resolve_dataset_repair_paths(row, dataset_path.parent)
                )
                plan.update(
                    {
                        "gt_source": planned_gt_source,
                        "gt_pyc": planned_gt_pyc,
                        "derived_pyc": planned_derived_pyc,
                        "derived_source": planned_derived_source,
                    }
                )
                if planned_gt_pyc is not None and planned_derived_pyc is not None:
                    plan["preflight"] = _semantic_repair_preflight(planned_gt_pyc, planned_derived_pyc)
            except Exception as exc:
                plan["preflight_error"] = f"{type(exc).__name__}: {exc}"
            preflight_plan[_dataset_row_key(row)] = plan
            sort_key = _preflight_easy_sort_key(plan.get("preflight"))
            scored_rows.append((sort_key, row))
            if (
                preflight_index == 1
                or preflight_index == preflight_total
                or preflight_index % preflight_progress_interval == 0
            ):
                status = "error" if "preflight_error" in plan else f"sort_key={sort_key}"
                print(
                    "[semantic_repair] preflight easy-first "
                    f"{preflight_index}/{preflight_total} "
                    f"file_hash={str(row.file_hash)[:16]} {status}",
                    flush=True,
                )
        row_items = [row for _, row in sorted(scored_rows, key=lambda item: item[0])]
        print(
            f"[semantic_repair] processing easy cases first using preflight scores for {len(row_items)} rows",
            flush=True,
        )

    with results_csv.open("w", newline="", encoding="utf-8") as handle, deferred_csv.open(
        "w", newline="", encoding="utf-8"
    ) as deferred_handle:
        writer = csv.DictWriter(handle, fieldnames=_dataset_fieldnames())
        deferred_writer = csv.DictWriter(deferred_handle, fieldnames=_dataset_deferred_fieldnames())
        writer.writeheader()
        deferred_writer.writeheader()

        for row in row_items:
            processed += 1
            gt_pyc = None
            derived_pyc = None
            derived_source = None
            preflight: dict[str, Any] | None = None
            try:
                sample_timeout_deadline = (
                    time.monotonic() + sample_timeout_seconds if sample_timeout_seconds > 0 else None
                )
                sample_hard_timeout_deadline = (
                    time.monotonic() + sample_hard_timeout_seconds if sample_hard_timeout_seconds > 0 else None
                )
                plan = preflight_plan.get(_dataset_row_key(row), {})
                gt_source = plan.get("gt_source")
                gt_pyc = plan.get("gt_pyc")
                derived_pyc = plan.get("derived_pyc")
                derived_source = plan.get("derived_source")
                if not plan:
                    gt_source, gt_pyc, derived_pyc, derived_source = _resolve_dataset_repair_paths(
                        row,
                        dataset_path.parent,
                    )
                preflight = plan.get("preflight")
                if gt_source is None or gt_pyc is None or derived_pyc is None or derived_source is None:
                    error_row = _dataset_error_row(row, gt_pyc, derived_pyc, derived_source, "Could not resolve gt_source, gt_pyc, derived_pyc, and/or derived_source")
                    writer.writerow(error_row)
                    append_log(
                        log_file,
                        {
                            "run_id": run_id,
                            "timestamp": now_iso(),
                            "mode": "semantic_repair",
                            "stage": "error",
                            **error_row,
                        },
                    )
                    failed += 1
                    continue

                if defer_preflight_risky_samples:
                    preflight = plan.get("preflight")
                    if preflight is None:
                        preflight = _semantic_repair_preflight(gt_pyc, derived_pyc)
                    defer_reason = _preflight_defer_reason(
                        preflight,
                        max_repair_targets=preflight_max_repair_targets,
                        max_initial_combined_distance=preflight_max_initial_combined_distance,
                        allow_missing_targets=preflight_allow_missing_targets,
                        allow_extra_targets=preflight_allow_extra_targets,
                    )
                    if defer_reason:
                        deferred_row = _dataset_deferred_row(
                            row,
                            gt_pyc,
                            derived_pyc,
                            derived_source,
                            defer_stage="preflight",
                            defer_reason=defer_reason,
                            preflight=preflight,
                            sample_timeout_seconds=sample_timeout_seconds,
                            sample_hard_timeout_seconds=sample_hard_timeout_seconds,
                            sample_timeout_min_improvement_delta=sample_timeout_min_improvement_delta,
                        )
                        deferred_writer.writerow(deferred_row)
                        append_log(
                            log_file,
                            {
                                "run_id": run_id,
                                "timestamp": now_iso(),
                                "mode": "semantic_repair",
                                "stage": "deferred",
                                "defer_policy": "preflight_risky_sample",
                                "preflight": preflight,
                                **deferred_row,
                            },
                        )
                        print(
                            f"[semantic_repair] file_hash={row.file_hash} -> deferred by preflight: {defer_reason}",
                            flush=True,
                        )
                        deferred += 1
                        continue

                row_output_dir = log_base / "semantic_repair" / str(row.source) / str(row.file_hash)
                row_output_dir.mkdir(parents=True, exist_ok=True)
                result_json_path = row_output_dir / "result.json"
                if fixer_name not in {"oracle", "llm"}:
                    raise ValueError(f"Unsupported fixer backend: {fixer_name}")
                fixer = (
                    OracleFragmentFixer(gt_pyc, gt_source)
                    if fixer_name == "oracle"
                    else LLMFragmentFixer(provider=llm_provider, model=llm_model)
                )
                loop = CodeObjectRepairLoop(fixer)
                print(f"[semantic_repair] file_hash={row.file_hash}", flush=True)
                result = loop.run(
                    gt_pyc=gt_pyc,
                    derived_pyc=derived_pyc,
                    derived_source=derived_source,
                    output_dir=row_output_dir,
                    log_file=log_file,
                    run_id=run_id,
                    file_hash=str(row.file_hash),
                    verify_with_pylingual=verify_with_pylingual,
                    verify_each_step_with_pylingual=verify_each_step_with_pylingual,
                    reject_non_improving_candidates=reject_non_improving_candidates,
                    max_iterations=max_iterations,
                    sample_timeout_deadline=sample_timeout_deadline,
                    sample_hard_timeout_deadline=sample_hard_timeout_deadline,
                    sample_timeout_seconds=sample_timeout_seconds,
                    sample_hard_timeout_seconds=sample_hard_timeout_seconds,
                    sample_timeout_min_improvement_delta=sample_timeout_min_improvement_delta,
                )
                result_json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
                result_row = _dataset_result_row(row, result, result_json_path)
                if result.get("sample_hard_timeout_reached"):
                    timed_out += 1
                    hard_timed_out += 1
                    print(
                        f"[semantic_repair] file_hash={row.file_hash} -> hard timeout reached after combined-distance improvement; kept best result",
                        flush=True,
                    )
                elif result.get("sample_timeout_checkpoint_count"):
                    print(
                        f"[semantic_repair] file_hash={row.file_hash} -> timeout checkpoint passed with combined-distance improvement; sample completed normally",
                        flush=True,
                    )
                writer.writerow(result_row)
                append_log(
                    log_file,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "mode": "semantic_repair",
                        "stage": "result",
                        "sample_timeout_seconds": sample_timeout_seconds,
                        "sample_hard_timeout_seconds": sample_hard_timeout_seconds,
                        "sample_timeout_min_improvement_delta": sample_timeout_min_improvement_delta,
                        "sample_timeout_reached": result.get("sample_timeout_reached"),
                        "sample_hard_timeout_reached": result.get("sample_hard_timeout_reached"),
                        "sample_timeout_checkpoint_count": result.get("sample_timeout_checkpoint_count"),
                        "sample_timed_out": result.get("sample_timed_out"),
                        "sample_timeout_action": result.get("sample_timeout_action"),
                        "sample_timeout_reason": result.get("sample_timeout_reason"),
                        "sample_timeout_best_combined_distance": result.get("sample_timeout_best_combined_distance"),
                        "sample_timeout_best_improvement_reason": result.get("sample_timeout_best_improvement_reason"),
                        **result_row,
                    },
                )
                repaired += 1
            except SemanticRepairTimeoutNoImprovement as exc:
                timeout_message = f"{type(exc).__name__}: {exc}"
                hard_timeout_reached = isinstance(exc, SemanticRepairHardTimeoutNoImprovement)
                print(
                    f"[semantic_repair] file_hash={row.file_hash} -> {exc}; skipping sample",
                    flush=True,
                )
                if defer_timeout_no_improvement:
                    deferred_row = _dataset_deferred_row(
                        row,
                        gt_pyc,
                        derived_pyc,
                        derived_source,
                        defer_stage="timeout_no_improvement",
                        defer_reason="sample timed out without combined-distance improvement",
                        preflight=preflight,
                        error_message=timeout_message,
                        sample_timeout_seconds=sample_timeout_seconds,
                        sample_hard_timeout_seconds=sample_hard_timeout_seconds,
                        sample_timeout_min_improvement_delta=sample_timeout_min_improvement_delta,
                    )
                    deferred_writer.writerow(deferred_row)
                    append_log(
                        log_file,
                        {
                            "run_id": run_id,
                            "timestamp": now_iso(),
                            "mode": "semantic_repair",
                            "stage": "deferred",
                            "defer_policy": "timeout_no_improvement",
                            "sample_hard_timeout_reached": hard_timeout_reached,
                            "sample_timeout_policy": "defer_without_combined_distance_improvement",
                            **deferred_row,
                        },
                    )
                    deferred += 1
                    timed_out += 1
                    if hard_timeout_reached:
                        hard_timed_out += 1
                    continue
                error_row = _dataset_error_row(row, gt_pyc, derived_pyc, derived_source, timeout_message)
                writer.writerow(error_row)
                append_log(
                    log_file,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "mode": "semantic_repair",
                        "stage": "timeout",
                        "sample_timeout_seconds": sample_timeout_seconds,
                        "sample_hard_timeout_seconds": sample_hard_timeout_seconds,
                        "sample_timeout_min_improvement_delta": sample_timeout_min_improvement_delta,
                        "sample_hard_timeout_reached": hard_timeout_reached,
                        "sample_timeout_policy": "skip_without_combined_distance_improvement",
                        **error_row,
                    },
                )
                failed += 1
                timed_out += 1
                if hard_timeout_reached:
                    hard_timed_out += 1
            except Exception as exc:
                error_row = _dataset_error_row(row, gt_pyc, derived_pyc, derived_source, f"{type(exc).__name__}: {exc}")
                writer.writerow(error_row)
                append_log(
                    log_file,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "mode": "semantic_repair",
                        "stage": "error",
                        **error_row,
                    },
                )
                failed += 1

    return {
        "dataset_path": str(dataset_path),
        "output_dir": str(log_base),
        "results_csv": str(results_csv),
        "deferred_csv": str(deferred_csv),
        "run_log": str(log_file),
        "run_id": run_id,
        "processed_rows": processed,
        "repaired_rows": repaired,
        "failed_rows": failed,
        "deferred_rows": deferred,
        "timed_out_rows": timed_out,
        "hard_timed_out_rows": hard_timed_out,
        "sample_timeout_seconds": sample_timeout_seconds,
        "sample_hard_timeout_seconds": sample_hard_timeout_seconds,
        "sample_timeout_min_improvement_delta": sample_timeout_min_improvement_delta,
        "defer_preflight_risky_samples": defer_preflight_risky_samples,
        "defer_timeout_no_improvement": defer_timeout_no_improvement,
        "process_easy_cases_first": process_easy_cases_first,
        "preflight_max_repair_targets": preflight_max_repair_targets,
        "preflight_max_initial_combined_distance": preflight_max_initial_combined_distance,
        "preflight_allow_missing_targets": preflight_allow_missing_targets,
        "preflight_allow_extra_targets": preflight_allow_extra_targets,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the per-code-object repair loop with a pluggable fragment fixer."
    )
    parser.add_argument("gt_pyc", type=Path, nargs="?", help="Ground-truth .pyc path")
    parser.add_argument("derived_pyc", type=Path, nargs="?", help="Derived .pyc path")
    parser.add_argument("derived_source", type=Path, nargs="?", help="Derived source .py path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for intermediate repaired files and fragments",
    )
    parser.add_argument(
        "--skip-pylingual-verification",
        action="store_true",
        help="Disable final and per-step PyLingual equivalence checks",
    )
    parser.add_argument(
        "--skip-step-verification",
        action="store_true",
        help="Disable per-step PyLingual checks while keeping final verification enabled",
    )
    parser.add_argument(
        "--keep-non-improving",
        action="store_true",
        help="Retain candidates even when they do not improve the measured state",
    )
    parser.add_argument(
        "--fixer",
        choices=("oracle", "llm"),
        default="oracle",
        help="Fragment fixer backend to use",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="Google",
        help="Provider from utils.providers for --fixer llm",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Model name from utils.providers for --fixer llm",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum semantic repair iterations over recomputed mismatch targets",
    )
    parser.add_argument(
        "--sample-timeout-seconds",
        type=int,
        default=SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS,
        help=(
            "Timeout checkpoint for one dataset-mode sample in seconds; "
            "skip only if no combined-distance improvement exists. "
            f"Defaults to {SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS}; set 0 to disable."
        ),
    )
    parser.add_argument(
        "--sample-hard-timeout-seconds",
        type=int,
        default=SEMANTIC_REPAIR_SAMPLE_HARD_TIMEOUT_SECONDS,
        help=(
            "Absolute hard cap for one dataset-mode sample in seconds. "
            f"Defaults to {SEMANTIC_REPAIR_SAMPLE_HARD_TIMEOUT_SECONDS}; set 0 to disable."
        ),
    )
    parser.add_argument(
        "--sample-timeout-min-improvement-delta",
        type=int,
        default=SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA,
        help=(
            "Minimum combined-distance decrease counted as timeout progress. "
            f"Defaults to {SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA}."
        ),
    )
    parser.add_argument(
        "--defer-preflight-risky-samples",
        action="store_true",
        help=(
            "In dataset mode, write samples that fail the cheap bytecode-distance preflight to "
            "semantic_repair_deferred_*.csv instead of processing them."
        ),
    )
    parser.add_argument(
        "--defer-timeout-no-improvement",
        action="store_true",
        help=(
            "In dataset mode, write timeout-without-improvement samples to "
            "semantic_repair_deferred_*.csv instead of the main results CSV."
        ),
    )
    parser.add_argument(
        "--process-easy-cases-first",
        action="store_true",
        help=(
            "In dataset mode, compute cheap preflight metrics and process likely easy samples first "
            "before larger or topology-mismatched samples."
        ),
    )
    parser.add_argument(
        "--preflight-max-repair-targets",
        type=int,
        default=SEMANTIC_REPAIR_PREFLIGHT_MAX_REPAIR_TARGETS,
        help=(
            "Maximum initial repair targets allowed by --defer-preflight-risky-samples. "
            f"Defaults to {SEMANTIC_REPAIR_PREFLIGHT_MAX_REPAIR_TARGETS}."
        ),
    )
    parser.add_argument(
        "--preflight-max-initial-combined-distance",
        type=int,
        default=SEMANTIC_REPAIR_PREFLIGHT_MAX_INITIAL_COMBINED_DISTANCE,
        help=(
            "Maximum initial combined distance allowed by --defer-preflight-risky-samples. "
            f"Defaults to {SEMANTIC_REPAIR_PREFLIGHT_MAX_INITIAL_COMBINED_DISTANCE}."
        ),
    )
    parser.add_argument(
        "--preflight-allow-missing-targets",
        action="store_true",
        help="Allow preflight samples with initial missing code-object targets.",
    )
    parser.add_argument(
        "--preflight-allow-extra-targets",
        action="store_true",
        help="Allow preflight samples with initial extra derived code-object targets.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full loop result as JSON",
    )
    parser.add_argument(
        "--dataset-mode",
        action="store_true",
        help="Run semantic repair for semantic_error rows in the env-configured dataset",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=BASE_DATASET_PATH,
        help=f"Dataset CSV path. Defaults to {BASE_DATASET_PATH}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for dataset-mode rows",
    )
    parser.add_argument(
        "--row-range",
        "--range",
        dest="row_range",
        type=str,
        default=None,
        help="Optional dataset-mode row range after filters, before --limit. Uses zero-based START:END slicing, e.g. 10:20, 10:, :20, or 10.",
    )
    parser.add_argument(
        "--file-hash",
        type=str,
        default=None,
        help="Optional file hash filter for dataset-mode",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional source filter for dataset-mode (for example VirusTotal, pylingual, or PyPi)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.fixer not in {"oracle", "llm"}:
        raise ValueError(f"Unsupported fixer backend: {args.fixer}")

    if args.dataset_mode:
        if args.output_dir is None:
            raise ValueError("--output-dir is required in --dataset-mode")
        result = run_dataset_repair_loop(
            fixer_name=args.fixer,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            limit=args.limit,
            file_hash=args.file_hash,
            source=args.source,
            row_range=args.row_range,
            verify_with_pylingual=not args.skip_pylingual_verification,
            verify_each_step_with_pylingual=not args.skip_step_verification,
            reject_non_improving_candidates=not args.keep_non_improving,
            max_iterations=args.max_iterations,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            sample_timeout_seconds=args.sample_timeout_seconds,
            sample_hard_timeout_seconds=args.sample_hard_timeout_seconds,
            sample_timeout_min_improvement_delta=args.sample_timeout_min_improvement_delta,
            defer_preflight_risky_samples=args.defer_preflight_risky_samples,
            defer_timeout_no_improvement=args.defer_timeout_no_improvement,
            process_easy_cases_first=args.process_easy_cases_first,
            preflight_max_repair_targets=args.preflight_max_repair_targets,
            preflight_max_initial_combined_distance=args.preflight_max_initial_combined_distance,
            preflight_allow_missing_targets=args.preflight_allow_missing_targets,
            preflight_allow_extra_targets=args.preflight_allow_extra_targets,
        )
    else:
        if args.gt_pyc is None or args.derived_pyc is None or args.derived_source is None:
            raise ValueError("gt_pyc, derived_pyc, and derived_source are required unless --dataset-mode is used")
        fixer = (
            OracleFragmentFixer(args.gt_pyc)
            if args.fixer == "oracle"
            else LLMFragmentFixer(provider=args.llm_provider, model=args.llm_model)
        )
        loop = CodeObjectRepairLoop(fixer)
        result = loop.run(
            gt_pyc=args.gt_pyc,
            derived_pyc=args.derived_pyc,
            derived_source=args.derived_source,
            output_dir=args.output_dir,
            verify_with_pylingual=not args.skip_pylingual_verification,
            verify_each_step_with_pylingual=not args.skip_step_verification,
            reject_non_improving_candidates=not args.keep_non_improving,
            max_iterations=args.max_iterations,
        )

    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
