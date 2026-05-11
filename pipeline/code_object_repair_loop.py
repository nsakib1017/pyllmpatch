from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
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
from utils.reattach_source_code_object import (
    _find_target_row,
    extract_source_segment,
    infer_source_from_pyc,
    repair_mismatching_code_objects,
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

INDENTATION_CONTRACT = """Indentation contract:
- Preserve the first line's relative indentation exactly as shown.
- If the fragment starts with `def`, `async def`, or `class`, do not dedent or reindent that header.
- Only adjust indentation inside the fragment when needed to make the repaired Python valid.
- Return a complete replacement for the provided fragment, not only the changed lines.
- Do not include source line-number prefixes such as `12|` in the returned fragment."""

BYTECODE_PROMPT_WINDOW_RADIUS = 12
BYTECODE_PROMPT_MAX_LINES_PER_SIDE = 30
BYTECODE_PROMPT_FULL_LISTING_MAX_INSTRUCTIONS = 80


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
    if not repair_context:
        return ""
    rejected_attempts = repair_context.get("rejected_attempts") or []
    if module_mode and rejected_attempts:
        rejected_attempts = rejected_attempts[-1:]
    rejected_text = "<none>"
    if rejected_attempts:
        rejected_text = "\n\n".join(
            (
                f"Attempt {item.get('attempt')} rejected: {item.get('acceptance_reason')}\n"
                f"Replacement:\n```python\n{item.get('replacement_text', '')}\n```"
            )
            for item in rejected_attempts
        )
    failed = repair_context.get("pylingual_failed_result") or {}
    return f"""Failure context:
- target_kind: {repair_context.get("target_kind")}
- qualname: {repair_context.get("qualname")}
- localized_line_number: {repair_context.get("localized_line_number")}
- failed_offset: {repair_context.get("failed_offset")}
- alignment_tag: {repair_context.get("alignment_tag")}
- pylingual_message: {failed.get("message")}

Instruction diff:
```text
{repair_context.get("instruction_diff", "<unavailable>")}
```

Previous rejected attempts:
{rejected_text}
"""


def _prompt_feature_flags(user_prompt: str, repair_context: dict | None) -> dict[str, bool]:
    return {
        "indentation_contract": "Indentation contract:" in user_prompt,
        "line_numbered_source": bool(re.search(r"(?m)^\d{3,}\| ", user_prompt)),
        "compact_metadata": "Identical fields:" in user_prompt or "co_firstlineno:" not in user_prompt,
        "localized_instruction_diff": _instruction_diff_available(repair_context),
        "bytecode_evidence_fallback": "Bytecode evidence fallback:" in user_prompt,
        "rejected_attempt_feedback": bool(repair_context and repair_context.get("rejected_attempts")),
        "gt_instruction_window": bool(repair_context and repair_context.get("gt_instruction_window")),
        "derived_instruction_window": bool(repair_context and repair_context.get("derived_instruction_window")),
        "telemetry_guidance": "Guidance from prior accepted repairs:" in user_prompt,
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

    repair_ops = _top_values([case.get("repair_operation") for case in similar_cases])
    target_kinds = _top_values([case.get("target_kind") for case in similar_cases])
    acceptance_reasons = _top_values([case.get("acceptance_reason") for case in similar_cases], limit=2)
    candidate_kinds = _top_values([_nested_get(case, ("reattachment", "candidate_kind")) for case in similar_cases])
    parse_transitions = _top_values([_nested_get(case, ("feature_delta", "parse_transition")) for case in similar_cases])
    names_added = _top_list_values(similar_cases, ("feature_delta", "names_added"))
    names_removed = _top_list_values(similar_cases, ("feature_delta", "names_removed"))
    statements_added = _top_list_values(similar_cases, ("feature_delta", "statement_types_added"))
    statements_removed = _top_list_values(similar_cases, ("feature_delta", "statement_types_removed"))
    control_flow_added = _top_list_values(similar_cases, ("feature_delta", "control_flow_added"))
    control_flow_removed = _top_list_values(similar_cases, ("feature_delta", "control_flow_removed"))
    distance_deltas = [
        _nested_get(case, ("semantic_delta", "combined_distance_delta"))
        for case in similar_cases
        if isinstance(_nested_get(case, ("semantic_delta", "combined_distance_delta")), int)
    ]

    lines = [
        "Guidance from prior accepted repairs:",
        f"- Matched accepted telemetry cases: {len(similar_cases)} for prompt variant {current_case.get('prompt_variant')}.",
    ]
    if repair_ops:
        lines.append(f"- Similar repair operations: {', '.join(repair_ops)}.")
    if target_kinds:
        lines.append(f"- Similar target kinds: {', '.join(target_kinds)}.")
    if acceptance_reasons:
        lines.append(f"- Common acceptance reasons: {'; '.join(acceptance_reasons)}.")
    if candidate_kinds:
        lines.append(f"- Successful reattachment kinds: {', '.join(candidate_kinds)}.")
    if parse_transitions:
        lines.append(f"- Parse transitions seen: {', '.join(parse_transitions)}.")
    if distance_deltas:
        best_delta = min(distance_deltas)
        median_delta = sorted(distance_deltas)[len(distance_deltas) // 2]
        lines.append(f"- Combined semantic distance deltas ranged down to {best_delta}; median matched delta was {median_delta}.")
    if names_added:
        lines.append(f"- Names often added: {', '.join(names_added)}.")
    if names_removed:
        lines.append(f"- Names often removed: {', '.join(names_removed)}.")
    if statements_added:
        lines.append(f"- Statement types often added: {', '.join(statements_added)}.")
    if statements_removed:
        lines.append(f"- Statement types often removed: {', '.join(statements_removed)}.")
    if control_flow_added:
        lines.append(f"- Control-flow nodes often added: {', '.join(control_flow_added)}.")
    if control_flow_removed:
        lines.append(f"- Control-flow nodes often removed: {', '.join(control_flow_removed)}.")

    operation = current_case.get("repair_operation")
    if operation == "insert_missing":
        lines.append("- For this missing-target variant, synthesize only the missing fragment that fits the provided insertion context.")
    elif operation == "repair_module_statement":
        lines.append("- For this module-statement variant, keep the edit localized to the shown top-level statement.")
    else:
        lines.append("- For this source-fragment variant, prefer the smallest local edit that improves bytecode alignment.")
    lines.append("- Treat telemetry as a weak prior; source, metadata, and bytecode evidence take precedence.")
    return "\n".join(lines)


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
        "telemetry_guidance": telemetry_guidance or None,
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
    )
    task_text = payload["task_text"]
    source_label = payload["source_label"]
    metadata_text = payload["metadata_context"]["metadata_text"]
    bytecode_evidence = payload["bytecode_evidence_fallback"]
    bytecode_section = f"\n\n{bytecode_evidence}" if bytecode_evidence else ""
    telemetry_guidance = payload["telemetry_guidance"]
    telemetry_section = f"\n\n{telemetry_guidance}" if telemetry_guidance else ""
    numbered_fragment = payload["line_numbered_source_fragment"]

    user_prompt = f"""Task: {task_text}

Target qualname: {qualname}

{INDENTATION_CONTRACT}

{source_label}:
```python
{numbered_fragment}
```

{_format_repair_context(repair_context, module_mode=qualname == "<module>")}

Code object metadata:
{metadata_text}{bytecode_section}{telemetry_section}

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


class LLMFragmentFixer(FragmentFixer):
    def __init__(self, *, provider: str = "Google", model: str = "gemini-2.5-flash-lite"):
        self.llm_config = find_llm_config(provider, model)
        self.calls: list[dict] = []
        self.prompt_output_dir: Path | None = None
        self._prompt_call_index = 0
        self.telemetry_records = _load_semantic_repair_telemetry_records()

    def set_prompt_output_dir(self, prompt_output_dir: Path | None) -> None:
        self.prompt_output_dir = None if prompt_output_dir is None else prompt_output_dir.expanduser().resolve()
        self._prompt_call_index = 0

    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        telemetry_guidance = generate_semantic_repair_guidance(
            qualname=qualname,
            derived_code_object=derived_code_object,
            repair_context=repair_context,
            telemetry_records=self.telemetry_records,
        )
        prompt_reconstruction_inputs = build_semantic_repair_prompt_payload(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
            repair_context=repair_context,
            telemetry_guidance=telemetry_guidance,
        )
        messages = build_semantic_repair_messages(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=derived_source_fragment,
            repair_context=repair_context,
            telemetry_guidance=telemetry_guidance,
        )
        prompt_record = {
            "provider": self.llm_config["provider"],
            "model": self.llm_config["name"],
            "qualname": qualname,
            "prompt_variant": SEMANTIC_PROMPT_VARIANT,
            "prompt_features": _prompt_feature_flags(messages[1]["content"] if len(messages) > 1 else "", repair_context),
            "prompt_reconstruction_inputs": prompt_reconstruction_inputs,
            "messages": messages,
            "system_prompt": messages[0]["content"] if messages else None,
            "user_prompt": messages[1]["content"] if len(messages) > 1 else None,
        }
        started = time.perf_counter()
        response = make_llm_call_from_config(messages, self.llm_config)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response_text = strip_code_fences(response)
        cleaned_response_text = strip_prompt_line_numbers(response_text)
        content = cleaned_response_text.strip()
        prompt_record.update(
            {
                "latency_ms": elapsed_ms,
                "usage": None if response is None else str(response.get("usage")),
                "response_text": response_text,
                "line_number_stripped_text": cleaned_response_text,
                "returned_text": content if content else derived_source_fragment,
            }
        )
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
        return content if content else derived_source_fragment


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
    ) -> str:
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
) -> dict:
    dataset_path = dataset_path.expanduser().resolve()
    if output_dir is None:
        run_id, log_base, log_file = build_run_paths(current_run_timestamp())
    else:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = output_dir.name
        log_base = output_dir
        log_file = output_dir / f"run_log_{run_id}_{dataset_path.stem}.jsonl"
    results_csv = log_base / f"semantic_repair_results_{dataset_path.stem}.csv"

    df = pd.read_csv(dataset_path)
    semantic_df = _filter_semantic_dataset_rows(df, source=source, file_hash=file_hash, row_range=row_range, limit=limit)

    processed = 0
    repaired = 0
    failed = 0

    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_dataset_fieldnames())
        writer.writeheader()

        for row in semantic_df.itertuples(index=False):
            processed += 1
            gt_source = fetch_pyllmpatch_source_path(row.file_hash, row.source)
            gt_pyc, derived_pyc, derived_source = fetch_pyllmpatch_repair_paths(row.file_hash, row.source)
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

            row_output_dir = log_base / "semantic_repair" / str(row.source) / str(row.file_hash)
            row_output_dir.mkdir(parents=True, exist_ok=True)
            result_json_path = row_output_dir / "result.json"
            try:
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
                )
                result_json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
                result_row = _dataset_result_row(row, result, result_json_path)
                writer.writerow(result_row)
                append_log(
                    log_file,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "mode": "semantic_repair",
                        "stage": "result",
                        **result_row,
                    },
                )
                repaired += 1
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
        "run_log": str(log_file),
        "run_id": run_id,
        "processed_rows": processed,
        "repaired_rows": repaired,
        "failed_rows": failed,
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
