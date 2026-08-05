from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.code_object_repair_loop import (
    CODE_OBJECT_PROMPT_FIELDS,
    SEMANTIC_PROMPT_VARIANT,
    SEMANTIC_REPAIR_CANDIDATE_COUNT,
    _candidate_strategy_specs,
    _load_semantic_action_patterns,
    _prompt_feature_flags,
    _replacement_hash,
    _replacement_structural_hash,
    build_semantic_repair_messages,
    build_semantic_repair_prompt_payload,
    generate_action_pattern_guidance,
    generate_semantic_repair_guidance,
)


class MetadataCodeObject:
    """Small attribute wrapper for serialized code object metadata."""

    def __init__(self, metadata: dict[str, Any] | None):
        metadata = metadata or {}
        for field in CODE_OBJECT_PROMPT_FIELDS:
            setattr(self, field, metadata.get(field))
        self.co_firstlineno = metadata.get("co_firstlineno")
        self.co_nlocals = len(metadata.get("co_varnames") or [])
        self.co_stacksize = metadata.get("co_stacksize")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str))
            handle.write("\n")


def _prompt_reconstruction_inputs(record: dict[str, Any]) -> dict[str, Any] | None:
    prompt = record.get("prompt")
    if not isinstance(prompt, dict):
        return None
    inputs = prompt.get("prompt_reconstruction_inputs")
    return inputs if isinstance(inputs, dict) else None


def _objects_from_inputs(inputs: dict[str, Any]) -> tuple[MetadataCodeObject | None, MetadataCodeObject | None]:
    metadata_context = inputs.get("metadata_context") if isinstance(inputs.get("metadata_context"), dict) else {}
    gt_metadata = metadata_context.get("gt_code_object_metadata") if isinstance(metadata_context, dict) else None
    derived_metadata = metadata_context.get("derived_code_object_metadata") if isinstance(metadata_context, dict) else None
    gt_object = MetadataCodeObject(gt_metadata) if isinstance(gt_metadata, dict) else None
    derived_object = MetadataCodeObject(derived_metadata) if isinstance(derived_metadata, dict) else None
    return gt_object, derived_object


def _base_repair_context(inputs: dict[str, Any]) -> dict[str, Any]:
    context = copy.deepcopy(inputs.get("repair_context") if isinstance(inputs.get("repair_context"), dict) else {})
    if inputs.get("source_start_line") is not None and context.get("source_lineno") is None:
        context["source_lineno"] = inputs.get("source_start_line")
    return context


def _safe_prior_candidate_outputs(source_fragment: str, replacement_text: str) -> list[dict[str, Any]]:
    source_hash = _replacement_hash(source_fragment)
    source_structural_hash = _replacement_structural_hash(source_fragment)
    replacement_hash = _replacement_hash(replacement_text)
    replacement_structural_hash = _replacement_structural_hash(replacement_text)
    if (
        (source_hash and source_hash == replacement_hash)
        or (
            source_structural_hash
            and replacement_structural_hash
            and source_structural_hash == replacement_structural_hash
        )
    ):
        return []
    return [
        {
            "strategy": "previous_candidate",
            "replacement_hash": source_hash,
            "replacement_structural_hash": source_structural_hash,
            "text": source_fragment,
        }
    ]


def _render_prompt_record(
    *,
    source_record: dict[str, Any],
    inputs: dict[str, Any],
    repair_context: dict[str, Any],
    telemetry_records: list[dict[str, Any]],
    action_patterns: list[dict[str, Any]],
    candidate_index: int,
    candidate_strategy: str,
    candidate_strategy_prompt: str,
    suppressed_action_pattern_guidance: bool,
    duplicate_retry_index: int = 0,
) -> dict[str, Any]:
    qualname = str(inputs.get("target_qualname") or source_record.get("qualname") or "")
    source_fragment = str(inputs.get("source_fragment") or source_record.get("extracted_before") or "")
    replacement_text = str(source_record.get("replacement_text") or "")
    gt_code_object, derived_code_object = _objects_from_inputs(inputs)

    telemetry_guidance = ""
    action_pattern_guidance = ""
    if not suppressed_action_pattern_guidance:
        telemetry_guidance = generate_semantic_repair_guidance(
            qualname=qualname,
            derived_code_object=derived_code_object,
            repair_context=repair_context,
            telemetry_records=telemetry_records,
        )
        action_pattern_guidance = generate_action_pattern_guidance(
            qualname=qualname,
            gt_code_object=gt_code_object,
            derived_code_object=derived_code_object,
            derived_source_fragment=source_fragment,
            repair_context=repair_context,
            action_patterns=action_patterns,
        )
        if action_pattern_guidance:
            telemetry_guidance = ""

    payload = build_semantic_repair_prompt_payload(
        qualname=qualname,
        gt_code_object=gt_code_object,
        derived_code_object=derived_code_object,
        derived_source_fragment=source_fragment,
        repair_context=repair_context,
        telemetry_guidance=telemetry_guidance,
        action_pattern_guidance=action_pattern_guidance,
    )
    messages = build_semantic_repair_messages(
        qualname=qualname,
        gt_code_object=gt_code_object,
        derived_code_object=derived_code_object,
        derived_source_fragment=source_fragment,
        repair_context=repair_context,
        telemetry_guidance=telemetry_guidance,
        action_pattern_guidance=action_pattern_guidance,
    )
    prompt = source_record.get("prompt") if isinstance(source_record.get("prompt"), dict) else {}
    user_prompt = messages[1]["content"] if len(messages) > 1 else ""
    return {
        "provider": "offline",
        "model": "current-semantic-prompt-template",
        "original_provider": prompt.get("provider"),
        "original_model": prompt.get("model"),
        "qualname": qualname,
        "prompt_variant": SEMANTIC_PROMPT_VARIANT,
        "candidate_index": candidate_index,
        "candidate_strategy": candidate_strategy,
        "candidate_strategy_prompt": candidate_strategy_prompt,
        "suppressed_action_pattern_guidance": suppressed_action_pattern_guidance,
        "prior_same_round_candidate_count": len(repair_context.get("_prior_candidate_outputs") or []),
        "duplicate_retry_index": duplicate_retry_index,
        "prompt_features": _prompt_feature_flags(user_prompt, repair_context),
        "prompt_reconstruction_inputs": payload,
        "messages": messages,
        "system_prompt": messages[0]["content"] if messages else None,
        "user_prompt": user_prompt,
        "response_text": replacement_text,
        "line_number_stripped_text": replacement_text,
        "returned_text": replacement_text,
        "replacement_hash": _replacement_hash(replacement_text),
        "replacement_structural_hash": _replacement_structural_hash(replacement_text),
        "prompt_refresh": {
            "source": "offline_rebuild",
            "template_variant": SEMANTIC_PROMPT_VARIANT,
            "label_source": "replacement_text",
        },
    }


def rebuild_records(
    records: list[dict[str, Any]],
    *,
    candidate_count: int,
    max_duplicate_retry_records: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    telemetry_records = [record for record in records if record.get("accepted")]
    action_patterns = _load_semantic_action_patterns()
    rebuilt: list[dict[str, Any]] = []
    stats = {
        "input_records": len(records),
        "skipped_missing_prompt_inputs": 0,
        "base_records": 0,
        "strategy_variant_records": 0,
        "duplicate_retry_records": 0,
    }

    for record in records:
        inputs = _prompt_reconstruction_inputs(record)
        replacement_text = str(record.get("replacement_text") or "")
        if not inputs or not replacement_text:
            stats["skipped_missing_prompt_inputs"] += 1
            continue
        source_fragment = str(inputs.get("source_fragment") or record.get("extracted_before") or "")
        base_context = _base_repair_context(inputs)
        strategies = _candidate_strategy_specs(candidate_count, base_context)
        prior_outputs = _safe_prior_candidate_outputs(source_fragment, replacement_text)

        for candidate_index, (strategy, strategy_prompt) in enumerate(strategies):
            context = copy.deepcopy(base_context)
            context["_candidate_index"] = candidate_index
            if len(strategies) > 1:
                context["_candidate_strategy"] = strategy
                context["_candidate_strategy_prompt"] = strategy_prompt
            if candidate_index > 0 and prior_outputs:
                context["_prior_candidate_outputs"] = prior_outputs
                context["_same_round_seen_hashes"] = {
                    item["replacement_hash"] for item in prior_outputs if item.get("replacement_hash")
                }
                context["_same_round_seen_structural_hashes"] = {
                    item["replacement_structural_hash"]
                    for item in prior_outputs
                    if item.get("replacement_structural_hash")
                }
            suppressed = strategy != "retrieval_guided"
            refreshed = copy.deepcopy(record)
            refreshed["prompt"] = _render_prompt_record(
                source_record=record,
                inputs=inputs,
                repair_context=context,
                telemetry_records=telemetry_records,
                action_patterns=action_patterns,
                candidate_index=candidate_index,
                candidate_strategy=strategy,
                candidate_strategy_prompt=strategy_prompt,
                suppressed_action_pattern_guidance=suppressed,
            )
            refreshed["prompt_refresh"] = {
                "source_record_qualname": record.get("qualname"),
                "source_prompt_variant": (record.get("prompt") or {}).get("prompt_variant")
                if isinstance(record.get("prompt"), dict)
                else None,
                "variant": "candidate_strategy",
            }
            rebuilt.append(refreshed)
            if candidate_index == 0:
                stats["base_records"] += 1
            else:
                stats["strategy_variant_records"] += 1

        if (
            prior_outputs
            and stats["duplicate_retry_records"] < max_duplicate_retry_records
            and len(strategies) > 1
        ):
            candidate_index = min(2, len(strategies) - 1)
            strategy, strategy_prompt = strategies[candidate_index]
            context = copy.deepcopy(base_context)
            context["_candidate_index"] = candidate_index
            context["_candidate_strategy"] = strategy
            context["_candidate_strategy_prompt"] = strategy_prompt
            context["_prior_candidate_outputs"] = prior_outputs
            context["_same_round_seen_hashes"] = {
                item["replacement_hash"] for item in prior_outputs if item.get("replacement_hash")
            }
            context["_same_round_seen_structural_hashes"] = {
                item["replacement_structural_hash"]
                for item in prior_outputs
                if item.get("replacement_structural_hash")
            }
            context["_force_distinct_retry"] = True
            context["_force_distinct_retry_source"] = "same_round_candidate"
            context["_duplicate_rejected_output_excerpt"] = source_fragment
            refreshed = copy.deepcopy(record)
            refreshed["prompt"] = _render_prompt_record(
                source_record=record,
                inputs=inputs,
                repair_context=context,
                telemetry_records=telemetry_records,
                action_patterns=action_patterns,
                candidate_index=candidate_index,
                candidate_strategy=strategy,
                candidate_strategy_prompt=strategy_prompt,
                suppressed_action_pattern_guidance=strategy != "retrieval_guided",
                duplicate_retry_index=1,
            )
            refreshed["prompt_refresh"] = {
                "source_record_qualname": record.get("qualname"),
                "source_prompt_variant": (record.get("prompt") or {}).get("prompt_variant")
                if isinstance(record.get("prompt"), dict)
                else None,
                "variant": "duplicate_retry",
            }
            rebuilt.append(refreshed)
            stats["duplicate_retry_records"] += 1

    return rebuilt, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild accepted semantic repair fine-tuning records with the current prompt template."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/accepted_codeobject_mining_raw.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/accepted_codeobject_mining_prompt_refresh.jsonl"),
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=SEMANTIC_REPAIR_CANDIDATE_COUNT,
        help="Number of current candidate strategy prompts to emit per source record.",
    )
    parser.add_argument(
        "--max-duplicate-retry-records",
        type=int,
        default=300,
        help="Maximum synthetic duplicate-retry prompt records to add.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _read_jsonl(args.input)
    rebuilt, stats = rebuild_records(
        records,
        candidate_count=max(1, min(args.candidate_count, 5)),
        max_duplicate_retry_records=max(0, args.max_duplicate_retry_records),
    )
    _write_jsonl(args.output, rebuilt)
    print(json.dumps({**stats, "output_records": len(rebuilt), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
