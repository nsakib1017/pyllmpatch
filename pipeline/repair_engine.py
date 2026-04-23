from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.file_helpers import read_file, strip_code_fences
from utils.providers import OPEN_LLM_MODELS, make_llm_call
from utils.token_helpers import Colors, count_tokens_safe
from utils.token_helpers import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FOR_LOCAL,
    SYSTEM_PROMPT_FOR_ROOT_CAUSE_ANALYSIS,
    USER_PROMPT_TEMPLATE_LOCAL,
    USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION,
    USER_PROMPT_TEMPLATE_ROOT_CAUSE,
    build_chat_messages,
    get_user_prompt,
)

RepairResult = Tuple[str, Dict[str, Any], bool, Optional[int], Optional[int], Optional[str]]


def make_call_to_local_llm(content: str, error: str, current_explanation: str, affected_file_path: Path, model_path: str, max_tokens: int):
    from model.inference import call_llm_with_message

    messages = build_chat_messages(
        code_snippet=content.strip("\n"),
        error_message=error,
        system_prompt=SYSTEM_PROMPT_FOR_LOCAL,
        user_prompt_template=USER_PROMPT_TEMPLATE_LOCAL if len(current_explanation) > 0 else USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION,
        current_explanation=current_explanation,
    )

    if not affected_file_path.exists():
        affected_file_path.mkdir(parents=True, exist_ok=True)

    out_path = affected_file_path / "last_message_to_llm_for_file.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=str)

    return call_llm_with_message(messages=messages, model_path=model_path, max_tokens=max_tokens)


def explain_current_code_syntax_error(content: str, error: str, model_path: str, max_tokens: int) -> str:
    from model.inference import call_llm_with_message

    messages = build_chat_messages(
        code_snippet=content.strip("\n"),
        error_message=error,
        system_prompt=SYSTEM_PROMPT_FOR_ROOT_CAUSE_ANALYSIS,
        user_prompt_template=USER_PROMPT_TEMPLATE_ROOT_CAUSE,
    )
    return call_llm_with_message(messages=messages, model_path=model_path, max_tokens=max_tokens)


def make_call_to_api_llm(content: str, model: dict, error: str):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_user_prompt(content, error)},
    ]
    return make_llm_call(prompt, model=model["name"], provider=model["provider"])


def process_file_in_single_run(content: str, model: dict, error: str, affected_file_path: Path, outer_idx: int) -> Tuple[str, Dict[str, Any]]:
    model_name = f"{model['provider']} - {model['name']}"
    print(f"{Colors.OKGREEN}    -> Content fits in a single run for {model_name}. Processing...{Colors.ENDC}")
    t0 = time.perf_counter()

    current_explanation = explain_current_code_syntax_error(content, error, model["model_path"], model["token_for_completion"]) if outer_idx != 0 else ""
    if model["name"] in {m["name"] for m in OPEN_LLM_MODELS}:
        llm_raw = make_call_to_local_llm(content, error, current_explanation, affected_file_path, model["model_path"], model["token_for_completion"])
    else:
        llm_raw = make_call_to_api_llm(content, model, error)

    try:
        llm_response = strip_code_fences(llm_raw) if llm_raw else content
    except Exception:
        llm_response = content

    llm_elapsed_ms = int((time.perf_counter() - t0) * 1000)
    metrics = {
        "fits_single_run": True,
        "llm_calls": 1,
        "llm_latency_ms_total": llm_elapsed_ms,
        "chunk_count": 1,
        "merge_passes": 0,
        "avg_chunk_tokens": count_tokens_safe(content, model["provider"], model["name"]),
        "max_chunk_tokens": count_tokens_safe(content, model["provider"], model["name"]),
    }
    return (llm_response if llm_response else content), metrics


def process_file_for_syntax_error_patching(
    initial_content: str,
    error_description,
    affected_file_path: Path,
    log_rec=None,
    llm=OPEN_LLM_MODELS[1],
    outer_idx=0,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if log_rec is None:
        log_rec = {}
    log_rec.update({"provider": llm["provider"], "model_name": llm["name"]})

    if initial_content is None:
        log_rec.update(
            {
                "skipped_due_to_missing_content": True,
                "skipped_due_to_token_limit": False,
                "input_token_count": None,
                "token_limit_for_completion": llm["token_for_completion"],
            }
        )
        return None

    input_token_count = count_tokens_safe(initial_content, llm["provider"], llm["name"])
    token_threshold = llm["token_for_completion"] - 5000

    if input_token_count > token_threshold:
        log_rec.update(
            {
                "skipped_due_to_token_limit": True,
                "skipped_due_to_missing_content": False,
                "input_token_count": input_token_count,
                "token_limit_for_completion": llm["token_for_completion"],
                "token_threshold_used": token_threshold,
            }
        )
        return None

    log_rec.update(
        {
            "skipped_due_to_token_limit": False,
            "skipped_due_to_missing_content": False,
            "input_token_count": input_token_count,
            "token_limit_for_completion": llm["token_for_completion"],
            "token_threshold_used": token_threshold,
        }
    )

    return process_file_in_single_run(initial_content, llm, error_description, affected_file_path, outer_idx)


def extract_bytecode_major_minor(src: str) -> Optional[str]:
    m = re.search(r"Bytecode version:\s*(\d+)\.(\d+)", src)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def attempt_repair(
    *,
    copy_dir: Path,
    error_description: str,
    llm,
    log_rec: Dict[str, Any],
    strategy_state: Dict[str, Dict[str, Any]],
    try_whole_file: bool,
    outer_idx: int,
    affected_file_path: Path,
    fetch_syntax_context,
) -> Optional[RepairResult]:
    strategies = ["syntax_context"]
    if try_whole_file:
        strategies.append("whole_file")

    for strategy in strategies:
        state = strategy_state.get(strategy)
        if not state:
            continue

        processed = None
        with_pin_point = False
        start_ln = None
        end_ln = None
        base_indent = None

        if not try_whole_file:
            from pipeline.logging_utils import extract_line_number

            error_line = extract_line_number(error_description)
            syntax_context = fetch_syntax_context(copy_dir, error_line, outer_idx)
            if not syntax_context:
                state["failures"] += 1
                continue

            initial_content, start_ln, end_ln, base_indent = syntax_context
            with_pin_point = True

            if not initial_content:
                with_pin_point = False
                initial_content = read_file(copy_dir)

            processed = process_file_for_syntax_error_patching(
                initial_content,
                error_description,
                affected_file_path,
                log_rec=log_rec,
                llm=llm,
                outer_idx=outer_idx,
            )
        else:
            initial_content = read_file(copy_dir)
            processed = process_file_for_syntax_error_patching(
                initial_content,
                error_description,
                affected_file_path,
                log_rec=log_rec,
                llm=llm,
                outer_idx=outer_idx,
            )

        if processed is None:
            state["failures"] += 1
            log_rec.update({"repair_attempt_skipped": True, "repair_skip_strategy": strategy})
            continue

        state["failures"] = 0
        final_code, llm_metrics = processed
        return final_code, llm_metrics, with_pin_point, start_ln, end_ln, base_indent

    return None
