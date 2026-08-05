from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import utils.providers as providers
from utils.file_helpers import read_file, strip_code_fences
from utils.providers import OPEN_LLM_MODELS, make_llm_call
from utils.token_helpers import Colors, count_tokens_safe
from utils.token_helpers import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FOR_LOCAL,
    SYSTEM_PROMPT_FOR_ROOT_CAUSE_ANALYSIS,
    USER_PROMPT_TEMPLATE_LOCAL,
    USER_PROMPT_TEMPLATE_LOCAL_RETRY,
    USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION,
    USER_PROMPT_TEMPLATE_ROOT_CAUSE,
    build_chat_messages,
    get_user_prompt,
)

RepairResult = Tuple[str, Dict[str, Any], bool, Optional[int], Optional[int], Optional[str], Optional[str]]


def make_call_to_local_llm(
    content: str,
    error: str,
    current_explanation: str,
    affected_file_path: Path,
    model_path: str,
    max_tokens: int,
    tokenizer_path: Optional[str] = None,
    generation_config: Optional[dict] = None,
    is_retry: bool = False,
    gt_context: str = "",
):
    # ERROR_FEEDBACK lever: on a retry, the LATEST compile error is the most
    # informative signal available and must be presented as authoritative
    # feedback rather than "for reference only" -- see
    # USER_PROMPT_TEMPLATE_LOCAL_RETRY. First attempt (is_retry=False, the
    # default) keeps the pre-existing WITH/WITHOUT-explanation template
    # selection unchanged.
    if is_retry:
        user_prompt_template = USER_PROMPT_TEMPLATE_LOCAL_RETRY
    elif len(current_explanation) > 0:
        user_prompt_template = USER_PROMPT_TEMPLATE_LOCAL
    else:
        user_prompt_template = USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION

    messages = build_chat_messages(
        code_snippet=content.strip("\n"),
        error_message=error,
        system_prompt=SYSTEM_PROMPT_FOR_LOCAL,
        user_prompt_template=user_prompt_template,
        current_explanation=current_explanation,
        gt_context=gt_context,
    )

    if not affected_file_path.exists():
        affected_file_path.mkdir(parents=True, exist_ok=True)

    out_path = affected_file_path / "last_message_to_llm_for_file.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=str)

    # Transport swap (mirrors utils/providers.py::make_llm_call_from_config): a running LOCAL
    # vllm-serve endpoint short-circuits the in-process engine so the syntactic pipeline can
    # share the same server as the semantic pipeline, instead of requiring vllm installed
    # in-process. localhost only; unset env falls through to the unchanged in-process path.
    server_url = os.getenv("SEMANTIC_VLLM_SERVER_URL", "").strip()
    if server_url:
        result = providers.call_local_vllm_server(
            messages,
            base_url=server_url,
            model=os.getenv("SEMANTIC_VLLM_SERVER_MODEL", "repair-model"),
            max_tokens=max_tokens,
            generation_config=generation_config,
        )
        return result.get("content") if result else None

    from model.inference import call_llm_with_message

    return call_llm_with_message(
        messages=messages,
        model_path=model_path,
        max_tokens=max_tokens,
        tokenizer_path=tokenizer_path,
    )


def explain_current_code_syntax_error(
    content: str,
    error: str,
    model_path: str,
    max_tokens: int,
    tokenizer_path: Optional[str] = None,
) -> str:
    from model.inference import call_llm_with_message

    messages = build_chat_messages(
        code_snippet=content.strip("\n"),
        error_message=error,
        system_prompt=SYSTEM_PROMPT_FOR_ROOT_CAUSE_ANALYSIS,
        user_prompt_template=USER_PROMPT_TEMPLATE_ROOT_CAUSE,
    )
    return call_llm_with_message(
        messages=messages,
        model_path=model_path,
        max_tokens=max_tokens,
        tokenizer_path=tokenizer_path,
    )


def make_call_to_api_llm(content: str, model: dict, error: str):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_user_prompt(content, error)},
    ]
    return make_llm_call(prompt, model=model["name"], provider=model["provider"])


def process_file_in_single_run(
    content: str,
    model: dict,
    error: str,
    affected_file_path: Path,
    enable_syntax_explanation: bool,
    is_retry: bool = False,
    gt_context: str = "",
    generation_config_override: Optional[dict] = None,
    max_tokens_override: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    model_name = f"{model['provider']} - {model['name']}"
    print(f"{Colors.OKGREEN}    -> Content fits in a single run for {model_name}. Processing...{Colors.ENDC}")
    t0 = time.perf_counter()

    current_explanation = ""
    if enable_syntax_explanation and model.get("model_path"):
        current_explanation = explain_current_code_syntax_error(
            content,
            error,
            model["model_path"],
            model["token_for_completion"],
            model.get("tokenizer_path"),
        )
    if model["name"] in {m["name"] for m in OPEN_LLM_MODELS}:
        # Best-of-N (SYNTACTIC_BEST_OF_N): candidate 0 (greedy) always passes
        # generation_config_override=None here, so it takes the model's own
        # generation_config exactly as before -- byte-identical to pre-best-of-N
        # behavior. Only the N-1 SAMPLED candidates the runner generates after a
        # failed greedy compile pass a non-None override, which takes precedence.
        effective_generation_config = (
            generation_config_override if generation_config_override is not None else model.get("generation_config")
        )
        # OUTPUT_OVERFLOW lever (SYNTACTIC_MAX_TOKENS, threaded from pipeline.runner as
        # max_tokens_override): None (the default) keeps using the model's own
        # token_for_completion, byte-identical to before this lever existed.
        effective_max_tokens = max_tokens_override if max_tokens_override is not None else model["token_for_completion"]
        llm_raw = make_call_to_local_llm(
            content,
            error,
            current_explanation,
            affected_file_path,
            model["model_path"],
            effective_max_tokens,
            model.get("tokenizer_path"),
            effective_generation_config,
            is_retry,
            gt_context,
        )
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
    expansion_level: int = 0,
    enable_syntax_explanation: bool = True,
    gt_context: str = "",
    generation_config_override: Optional[dict] = None,
    max_tokens_override: Optional[int] = None,
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

    # ERROR_FEEDBACK lever: expansion_level is the per-target attempt/widening
    # counter threaded in from the retry loop (pipeline/runner.py -- 0 on the
    # first attempt, >0 once a prior attempt's compile has failed and fed
    # back an updated error_description). Treat any expansion_level > 0 as
    # "this is a retry" so the LLM sees the latest compile error framed as
    # authoritative feedback instead of the first-attempt template.
    is_retry = expansion_level > 0

    return process_file_in_single_run(
        initial_content,
        llm,
        error_description,
        affected_file_path,
        enable_syntax_explanation=enable_syntax_explanation,
        is_retry=is_retry,
        gt_context=gt_context,
        generation_config_override=generation_config_override,
        max_tokens_override=max_tokens_override,
    )


def attempt_repair(
    *,
    copy_dir: Path,
    error_description: str,
    llm,
    log_rec: Dict[str, Any],
    strategy_state: Dict[str, Dict[str, Any]],
    try_whole_file: bool,
    expansion_level: int,
    affected_file_path: Path,
    segment_syntax_context,
    enable_syntax_explanation: bool,
    gt_context: str = "",
    generation_config_override: Optional[dict] = None,
    max_tokens_override: Optional[int] = None,
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
        anchor_indent = None

        if not try_whole_file:
            from pipeline.logging_utils import extract_line_number

            error_line = extract_line_number(error_description)
            syntax_segment = segment_syntax_context(copy_dir, error_line, error_description, expansion_level)
            if not syntax_segment:
                state["failures"] += 1
                continue

            log_rec.update(
                {
                    "segment_kind": syntax_segment.segment_kind,
                    "segment_line_roles": list(syntax_segment.line_roles),
                }
            )
            initial_content = syntax_segment.text
            start_ln = syntax_segment.start_line
            end_ln = syntax_segment.end_line
            base_indent = syntax_segment.base_indent
            anchor_indent = syntax_segment.anchor_indent
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
                expansion_level=expansion_level,
                enable_syntax_explanation=enable_syntax_explanation,
                gt_context=gt_context,
                generation_config_override=generation_config_override,
                max_tokens_override=max_tokens_override,
            )
        else:
            initial_content = read_file(copy_dir)
            processed = process_file_for_syntax_error_patching(
                initial_content,
                error_description,
                affected_file_path,
                log_rec=log_rec,
                llm=llm,
                expansion_level=expansion_level,
                enable_syntax_explanation=enable_syntax_explanation,
                gt_context=gt_context,
                generation_config_override=generation_config_override,
                max_tokens_override=max_tokens_override,
            )

        if processed is None:
            state["failures"] += 1
            log_rec.update({"repair_attempt_skipped": True, "repair_skip_strategy": strategy})
            continue

        state["failures"] = 0
        final_code, llm_metrics = processed
        return final_code, llm_metrics, with_pin_point, start_ln, end_ln, base_indent, anchor_indent

    return None


def select_best_of_n_candidate(
    *,
    best_of_n: int,
    generate_greedy: Callable[[], Any],
    generate_sampled: Callable[[], Any],
    compiles: Callable[[Any], bool],
) -> Tuple[Any, bool]:
    greedy_candidate = generate_greedy()
    if compiles(greedy_candidate):
        return greedy_candidate, True

    if best_of_n > 1:
        for _ in range(best_of_n - 1):
            sampled_candidate = generate_sampled()
            if compiles(sampled_candidate):
                return sampled_candidate, True

    return greedy_candidate, False


def extract_bytecode_major_minor(src: str) -> Optional[str]:
    m = re.search(r"Bytecode version:\s*(\d+)\.(\d+)", src)
    return f"{m.group(1)}.{m.group(2)}" if m else None
