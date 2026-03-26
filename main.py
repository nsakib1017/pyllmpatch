# Read dataset_summary.csv into a pandas DataFrame
import gc
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from typing import Any
import difflib  
import json     
import time     
import uuid     
from datetime import datetime, timezone  
import shutil
from typing import Tuple, Optional, Dict, Any  

from utils.delete_only_compilation import delete_lines_until_compilable_with_oracle
from utils.token_helpers import *
from utils.providers import *
from utils.chunk_helpers import *
from utils.file_helpers import *
from utils.generate_bytecode import *

from model.inference import *
from utils.indentation_fixer import *
from dataclasses import asdict
import traceback
from dotenv import load_dotenv
load_dotenv()


RepairResult = Tuple[str, Dict[str, Any], bool, Optional[int], Optional[int], Optional[int]]


PROJECT_ROOT_DIR = Path(str(os.getenv("PROJECT_ROOT_DIR")))
ROOT_FOR_FILES = Path(str(os.getenv("ROOT_FOR_FILES")))
BASE_DIR_PYTHON_FILES_PYLINGUAL = ROOT_FOR_FILES / str(os.getenv("BASE_DIR_PYTHON_FILES_PYLINGUAL"))
BASE_DIR_PYTHON_FILES_PYPI = ROOT_FOR_FILES / str(os.getenv("BASE_DIR_PYTHON_FILES_PYPI"))
BASE_DATASET_NAME = str(os.getenv("BASE_DATASET_NAME"))
BASE_DATASET_PATH = PROJECT_ROOT_DIR / "dataset" / BASE_DATASET_NAME

MAX_EXAMPLE_RUNTIME_SEC = int(float(os.getenv("MAX_EXAMPLE_RUNTIME_MIN")) * 60 * 60)
def prepare_snippets_for_repair(
    previous_run_log_path: str = "",
    only_previously_failed: bool = False,
):
    if not previous_run_log_path:
        return read_csv_file(f"{BASE_DATASET_PATH}")

    df_previous_log = pd.read_json(previous_run_log_path, lines=True)
    df_balanced = read_csv_file(f"{BASE_DATASET_PATH}")

    # Normalize types and remove NaNs/dupes just in case
    df_previous_log["file_hash"] = df_previous_log["file_hash"].astype(str)
    df_balanced["file_hash"] = df_balanced["file_hash"].astype(str)

    if only_previously_failed:
        # File hashes that never compiled successfully
        success_by_file = (
            df_previous_log
            .groupby("file_hash")["compiled_success"]
            .any()
        )
        failed_hashes = success_by_file[~success_by_file].index
        mask = df_balanced["file_hash"].isin(set(failed_hashes))
    else:
        # Original behavior: exclude all previously seen file_hashes
        prev_hashes = (
            df_previous_log["file_hash"]
            .dropna()
            .drop_duplicates()
        )
        mask = ~df_balanced["file_hash"].isin(set(prev_hashes))

    return df_balanced.loc[mask].copy()


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def make_call_to_local_llm(content: str,  error: str, current_explanation: str, affected_file_path: Path, model_path: str, max_tokens: int):
    messages = build_chat_messages(code_snippet=content.strip("\n"), error_message=error, system_prompt=SYSTEM_PROMPT_FOR_LOCAL, user_prompt_template=USER_PROMPT_TEMPLATE_LOCAL if len(current_explanation) > 0 else USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION, current_explanation=current_explanation)

    if not affected_file_path.exists():
                affected_file_path.mkdir(parents=True, exist_ok=True)

    out_path = affected_file_path / "last_message_to_llm_for_file.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=str)

    llm_raw = call_llm_with_message(messages=messages, model_path=model_path, max_tokens=max_tokens)
    return llm_raw


def explain_current_code_syntax_error(content: str, error: str, model_path: str, max_tokens: int) -> str:
    messages = build_chat_messages(code_snippet=content.strip("\n"), error_message=error, system_prompt=SYSTEM_PROMPT_FOR_ROOT_CAUSE_ANALYSIS, user_prompt_template=USER_PROMPT_TEMPLATE_ROOT_CAUSE)
    llm_raw = call_llm_with_message(messages=messages, model_path=model_path, max_tokens=max_tokens)
    # print("Explanation of syntax error root cause:\n", llm_raw)
    return llm_raw

def make_call_to_api_llm(content: str, model: dict, error: str):
    prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_user_prompt(content, error)}]
        # print(prompt[1]["content"])

    llm_raw = make_llm_call(prompt, model=model['name'], provider=model['provider'])
    return llm_raw



def process_file_in_single_run(content: str, model: dict, error: str, affected_file_path: Path, outer_idx: int) -> Tuple[str, Dict[str, Any]]:
    """Simulates processing a file that fits in the context window."""
    model_name = f"{model['provider']} - {model['name']}"
    print(f"{Colors.OKGREEN}    -> Content fits in a single run for {model_name}. Processing...{Colors.ENDC}")
    t0 = time.perf_counter()

    current_explanation = explain_current_code_syntax_error(content, error, model["model_path"], model['token_for_completion']) if outer_idx != 0 else ""
    # print(f"    -> Explanation of current syntax error:\n{current_explanation}\n", outer_idx)
    if model["name"] in {m["name"] for m in OPEN_LLM_MODELS}:
        llm_raw = make_call_to_local_llm(content, error, current_explanation, affected_file_path, model["model_path"], model['token_for_completion'])
    else:
        llm_raw = make_call_to_api_llm(content, model, error)
    try:
        llm_response = strip_code_fences(llm_raw) if llm_raw else content
    except Exception:
        llm_response = content
    
    # print(llm_response)
    # sys.exit(0)

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


# def process_and_merge_chunks(chunks: List[str], model: dict, error, retry_attempt=False, previuos_chunks=[], previous_error="") -> Tuple[str, Dict[str, Any]]:
#     """
#     Simulates sending each chunk to an LLM and then merges the results back together.
#     """
#     model_name = f"{model['provider']} - {model['name']}"
#     print(f"   {Colors.OKCYAN} -> Processing and merging for {len(chunks)} chunks with {model_name}...{Colors.ENDC}")
    
#     corrected_chunks = []
#     total_latency_ms = 0
#     max_chunk_tokens = 0
#     llm_calls = 0

#     for i, chunk in enumerate(chunks):
#         ctoks = count_tokens_safe(chunk,  model["provider"], model["name"])
#         max_chunk_tokens = max(max_chunk_tokens, ctoks)
#         print(f"        -> Processing chunk {i+1}/{len(chunks)} ({ctoks} tokens)...")
#         prompt = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": get_user_prompt_chunk(chunk, error, retry_attempt, previuos_chunks[i] if previuos_chunks and len(previuos_chunks) > 0 else "", previous_error)}
#         ]
#         print(prompt[1]["content"])
#         t0 = time.perf_counter()
#         llm_response = make_llm_call(prompt, model=model['name'], provider=model['provider'])
#         total_latency_ms += int((time.perf_counter() - t0) * 1000)
#         llm_calls += 1
#         llm_usage = llm_response.get('usage', {}) if llm_response else {}
#         llm_raw = llm_response.get('content', {}) if llm_response else {}
#         # total_in += count_tokens_safe(prompt[0]["content"]) + count_tokens_safe(prompt[1]["content"])
#         try:
#             corrected_chunk = strip_code_fences(llm_raw) if llm_raw else chunk
#         except Exception:
#             corrected_chunk = llm_raw
#         # total_out += count_tokens_safe(corrected_chunk)

#         corrected_chunks.append(corrected_chunk)

#     final_content = "\n".join(corrected_chunks)

#     print(f"    {Colors.OKGREEN}    -> All chunks processed and merged.{Colors.ENDC}")
#     print(f"    -> Final merged content token count: {count_tokens(final_content, model['provider'], model['name'])}")

#     metrics = {
#         "fits_single_run": False,
#         "chunk_count": len(chunks),
#         "merge_passes": 1,  # based on your call to chunk_top_level_objects_lenient_merged(..., extra_merge_passes=1)
#         "avg_chunk_tokens": int(sum(count_tokens_safe(c,  model["provider"], model["name"]) for c in chunks) / max(1, len(chunks))),
#         "max_chunk_tokens": max_chunk_tokens,
#         "prompt_token_count": getattr(llm_usage, 'prompt_token_count', None),
#         "candidates_token_count": getattr(llm_usage, 'candidates_token_count', None),
#         "total_token_count": getattr(llm_usage, 'total_token_count', None),
#         "llm_calls": llm_calls,
#         "llm_latency_ms_total": total_latency_ms,
#     }
#     return final_content, metrics, corrected_chunks


def process_file_for_syntax_error_patching(
    initial_content: str,
    error_description,
    affected_file_path: Path,
    log_rec={},
    llm=OPEN_LLM_MODELS[1],
    outer_idx=0
) -> Optional[Tuple[str, Dict[str, Any]]]:
    log_rec.update({"provider": llm["provider"], "model_name": llm["name"]})

    if initial_content is None:
        log_rec.update({
            "skipped_due_to_missing_content": True,
            "skipped_due_to_token_limit": False,
            "input_token_count": None,
            "token_limit_for_completion": llm["token_for_completion"],
        })
        return None

    input_token_count = count_tokens_safe(initial_content, llm["provider"], llm["name"])
    token_threshold = llm["token_for_completion"] - 5000

    if input_token_count > token_threshold:
        log_rec.update({
            "skipped_due_to_token_limit": True,
            "skipped_due_to_missing_content": False,
            "input_token_count": input_token_count,
            "token_limit_for_completion": llm["token_for_completion"],
            "token_threshold_used": token_threshold,
        })
        return None

    log_rec.update({
        "skipped_due_to_token_limit": False,
        "skipped_due_to_missing_content": False,
        "input_token_count": input_token_count,
        "token_limit_for_completion": llm["token_for_completion"],
        "token_threshold_used": token_threshold,
    })

    final_code, metrics = process_file_in_single_run(
        initial_content, llm, error_description, affected_file_path, outer_idx
    )
    return final_code, metrics

def extract_bytecode_major_minor(src: str) -> Optional[str]:
    m = re.search(r"Bytecode version:\s*(\d+)\.(\d+)", src)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def clean_error_message(error_message):
    import re
    if not error_message:
        return None
    cleaned_message = error_message.lower()
    cleaned_message = cleaned_message.strip()
    cleaned_message = re.sub(r'\(.*?\)', '', cleaned_message)
    cleaned_message = cleaned_message.strip()
    return cleaned_message

def compile_new_pyc(py_file_content, py_file_dir, out_file_base_dir, version=None):
    with open(py_file_dir, "w", encoding="utf-8") as f:
        f.write(py_file_content)
    try:
        compile_version(py_file_dir, out_file_base_dir, version)
        return {"is_compiled": True, "error_description": None}
    except CompileError as e:
        return {"is_compiled": False, "error_description": str(e)}



def _append_log(path: Path, record: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as _:
        # non-fatal
        pass


def failure_cleanup(affected_file_path, final_code,  idx, error_word, error_message):
    os.remove(str(affected_file_path / f"syntax_repaired_{idx}.py"))
    with open(str(affected_file_path / f"syntax_failed_repaired_{idx}.py"), "w", encoding="utf-8") as f:
        f.write(final_code)
        f.close()     
    log_rec.update({"compile_error_word_after": error_word, "path_out": str(affected_file_path / f"syntax_failed_repaired_{idx}.py"), "compile_error_message_after": clean_error_message(error_message) if error_message else None,})


def extract_line_number(error_msg: str):
    # Find patterns like: "line 131"
    m = re.search(r"line\s+(\d+)", error_msg)
    if m:
        return int(m.group(1))
    return None

def attempt_repair(
    *,
    copy_dir: Path,
    error_description: str,
    log_base: Path,
    file_hash: str,
    llm,
    log_rec: Dict[str, Any],
    strategy_state: Dict[str, Dict[str, Any]],
    try_whole_file: bool = True,
    outer_idx: int = 0,
    affected_file_path: Path
) -> Optional[RepairResult]:
    # print("here")
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
            # print("here 2")
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

        elif try_whole_file:
            # print("here")
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
            log_rec.update({
                "repair_attempt_skipped": True,
                "repair_skip_strategy": strategy,
            })
            continue

        # success
        state["failures"] = 0

        final_code, llm_metrics = processed

        return (
            final_code,
            llm_metrics,
            with_pin_point,
            start_ln,
            end_ln,
            base_indent,
        )

    return None

def get_file_path_base_dir(decompiler_name: str, dataset_name: str, file_hash: str) -> Path:
    if "pylingual" in decompiler_name.lower():
        if "pylingual" in dataset_name.lower():
            return BASE_DIR_PYTHON_FILES_PYLINGUAL / file_hash / "decompiler_output"
        if "pypi" in dataset_name.lower():
             return BASE_DIR_PYTHON_FILES_PYPI / file_hash / "decompiled_output"
    elif "pycdc" in decompiler_name.lower():
        if "pylingual" in dataset_name.lower():
            return BASE_DIR_PYTHON_FILES_PYLINGUAL / file_hash / "pycdc_decompilation_output"
        if "pypi" in dataset_name.lower():
             return BASE_DIR_PYTHON_FILES_PYPI / file_hash / "decompiled_output_pycdc"
    else:
        return None

if __name__ == "__main__":
    previuos_run_log_path = Path(str(os.getenv("PREVIOUS_RUN_LOG_PATH"))) if os.getenv("PREVIOUS_RUN_LOG_PATH") else None
    df_syntax_error_balanced = prepare_snippets_for_repair(
        str(previuos_run_log_path) if previuos_run_log_path else "",
        only_previously_failed=False
    )
    df_syntax_error_balanced = df_syntax_error_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("DataFrame shape:", df_syntax_error_balanced.shape)


    DELETE_ONLY_MODE = str(os.getenv("DELETE_ONLY_MODE", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    ENABLE_DELETE_ONLY_FALLBACK = str(os.getenv("ENABLE_DELETE_ONLY_FALLBACK", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
    USE_LOCAL_LLM = str(os.getenv("USE_LOCAL_LLM", "True")).strip().lower() in {"1", "true", "yes", "y", "on"}
    CONFIG_IDX_START = int(os.getenv("CONFIG_IDX_START", 0))
    CONFIG_IDX_RANGE = int(os.getenv("CONFIG_IDX_RANGE", 1))


    DELETE_ONLY_INFINITE_ITERS = str(os.getenv("DELETE_ONLY_INFINITE_ITERS", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    DELETE_ONLY_MAX_ITERS = (10**9) if DELETE_ONLY_INFINITE_ITERS else int(os.getenv("DELETE_ONLY_MAX_ITERS", "5000"))

    DELETE_ONLY_BASE_WINDOW = int(os.getenv("DELETE_ONLY_BASE_WINDOW", "1"))
    DELETE_ONLY_MAX_DELETED_RATIO = float(os.getenv("DELETE_ONLY_MAX_DELETED_RATIO", "0.95"))
    MAX_WHOLE_FILE_BYTES = int(os.getenv("MAX_WHOLE_FILE_BYTES", str(1 * 1024 * 1024)))

    if DELETE_ONLY_MODE:
        print(f"{Colors.WARNING}DELETE_ONLY_MODE is enabled, LLMs will not be called{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}DELETE_ONLY_MODE is disabled, LLMs may be called{Colors.ENDC}")

    if DELETE_ONLY_INFINITE_ITERS:
        print(f"{Colors.WARNING}DELETE_ONLY_INFINITE_ITERS is enabled, delete-only may run for a very long time{Colors.ENDC}")

    for outer_idx in range(CONFIG_IDX_START, CONFIG_IDX_RANGE):
        run_id = uuid.uuid4().hex
        batch_start_ts = _now_iso()
        LOG_BASE = Path(f"results/experiment_outputs/{ts}/{run_id}")

        if not LOG_BASE.exists():
            LOG_BASE.mkdir(parents=True, exist_ok=True)
        LOG_FILE = LOG_BASE / f"run_log_{run_id}_with_config_{outer_idx}_{BASE_DATASET_PATH.name.split('.')[0]}.jsonl"
        count_idx = 0
        for idx, row in df_syntax_error_balanced.iterrows():
            copy_dir = None
            AFFECTED_FILE_PATH = None
            log_rec = None

            is_compiled = False
            file_hash = norm_str(row.get("file_hash"))
            file_name = norm_str(row.get("file"))
            file_dir = Path(row.get("file_path")) if row.get("file_path") else None
            decompiler_name = row.get("decompiler")
            dataset_name = row.get("dataset")
            bytecode_version = str(row.get("bytecode_version")) if row.get("bytecode_version") is not None else ""

            header = f"\n# --- Processing Content from file (with config {outer_idx}): {str(file_dir)} ({count_idx+1}/{len(df_syntax_error_balanced)}) --- #\n"
            footer = f"\n# --- End of Processing Content from file (with config {outer_idx}): {str(file_dir)} ({count_idx+1}/{len(df_syntax_error_balanced)}) --- #\n"

            print(header)

            try:
                if not file_dir or not file_dir.exists() or not decompiler_name or not dataset_name or not file_hash or not file_name or not bytecode_version:
                    continue

                try:
                    input_file_size_bytes = file_dir.stat().st_size
                except Exception:
                    input_file_size_bytes = None

                if input_file_size_bytes is not None and input_file_size_bytes > MAX_WHOLE_FILE_BYTES:
                    skip_log = {
                        "run_id": run_id,
                        "timestamp": _now_iso(),
                        "row_index": int(idx),
                        "file_hash": file_hash,
                        "file_name": file_name,
                        "path_in": str(file_dir),
                        "bytecode_version": bytecode_version,
                        "decompiler": decompiler_name,
                        "dataset": dataset_name,
                        "compiled_success": False,
                        "skipped_due_to_file_size_guard": True,
                        "input_file_size_bytes": int(input_file_size_bytes),
                        "max_whole_file_bytes": int(MAX_WHOLE_FILE_BYTES),
                    }
                    _append_log(LOG_FILE, skip_log)
                    print(f"{Colors.WARNING}Skipping oversized file: {file_dir} ({input_file_size_bytes} bytes > {MAX_WHOLE_FILE_BYTES}){Colors.ENDC}")
                    continue

                path_to_err_file = str(file_dir)

                initial_error_description = row.get("error_description")
                error_message_processed = row.get("error_msg_clean")
                error_word = row.get("syntactic_error_word")

                file_path_base_dir = get_file_path_base_dir(decompiler_name, dataset_name, file_hash)
                if file_path_base_dir is None:
                    continue

                copy_dir = file_path_base_dir / f"copy_for_run_id_{run_id}_of_{file_name}"
                copy_file(path_to_err_file, copy_dir)

                version = bytecode_version
                compilation_result = None

                max_retries = int(os.getenv("NO_OF_MAX_RETRIES")) if os.getenv("NO_OF_MAX_RETRIES") else 0
                total_attempts_completed = 0
                error_list = []

                log_rec = {
                    "run_id": run_id,
                    "timestamp": _now_iso(),
                    "file_hash": file_hash,
                    "file_name": file_name,
                    "row_index": int(idx),
                    "total_attempts_allowed": max_retries + 1,
                    "retries_allowed": max_retries,
                    "path_in": str(path_to_err_file),
                    "bytecode_version": version,
                    "decompiler": decompiler_name,
                    "dataset": dataset_name,
                    "input_file_size_bytes": int(input_file_size_bytes) if input_file_size_bytes is not None else None,
                    "max_whole_file_bytes": int(MAX_WHOLE_FILE_BYTES),
                    "compile_error_word_before": row.get("error"),
                    "compile_error_message_before": row.get("error_message"),
                    "delete_only_mode": bool(DELETE_ONLY_MODE),
                    "delete_only_infinite_iters": bool(DELETE_ONLY_INFINITE_ITERS),
                    "delete_only_max_iters": int(DELETE_ONLY_MAX_ITERS),
                    "delete_only_base_window": int(DELETE_ONLY_BASE_WINDOW),
                    "delete_only_max_deleted_ratio": float(DELETE_ONLY_MAX_DELETED_RATIO),
                }

                t_begin = time.monotonic()
                retry_attempt = False
                previuos_response = None
                previous_error = ""

                AFFECTED_FILE_PATH = LOG_BASE / decompiler_name / dataset_name / bytecode_version / file_hash
                if not AFFECTED_FILE_PATH.exists():
                    AFFECTED_FILE_PATH.mkdir(parents=True, exist_ok=True)

                while not is_compiled:
                    out_py_path = str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.py")
                    out_pyc_path = str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.pyc")
                    err_txt_path = str(AFFECTED_FILE_PATH / f"syntax_failed_repaired_{file_name[:-3]}_error.txt")

                    elapsed = time.monotonic() - t_begin

                    if DELETE_ONLY_MODE:
                        try:
                            t0 = time.perf_counter()

                            compilation_candidate = read_file(copy_dir) or read_file(path_to_err_file) or ""

                            fixed_code, del_log, last_res = delete_lines_until_compilable_with_oracle(
                                py_file_content=compilation_candidate,
                                compile_check=compile_new_pyc,
                                extract_line_number=extract_line_number,
                                get_error_word_message_from_content=get_error_word_message_from_content,
                                out_py_path=out_py_path,
                                out_pyc_path=out_pyc_path,
                                version=version,
                                base_window=DELETE_ONLY_BASE_WINDOW,
                                max_iters=DELETE_ONLY_MAX_ITERS,
                                max_deleted_ratio=DELETE_ONLY_MAX_DELETED_RATIO,
                                min_remaining_lines=1,
                                err_txt_path=err_txt_path,
                            )

                            compile_ms = int((time.perf_counter() - t0) * 1000)
                            is_compiled = bool(last_res.get("is_compiled", False))
                            initial_error_description = last_res.get("error_description")

                            log_rec.update({
                                "delete_only_fallback_used": False,
                                "delete_only_deletions": len(del_log),
                                "compiled_success": is_compiled,
                                "total_attempts_completed": 1,
                                "compile_latency_ms": compile_ms,
                                "fits_single_run": None,
                                "avg_chunk_tokens": None,
                                "max_chunk_tokens": None,
                                "llm_calls": 0,
                                "llm_latency_ms_total": 0,
                            })

                            try:
                                with open(str(AFFECTED_FILE_PATH / f"delete_only_log_{file_name[:-3]}.jsonl"), "w", encoding="utf-8") as f:
                                    for rec in del_log:
                                        f.write(str(asdict(rec)) + "\n")
                            except Exception:
                                pass

                            if is_compiled:
                                log_rec.update({"path_out": out_py_path})
                                _append_log(LOG_FILE, log_rec)
                            else:
                                with open(err_txt_path, "w", encoding="utf-8") as f:
                                    f.write(initial_error_description or "Unknown error")
                                error_word, error_message = get_error_word_message_from_content(err_txt_path)
                                failure_cleanup(AFFECTED_FILE_PATH, compilation_candidate, file_name[:-3], error_word, error_message)
                                _append_log(LOG_FILE, log_rec)

                            break
                        except Exception:
                            raise

                    try:
                        half_retries = max_retries // 2
                        is_late_retry = total_attempts_completed >= half_retries
                        is_final_outer = outer_idx == 2
                        use_whole_file_for_remote = not USE_LOCAL_LLM
                        try_whole_file = (is_late_retry and is_final_outer) or use_whole_file_for_remote
                        
                        llm_map = OPEN_LLM_MODELS if USE_LOCAL_LLM else LLM_MODELS
                        idx_str = os.getenv("LOCAL_LLM_IDX")
                        try:
                            idx = int(idx_str) if idx_str is not None else 0
                        except ValueError:
                            idx = 0
                        if not (0 <= idx < len(llm_map)):
                            idx = 0
                        selected_llm = llm_map[idx]


                        repair_result = attempt_repair(
                            copy_dir=copy_dir,
                            error_description=initial_error_description,
                            log_base=LOG_BASE,
                            file_hash=file_hash,
                            llm=selected_llm,
                            log_rec=log_rec,
                            strategy_state={"syntax_context": {"failures": 0}, "whole_file": {"failures": 0}},
                            try_whole_file=try_whole_file,
                            outer_idx=outer_idx,
                            affected_file_path=AFFECTED_FILE_PATH
                        )

                        if repair_result is None:
                            log_rec.update({
                                "compiled_success": False,
                                "total_attempts_completed": total_attempts_completed,
                                "repair_not_attempted_or_failed_precheck": True,
                            })
                            _append_log(LOG_FILE, log_rec)
                            break

                        final_code, llm_metrics, with_pin_point, start_ln, end_ln, base_indent = repair_result

                    except Exception as e:
                        print(f"Error during repair attempt: {e}")
                        log_rec.update({
                            "compiled_success": False,
                            "repair_exception": str(e),
                        })
                        _append_log(LOG_FILE, log_rec)
                        break

                    t0 = time.perf_counter()
                    final_code = (final_code or "").strip()

                    try:
                        if final_code:
                            if with_pin_point:
                                final_code = align_indentation(final_code, base_indent)
                                reattach_block(copy_dir, start_ln, end_ln, final_code)
                            else:
                                create_file_from_response(copy_dir, final_code)
                            compilation_candidate = read_file(copy_dir)
                        else:
                            copied_content = read_file(copy_dir)
                            if copied_content:
                                compilation_candidate = copied_content
                            else:
                                compilation_candidate = read_file(path_to_err_file)

                        compilation_result = compile_new_pyc(
                            compilation_candidate,
                            out_py_path,
                            out_pyc_path,
                            version
                        )
                        compile_ms = int((time.perf_counter() - t0) * 1000)
                        is_compiled = compilation_result["is_compiled"]
                        initial_error_description = compilation_result["error_description"]

                        if not is_compiled:
                            print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying ({total_attempts_completed+1}/{int(os.getenv('NO_OF_MAX_RETRIES'))+1}).... {Colors.ENDC}")

                        total_attempts_completed += 1
                        max_retries -= 1

                        log_rec.update({
                            "fits_single_run": llm_metrics.get("fits_single_run"),
                            "avg_chunk_tokens": llm_metrics.get("avg_chunk_tokens"),
                            "max_chunk_tokens": llm_metrics.get("max_chunk_tokens"),
                            "llm_calls": llm_metrics.get("llm_calls"),
                            "llm_latency_ms_total": llm_metrics.get("llm_latency_ms_total"),
                            "compiled_success": bool(is_compiled),
                            "total_attempts_completed": total_attempts_completed,
                            "compile_latency_ms": compile_ms,
                        })

                        if is_compiled:
                            log_rec.update({"path_out": out_py_path})
                            _append_log(LOG_FILE, log_rec)

                        else:
                            with open(err_txt_path, "w", encoding="utf-8") as f:
                                f.write(initial_error_description or "Unknown error")
                            error_word, error_message = get_error_word_message_from_content(err_txt_path)

                    except Exception:
                        compile_ms = int((time.perf_counter() - t0) * 1000)
                        is_compiled = False
                        print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying ({total_attempts_completed+1}/{int(os.getenv('NO_OF_MAX_RETRIES'))+1}).... {Colors.ENDC}")

                    if (max_retries < 0 or elapsed > MAX_EXAMPLE_RUNTIME_SEC):
                        print(f"{Colors.FAIL}    -> Max retries reached. Could not compile the file. {Colors.ENDC}")

                        if ENABLE_DELETE_ONLY_FALLBACK:
                            print(f"{Colors.WARNING}    -> Engaging delete-only fallback for the last llm output...  {Colors.ENDC}")
                            try:
                                llm_snapshot_path = AFFECTED_FILE_PATH / f"llm_last_output_{file_name[:-3]}.py"
                                with open(str(llm_snapshot_path), "w", encoding="utf-8") as f:
                                    f.write(compilation_candidate or "")

                                delete_only_input_path = AFFECTED_FILE_PATH / f"delete_only_input_from_llm_{file_name[:-3]}.py"
                                shutil.copyfile(str(llm_snapshot_path), str(delete_only_input_path))

                                delete_only_out_py_path = str(AFFECTED_FILE_PATH / f"delete_only_repaired_from_llm_{file_name[:-3]}.py")
                                delete_only_out_pyc_path = str(AFFECTED_FILE_PATH / f"delete_only_repaired_from_llm_{file_name[:-3]}.pyc")
                                delete_only_err_txt_path = str(AFFECTED_FILE_PATH / f"delete_only_from_llm_{file_name[:-3]}_error.txt")

                                fixed_code, del_log, last_res = delete_lines_until_compilable_with_oracle(
                                    py_file_content=read_file(str(delete_only_input_path)) or "",
                                    compile_check=compile_new_pyc,
                                    extract_line_number=extract_line_number,
                                    get_error_word_message_from_content=get_error_word_message_from_content,
                                    out_py_path=delete_only_out_py_path,
                                    out_pyc_path=delete_only_out_pyc_path,
                                    version=version,
                                    base_window=DELETE_ONLY_BASE_WINDOW,
                                    max_iters=DELETE_ONLY_MAX_ITERS,
                                    max_deleted_ratio=DELETE_ONLY_MAX_DELETED_RATIO,
                                    min_remaining_lines=1,
                                    err_txt_path=delete_only_err_txt_path,
                                )

                                log_rec.update({
                                    "delete_only_fallback_used": True,
                                    "delete_only_deletions": len(del_log),
                                    "llm_last_output_snapshot_path": str(llm_snapshot_path),
                                    "delete_only_input_path": str(delete_only_input_path),
                                    "delete_only_output_path": str(delete_only_out_py_path),
                                })

                                if last_res.get("is_compiled"):
                                    is_compiled = True
                                    log_rec.update({
                                        "compiled_success": True,
                                        "path_out": delete_only_out_py_path,
                                    })

                                    try:
                                        with open(str(AFFECTED_FILE_PATH / f"delete_only_log_from_llm_{file_name[:-3]}.jsonl"), "w", encoding="utf-8") as f:
                                            for rec in del_log:
                                                f.write(str(asdict(rec)) + "\n")
                                    except Exception:
                                        pass
                                else:
                                    log_rec.update({
                                        "delete_only_failed_error": last_res.get("error_description"),
                                    })

                            except Exception as e:
                                log_rec.update({
                                    "delete_only_fallback_used": True,
                                    "delete_only_fallback_exception": str(e),
                                })
                        else:
                            log_rec.update({"delete_only_fallback_used": False})

                        try:
                            failure_cleanup(AFFECTED_FILE_PATH, compilation_candidate, file_name[:-3], error_word, error_message)
                            if elapsed > MAX_EXAMPLE_RUNTIME_SEC:
                                log_rec.update({"compiled_success": None})
                            _append_log(LOG_FILE, log_rec)
                        finally:
                            break

            except KeyboardInterrupt:
                raise
            except Exception as e:
                exc_tb = traceback.format_exc()
                print(f"{Colors.FAIL}Unexpected exception for sample {file_hash or '<unknown>'}: {e}{Colors.ENDC}")

                crash_log = {
                    "run_id": run_id,
                    "timestamp": _now_iso(),
                    "row_index": int(idx),
                    "file_hash": file_hash,
                    "file_name": file_name,
                    "path_in": str(file_dir) if file_dir else None,
                    "bytecode_version": bytecode_version,
                    "decompiler": decompiler_name,
                    "dataset": dataset_name,
                    "compiled_success": False,
                    "unexpected_exception": str(e),
                    "unexpected_exception_traceback": exc_tb,
                }

                if isinstance(log_rec, dict):
                    crash_log.update(log_rec)
                    crash_log["compiled_success"] = False
                    crash_log["unexpected_exception"] = str(e)
                    crash_log["unexpected_exception_traceback"] = exc_tb

                try:
                    _append_log(LOG_FILE, crash_log)
                except Exception:
                    pass

                try:
                    if AFFECTED_FILE_PATH is not None:
                        with open(AFFECTED_FILE_PATH / f"unexpected_exception_{file_name or 'unknown'}.txt", "w", encoding="utf-8") as f:
                            f.write(exc_tb)
                except Exception:
                    pass

            finally:
                if copy_dir is not None:
                    try:
                        os.unlink(copy_dir)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass

                count_idx += 1
                print(footer)