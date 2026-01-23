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

from typing import Tuple, Optional, Dict, Any  

from utils.token_helpers import *
from utils.providers import *
from utils.chunk_helpers import *
from utils.file_helpers import *
from utils.generate_bytecode import *

from model.inference import *
from utils.indentation_fixer import *


from dotenv import load_dotenv
load_dotenv()


RepairResult = Tuple[str, Dict[str, Any], bool, Optional[int], Optional[int], Optional[int]]



BASE_DIR_PYTHON_FILES = Path(str(os.getenv("BASE_DIR_PYTHON_FILES")))
MAX_EXAMPLE_RUNTIME_SEC = 2 * 60 * 60  # 2 hours
def prepare_snippets_for_repair(
    previous_run_log_path: str = "",
    only_previously_failed: bool = False,
):
    if not previous_run_log_path:
        return read_csv_file(
            f"{os.getenv('PROJECT_ROOT_DIR')}/dataset/{os.getenv('BASE_DATASET_NAME')}"
        )

    df_previous_log = pd.read_json(previous_run_log_path, lines=True)
    df_balanced = read_csv_file(
        f"{os.getenv('PROJECT_ROOT_DIR')}/dataset/{os.getenv('BASE_DATASET_NAME')}"
    )

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

def make_call_to_local_llm(content: str,  error: str, current_explanation: str, affected_file_path: Path, model_path: str,):
    messages = build_chat_messages(code_snippet=content.strip("\n"), error_message=error, system_prompt=SYSTEM_PROMPT_FOR_LOCAL, user_prompt_template=USER_PROMPT_TEMPLATE_LOCAL if len(current_explanation) > 0 else USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION, current_explanation=current_explanation)

    if not affected_file_path.exists():
                affected_file_path.mkdir(parents=True, exist_ok=True)

    out_path = affected_file_path / "last_message_to_llm_for_file.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=str)

    llm_raw = call_llm_with_message(messages=messages, model_path=model_path)
    return llm_raw


def explain_current_code_syntax_error(content: str, error: str, model_path: str) -> str:
    messages = build_chat_messages(code_snippet=content.strip("\n"), error_message=error, system_prompt=SYSTEM_PROMPT_FOR_ROOT_CAUSE_ANALYSIS, user_prompt_template=USER_PROMPT_TEMPLATE_ROOT_CAUSE)
    llm_raw = call_llm_with_message(messages=messages, model_path=model_path)
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

    current_explanation = explain_current_code_syntax_error(content, error, model["model_path"]) if outer_idx != 0 else ""
    # print(f"    -> Explanation of current syntax error:\n{current_explanation}\n", outer_idx)
    if model["name"] in {m["name"] for m in OPEN_LLM_MODELS}:
        llm_raw = make_call_to_local_llm(content, error, current_explanation, affected_file_path, model["model_path"])
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


def process_file_for_syntax_error_patching(initial_content: str, error_description, affected_file_path: Path, log_rec={}, llm=OPEN_LLM_MODELS[1], outer_idx=0) -> Optional[Tuple[str, Dict[str, Any]]]:
    log_rec.update({"provider": llm["provider"], "model_name": llm["name"]})
    if initial_content is not None and not count_tokens_safe(initial_content,  llm["provider"], llm["name"]) > llm['token_for_completion'] - 5000:
            final_code, metrics = process_file_in_single_run(initial_content, llm, error_description, affected_file_path, outer_idx)
            return final_code, metrics
        # else:
        #     # chunks = chunk_top_level_objects_lenient_merged(
        #     #     initial_content,
        #     #     max_tokens_per_chunk=llm['token_for_completion'] - 5000,
        #     #     headroom_tokens=1000,
        #     #     extra_merge_passes=1
        #     # )
        #     # for chunk in chunks:
        #     #     if count_tokens_safe(chunk,  llm["provider"], llm["name"]) > llm['token_for_completion'] - 5000:
        #     #         return None
        #     # final_code, metrics, corrected_chunks = process_and_merge_chunks(chunks, llm, error_description, retry_attempt, previuos_response, previous_error)
        #     return None  
    else:
        return None

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
    try_whole_file: bool = False,
    outer_idx: int = 0,
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
                log_base / file_hash,
                log_rec=log_rec,
                llm=llm,
                outer_idx=outer_idx,
            )

        elif try_whole_file:
            initial_content = read_file(copy_dir)
            processed = process_file_for_syntax_error_patching(
                initial_content,
                error_description,
                log_base / file_hash,
                log_rec=log_rec,
                llm=llm,
                outer_idx=outer_idx,
            )

        if processed is None:
            state["failures"] += 1
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




if __name__ == "__main__":
    previuos_run_log_path = Path(str(os.getenv("PREVIOUS_RUN_LOG_PATH"))) if os.getenv("PREVIOUS_RUN_LOG_PATH") else None
    df_syntax_error_balanced = prepare_snippets_for_repair(str(previuos_run_log_path) if previuos_run_log_path else "", only_previously_failed=False)
    df_syntax_error_balanced = df_syntax_error_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print('DataFrame shape:', df_syntax_error_balanced.shape)
    
    for outer_idx in range(0,3):  # <-- new outer loop

        run_id = uuid.uuid4().hex
        batch_start_ts = _now_iso()
        LOG_BASE = Path(f"results/experiment_outputs/{ts}/{run_id}")

        if not LOG_BASE.exists():
            LOG_BASE.mkdir(parents=True, exist_ok=True)
        LOG_FILE = LOG_BASE / f"run_log_{run_id}.jsonl"
        count_idx = 0

        for idx, row in df_syntax_error_balanced.iterrows():

            is_compiled = False
            file_hash = norm_str(row.get("file_hash"))
            file_name = norm_str(row.get("file"))
            file_dir = BASE_DIR_PYTHON_FILES / file_hash / file_name

            header = f"\n# --- Processing Content from file (with config {outer_idx}): {str(file_dir)} ({count_idx+1}/{len(df_syntax_error_balanced)}) --- #\n"
            footer = f"\n# --- End of Processing Content from file (with config {outer_idx}): {str(file_dir)} ({count_idx+1}/{len(df_syntax_error_balanced)}) --- #\n"

            print(header)
            path_to_err_file = str(file_dir)

            initial_error_description = row.get("error_description")
            error_message_processed = row.get("error_msg_clean")
            error_word = row.get("syntactic_error_word")

            copy_dir = BASE_DIR_PYTHON_FILES / file_hash / f"copy_for_run_id_{run_id}_of_{file_name}"
            copy_file(path_to_err_file, copy_dir)
            whole_content = read_file(copy_dir)
            version = extract_bytecode_major_minor(whole_content)
            compilation_result = None

            max_retries = int(os.getenv("NO_OF_MAX_RETRIES")) if os.getenv("NO_OF_MAX_RETRIES") else 0
            total_attempts_completed = 0
            error_list = []

            log_rec = {
                "run_id": run_id,
                "timestamp": _now_iso(),
                "file_hash": file_hash,
                "total_attempts_allowed": max_retries + 1,
                "retries_allowed": max_retries,
                "path_in": str(path_to_err_file),
                "bytecode_version": version,
                # "provider": LLM_MODELS[3]["provider"],
                # "model_name": LLM_MODELS[3]["name"],
                # "compile_error_word_before": error_word,
                "compile_error_description_before": error_message_processed,
            }

            t_begin = time.monotonic()
            retry_attempt = False
            previuos_response = None
            previous_error = ""

            while not is_compiled:
                try:
                    # print(OPEN_LLM_MODELS[0]['model_path'])
                    final_code, llm_metrics, with_pin_point, start_ln, end_ln, base_indent = attempt_repair(
                        copy_dir=copy_dir,
                        error_description=initial_error_description,
                        log_base=LOG_BASE,
                        file_hash=file_hash,
                        llm=OPEN_LLM_MODELS[3],
                        log_rec=log_rec,
                        strategy_state={"syntax_context": {"failures": 0}, "whole_file": {"failures": 0}},
                        try_whole_file=True if (total_attempts_completed > (int(max_retries/2) - 1)) else False,
                        outer_idx=outer_idx,
                    )
                except Exception as e:
                    print(f"Error during repair attempt: {e}")
                    break

                AFFECTED_FILE_PATH = LOG_BASE / file_hash
                if not AFFECTED_FILE_PATH.exists():
                    AFFECTED_FILE_PATH.mkdir(parents=True, exist_ok=True)

                elapsed = time.monotonic() - t_begin
                t0 = time.perf_counter()
                final_code = final_code.strip()

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

                try:
                    compilation_result = compile_new_pyc(
                        compilation_candidate,
                        str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.py"),
                        str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.pyc"),
                        version
                    )
                    compile_ms = int((time.perf_counter() - t0) * 1000)
                    is_compiled = compilation_result["is_compiled"]
                    initial_error_description = compilation_result["error_description"]
                    if not is_compiled:
                        print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying ({total_attempts_completed+1}/{int(os.getenv('NO_OF_MAX_RETRIES'))+1}).... {Colors.ENDC}")
                except Exception as e:
                    compile_ms = int((time.perf_counter() - t0) * 1000)
                    print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying ({total_attempts_completed+1}/{int(os.getenv('NO_OF_MAX_RETRIES'))+1}).... {Colors.ENDC}")


                total_attempts_completed += 1
                max_retries -= 1

                log_rec.update({
                    "fits_single_run": llm_metrics.get("fits_single_run"),
                    # "chunk_count": llm_metrics.get("chunk_count"),
                    # "merge_passes": llm_metrics.get("merge_passes"),
                    "avg_chunk_tokens": llm_metrics.get("avg_chunk_tokens"),
                    "max_chunk_tokens": llm_metrics.get("max_chunk_tokens"),
                    "llm_calls": llm_metrics.get("llm_calls"),
                    "llm_latency_ms_total": llm_metrics.get("llm_latency_ms_total"),
                    # "prompt_token_count": llm_metrics.get("prompt_token_count"),
                    # "candidates_token_count": llm_metrics.get("candidates_token_count"),
                    # "total_token_count": llm_metrics.get("total_token_count"),
                    "compiled_success": bool(is_compiled),
                    "total_attempts_completed": total_attempts_completed,
                    "compile_latency_ms": compile_ms,
                })

                if is_compiled:
                    log_rec.update({"path_out": str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.py")})
                    os.unlink(copy_dir)
                    _append_log(LOG_FILE, log_rec)
                elif not is_compiled:
                    with open(str(AFFECTED_FILE_PATH / f"syntax_failed_repaired_{file_name[:-3]}_error.txt"), "w", encoding="utf-8") as f:
                        f.write(initial_error_description or "Unknown error")
                        f.close()
                    error_word, error_message = get_error_word_message_from_content(str(AFFECTED_FILE_PATH / f"syntax_failed_repaired_{file_name[:-3]}_error.txt"))
                if max_retries < 0 or elapsed > MAX_EXAMPLE_RUNTIME_SEC:
                    print(f"{Colors.FAIL}    -> Max retries reached. Could not compile the file. {Colors.ENDC}")
                    try:
                        failure_cleanup(AFFECTED_FILE_PATH, compilation_candidate, file_name[:-3], error_word, error_message)
                        _append_log(LOG_FILE, log_rec)
                        os.unlink(copy_dir)
                    except FileNotFoundError:
                        pass
                    finally:
                        break

            count_idx += 1
            print(footer)
            # break  # for testing only one file per outer loop

