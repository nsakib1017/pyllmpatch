# Read dataset_summary.csv into a pandas DataFrame
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

import difflib  
import json     
import time     
import uuid     
from datetime import datetime, timezone  

from typing import List, Tuple, Optional, Dict, Any  

from utils.token_helpers import *
from utils.providers import *
from utils.chunk_helpers import *
from utils.file_helpers import *
from utils.generate_bytecode import *

from dotenv import load_dotenv
load_dotenv()

def read_csv_file(file_name: str) -> pd.DataFrame:
    script_dir = Path(__file__).parent
    csv_path = script_dir / file_name
    print(csv_path)
    print(f"Trying to read: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    return df

def prepare_snippets_for_repair(previous_run_log_path: str = ""):
    if not previous_run_log_path:
        return read_csv_file('./dataset/balanced_sample.csv')
    df_previous_log = pd.read_json(previous_run_log_path, lines=True)
    df_balanced = read_csv_file('./dataset/balanced_sample.csv')  # your helper

    # Normalize types and remove NaNs/dupes just in case
    prev_hashes = (
        df_previous_log['file_hash']
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    df_balanced['file_hash'] = df_balanced['file_hash'].astype(str)
    mask = ~df_balanced['file_hash'].isin(set(prev_hashes))
    return df_balanced.loc[mask].copy()

BASE_DIR = Path("../decompiler_workspace")

def get_error_word_message_from_content(filepath):
    FILE_LINE_RE = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in (.+))?')
    ERROR_LINE_RE = re.compile(r'^\s*(\w+(?:Error|Exception))(?:\s*:\s*(.*))?$')
    SORRY_LINE_RE = re.compile(r'^\s*Sorry:\s*(\w+(?:Error|Exception))\s*:\s*(.*?)\s*\(([^,]+),\s*line\s*(\d+)\)\s*$')
    error_word = []
    messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            m_file = FILE_LINE_RE.match(s)
            if m_file:
                continue
            m_err = ERROR_LINE_RE.match(s)
            if m_err:
                error_word.append(m_err.group(1))
                messages.append((m_err.group(2) or "").strip())
                break
            m_sorry = SORRY_LINE_RE.match(line)
            if m_sorry:
                error_word.append(m_sorry.group(1))
                messages.append((m_sorry.group(2) or "").strip())
                break
            
    error_word = error_word[0] if len(error_word) > 0 else None
    message = messages[0] if len(messages) > 0 else None

    return error_word, message


def strip_code_fences(text: str) -> str:
    fence = r"(?:```|~~~)"
    lang_until_eol = r"[^\r\n]*"  
    paired_re = re.compile(
        rf"(?P<f>{fence})[ \t]*{lang_until_eol}[ \t]*(?:\r?\n)?"
        rf"(?P<body>.*?)"
        rf"(?:\r?\n)?(?P=f)[ \t]*",
        re.DOTALL,
    )
    text = paired_re.sub(lambda m: m.group("body"), text)
    leading_open_re = re.compile(
        rf"^\ufeff?(?:{fence})[ \t]*{lang_until_eol}[ \t]*(?:\r?\n)?",
        re.DOTALL,
    )
    text = leading_open_re.sub("", text, count=1)
    trailing_close_re = re.compile(
        rf"(?:\r?\n)?(?:{fence})[ \t]*\Z",
        re.DOTALL,
    )
    text = trailing_close_re.sub("", text, count=1)

    return text



def _count_fences(text: str) -> int:
    return len(re.findall(r"(?:```|~~~)[^\n]*\n.*?(?:```|~~~)", text, flags=re.DOTALL))

def _diff_lines(a: str, b: str) -> int:
    return sum(1 for ln in difflib.ndiff(a.splitlines(), b.splitlines()) if ln[:1] in {"+", "-"})

def _parse_error_type(msg: Optional[str]) -> Optional[str]:
    if not msg:
        return None
    # pick first non-empty line and extract final token before ':' as error type
    first = next((ln for ln in str(msg).splitlines() if ln.strip()), "")
    m = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*Error)\b", first)
    return m.group(1) if m else first[:120]

def _count_tokens_safe(text: Optional[str]) -> int:
    try:
        return count_tokens(text or "")
    except Exception:
        return len(text or "")

def _now_iso() -> str:  # ISO8601 with timezone naive (UTC-like)
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def process_file_in_single_run(content: str, model: dict, error: str, previous_error_summary_prompt="") -> Tuple[str, Dict[str, Any]]:
    """Simulates processing a file that fits in the context window."""
    model_name = f"{model['provider']} - {model['name']}"
    print(f"{Colors.OKGREEN}    -> Content fits in a single run for {model_name}. Processing...{Colors.ENDC}")
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_user_prompt(content, error)}
    ]
    # print(prompt[1]["content"])
    t0 = time.perf_counter()
    llm_raw = make_llm_call(prompt, model=model['name'], provider=model['provider'])
    # print(llm_raw)
    llm_elapsed_ms = int((time.perf_counter() - t0) * 1000)
    try:
        llm_response = strip_code_fences(llm_raw.get('content', {})) if llm_raw else content
    except Exception:
        llm_response = llm_raw.get('content', {}) if llm_raw else content

    llm_usage = llm_raw.get('usage', {}) if llm_raw else {}
    # llm_raw = llm_response.get('content', {}) if llm_response else {}
    # llm_usage = llm_raw.get('usage', {}) if llm_raw else {}
    metrics = {
        "fits_single_run": True,
        "llm_calls": 1,
        "llm_latency_ms_total": llm_elapsed_ms,
        "prompt_token_count": getattr(llm_usage, 'prompt_token_count', None),
        "candidates_token_count": getattr(llm_usage, 'candidates_token_count', None),
        "total_token_count": getattr(llm_usage, 'total_token_count', None),
        "chunk_count": 1,
        "merge_passes": 0,
        "avg_chunk_tokens": _count_tokens_safe(content),
        "max_chunk_tokens": _count_tokens_safe(content),
    }
    return (llm_response if llm_response else content), metrics


def process_and_merge_chunks(chunks: List[str], model: dict, error, previous_error_summary_prompt="") -> Tuple[str, Dict[str, Any]]:
    """
    Simulates sending each chunk to an LLM and then merges the results back together.
    """
    model_name = f"{model['provider']} - {model['name']}"
    print(f"   {Colors.OKCYAN} -> Processing and merging for {len(chunks)} chunks with {model_name}...{Colors.ENDC}")
    
    corrected_chunks = []
    total_in = 0
    total_out = 0
    total_latency_ms = 0
    max_chunk_tokens = 0
    llm_calls = 0

    for i, chunk in enumerate(chunks):
        ctoks = _count_tokens_safe(chunk)
        max_chunk_tokens = max(max_chunk_tokens, ctoks)
        print(f"        -> Processing chunk {i+1}/{len(chunks)} ({ctoks} tokens)...")
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_user_prompt(chunk, error, previous_error_summary_prompt)}
        ]
        # print(prompt[1]["content"])
        t0 = time.perf_counter()
        llm_response = make_llm_call(prompt, model=model['name'], provider=model['provider'])
        total_latency_ms += int((time.perf_counter() - t0) * 1000)
        llm_calls += 1
        llm_usage = llm_response.get('usage', {}) if llm_response else {}
        llm_raw = llm_response.get('content', {}) if llm_response else {}
        # total_in += _count_tokens_safe(prompt[0]["content"]) + _count_tokens_safe(prompt[1]["content"])
        try:
            corrected_chunk = strip_code_fences(llm_raw.get('content', {})) if llm_raw else chunk
        except Exception:
            corrected_chunk = llm_raw or chunk
        # total_out += _count_tokens_safe(corrected_chunk)

        corrected_chunks.append(corrected_chunk)

    final_content = "\n".join(corrected_chunks)

    print(f"    {Colors.OKGREEN} -> All chunks processed and merged.{Colors.ENDC}")
    print(f"    -> Final merged content token count: {count_tokens(final_content)}")

    metrics = {
        "fits_single_run": False,
        "chunk_count": len(chunks),
        "merge_passes": 1,  # based on your call to chunk_top_level_objects_lenient_merged(..., extra_merge_passes=1)
        "avg_chunk_tokens": int(sum(_count_tokens_safe(c) for c in chunks) / max(1, len(chunks))),
        "max_chunk_tokens": max_chunk_tokens,
        "prompt_token_count": getattr(llm_usage, 'prompt_token_count', None),
        "candidates_token_count": getattr(llm_usage, 'candidates_token_count', None),
        "total_token_count": getattr(llm_usage, 'total_token_count', None),
        "llm_calls": llm_calls,
        "llm_latency_ms_total": total_latency_ms,
        # "total_input_tokens": total_in,
        # "total_output_tokens": total_out,
    }
    return final_content, metrics


def process_file_for_syntax_error_patching(content: str, error_description, error_list, log_rec, llm=LLM_MODELS[4]) -> Optional[Tuple[str, Dict[str, Any]]]:
    log_rec.update({"provider": llm["provider"], "model_name": llm["name"]})
    
    if content is not None:
        does_not_fit_model = check_context_windows(content, error_description, llm)
        if not does_not_fit_model:
            final_code, metrics = process_file_in_single_run(content, llm, error_description)
        else:
            chunks = chunk_top_level_objects_lenient_merged(
                content,
                max_tokens_per_chunk=llm['token_for_completion'] - 5000,
                headroom_tokens=1000,
                extra_merge_passes=1
            )
            for chunk in chunks:
                if _count_tokens_safe(chunk) > llm['token_for_completion'] - 5000:
                    return None
            final_code, metrics = process_and_merge_chunks(chunks, llm, error_description)
        return final_code, metrics
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


def filter_not_run_false_only_points(path: str):
    df_false_only = read_csv_file(path)
    df_false_only = df_false_only[df_false_only['compiled_success'] == True]
    prev_hashes = (
        df_false_only['file_hash']
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    df_all['file_hash'] = df_all['file_hash'].astype(str)

    # Anti-join: keep rows in df_balanced whose file_hash is NOT in previous
    mask = ~df_all['file_hash'].isin(set(prev_hashes))
    return df_all.loc[mask].copy()

if __name__ == "__main__":
    # previuos_run_log_path = Path("results/experiment_outputs/20251009T033339Z/f5d5e521eee14823961dacd3b6489f69/run_log_f5d5e521eee14823961dacd3b6489f69.jsonl")
    df_all = prepare_snippets_for_repair()
    df_syntax_error_balanced = filter_not_run_false_only_points('./dataset/cleaned_results_with_no_retry.csv')
    df_syntax_error_balanced = df_syntax_error_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    # df_syntax_error_balanced = df_syntax_error_balanced.head(50) 
    print('DataFrame shape:', df_syntax_error_balanced.shape)
    run_id = uuid.uuid4().hex
    batch_start_ts = _now_iso()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") 
    LOG_BASE = Path(f"results/experiment_outputs/{ts}/{run_id}")

    if not LOG_BASE.exists():
        LOG_BASE.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_BASE / f"run_log_{run_id}.jsonl"
    count_idx = 0

    for idx, row in df_syntax_error_balanced.iterrows():

        is_compiled = False
        file_hash = norm_str(row.get("file_hash"))
        file_dir = BASE_DIR / file_hash / "decompiler_output"
        

        # highest_indented_file returns Optional[Tuple[Path, int]]
        res = highest_indented_file(file_dir, pattern="indented_*.py")
        header = f"\n# --- Processing Content from file: {str(res[0])} ({count_idx+1}/{len(df_syntax_error_balanced)}) --- #\n"
        footer = f"\n# --- End of Processing Content from file: {str(res[0])} ({count_idx+1}/{len(df_syntax_error_balanced)}) --- #\n"

        print(header)
        path_to_err_file = res[0]
        content = read_file(path_to_err_file)
        initial_content = content
        error_description = row.get("syntactic_error_description") 
        error_message_processed =  row.get("precessed_error_message") 
        error_word = row.get("syntactic_error_word")
        compilation_result = None

        max_retries = int(os.getenv("NO_OF_MAX_RETRIES")) if os.getenv("NO_OF_MAX_RETRIES") else 0  # max attempts to recompile
        total_attempts_completed = 0  
        error_list = []

        log_rec: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": _now_iso(),
            "file_hash": file_hash,
            "total_attempts_allowed": max_retries + 1,
            "retries_allowed": max_retries,
            "path_in": str(path_to_err_file),
            "bytecode_version": extract_bytecode_major_minor(content),
            # "provider": LLM_MODELS[3]["provider"],
            # "model_name": LLM_MODELS[3]["name"],
            "compile_error_word_before": error_word,
            "compile_error_description_before": error_message_processed,
            "fences_stripped": _count_fences(content),
        }
            # LLM repair
        version = extract_bytecode_major_minor(content)
        while not is_compiled:
            error_list.append(error_description)
            processed = process_file_for_syntax_error_patching(content, error_description, error_list, log_rec=log_rec, llm=LLM_MODELS[4] if max_retries >= 4 else LLM_MODELS[5])
            if processed is None:
                break

            AFFECTED_FILE_PATH = LOG_BASE / file_hash
            if not AFFECTED_FILE_PATH.exists():
                AFFECTED_FILE_PATH.mkdir(parents=True, exist_ok=True)
            
            final_code, llm_metrics = processed  

            # compile with timing
            t0 = time.perf_counter()
            try:
                compilation_result = compile_new_pyc(
                    final_code,
                    str(AFFECTED_FILE_PATH / f"syntax_repaired_{res[1]}.py"),
                    str(AFFECTED_FILE_PATH / f"syntax_repaired_{res[1]}.pyc"),
                    version
                )
                compile_ms = int((time.perf_counter() - t0) * 1000)
                is_compiled = compilation_result["is_compiled"]
                error_description = compilation_result["error_description"]
                if not is_compiled:
                    print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying.... {Colors.ENDC}")
            except Exception as e:
                compile_ms = int((time.perf_counter() - t0) * 1000)
                print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying.... {Colors.ENDC}")
                # error_description = str(e)


            total_attempts_completed += 1
            max_retries -= 1

            log_rec.update({
                "fits_single_run": llm_metrics.get("fits_single_run"),
                "chunk_count": llm_metrics.get("chunk_count"),
                "merge_passes": llm_metrics.get("merge_passes"),
                "avg_chunk_tokens": llm_metrics.get("avg_chunk_tokens"),
                "max_chunk_tokens": llm_metrics.get("max_chunk_tokens"),
                "llm_calls": llm_metrics.get("llm_calls"),
                "llm_latency_ms_total": llm_metrics.get("llm_latency_ms_total"),
                "prompt_token_count": llm_metrics.get("prompt_token_count"),
                "candidates_token_count": llm_metrics.get("candidates_token_count"),
                "total_token_count": llm_metrics.get("total_token_count"),
                "compiled_success": bool(is_compiled),
                "total_attempts_completed": total_attempts_completed,
                "compile_latency_ms": compile_ms,
                "compile_error_word_after": None,
                "compile_error_message_after": None,
                "diff_lines": _diff_lines(final_code, initial_content),
            })
            if is_compiled:
                log_rec.update({"path_out": str(AFFECTED_FILE_PATH / f"syntax_repaired_{res[1]}.py")})
                _append_log(LOG_FILE, log_rec)
            elif not is_compiled:

                with open(str(AFFECTED_FILE_PATH / f"syntax_failed_repaired_{res[1]}_error.txt"), "w", encoding="utf-8") as f:
                    f.write(error_description or "Unknown error")
                    f.close()
                error_word, error_message = get_error_word_message_from_content(str(AFFECTED_FILE_PATH / f"syntax_failed_repaired_{res[1]}_error.txt"))
                if max_retries >= 0:
                    content = final_code  # retry with the latest code
                else:
                    print(f"{Colors.FAIL}    -> Max retries reached. Could not compile the file. {Colors.ENDC}")
                    try:
                        failure_cleanup(AFFECTED_FILE_PATH, final_code, res[1], error_word, error_message)
                        _append_log(LOG_FILE, log_rec)
                    except FileNotFoundError:
                        pass
                    finally:
                        break
        count_idx += 1
        print(footer)
        # break

        
