# Read dataset_summary.csv into a pandas DataFrame
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

from typing import List, Tuple, Optional, Dict, Any  

from utils.token_helpers import *
from utils.providers import *
from utils.chunk_helpers import *
from utils.file_helpers import *
from utils.generate_bytecode import *

from model.inference import fix_python_syntax
from utils.indentation_fixer import *


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
    df_balanced = read_csv_file('./dataset/decompiled_syntax_errors.csv')  # your helper

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

BASE_DIR = Path("/home/diogenes/pylingual_colaboration/pypi_downloaded")

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


def strip_code_fences(payload: Any) -> str:
    """
    Strips the code fences (```python, ~~~, etc.) from a string,
    and handles different formats like None, dict, list, bytes, etc.
    """
    # ---- 1) Normalize to text ------------------------------------------------
    def _normalize(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            # Common LLM shapes
            val = x.get("content") or x.get("text") or ""
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="ignore")
            return str(val) if not isinstance(val, str) else val
        if isinstance(x, list):
            parts = []
            for item in x:
                if isinstance(item, dict):
                    v = item.get("content") or item.get("text") or ""
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", errors="ignore")
                    parts.append(v if isinstance(v, str) else str(v))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(x)

    text = _normalize(payload)

    # ---- 2) Strip fences (regex patterns) ----------------------------------
    fence = r"(?:```|~~~)"
    lang_until_eol = r"[^\r\n]*"  # Match language until end of line
    # Refined regex that also ensures line breaks are handled
    paired_re = re.compile(
        rf"(?P<f>{fence})[ \t]*{lang_until_eol}[ \t]*(?:\r?\n)?"
        rf"(?P<body>.*?)"
        rf"(?:\r?\n)?(?P=f)[ \t]*", re.DOTALL
    )

    # This should strip the code block properly
    text = paired_re.sub(lambda m: m.group("body"), text)

    # ---- 3) Remove leading fence at the start of the string -----------------
    leading_open_re = re.compile(
        rf"^\ufeff?(?:{fence})[ \t]*{lang_until_eol}[ \t]*(?:\r?\n)?",
        re.DOTALL,
    )
    text = leading_open_re.sub("", text, count=1)

    # ---- 4) Remove trailing fence at the end of the string -----------------
    trailing_close_re = re.compile(
        rf"(?:\r?\n)?(?:{fence})[ \t]*\Z",
        re.DOTALL,
    )
    text = trailing_close_re.sub("", text, count=1)

    return text.strip()





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

def _count_tokens_safe(text: Optional[str], provider, model_name) -> int:
    try:
        return count_tokens(text or "", provider, model_name)
    except Exception:
        return len(text or "")

def _now_iso() -> str:  # ISO8601 with timezone naive (UTC-like)
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def process_file_in_single_run(content: str, model: dict, error: str, retry_attempt=False, previuos_response="", previous_error="") -> Tuple[str, Dict[str, Any]]:
    """Simulates processing a file that fits in the context window."""
    model_name = f"{model['provider']} - {model['name']}"
    print(f"{Colors.OKGREEN}    -> Content fits in a single run for {model_name}. Processing...{Colors.ENDC}")
    context_indent = detect_context_indent(content.strip("\n"))
    normalized_code = normalize_indentation(content.strip("\n"), context_indent)
    t0 = time.perf_counter()
    if model["name"] in ['qwen-v2.5-coder-7b']:
        messages = build_chat_messages(
            code_snippet=normalized_code,
            error_message=error,
            system_prompt=SYSTEM_PROMPT_FOR_LOCAL,
            user_prompt_template=USER_PROMPT_TEMPLATE_LOCAL,
        )
        print(content)
        # print(normalized_code)
        print('--------------------------------------------------')
        llm_raw = fix_python_syntax(messages=messages)

        try:
            llm_response = strip_code_fences(llm_raw) if llm_raw else content
        except Exception:
            llm_response = content

         #reapply_context_indent(llm_raw, context_indent)
        
        # sys.exit(0)
    else:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_user_prompt(content, error, retry_attempt, previuos_response, previous_error)}
        ]
        # print(prompt[1]["content"])

        llm_raw = make_llm_call(prompt, model=model['name'], provider=model['provider'])
        # print(llm_raw)
        try:
            llm_response = strip_code_fences(llm_raw) if llm_raw else content
        except Exception:
            llm_response = content
    try:
        llm_usage = llm_raw.get('usage', {}) if llm_raw else {}
    except Exception:
        llm_usage = {}
    # llm_raw = llm_response.get('content', {}) if llm_response else {}
    # llm_usage = llm_raw.get('usage', {}) if llm_raw else {}
    llm_elapsed_ms = int((time.perf_counter() - t0) * 1000)
    metrics = {
        "fits_single_run": True,
        "llm_calls": 1,
        "llm_latency_ms_total": llm_elapsed_ms,
        "prompt_token_count": getattr(llm_usage, 'prompt_token_count', None),
        "candidates_token_count": getattr(llm_usage, 'candidates_token_count', None),
        "total_token_count": getattr(llm_usage, 'total_token_count', None),
        "chunk_count": 1,
        "merge_passes": 0,
        "avg_chunk_tokens": _count_tokens_safe(content, model["provider"], model["name"]),
        "max_chunk_tokens": _count_tokens_safe(content, model["provider"], model["name"]),
    }
    return (llm_response if llm_response else content), metrics


def process_and_merge_chunks(chunks: List[str], model: dict, error, retry_attempt=False, previuos_chunks=[], previous_error="") -> Tuple[str, Dict[str, Any]]:
    """
    Simulates sending each chunk to an LLM and then merges the results back together.
    """
    model_name = f"{model['provider']} - {model['name']}"
    print(f"   {Colors.OKCYAN} -> Processing and merging for {len(chunks)} chunks with {model_name}...{Colors.ENDC}")
    
    corrected_chunks = []
    total_latency_ms = 0
    max_chunk_tokens = 0
    llm_calls = 0

    for i, chunk in enumerate(chunks):
        ctoks = _count_tokens_safe(chunk,  model["provider"], model["name"])
        max_chunk_tokens = max(max_chunk_tokens, ctoks)
        print(f"        -> Processing chunk {i+1}/{len(chunks)} ({ctoks} tokens)...")
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_user_prompt_chunk(chunk, error, retry_attempt, previuos_chunks[i] if previuos_chunks and len(previuos_chunks) > 0 else "", previous_error)}
        ]
        print(prompt[1]["content"])
        t0 = time.perf_counter()
        llm_response = make_llm_call(prompt, model=model['name'], provider=model['provider'])
        total_latency_ms += int((time.perf_counter() - t0) * 1000)
        llm_calls += 1
        llm_usage = llm_response.get('usage', {}) if llm_response else {}
        llm_raw = llm_response.get('content', {}) if llm_response else {}
        # total_in += _count_tokens_safe(prompt[0]["content"]) + _count_tokens_safe(prompt[1]["content"])
        try:
            corrected_chunk = strip_code_fences(llm_raw) if llm_raw else chunk
        except Exception:
            corrected_chunk = llm_raw
        # total_out += _count_tokens_safe(corrected_chunk)

        corrected_chunks.append(corrected_chunk)

    final_content = "\n".join(corrected_chunks)

    print(f"    {Colors.OKGREEN}    -> All chunks processed and merged.{Colors.ENDC}")
    print(f"    -> Final merged content token count: {count_tokens(final_content, model['provider'], model['name'])}")

    metrics = {
        "fits_single_run": False,
        "chunk_count": len(chunks),
        "merge_passes": 1,  # based on your call to chunk_top_level_objects_lenient_merged(..., extra_merge_passes=1)
        "avg_chunk_tokens": int(sum(_count_tokens_safe(c,  model["provider"], model["name"]) for c in chunks) / max(1, len(chunks))),
        "max_chunk_tokens": max_chunk_tokens,
        "prompt_token_count": getattr(llm_usage, 'prompt_token_count', None),
        "candidates_token_count": getattr(llm_usage, 'candidates_token_count', None),
        "total_token_count": getattr(llm_usage, 'total_token_count', None),
        "llm_calls": llm_calls,
        "llm_latency_ms_total": total_latency_ms,
    }
    return final_content, metrics, corrected_chunks


def process_file_for_syntax_error_patching(initial_content: str, error_description, retry_attempt, previuos_response, previous_error, log_rec, llm=LLM_MODELS[4]) -> Optional[Tuple[str, Dict[str, Any]]]:
    log_rec.update({"provider": llm["provider"], "model_name": llm["name"]})
    corrected_chunks = []
    if initial_content is not None:
        does_not_fit_model = check_context_windows(initial_content, llm)
        if not does_not_fit_model:
            final_code, metrics = process_file_in_single_run(initial_content, llm, error_description, retry_attempt, previuos_response, previous_error)
        else:
            chunks = chunk_top_level_objects_lenient_merged(
                initial_content,
                max_tokens_per_chunk=llm['token_for_completion'] - 5000,
                headroom_tokens=1000,
                extra_merge_passes=1
            )
            for chunk in chunks:
                if _count_tokens_safe(chunk,  llm["provider"], llm["name"]) > llm['token_for_completion'] - 5000:
                    return None
            final_code, metrics, corrected_chunks = process_and_merge_chunks(chunks, llm, error_description, retry_attempt, previuos_response, previous_error)
        return final_code, metrics, corrected_chunks
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

def extract_line_number(error_msg: str):
    # Find patterns like: "line 131"
    m = re.search(r"line\s+(\d+)", error_msg)
    if m:
        return int(m.group(1))
    return None


def copy_file(src: Path | str, dst: Path | str) -> None:
    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

if __name__ == "__main__":
    # previuos_run_log_path = Path("results/experiment_outputs/20251109T223554Z/482d928cb2be4eefb2c948f5740f162c/run_log_482d928cb2be4eefb2c948f5740f162c.jsonl")
    df_all = prepare_snippets_for_repair()
    # df_syntax_error_balanced = prepare_snippets_for_repair()
    #df_syntax_error_balanced = filter_not_run_false_only_points('./dataset/cleaned_results_with_no_retry.csv')
    df_syntax_error_balanced = read_csv_file('./dataset/decompiled_syntax_errors.csv')
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
        file_name = norm_str(row.get("file"))
        file_dir = BASE_DIR / file_hash / file_name
        

        # highest_indented_file returns Optional[Tuple[Path, int]]
        # res = highest_indented_file(file_dir, pattern="indented_*.py")
        header = f"\n# --- Processing Content from file: {str(file_dir)} ({count_idx+1}/{len(df_syntax_error_balanced)}) --- #\n"
        footer = f"\n# --- End of Processing Content from file: {str(file_dir)} ({count_idx+1}/{len(df_syntax_error_balanced)}) --- #\n"

        print(header)
        path_to_err_file = str(file_dir)

        # initial_error_description = row.get("syntactic_error_description") 
        initial_error_description = row.get("error_description") 
        #error_message_processed =  row.get("precessed_error_message") 
        error_message_processed =  row.get("error_msg_clean") 
        error_word = row.get("syntactic_error_word")
        error_line_number = extract_line_number(initial_error_description)
        # print(f" line: {error_line_number}")
        copy_dir  = BASE_DIR / file_hash / f"copy_of_{file_name}"
        copy_file(path_to_err_file, copy_dir)
        whole_content = read_file(copy_dir)
        version = extract_bytecode_major_minor(whole_content)
        results = fetch_context_lines(Path(copy_dir), error_line_number)
        # print(content)
        # sys.exit(0)
        if results:
            content, start_ln, end_ln = results
        initial_content = content
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
            "bytecode_version": version,
            # "provider": LLM_MODELS[3]["provider"],
            # "model_name": LLM_MODELS[3]["name"],
            "compile_error_word_before": error_word,
            "compile_error_description_before": error_message_processed,
            "fences_stripped": _count_fences(content),
        }
            # LLM repair
  
        retry_attempt = False 
        previuos_response=None 
        previous_error=""
        # error_description = initial_error_description
        # Retry attempt logic set to false for local LLM for now
        while not is_compiled:
            processed = process_file_for_syntax_error_patching(initial_content, initial_error_description, False, previuos_response, previous_error, log_rec=log_rec, llm=LLM_MODELS[6])
            if processed is None:
                break

            AFFECTED_FILE_PATH = LOG_BASE / file_hash
            if not AFFECTED_FILE_PATH.exists():
                AFFECTED_FILE_PATH.mkdir(parents=True, exist_ok=True)
            
            final_code, llm_metrics, chunk_responses = processed  
            print(final_code)

            # compile with timing
            t0 = time.perf_counter()
            compilation_candidate = final_code
            if LLM_MODELS[6]["name"] in ['qwen-v2.5-coder-7b']:
                compilation_candidate = reattach_context_lines(Path(copy_dir), final_code, start_ln, end_ln)
            with open(copy_dir, "w", encoding="utf-8") as f:
                f.write(compilation_candidate)
                f.close()
            try:
                compilation_result = compile_new_pyc(
                    compilation_candidate,
                    str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.py"),
                    str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.pyc"),
                    version
                )
                compile_ms = int((time.perf_counter() - t0) * 1000)
                is_compiled = compilation_result["is_compiled"]
                # previous_error = compilation_result["error_description"]
                initial_error_description = compilation_result["error_description"]
                if not is_compiled:
                    print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying.... {Colors.ENDC}")
                    error_line_number = extract_line_number(initial_error_description)
                    results = fetch_context_lines(copy_dir, error_line_number)
                    # print(content)
                    # sys.exit(0)
                    if results:
                        initial_content, start_ln, end_ln = results
                        pass
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
                "diff_lines": _diff_lines(compilation_candidate, initial_content),
            })
            if is_compiled:
                log_rec.update({"path_out": str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.py")})
                os.unlink(copy_dir)
                _append_log(LOG_FILE, log_rec)
            elif not is_compiled:

                with open(str(AFFECTED_FILE_PATH / f"syntax_failed_repaired_{file_name[:-3]}_error.txt"), "w", encoding="utf-8") as f:
                    f.write(previous_error or "Unknown error")
                    f.close()
                error_word, error_message = get_error_word_message_from_content(str(AFFECTED_FILE_PATH / f"syntax_failed_repaired_{file_name[:-3]}_error.txt"))
                if max_retries >= 0:
                    previuos_response = compilation_candidate if len(chunk_responses) == 0 else chunk_responses
                    retry_attempt = True
                else:
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
        break # Remove this break to process all files

        
