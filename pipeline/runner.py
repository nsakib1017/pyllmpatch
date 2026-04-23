from __future__ import annotations

import os
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from pipeline.config import (
    BASE_DIR_PYTHON_FILES_PYLINGUAL,
    BASE_DIR_PYTHON_FILES_PYPI,
    BASE_DATASET_PATH,
    MAX_EXAMPLE_RUNTIME_SEC,
    RuntimeConfig,
    build_run_paths,
    now_iso,
)
from pipeline.dataset import prepare_snippets_for_repair
from pipeline.logging_utils import append_log, extract_line_number, failure_cleanup
from utils.delete_only_compilation import delete_lines_until_compilable_with_oracle
from utils.file_helpers import align_indentation, copy_file, create_file_from_response, fetch_syntax_context, get_error_word_message_from_content, norm_str, read_file, reattach_block
from utils.generate_bytecode import CompileError, compile_version
from utils.providers import Colors


def compile_new_pyc(py_file_content, py_file_dir, out_file_base_dir, version=None):
    with open(py_file_dir, "w", encoding="utf-8") as f:
        f.write(py_file_content)
    try:
        compile_version(py_file_dir, out_file_base_dir, version)
        return {"is_compiled": True, "error_description": None}
    except CompileError as e:
        return {"is_compiled": False, "error_description": str(e)}


def get_file_path_base_dir(decompiler_name: str, dataset_name: str, file_hash: str) -> Path | None:
    if "pylingual" in decompiler_name.lower():
        if "pylingual" or "malware" in dataset_name.lower():
            return BASE_DIR_PYTHON_FILES_PYLINGUAL / file_hash / "decompiler_output"
        if "pypi" in dataset_name.lower():
            return BASE_DIR_PYTHON_FILES_PYPI / file_hash / "decompiled_output"
    elif "pycdc" in decompiler_name.lower():
        if "pylingual" or "malware" in dataset_name.lower():
            return BASE_DIR_PYTHON_FILES_PYLINGUAL / file_hash / "pycdc_decompilation_output"
        if "pypi" in dataset_name.lower():
            return BASE_DIR_PYTHON_FILES_PYPI / file_hash / "decompiled_output_pycdc"
    return None


def select_llm(use_local_llm: bool, local_llm_idx: int):
    from utils.providers import LLM_MODELS, OPEN_LLM_MODELS

    llm_map = OPEN_LLM_MODELS if use_local_llm else LLM_MODELS
    idx = local_llm_idx
    if not (0 <= idx < len(llm_map)):
        idx = 0
    return llm_map[idx]


def print_runtime_mode(config: RuntimeConfig) -> None:
    if config.delete_only_mode:
        print(f"{Colors.WARNING}DELETE_ONLY_MODE is enabled, LLMs will not be called{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}DELETE_ONLY_MODE is disabled, LLMs may be called{Colors.ENDC}")

    if config.delete_only_infinite_iters:
        print(f"{Colors.WARNING}DELETE_ONLY_INFINITE_ITERS is enabled, delete-only may run for a very long time{Colors.ENDC}")


def run_experiment(config: RuntimeConfig) -> None:
    from pipeline.repair_engine import attempt_repair

    previous_run_log = str(config.previous_run_log_path) if config.previous_run_log_path else ""
    df = prepare_snippets_for_repair(previous_run_log, only_previously_failed=False)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("DataFrame shape:", df.shape)
    print_runtime_mode(config)

    for outer_idx in range(config.config_idx_start, config.config_idx_range):
        run_id, log_base, log_file = build_run_paths(config.run_timestamp)
        log_file = log_base / f"run_log_{run_id}_with_config_{outer_idx}_{BASE_DATASET_PATH.name.split('.')[0]}.jsonl"

        count_idx = 0
        for idx, row in df.iterrows():
            copy_dir = None
            affected_file_path = None
            log_rec = None
            is_compiled = False

            file_hash = norm_str(row.get("file_hash"))
            file_name = norm_str(row.get("file"))
            file_dir = Path(row.get("file_path")) if row.get("file_path") else None
            decompiler_name = row.get("decompiler")
            dataset_name = row.get("dataset")
            bytecode_version = str(row.get("bytecode_version")) if row.get("bytecode_version") is not None else ""

            header = f"\n# --- Processing Content from file (with config {outer_idx}): {str(file_dir)} ({count_idx+1}/{len(df)}) --- #\n"
            footer = f"\n# --- End of Processing Content from file (with config {outer_idx}): {str(file_dir)} ({count_idx+1}/{len(df)}) --- #\n"
            print(header)

            try:
                if not file_dir or not file_dir.exists() or not decompiler_name or not dataset_name or not file_hash or not file_name or not bytecode_version:
                    continue

                try:
                    input_file_size_bytes = file_dir.stat().st_size
                except Exception:
                    input_file_size_bytes = None

                if input_file_size_bytes is not None and input_file_size_bytes > config.max_whole_file_bytes:
                    skip_log = {
                        "run_id": run_id,
                        "timestamp": now_iso(),
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
                        "max_whole_file_bytes": int(config.max_whole_file_bytes),
                    }
                    append_log(log_file, skip_log)
                    print(f"{Colors.WARNING}Skipping oversized file: {file_dir} ({input_file_size_bytes} bytes > {config.max_whole_file_bytes}){Colors.ENDC}")
                    continue

                path_to_err_file = str(file_dir)
                initial_error_description = row.get("error_description")
                error_word = row.get("syntactic_error_word")
                file_path_base_dir = get_file_path_base_dir(decompiler_name, dataset_name, file_hash)
                if file_path_base_dir is None:
                    continue

                copy_dir = file_path_base_dir / f"copy_for_run_id_{run_id}_of_{file_name}"
                copy_file(path_to_err_file, copy_dir)

                version = bytecode_version
                max_retries = config.max_retries_default
                total_attempts_completed = 0

                log_rec = {
                    "run_id": run_id,
                    "timestamp": now_iso(),
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
                    "max_whole_file_bytes": int(config.max_whole_file_bytes),
                    "compile_error_word_before": row.get("error"),
                    "compile_error_message_before": row.get("error_message"),
                    "delete_only_mode": bool(config.delete_only_mode),
                    "delete_only_infinite_iters": bool(config.delete_only_infinite_iters),
                    "delete_only_max_iters": int(config.delete_only_max_iters),
                    "delete_only_base_window": int(config.delete_only_base_window),
                    "delete_only_max_deleted_ratio": float(config.delete_only_max_deleted_ratio),
                }

                t_begin = time.monotonic()
                compilation_candidate = ""

                affected_file_path = log_base / decompiler_name / dataset_name / bytecode_version / file_hash
                affected_file_path.mkdir(parents=True, exist_ok=True)

                while not is_compiled:
                    out_py_path = str(affected_file_path / f"syntax_repaired_{file_name[:-3]}.py")
                    out_pyc_path = str(affected_file_path / f"syntax_repaired_{file_name[:-3]}.pyc")
                    err_txt_path = str(affected_file_path / f"syntax_failed_repaired_{file_name[:-3]}_error.txt")
                    elapsed = time.monotonic() - t_begin

                    if config.delete_only_mode:
                        compilation_candidate = read_file(copy_dir) or read_file(path_to_err_file) or ""
                        fixed_code, del_log, last_res = delete_lines_until_compilable_with_oracle(
                            py_file_content=compilation_candidate,
                            compile_check=compile_new_pyc,
                            extract_line_number=extract_line_number,
                            get_error_word_message_from_content=get_error_word_message_from_content,
                            out_py_path=out_py_path,
                            out_pyc_path=out_pyc_path,
                            version=version,
                            base_window=config.delete_only_base_window,
                            max_iters=config.delete_only_max_iters,
                            max_deleted_ratio=config.delete_only_max_deleted_ratio,
                            min_remaining_lines=1,
                            err_txt_path=err_txt_path,
                        )

                        is_compiled = bool(last_res.get("is_compiled", False))
                        initial_error_description = last_res.get("error_description")
                        log_rec.update(
                            {
                                "delete_only_fallback_used": False,
                                "delete_only_deletions": len(del_log),
                                "compiled_success": is_compiled,
                                "total_attempts_completed": 1,
                                "fits_single_run": None,
                                "avg_chunk_tokens": None,
                                "max_chunk_tokens": None,
                                "llm_calls": 0,
                                "llm_latency_ms_total": 0,
                            }
                        )

                        try:
                            with open(str(affected_file_path / f"delete_only_log_{file_name[:-3]}.jsonl"), "w", encoding="utf-8") as f:
                                for rec in del_log:
                                    f.write(str(asdict(rec)) + "\n")
                        except Exception:
                            pass

                        if is_compiled:
                            log_rec.update({"path_out": out_py_path})
                            append_log(log_file, log_rec)
                        else:
                            with open(err_txt_path, "w", encoding="utf-8") as f:
                                f.write(initial_error_description or "Unknown error")
                            error_word, error_message = get_error_word_message_from_content(err_txt_path)
                            failure_cleanup(affected_file_path, compilation_candidate, file_name[:-3], error_word, error_message, log_rec)
                            append_log(log_file, log_rec)
                        break

                    half_retries = max_retries // 2
                    is_late_retry = total_attempts_completed >= half_retries
                    is_final_outer = outer_idx == 2
                    use_whole_file_for_remote = not config.use_local_llm
                    try_whole_file = (is_late_retry and is_final_outer) or use_whole_file_for_remote
                    selected_llm = select_llm(config.use_local_llm, config.local_llm_idx)

                    try:
                        repair_result = attempt_repair(
                            copy_dir=copy_dir,
                            error_description=initial_error_description,
                            llm=selected_llm,
                            log_rec=log_rec,
                            strategy_state={"syntax_context": {"failures": 0}, "whole_file": {"failures": 0}},
                            try_whole_file=try_whole_file,
                            outer_idx=outer_idx,
                            affected_file_path=affected_file_path,
                            fetch_syntax_context=fetch_syntax_context,
                        )
                        if repair_result is None:
                            log_rec.update(
                                {
                                    "compiled_success": False,
                                    "total_attempts_completed": total_attempts_completed,
                                    "repair_not_attempted_or_failed_precheck": True,
                                }
                            )
                            append_log(log_file, log_rec)
                            break
                        final_code, llm_metrics, with_pin_point, start_ln, end_ln, base_indent = repair_result
                    except Exception as e:
                        print(f"Error during repair attempt: {e}")
                        log_rec.update({"compiled_success": False, "repair_exception": str(e)})
                        append_log(log_file, log_rec)
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
                            compilation_candidate = copied_content if copied_content else read_file(path_to_err_file)

                        compilation_result = compile_new_pyc(compilation_candidate, out_py_path, out_pyc_path, version)
                        compile_ms = int((time.perf_counter() - t0) * 1000)
                        is_compiled = compilation_result["is_compiled"]
                        initial_error_description = compilation_result["error_description"]

                        if not is_compiled:
                            print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying ({total_attempts_completed+1}/{config.max_retries_default+1}).... {Colors.ENDC}")

                        total_attempts_completed += 1
                        max_retries -= 1

                        log_rec.update(
                            {
                                "fits_single_run": llm_metrics.get("fits_single_run"),
                                "avg_chunk_tokens": llm_metrics.get("avg_chunk_tokens"),
                                "max_chunk_tokens": llm_metrics.get("max_chunk_tokens"),
                                "llm_calls": llm_metrics.get("llm_calls"),
                                "llm_latency_ms_total": llm_metrics.get("llm_latency_ms_total"),
                                "compiled_success": bool(is_compiled),
                                "total_attempts_completed": total_attempts_completed,
                                "compile_latency_ms": compile_ms,
                            }
                        )

                        if is_compiled:
                            log_rec.update({"path_out": out_py_path})
                            append_log(log_file, log_rec)
                        else:
                            with open(err_txt_path, "w", encoding="utf-8") as f:
                                f.write(initial_error_description or "Unknown error")
                            error_word, error_message = get_error_word_message_from_content(err_txt_path)
                    except Exception:
                        is_compiled = False
                        print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying ({total_attempts_completed+1}/{config.max_retries_default+1}).... {Colors.ENDC}")

                    if max_retries < 0 or elapsed > MAX_EXAMPLE_RUNTIME_SEC:
                        print(f"{Colors.FAIL}    -> Max retries reached. Could not compile the file. {Colors.ENDC}")

                        if config.enable_delete_only_fallback:
                            print(f"{Colors.WARNING}    -> Engaging delete-only fallback for the last llm output...  {Colors.ENDC}")
                            try:
                                llm_snapshot_path = affected_file_path / f"llm_last_output_{file_name[:-3]}.py"
                                with open(str(llm_snapshot_path), "w", encoding="utf-8") as f:
                                    f.write(compilation_candidate or "")

                                delete_only_input_path = affected_file_path / f"delete_only_input_from_llm_{file_name[:-3]}.py"
                                shutil.copyfile(str(llm_snapshot_path), str(delete_only_input_path))

                                delete_only_out_py_path = str(affected_file_path / f"delete_only_repaired_from_llm_{file_name[:-3]}.py")
                                delete_only_out_pyc_path = str(affected_file_path / f"delete_only_repaired_from_llm_{file_name[:-3]}.pyc")
                                delete_only_err_txt_path = str(affected_file_path / f"delete_only_from_llm_{file_name[:-3]}_error.txt")

                                fixed_code, del_log, last_res = delete_lines_until_compilable_with_oracle(
                                    py_file_content=read_file(str(delete_only_input_path)) or "",
                                    compile_check=compile_new_pyc,
                                    extract_line_number=extract_line_number,
                                    get_error_word_message_from_content=get_error_word_message_from_content,
                                    out_py_path=delete_only_out_py_path,
                                    out_pyc_path=delete_only_out_pyc_path,
                                    version=version,
                                    base_window=config.delete_only_base_window,
                                    max_iters=config.delete_only_max_iters,
                                    max_deleted_ratio=config.delete_only_max_deleted_ratio,
                                    min_remaining_lines=1,
                                    err_txt_path=delete_only_err_txt_path,
                                )

                                log_rec.update(
                                    {
                                        "delete_only_fallback_used": True,
                                        "delete_only_deletions": len(del_log),
                                        "llm_last_output_snapshot_path": str(llm_snapshot_path),
                                        "delete_only_input_path": str(delete_only_input_path),
                                        "delete_only_output_path": str(delete_only_out_py_path),
                                    }
                                )

                                if last_res.get("is_compiled"):
                                    is_compiled = True
                                    log_rec.update({"compiled_success": True, "path_out": delete_only_out_py_path})
                                    try:
                                        with open(str(affected_file_path / f"delete_only_log_from_llm_{file_name[:-3]}.jsonl"), "w", encoding="utf-8") as f:
                                            for rec in del_log:
                                                f.write(str(asdict(rec)) + "\n")
                                    except Exception:
                                        pass
                                else:
                                    log_rec.update({"delete_only_failed_error": last_res.get("error_description")})
                            except Exception as e:
                                log_rec.update({"delete_only_fallback_used": True, "delete_only_fallback_exception": str(e)})
                        else:
                            log_rec.update({"delete_only_fallback_used": False})

                        try:
                            failure_cleanup(affected_file_path, compilation_candidate, file_name[:-3], error_word, error_message, log_rec)
                            if elapsed > MAX_EXAMPLE_RUNTIME_SEC:
                                log_rec.update({"compiled_success": None})
                            append_log(log_file, log_rec)
                        finally:
                            break

            except KeyboardInterrupt:
                raise
            except Exception as e:
                exc_tb = traceback.format_exc()
                print(f"{Colors.FAIL}Unexpected exception for sample {file_hash or '<unknown>'}: {e}{Colors.ENDC}")

                crash_log = {
                    "run_id": run_id,
                    "timestamp": now_iso(),
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

                append_log(log_file, crash_log)

                try:
                    if affected_file_path is not None:
                        with open(affected_file_path / f"unexpected_exception_{file_name or 'unknown'}.txt", "w", encoding="utf-8") as f:
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
