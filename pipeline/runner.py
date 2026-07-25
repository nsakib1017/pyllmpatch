from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from pipeline.config import (
    BASE_DATASET_PATH,
    MAX_EXAMPLE_RUNTIME_SEC,
    RuntimeConfig,
    build_run_paths,
    now_iso,
)
from pipeline.dataset import filter_dataset_rows, prepare_snippets_for_repair, resolve_syntax_dataset_source_path
from pipeline.logging_utils import append_log, extract_line_number, failure_cleanup
from utils.delete_only_compilation import delete_only_best_of, delete_lines_until_compilable_with_oracle
from utils.file_helpers import (
    SyntaxSegment,
    align_indentation,
    copy_file,
    create_file_from_response,
    get_error_word_message_from_content,
    norm_str,
    read_file,
    reattach_block,
    segment_syntax_context,
)
from utils.generate_bytecode import CompileError, compile_version
from utils.gt_syntactic_context import build_gt_object_context
from utils.providers import Colors
from utils.syntactic_prepass import SyntaxErrorInfo, cause_aware_window, codeobject_window, reattach_window
from utils.version import PythonVersion

SYNTACTIC_DETERMINISTIC_PREPASS_ENV = "SYNTACTIC_DETERMINISTIC_PREPASS"
SYNTACTIC_CODEOBJECT_WINDOW_ENV = "SYNTACTIC_CODEOBJECT_WINDOW"
SYNTACTIC_GT_CONTEXT_ENV = "SYNTACTIC_GT_CONTEXT"
SYNTACTIC_BEST_OF_N_ENV = "SYNTACTIC_BEST_OF_N"
SYNTACTIC_REPETITION_PENALTY_ENV = "SYNTACTIC_REPETITION_PENALTY"
SYNTACTIC_MAX_TOKENS_ENV = "SYNTACTIC_MAX_TOKENS"


def compile_new_pyc(py_file_content, py_file_dir, out_file_base_dir, version=None):
    with open(py_file_dir, "w", encoding="utf-8") as f:
        f.write(py_file_content)
    try:
        compile_version(py_file_dir, out_file_base_dir, version)
        return {"is_compiled": True, "error_description": None}
    except CompileError as e:
        return {"is_compiled": False, "error_description": str(e)}


def deterministic_prepass_enabled() -> bool:
    """Task 6, bullet 1: env flag gate, default ON. Read fresh every call so
    tests (and, in principle, ops) can toggle it without re-importing."""
    return os.getenv(SYNTACTIC_DETERMINISTIC_PREPASS_ENV, "1") == "1"


def syntactic_codeobject_window_enabled() -> bool:
    """Gate for the code-object-isolation window provider, default ON. The
    measured LLM-solo compile rate was 75% vs 52% for the prior minimal/
    cause-aware window (+23pts, nearly doubling 3.15: 37%->74%), so this is
    the default; set SYNTACTIC_CODEOBJECT_WINDOW=0 to fall back to the prior
    window. Read fresh every call so tests can toggle it without re-importing."""
    return os.getenv(SYNTACTIC_CODEOBJECT_WINDOW_ENV, "1") != "0"


def syntactic_gt_context_enabled() -> bool:
    """Gate for the GT-object-context prompt lever, default OFF. When ON, and a
    ground-truth .pyc path is available for the current dataset row, the LLM is shown a
    short brief of the TRUE structure of the code object enclosing the error (read from
    the GT .pyc via utils.gt_syntactic_context.build_gt_object_context) -- telling it what
    a decompiler-garbled object should BE, rather than asking it to guess blind. Set
    SYNTACTIC_GT_CONTEXT=1 to enable; unset/"0"/anything else keeps today's behavior byte-
    identical. Read fresh every call so tests can toggle it without re-importing."""
    return os.getenv(SYNTACTIC_GT_CONTEXT_ENV, "0") == "1"


def syntactic_best_of_n() -> int:
    """Gate + magnitude for compile-gated best-of-N sampling on the syntactic-repair LLM
    path (SYNTACTIC_BEST_OF_N, default "1" == today's single-greedy-call behavior, BYTE-
    IDENTICAL: this env flag unset/=="1" never generates a sampled candidate, no matter
    whether the greedy candidate compiles). N>1: the greedy candidate (temperature 0, the
    exact same call production already makes) is tried first; only if it fails to compile
    are up to N-1 additional SAMPLED candidates (see sampling_generation_config) generated
    and compile-checked one at a time, accepting the first that compiles -- see
    pipeline.repair_engine.select_best_of_n_candidate for the selection logic and
    run_experiment's while-loop for the reattach/compile wiring. Non-integer or <1 values
    clamp to 1 (defensive: never silently disables the LLM call, never goes negative). Read
    fresh every call so tests can toggle it without re-importing."""
    raw = os.getenv(SYNTACTIC_BEST_OF_N_ENV, "1")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 1
    return n if n >= 1 else 1


def sampling_generation_config(base_generation_config: dict | None) -> dict:
    """Best-of-N's SAMPLED (non-greedy) generation_config: merges do_sample=True /
    temperature=0.7 / top_p=0.9 on top of the model's own generation_config (so e.g.
    max_new_tokens is preserved), per the user requirement of "temperature ~0.7, top_p
    ~0.9". Candidate 0 (greedy) never calls this -- it keeps using the model's own
    generation_config exactly as before."""
    merged = dict(base_generation_config or {})
    merged.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9})
    return merged


def syntactic_repetition_penalty() -> float:
    """OUTPUT_OVERFLOW anti-repetition-loop lever (SYNTACTIC_REPETITION_PENALTY, default
    "1.0" == OFF/no penalty, BYTE-IDENTICAL to behavior before this lever existed). The
    single biggest residual syntactic-repair failure mode is the local model blowing its
    output-token budget via a REPETITION LOOP (re-emitting the same lines verbatim) before it
    ever emits a valid completion -- a decode/harness fault, not a knowledge gap. A
    repetition_penalty > 1.0 discourages the decoder from re-selecting already-generated
    tokens, directly countering that failure shape. See syntactic_generation_config for how
    this value is merged into the generation_config passed to BOTH the greedy first candidate
    and any best-of-N sampled candidates. Garbage/unparsable env values fall back to 1.0
    (defensive: never silently changes decoding behavior on a bad env value). Read fresh every
    call so tests can toggle it without re-importing."""
    raw = os.getenv(SYNTACTIC_REPETITION_PENALTY_ENV, "1.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def syntactic_max_tokens_override() -> int | None:
    """OUTPUT_OVERFLOW budget lever (SYNTACTIC_MAX_TOKENS, default unset -> None == OFF,
    BYTE-IDENTICAL: the syntactic LLM call keeps using the selected model's own
    `token_for_completion` exactly as before). When set, overrides the max_tokens budget
    passed to the syntactic-repair LLM call (see
    pipeline.repair_engine.process_file_in_single_run). Unset, blank, or unparsable values all
    return None rather than raising or silently substituting some magic number. Read fresh
    every call so tests can toggle it without re-importing."""
    raw = os.getenv(SYNTACTIC_MAX_TOKENS_ENV, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def syntactic_generation_config(base_generation_config: dict | None) -> dict | None:
    """Merges the OUTPUT_OVERFLOW repetition-penalty lever (syntactic_repetition_penalty) on
    TOP of `base_generation_config`, COMPOSING with -- never replacing -- whatever sampling
    settings are already present. Call this on the model's own generation_config for the
    GREEDY candidate, or on sampling_generation_config(...)'s output for a SAMPLED best-of-N
    candidate; either way the caller's do_sample/temperature/top_p settings pass through
    untouched and only repetition_penalty is added (or overwritten if already present).

    Default (syntactic_repetition_penalty() == 1.0, i.e. SYNTACTIC_REPETITION_PENALTY unset):
    a total no-op -- returns `base_generation_config` itself, unchanged (same object, same
    None-ness), so default behavior is byte-identical to before this lever existed."""
    penalty = syntactic_repetition_penalty()
    if penalty == 1.0:
        return base_generation_config
    merged = dict(base_generation_config or {})
    merged["repetition_penalty"] = penalty
    return merged


def maybe_gt_context(gt_pyc_value: str | None, error_description, copy_dir) -> str:
    """Pure, best-effort wrapper around utils.gt_syntactic_context.build_gt_object_context,
    gated by syntactic_gt_context_enabled(). Returns "" whenever the flag is off, no GT
    .pyc path is available for this row, the current compile error doesn't carry a
    resolvable line number, or the full current-file source can't be read -- matching
    build_gt_object_context's own "never raises, empty string on any failure" contract, so
    callers never need to special-case any of this themselves."""
    if not syntactic_gt_context_enabled():
        return ""
    if not gt_pyc_value or not Path(gt_pyc_value).is_file():
        return ""

    error_line = extract_line_number(error_description)
    if error_line is None:
        return ""

    full_source = read_file(copy_dir)
    if not full_source:
        return ""

    return build_gt_object_context(full_source, error_line, gt_pyc_value)


def maybe_prepass(source: str, version: str, compile_version_fn):
    """Pure, testable wrapper around utils.syntactic_prepass.run_syntactic_prepass.

    Behind env flag SYNTACTIC_DETERMINISTIC_PREPASS (default "1"): runs the
    deterministic mechanical pre-pass over `source`, probing candidates with
    `compile_version_fn(candidate_source: str, version: str) -> None` (raises
    on failure -- the same contract `utils.syntactic_prepass.probe_syntax`
    expects of a compile_fn). Flag "0" => no-op passthrough: (source, False, []),
    matching the pre-Task-6 code path byte-for-byte.
    """
    if not deterministic_prepass_enabled():
        return source, False, []

    from utils.syntactic_prepass import run_syntactic_prepass

    def _cf(s: str) -> None:
        compile_version_fn(s, version)

    r = run_syntactic_prepass(source, compile_fn=_cf)
    return r.source, r.compiled, r.operations


def _line_roles_for_window(line_count: int) -> tuple[str, ...]:
    """Mirrors utils.file_helpers._syntax_segment_roles without importing a
    private symbol from another module."""
    if line_count <= 0:
        return tuple()
    if line_count == 1:
        return ("B/E",)
    roles = ["I"] * line_count
    roles[0] = "B"
    roles[-1] = "E"
    return tuple(roles)


def minimal_window_syntax_context(
    file_path,
    error_line,
    error_description: str | None = None,
    expansion_level: int = 0,
):
    """Task 6, bullet 3 (MINIMAL WINDOW for the LLM residual -- user
    requirement): a `segment_syntax_context`-compatible provider that hands
    the LLM the tightest self-contained snippet around the error (the
    failing logical line/statement plus any needed continuation lines),
    widening by one enclosing block per retry via `expansion_level`, instead
    of the larger default `segment_syntax_context` window. Cause-aware
    (`cause_aware_window`): when the true cause of the error sits above the
    reported symptom line (e.g. "unexpected indent" whose cause is the
    preceding line, or a def/class header whose decorators sit above it),
    the window is anchored on that cause line, not just the symptom, while
    staying minimal when there is no cause above. Falls back to
    `segment_syntax_context` when `cause_aware_window` cannot form a
    parseable (non-empty) unit -- including when `error_line` is unknown,
    which matches `segment_syntax_context`'s own None-error_line behavior
    rather than guessing a line.
    """
    if error_line is None:
        return segment_syntax_context(file_path, error_line, error_description, expansion_level)

    content = read_file(file_path)
    if not content:
        return segment_syntax_context(file_path, error_line, error_description, expansion_level)

    error_info = SyntaxErrorInfo(lineno=error_line, offset=None, msg=error_description or "")
    window = cause_aware_window(content, error_info, expansion=expansion_level)

    if not window.text.strip() or window.start_line > window.end_line:
        return segment_syntax_context(file_path, error_line, error_description, expansion_level)

    line_count = max(1, window.end_line - window.start_line + 1)
    return SyntaxSegment(
        text=window.text,
        start_line=window.start_line,
        end_line=window.end_line,
        base_indent=window.indent,
        anchor_indent=window.indent,
        segment_kind="statement" if line_count == 1 else "block",
        line_roles=_line_roles_for_window(line_count),
    )


def codeobject_window_syntax_context(
    file_path,
    error_line,
    error_description: str | None = None,
    expansion_level: int = 0,
):
    """Opt-in alternative to minimal_window_syntax_context (gated by
    SYNTACTIC_CODEOBJECT_WINDOW): hands the LLM the enclosing code object
    (nearest def/class that actually contains the error line) instead of
    the tighter cause-aware line window, isolating repair at object
    granularity. Mirrors minimal_window_syntax_context exactly -- same
    error_line-is-None / empty-window fallback to segment_syntax_context --
    but builds the window via codeobject_window, which itself degenerates
    to cause_aware_window at module scope or when the enclosing object is
    too large (see utils.syntactic_prepass.OBJECT_WINDOW_MAX_LINES).
    """
    if error_line is None:
        return segment_syntax_context(file_path, error_line, error_description, expansion_level)

    content = read_file(file_path)
    if not content:
        return segment_syntax_context(file_path, error_line, error_description, expansion_level)

    error_info = SyntaxErrorInfo(lineno=error_line, offset=None, msg=error_description or "")
    window = codeobject_window(content, error_info, expansion=expansion_level)

    if not window.text.strip() or window.start_line > window.end_line:
        return segment_syntax_context(file_path, error_line, error_description, expansion_level)

    line_count = max(1, window.end_line - window.start_line + 1)
    return SyntaxSegment(
        text=window.text,
        start_line=window.start_line,
        end_line=window.end_line,
        base_indent=window.indent,
        anchor_indent=window.indent,
        segment_kind="statement" if line_count == 1 else "block",
        line_roles=_line_roles_for_window(line_count),
    )


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


def run_experiment(config: RuntimeConfig, *, source: str | None = None, limit: int | None = None) -> None:
    from pipeline.repair_engine import attempt_repair, select_best_of_n_candidate

    previous_run_log = str(config.previous_run_log_path) if config.previous_run_log_path else ""
    df = prepare_snippets_for_repair(previous_run_log, only_previously_failed=False)
    if "error_type" in df.columns:
        df = df[df["error_type"].astype(str) == "syntactic_error"].copy()
    df = filter_dataset_rows(df, source=source, limit=limit)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("DataFrame shape:", df.shape)
    print_runtime_mode(config)

    run_id, log_base, log_file = build_run_paths(config.run_timestamp)
    log_file = log_base / f"run_log_{run_id}_{BASE_DATASET_PATH.name.split('.')[0]}.jsonl"

    count_idx = 0
    for idx, row in df.iterrows():
        copy_dir = None
        affected_file_path = None
        log_rec = None
        is_compiled = False

        file_hash = norm_str(row.get("file_hash"))
        file_path_value = norm_str(row.get("file_path"))
        file_name = norm_str(row.get("file"))
        source_name = norm_str(row.get("source")) or "syntactic"
        decompiler_name = norm_str(row.get("decompiler")) or source_name
        dataset_name = norm_str(row.get("dataset")) or str(row.get("error_type") or "syntactic_error")
        raw_bytecode_version = row.get("bytecode_version")
        bytecode_version = ""
        if raw_bytecode_version is not None:
            try:
                bytecode_version = PythonVersion(raw_bytecode_version).as_str()
            except Exception:
                bytecode_version = norm_str(raw_bytecode_version) or ""

        file_dir = Path(file_path_value) if file_path_value else None
        if file_dir is None:
            resolved_source_path = resolve_syntax_dataset_source_path(row)
            file_dir = Path(resolved_source_path) if resolved_source_path else None

        if file_dir is not None and not file_name:
            file_name = file_dir.name

        header = f"\n# --- Processing Content from file: {str(file_dir)} ({count_idx+1}/{len(df)}) --- #\n"
        footer = f"\n# --- End of Processing Content from file: {str(file_dir)} ({count_idx+1}/{len(df)}) --- #\n"
        print(header)

        try:
            if not file_dir or not file_dir.exists() or not file_hash or not file_name or not bytecode_version:
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
            initial_error_description = row.get("error_description") or row.get("error_message") or row.get("error")
            error_word = row.get("syntactic_error_word") or row.get("error")
            gt_pyc_value = norm_str(row.get("gt_pyc"))
            copy_dir = log_base / decompiler_name / dataset_name / bytecode_version / file_hash / f"copy_for_run_id_{run_id}_of_{file_name}"
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

                # --- Task 6, bullet 2: deterministic mechanical pre-pass, tried
                # BEFORE the LLM. If it makes the file compile, write it via the
                # same compile_new_pyc path used everywhere else in this loop and
                # break WITHOUT calling the LLM. Gated entirely behind
                # SYNTACTIC_DETERMINISTIC_PREPASS (default "1"); when the flag is
                # "0" this whole block is skipped and the loop below is untouched.
                if deterministic_prepass_enabled():
                    prepass_current_source = read_file(copy_dir)
                    if not prepass_current_source:
                        prepass_current_source = compilation_candidate or read_file(path_to_err_file) or ""

                    with tempfile.TemporaryDirectory(prefix="syntactic_prepass_probe_") as prepass_tmp_dir:
                        prepass_tmp_py = os.path.join(prepass_tmp_dir, "probe.py")
                        prepass_tmp_pyc = os.path.join(prepass_tmp_dir, "probe.pyc")

                        def _prepass_probe_compile_fn(candidate_source: str, candidate_version: str) -> None:
                            probe_result = compile_new_pyc(candidate_source, prepass_tmp_py, prepass_tmp_pyc, candidate_version)
                            if not probe_result["is_compiled"]:
                                raise CompileError(probe_result["error_description"] or "syntactic prepass probe: compile failed")

                        prepass_fixed_source, prepass_is_compiled, prepass_ops = maybe_prepass(
                            prepass_current_source, version, _prepass_probe_compile_fn
                        )

                    if prepass_is_compiled:
                        compilation_result = compile_new_pyc(prepass_fixed_source, out_py_path, out_pyc_path, version)
                        is_compiled = compilation_result["is_compiled"]
                        initial_error_description = compilation_result["error_description"]
                        attempt_number = total_attempts_completed + 1
                        total_attempts_completed = attempt_number

                        log_rec.update(
                            {
                                "fits_single_run": None,
                                "avg_chunk_tokens": None,
                                "max_chunk_tokens": None,
                                "llm_calls": 0,
                                "llm_latency_ms_total": 0,
                                "compiled_success": bool(is_compiled),
                                "total_attempts_completed": attempt_number,
                                "deterministic_prepass_used": True,
                                "deterministic_prepass_operations": list(prepass_ops),
                            }
                        )

                        if is_compiled:
                            compilation_candidate = prepass_fixed_source
                            log_rec.update({"path_out": out_py_path})
                            append_log(log_file, log_rec)
                            break
                        # Defensive: maybe_prepass and the explicit write-through both
                        # route through compile_new_pyc/compile_version, so they should
                        # never disagree -- but if they somehow do, fall through to the
                        # LLM path below rather than trusting a stale "compiled" signal.
                    elif prepass_ops:
                        log_rec.update(
                            {
                                "deterministic_prepass_used": True,
                                "deterministic_prepass_operations": list(prepass_ops),
                                "deterministic_prepass_deferred": True,
                            }
                        )

                half_retries = max_retries // 2
                is_late_retry = total_attempts_completed >= half_retries
                try_whole_file = config.enable_whole_file_repair and is_late_retry
                selected_llm = select_llm(config.use_local_llm, config.local_llm_idx)

                # Task 6, bullet 3 (MINIMAL WINDOW for the LLM residual): when the
                # prepass has deferred and the LLM is about to be called, localize
                # with the tightest snippet (minimal_window_syntax_context, backed
                # by cause_aware_window) instead of the larger default
                # segment_syntax_context window. Gated on the same flag so a "0"
                # setting leaves the pre-Task-6 window provider untouched.
                llm_use_minimal_window = deterministic_prepass_enabled()
                # Opt-in code-object-isolation window (SYNTACTIC_CODEOBJECT_WINDOW,
                # default OFF): when enabled, supersedes the minimal-window choice
                # above -- the LLM sees the enclosing def/class instead. Flag OFF
                # leaves the pre-existing minimal_window_syntax_context/
                # segment_syntax_context selection byte-identical.
                llm_use_codeobject_window = syntactic_codeobject_window_enabled()
                if llm_use_codeobject_window:
                    syntax_context_provider = codeobject_window_syntax_context
                else:
                    syntax_context_provider = minimal_window_syntax_context if llm_use_minimal_window else segment_syntax_context

                # GT-context lever (SYNTACTIC_GT_CONTEXT, default OFF): computed once per
                # attempt, from the FULL current file source and the full-file error line,
                # so it stays in sync with each retry's latest compile error/source state.
                # maybe_gt_context is a total no-op ("") when the flag is off, matching
                # today's behavior byte-for-byte.
                gt_context = maybe_gt_context(gt_pyc_value, initial_error_description, copy_dir)

                # OUTPUT_OVERFLOW levers (SYNTACTIC_REPETITION_PENALTY / SYNTACTIC_MAX_TOKENS):
                # read once per attempt, applied to BOTH the greedy call below and any sampled
                # best-of-N candidate (_generate_sampled, further down). Default (penalty 1.0,
                # no override) makes syntactic_generation_config(...) a no-op and
                # max_tokens_override_value None -- byte-identical to pre-lever behavior.
                max_tokens_override_value = syntactic_max_tokens_override()

                try:
                    repair_result = attempt_repair(
                        copy_dir=copy_dir,
                        error_description=initial_error_description,
                        llm=selected_llm,
                        log_rec=log_rec,
                        strategy_state={"syntax_context": {"failures": 0}, "whole_file": {"failures": 0}},
                        try_whole_file=try_whole_file,
                        expansion_level=total_attempts_completed,
                        affected_file_path=affected_file_path,
                        segment_syntax_context=syntax_context_provider,
                        enable_syntax_explanation=config.enable_syntax_explanation,
                        gt_context=gt_context,
                        generation_config_override=syntactic_generation_config(selected_llm.get("generation_config")),
                        max_tokens_override=max_tokens_override_value,
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
                    final_code, llm_metrics, with_pin_point, start_ln, end_ln, base_indent, anchor_indent = repair_result
                except Exception as e:
                    print(f"Error during repair attempt: {e}")
                    log_rec.update({"compiled_success": False, "repair_exception": str(e)})
                    append_log(log_file, log_rec)
                    break

                final_code = (final_code or "").strip()
                attempt_number = total_attempts_completed + 1

                # --- Best-of-N (SYNTACTIC_BEST_OF_N): compile-gate `final_code` (the
                # greedy candidate) and, only if it fails to compile, generate up to
                # best_of_n - 1 additional SAMPLED candidates (same window/attempt,
                # temperature ~0.7 / top_p ~0.9) and compile-check each, accepting the
                # first that compiles. best_of_n == 1 (the default) makes exactly the
                # SAME single compile_new_pyc call this loop always made -- no sampled
                # attempt_repair call is ever issued, byte-identical to before.
                pre_attempt_snapshot = read_file(copy_dir)
                best_of_n = syntactic_best_of_n()
                last_applied: dict = {}

                def _apply_candidate_and_compile(candidate_final_code: str) -> dict:
                    """Restore copy_dir to `pre_attempt_snapshot` (this attempt's
                    pre-reattach state) and splice `candidate_final_code` into it via
                    the same reattach strategy the single-candidate path always used,
                    then compile-check the result. Restoring first is what makes trying
                    several candidates against the SAME window/error safe -- without it,
                    a rejected candidate's splice would corrupt the line numbers/content
                    the next candidate's reattach depends on."""
                    if pre_attempt_snapshot is not None:
                        with open(copy_dir, "w", encoding="utf-8") as f:
                            f.write(pre_attempt_snapshot)

                    t_candidate = time.perf_counter()
                    candidate_code = (candidate_final_code or "").strip()
                    candidate_is_compiled = False
                    candidate_compilation_candidate = None
                    candidate_error_description = initial_error_description
                    candidate_compile_ms = None
                    candidate_compile_exception = None

                    try:
                        if candidate_code:
                            if with_pin_point:
                                # Task 6, bullet 3: splice back with reattach_window when
                                # the segment actually came from minimal_window_syntax_context
                                # (verified by deterministically recomputing the same window
                                # from the same inputs -- pure function, so an exact
                                # start/end match means it was the source of start_ln/end_ln;
                                # a mismatch means the provider fell back to
                                # segment_syntax_context, which uses the legacy
                                # align_indentation + reattach_block splice unchanged).
                                # Recomputed with cause_aware_window -- must be the SAME
                                # function minimal_window_syntax_context used to produce
                                # the segment, or the start/end match below is meaningless.
                                # (When llm_use_codeobject_window is set, the recompute
                                # uses codeobject_window instead, mirroring
                                # syntax_context_provider so producer and recompute stay
                                # consistent.)
                                reattached_via_minimal_window = False
                                if llm_use_minimal_window or llm_use_codeobject_window:
                                    recompute_error_line = extract_line_number(initial_error_description)
                                    recompute_content = read_file(copy_dir)
                                    if recompute_error_line is not None and recompute_content:
                                        recompute_error_info = SyntaxErrorInfo(
                                            lineno=recompute_error_line, offset=None, msg=initial_error_description or ""
                                        )
                                        if llm_use_codeobject_window:
                                            recompute_window = codeobject_window(
                                                recompute_content,
                                                recompute_error_info,
                                                expansion=total_attempts_completed,
                                            )
                                        else:
                                            recompute_window = cause_aware_window(
                                                recompute_content,
                                                recompute_error_info,
                                                expansion=total_attempts_completed,
                                            )
                                        if (
                                            recompute_window.text.strip()
                                            and recompute_window.start_line <= recompute_window.end_line
                                            and recompute_window.start_line == start_ln
                                            and recompute_window.end_line == end_ln
                                        ):
                                            new_full_source = reattach_window(recompute_content, recompute_window, candidate_code)
                                            with open(copy_dir, "w", encoding="utf-8") as f:
                                                f.write(new_full_source)
                                            reattached_via_minimal_window = True
                                if not reattached_via_minimal_window:
                                    aligned_code = align_indentation(candidate_code, base_indent, anchor_indent)
                                    reattach_block(copy_dir, start_ln, end_ln, aligned_code)
                            else:
                                create_file_from_response(copy_dir, candidate_code)
                            candidate_compilation_candidate = read_file(copy_dir)
                        else:
                            copied_content = read_file(copy_dir)
                            candidate_compilation_candidate = copied_content if copied_content else read_file(path_to_err_file)

                        compilation_result = compile_new_pyc(candidate_compilation_candidate, out_py_path, out_pyc_path, version)
                        candidate_compile_ms = int((time.perf_counter() - t_candidate) * 1000)
                        candidate_is_compiled = compilation_result["is_compiled"]
                        candidate_error_description = compilation_result["error_description"]
                    except Exception:
                        candidate_is_compiled = False
                        candidate_compile_exception = traceback.format_exc()

                    return {
                        "is_compiled": candidate_is_compiled,
                        "compilation_candidate": candidate_compilation_candidate,
                        "error_description": candidate_error_description,
                        "compile_ms": candidate_compile_ms,
                        "compile_exception": candidate_compile_exception,
                    }

                def _compiles(candidate_code: str) -> bool:
                    result = _apply_candidate_and_compile(candidate_code)
                    last_applied["candidate"] = candidate_code
                    last_applied["result"] = result
                    return bool(result["is_compiled"])

                def _generate_sampled() -> str:
                    # Restore BEFORE generating too: segment_syntax_context/codeobject_window
                    # etc. read copy_dir fresh to build the prompt, so a leftover mutation from
                    # the previous (rejected) candidate must not leak into the next prompt.
                    if pre_attempt_snapshot is not None:
                        with open(copy_dir, "w", encoding="utf-8") as f:
                            f.write(pre_attempt_snapshot)
                    try:
                        sampled_result = attempt_repair(
                            copy_dir=copy_dir,
                            error_description=initial_error_description,
                            llm=selected_llm,
                            log_rec=log_rec,
                            strategy_state={"syntax_context": {"failures": 0}, "whole_file": {"failures": 0}},
                            try_whole_file=try_whole_file,
                            expansion_level=total_attempts_completed,
                            affected_file_path=affected_file_path,
                            segment_syntax_context=syntax_context_provider,
                            enable_syntax_explanation=config.enable_syntax_explanation,
                            gt_context=gt_context,
                            generation_config_override=syntactic_generation_config(
                                sampling_generation_config(selected_llm.get("generation_config"))
                            ),
                            max_tokens_override=max_tokens_override_value,
                        )
                    except Exception:
                        sampled_result = None
                    if not sampled_result:
                        return ""
                    return (sampled_result[0] or "").strip()

                selected_candidate, candidate_compiled = select_best_of_n_candidate(
                    best_of_n=best_of_n,
                    generate_greedy=lambda: final_code,
                    generate_sampled=_generate_sampled,
                    compiles=_compiles,
                )

                if last_applied.get("candidate") != selected_candidate:
                    # Only reachable when best_of_n > 1 and every candidate failed: the
                    # last _compiles() call left copy_dir mutated with a losing SAMPLED
                    # candidate, not the greedy one select_best_of_n_candidate falls back
                    # to returning. Re-apply the greedy candidate so copy_dir and the
                    # result below end up in EXACTLY the state a single greedy attempt
                    # (today's behavior, and N=1) would have left.
                    _compiles(selected_candidate)

                apply_result = last_applied["result"]
                is_compiled = apply_result["is_compiled"]
                compilation_candidate = apply_result["compilation_candidate"]
                initial_error_description = apply_result["error_description"]
                compile_ms = apply_result["compile_ms"]
                compile_exception = apply_result["compile_exception"]

                if not is_compiled:
                    print(f"{Colors.WARNING}    -> Re-compilation failed for file. Retrying ({attempt_number}/{config.max_retries_default+1}).... {Colors.ENDC}")

                if compile_exception is None:
                    log_rec.update(
                        {
                            "fits_single_run": llm_metrics.get("fits_single_run"),
                            "avg_chunk_tokens": llm_metrics.get("avg_chunk_tokens"),
                            "max_chunk_tokens": llm_metrics.get("max_chunk_tokens"),
                            "llm_calls": llm_metrics.get("llm_calls"),
                            "llm_latency_ms_total": llm_metrics.get("llm_latency_ms_total"),
                            "compiled_success": bool(is_compiled),
                            "total_attempts_completed": attempt_number,
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

                total_attempts_completed = attempt_number
                max_retries -= 1
                if not is_compiled and compile_exception:
                    log_rec.update({"compile_exception": compile_exception})

                if max_retries < 0 or elapsed > MAX_EXAMPLE_RUNTIME_SEC:
                    print(f"{Colors.FAIL}    -> Max retries reached. Could not compile the file. {Colors.ENDC}")

                    if config.enable_delete_only_fallback:
                        print(f"{Colors.WARNING}    -> Engaging delete-only fallback (best-of original + last llm output)...  {Colors.ENDC}")
                        try:
                            llm_snapshot_path = affected_file_path / f"llm_last_output_{file_name[:-3]}.py"
                            with open(str(llm_snapshot_path), "w", encoding="utf-8") as f:
                                f.write(compilation_candidate or "")

                            delete_only_input_path = affected_file_path / f"delete_only_input_from_llm_{file_name[:-3]}.py"
                            shutil.copyfile(str(llm_snapshot_path), str(delete_only_input_path))

                            delete_only_out_py_path = str(affected_file_path / f"delete_only_repaired_from_llm_{file_name[:-3]}.py")
                            delete_only_out_pyc_path = str(affected_file_path / f"delete_only_repaired_from_llm_{file_name[:-3]}.pyc")
                            delete_only_err_txt_path = str(affected_file_path / f"delete_only_from_llm_{file_name[:-3]}_error.txt")

                            # Best-of-both: the LLM's last output is frequently
                            # corrupted beyond what line-deletion can recover
                            # (mask tokens, unbalanced constructs), while the
                            # pristine ORIGINAL decompiled source (never mutated
                            # by this pipeline) almost always deletes down to
                            # something that compiles. Run both, keep the better
                            # result -- see delete_only_best_of's docstring for
                            # the winner rule.
                            fixed_code, del_log, last_res, winning_label = delete_only_best_of(
                                [
                                    ("original", read_file(path_to_err_file) or ""),
                                    ("llm", compilation_candidate or ""),
                                ],
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
                                    "delete_only_source_used": winning_label,
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
