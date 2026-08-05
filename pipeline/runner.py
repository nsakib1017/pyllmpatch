from __future__ import annotations

import os
import sys
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
from pipeline.logging_utils import append_log, choose_initial_error, extract_line_number, failure_cleanup, window_exceeds_token_budget
from utils.delete_only_compilation import delete_only_best_of, delete_lines_until_compilable_with_oracle
from utils.file_helpers import (
    SyntaxSegment,
    align_indentation,
    clamp_syntax_segment,
    copy_file,
    create_file_from_response,
    get_error_word_message_from_content,
    norm_str,
    read_file,
    reattach_block,
    segment_syntax_context,
)
from utils.generate_bytecode import CompileError, compile_version
from utils.gt_syntactic_context import build_gt_object_context, harvest_constant_sequences
from utils.providers import Colors
from utils.syntactic_prepass import (SyntaxErrorInfo, cause_aware_window, codeobject_window,
                                     elide_long_string_literals, is_truncated_literal_line,
                                     reattach_window, restore_and_verify,
                                     splice_truncated_literals)
from utils.exhaustive_repair import SearchBudget, exhaustive_repair
from utils.structural_repair import STRUCT_OPS
from utils.syntactic_sweeps import CONTROL_FLOW_REWRITES, PURE_OPS, SYNTH_OPS, parse_error, run_stack
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
    return os.getenv(SYNTACTIC_DETERMINISTIC_PREPASS_ENV, "1") == "1"


def syntactic_codeobject_window_enabled() -> bool:
    return os.getenv(SYNTACTIC_CODEOBJECT_WINDOW_ENV, "1") != "0"


def syntactic_gt_context_enabled() -> bool:
    return os.getenv(SYNTACTIC_GT_CONTEXT_ENV, "0") == "1"


def syntactic_best_of_n() -> int:
    raw = os.getenv(SYNTACTIC_BEST_OF_N_ENV, "1")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 1
    return n if n >= 1 else 1


def sampling_generation_config(base_generation_config: dict | None) -> dict:
    merged = dict(base_generation_config or {})
    merged.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9})
    return merged


def syntactic_repetition_penalty() -> float:
    raw = os.getenv(SYNTACTIC_REPETITION_PENALTY_ENV, "1.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def syntactic_max_tokens_override() -> int | None:
    raw = os.getenv(SYNTACTIC_MAX_TOKENS_ENV, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _clamped_context_provider(provider, max_tokens, count_tokens):
    if not max_tokens or max_tokens <= 0:
        return provider

    def wrapped(path, error_line, error_description, expansion_level):
        segment = provider(path, error_line, error_description, expansion_level)
        return clamp_syntax_segment(segment, error_line, max_tokens, count_tokens)

    return wrapped


def syntactic_max_window_tokens() -> int:
    raw = os.getenv("SYNTACTIC_MAX_WINDOW_TOKENS", "2048").strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 2048


_WINDOW_TOKENIZER = None
_WINDOW_TOKENIZER_FAILED = False


def count_window_tokens(text: str) -> int:
    global _WINDOW_TOKENIZER, _WINDOW_TOKENIZER_FAILED
    if not text:
        return 0
    n = len(text)
    if n > 40000:
        return 10 ** 9  # giant blob -- definitely over budget, don't tokenize it
    if _WINDOW_TOKENIZER_FAILED:
        return int(n / 3.3)
    if _WINDOW_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            _WINDOW_TOKENIZER = AutoTokenizer.from_pretrained(
                os.getenv("SYNTACTIC_WINDOW_TOKENIZER", "unsloth/qwen3-coder-30b-a3b-instruct"),
                trust_remote_code=True,
            )
        except Exception:
            _WINDOW_TOKENIZER_FAILED = True
            return int(n / 3.3)
    try:
        return len(_WINDOW_TOKENIZER(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return int(n / 3.3)


def syntactic_generation_config(base_generation_config: dict | None) -> dict | None:
    penalty = syntactic_repetition_penalty()
    if penalty == 1.0:
        return base_generation_config
    merged = dict(base_generation_config or {})
    merged["repetition_penalty"] = penalty
    return merged


def maybe_gt_context(gt_pyc_value: str | None, error_description, copy_dir) -> str:
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


GT_LITERAL_SPLICE_ENV = "SYNTACTIC_GT_LITERAL_SPLICE"
SWEEP_CASCADE_ENV = "SYNTACTIC_SWEEP_CASCADE"
PAYLOAD_ELISION_ENV = "SYNTACTIC_PAYLOAD_ELISION"

# Literals below this are left alone: they cost little window and eliding them would churn the
# source for no benefit. The un-windowable files carry payloads orders of magnitude larger.
_ELISION_MIN_LITERAL_CHARS = 2000


def maybe_elide_payloads(source):
    if os.getenv(PAYLOAD_ELISION_ENV, "1") != "1":
        return source, {}
    if not source:
        return source, {}
    try:
        return elide_long_string_literals(source, max_literal_chars=_ELISION_MIN_LITERAL_CHARS)
    except Exception:
        return source, {}

_MAX_SPLICE_PASSES = 8


def control_flow_rewrites_in(operations):
    if not operations:
        return []
    return [op for op in operations if op in CONTROL_FLOW_REWRITES]


def _splice_to_fixpoint(source, gt_sequences):
    if not gt_sequences:
        return source, []
    current, operations = source, []
    for _ in range(_MAX_SPLICE_PASSES):
        spliced = splice_truncated_literals(current, parse_error(current), gt_sequences)
        if not spliced or spliced == current:
            break
        current = spliced
        operations.append("splice_truncated_literals")
    return current, operations


POST_LLM_EXHAUSTIVE_ENV = "SYNTACTIC_POST_LLM_EXHAUSTIVE"

# Every deterministic rewrite we have, as uniform (name, fn) pairs, so the search covers all orders.
_ALL_DETERMINISTIC_OPS = None


def _all_deterministic_ops():
    global _ALL_DETERMINISTIC_OPS
    if _ALL_DETERMINISTIC_OPS is None:
        from utils.syntactic_prepass import _OPERATORS as _PREPASS_OPS

        def _wrap(fn):
            return lambda source, err: fn(source, parse_error(source))

        _ALL_DETERMINISTIC_OPS = (
            [("sweep:" + o.name, _wrap(o.fn)) for o in PURE_OPS + SYNTH_OPS]
            + [("struct:" + o.name, o.fn) for o in STRUCT_OPS]
            + [("prepass:" + f.__name__, _wrap(f)) for f in _PREPASS_OPS]
        )
    return _ALL_DETERMINISTIC_OPS


def maybe_post_llm_repair(source, version, compile_version_fn):
    if os.getenv(POST_LLM_EXHAUSTIVE_ENV, "1") != "1":
        return source, False, []
    if not source:
        return source, False, []
    # Validate the PROBE before searching. `compile_version` takes (py_path, out_path, version) --
    # file paths -- and wiring it here as a (source, version) probe raised TypeError on every call,
    # which the blanket except below swallowed into a silent permanent "no fix". A probe that cannot
    # accept known-good source is a WIRING bug, not a repair failure, and must be visible.
    try:
        compile_version_fn("x = 1\n", version)
    except TypeError as exc:
        print(f"[post-llm] BROKEN PROBE, layer disabled: {exc}", file=sys.stderr, flush=True)
        return source, False, []
    except Exception:
        pass  # a probe that rejects valid source for version reasons is still usable
    try:
        repaired, path = exhaustive_repair(
            source, compile_version_fn, version, _all_deterministic_ops(),
            budget=SearchBudget(max_depth=4, max_states=250),
        )
        if repaired is None:
            return source, False, []
        return repaired, True, list(path)
    except Exception:
        return source, False, []


_MAX_STRUCT_ROUNDS = 8


def _structural_fixpoint(source, rounds=_MAX_STRUCT_ROUNDS):
    current, fired = source, []
    for _ in range(rounds):
        changed = False
        for op in STRUCT_OPS:
            candidate = op.fn(current, None)
            if candidate and candidate != current:
                current = candidate
                fired.append(op.name)
                changed = True
                if parse_error(current) is None:
                    return current, fired
        if not changed:
            break
    return current, fired


def maybe_sweep_cascade(source, version, compile_version_fn, gt_sequences=None):
    if os.getenv(SWEEP_CASCADE_ENV, "1") != "1":
        return source, False, [], 0
    if not source:
        return source, False, [], 0

    try:
        if parse_error(source) is None:
            return source, False, [], 0  # already parses -- not this stack's business

        spliced, operations = _splice_to_fixpoint(source, gt_sequences)

        def _adopt(candidate, fired, stage):
            if parse_error(candidate) is not None:
                return None
            try:
                compile_version_fn(candidate, version)
            except Exception:
                return None  # host accepted it, the target did not
            return candidate, True, operations + list(fired), stage

        after_pure, pure_fired = run_stack(spliced, PURE_OPS)
        adopted = _adopt(after_pure, pure_fired, 1)
        if adopted:
            return adopted

        after_synth, synth_fired = run_stack(after_pure, SYNTH_OPS)
        adopted = _adopt(after_synth, list(pure_fired) + list(synth_fired), 2)
        if adopted:
            return adopted

        # STAGE 3 -- PyLingual control-flow RECONSTRUCTION defects (utils.structural_repair):
        # a compound header whose body was never emitted, a for-loop flattened into comprehension
        # shape, a doubled for-clause. Measured on the 410-file residual: rescues 10 (2.4%), all
        # with ZERO lines deleted.
        # It is adopt-only-if-it-parses for a hard reason: the same measurement showed the pass
        # makes files WORSE on average (median defects 61 -> 74; 76 of 150 files worsened, 4
        # improved, and NOT ONE crossed into LLM-winnable range). `renest_flattened_for` fires on
        # 354 of 410 files but is right on ~10, and each wrong split manufactures a new broken line.
        # So this must never be forwarded as a partial "LLM aid" -- only a complete fix is taken.
        after_struct, struct_fired = _structural_fixpoint(after_synth)
        if struct_fired:
            adopted = _adopt(
                after_struct, list(pure_fired) + list(synth_fired) + struct_fired, 3
            )
            if adopted:
                return adopted

        return source, False, [], 0
    except Exception:
        return source, False, [], 0


def maybe_gt_sequences(gt_pyc_value, source):
    if os.getenv(GT_LITERAL_SPLICE_ENV, "1") != "1":
        return []
    if not gt_pyc_value or not source:
        return []
    try:
        if not any(is_truncated_literal_line(line) for line in str(source).splitlines()):
            return []
        return harvest_constant_sequences(gt_pyc_value) or []
    except Exception:
        return []


def elide_to_fit_window(source, error_line, max_tokens, count_tokens):
    if os.getenv(PAYLOAD_ELISION_ENV, "1") != "1":
        return source, {}
    if not source or not max_tokens or max_tokens <= 0 or not error_line:
        return source, {}
    try:
        lines = source.splitlines(keepends=True)
        index = int(error_line) - 1
        if not (0 <= index < len(lines)):
            return source, {}
        if count_tokens(lines[index]) <= max_tokens:
            return source, {}  # windowable already -- not our business

        elided, mapping = maybe_elide_payloads(source)
        if not mapping:
            return source, {}

        elided_lines = elided.splitlines(keepends=True)
        if not (0 <= index < len(elided_lines)):
            return source, {}
        if count_tokens(elided_lines[index]) > max_tokens:
            # Still oversized: the bulk is not a string payload (a giant numeric tuple, a
            # thousand-argument call). Churning the source buys nothing -- leave it out of scope.
            return source, {}
        return elided, mapping
    except Exception:
        return source, {}


def finalize_and_compile(source, out_py_path, out_pyc_path, version, elision_mapping=None):
    if elision_mapping:
        restored, ok = restore_and_verify(source, elision_mapping)
        if not ok:
            return {
                "is_compiled": False,
                "error_description": "payload restoration failed: placeholder lost during repair",
                "payload_restoration_failed": True,
            }
        source = restored
    return compile_new_pyc(source, out_py_path, out_pyc_path, version)


def maybe_prepass(source: str, version: str, compile_version_fn, gt_sequences=None):
    if not deterministic_prepass_enabled():
        return source, False, []

    from utils.syntactic_prepass import run_syntactic_prepass

    def _cf(s: str) -> None:
        compile_version_fn(s, version)

    r = run_syntactic_prepass(source, compile_fn=_cf, gt_sequences=gt_sequences)
    return r.source, r.compiled, r.operations


def _line_roles_for_window(line_count: int) -> tuple[str, ...]:
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
    def _process_row(idx, row) -> None:
        nonlocal count_idx
        # rebound inside _run_delete_only_mode; declared so nonlocal can bind
        del_log = None
        fixed_code = None
        last_res = None
        rec = None

        def _run_delete_only_mode() -> bool:
            """Delete-only mode: skip the LLM entirely and delete until it compiles. True means: stop retrying."""
            nonlocal compilation_candidate, del_log, error_message, error_word, f, fixed_code
            nonlocal initial_error_description, is_compiled, last_res, rec
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
            return True
            return False

        def _finalise_with_fallback() -> bool:
            """Retry budget or time is exhausted: take the delete-only / forced-compile fallback. True means: stop retrying."""
            nonlocal del_log, f, fixed_code, is_compiled, last_res, rec
            print(f"{Colors.FAIL}    -> Max retries reached. Could not compile the file. {Colors.ENDC}")

            if config.enable_forced_compile_fallback:
                print(f"{Colors.WARNING}    -> Engaging forced-compile fallback (neutralise unparseable code, NO deletion)...  {Colors.ENDC}")
                try:
                    from utils.forced_compile import force_compile, _whole_file_literal

                    fc_out_py_path = str(affected_file_path / f"forced_compile_{file_name[:-3]}.py")
                    fc_out_pyc_path = str(affected_file_path / f"forced_compile_{file_name[:-3]}.pyc")
                    original_src = read_file(path_to_err_file) or ""
                    fc_text, fc_led = force_compile(original_src, None, version)

                    # Restore any elided payloads before the authoritative compile.
                    if elision_mapping:
                        _fc_restored, _fc_ok = restore_and_verify(fc_text, elision_mapping)
                        if _fc_ok:
                            fc_text = _fc_restored

                    # compile_new_pyc(content, out_py_path, out_pyc_path, version) writes the
                    # .py from the content string itself -- pass fc_text, not the path.
                    fc_res = compile_new_pyc(fc_text, fc_out_py_path, fc_out_pyc_path, version)

                    if not fc_res.get("is_compiled"):
                        # force_compile's internal guarantee is host-version; if the TARGET
                        # version still rejects it, the whole-file literal compiles at every
                        # version and preserves every byte. Guarantees this fallback compiles.
                        fc_text = _whole_file_literal(original_src)
                        fc_res = compile_new_pyc(fc_text, fc_out_py_path, fc_out_pyc_path, version)
                        fc_led["fallback"] = "whole-file"
                        fc_led["lines_in_neutralised_chunks"] = original_src.count("\n")

                    log_rec.update({
                        "forced_compile_used": True,
                        "forced_compile_neutralised_lines": int(fc_led.get("lines_in_neutralised_chunks", 0)),
                        "forced_compile_fallback_kind": fc_led.get("fallback"),
                        "forced_compile_output_path": fc_out_py_path,
                        "delete_only_fallback_used": False,
                    })
                    if fc_res.get("is_compiled"):
                        is_compiled = True
                        log_rec.update({"compiled_success": True, "path_out": fc_out_py_path})
                    else:
                        log_rec.update({"forced_compile_failed": True})
                except Exception as e:
                    log_rec.update({"forced_compile_used": True, "forced_compile_exception": str(e)})
            elif config.enable_delete_only_fallback:
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
                        # The fallback writes its own output path, bypassing
                        # finalize_and_compile, so an elided file rescued here would ship
                        # with placeholders where its payload belongs. Restore in place;
                        # if a placeholder was lost, this is not a repair.
                        if elision_mapping:
                            _do_src = read_file(delete_only_out_py_path) or ""
                            _do_restored, _do_ok = restore_and_verify(_do_src, elision_mapping)
                            if _do_ok:
                                with open(delete_only_out_py_path, "w", encoding="utf-8") as f:
                                    f.write(_do_restored)
                            else:
                                last_res = {
                                    "is_compiled": False,
                                    "error_description": "payload restoration failed after delete-only fallback",
                                }
                                log_rec.update({"payload_restoration_failed": True})

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
                # The failure artifact is a diagnostic the analyst reads; write the real
                # source into it, not placeholders. Restoration failure is irrelevant here
                # -- the file already failed -- so keep whatever we have.
                _failed_source = compilation_candidate
                if elision_mapping:
                    _failed_source = restore_and_verify(compilation_candidate, elision_mapping)[0]
                failure_cleanup(affected_file_path, _failed_source, file_name[:-3], error_word, error_message, log_rec)
                if elapsed > MAX_EXAMPLE_RUNTIME_SEC:
                    log_rec.update({"compiled_success": None})
                append_log(log_file, log_rec)
            finally:
                return True
            return False

        # rebound inside _try_deterministic_prepass; declared so nonlocal can bind
        compilation_result = None

        def _try_deterministic_prepass() -> bool:
            """Run the mechanical pre-pass before the LLM. True means: stop retrying."""
            nonlocal attempt_number, compilation_candidate, compilation_result, initial_error_description, is_compiled, total_attempts_completed
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

                # GT co_consts splice: only harvested when this file actually carries a
                # truncated constant sequence, so the .pyc load is paid for by the rows
                # that can use it (see maybe_gt_sequences).
                prepass_gt_sequences = maybe_gt_sequences(gt_pyc_value, prepass_current_source)

                # Two-stage sweep cascade FIRST: it converts 30.2% of non-parsing files
                # against the shipped error-driven prepass's 0.9%, because it sweeps the
                # whole file to a fixpoint instead of chasing the single error the parser
                # currently reports. Only when it cannot finish the file does the shipped
                # prepass get its turn -- it still owns two operators the cascade excludes.
                cascade_source, cascade_compiled, cascade_ops, cascade_stage = maybe_sweep_cascade(
                    prepass_current_source, version, _prepass_probe_compile_fn,
                    gt_sequences=prepass_gt_sequences,
                )

                if cascade_compiled:
                    prepass_fixed_source, prepass_is_compiled, prepass_ops = (
                        cascade_source, True, cascade_ops
                    )
                else:
                    prepass_fixed_source, prepass_is_compiled, prepass_ops = maybe_prepass(
                        prepass_current_source, version, _prepass_probe_compile_fn,
                        gt_sequences=prepass_gt_sequences,
                    )

            if prepass_is_compiled:
                compilation_result = finalize_and_compile(prepass_fixed_source, out_py_path, out_pyc_path, version, elision_mapping)
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
                        # Sub-reporting: which stage carried the file, and whether the fix
                        # rewrote control flow. Stage 2 SYNTHESIZES a handler (genuine --
                        # nothing is deleted -- but it is scaffolding, not recovery), and
                        # `clause_to_if_true` preserves every line while changing what the
                        # code does. Both must stay separable from a plain stage-1 repair.
                        "deterministic_cascade_stage": cascade_stage,
                        "deterministic_control_flow_rewrites": control_flow_rewrites_in(prepass_ops),
                    }
                )

                if is_compiled:
                    compilation_candidate = prepass_fixed_source
                    log_rec.update({"path_out": out_py_path})
                    append_log(log_file, log_rec)
                    return True
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
            return False

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
                return

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
                return

            path_to_err_file = str(file_dir)
            initial_error_description = (
                row.get("error_description")
                or row.get("error_message")
                or row.get("error")
                or row.get("syntactic_error_message")
            )
            error_word = row.get("syntactic_error_word") or row.get("error")
            gt_pyc_value = norm_str(row.get("gt_pyc"))
            copy_dir = log_base / decompiler_name / dataset_name / bytecode_version / file_hash / f"copy_for_run_id_{run_id}_of_{file_name}"
            copy_file(path_to_err_file, copy_dir)

            # The row's error may lack a resolvable line number (e.g. the malware manifest
            # stores only "unexpected indent"); the repair loop localizes via
            # extract_line_number and can't proceed without one. Derive a line-numbered error
            # from the file itself -- the source of truth for its current syntax error -- via a
            # single compile probe, only when the row error has no line number.
            if extract_line_number(initial_error_description) is None:
                _probe_source = read_file(copy_dir) or ""
                if _probe_source:
                    with tempfile.TemporaryDirectory(prefix="init_err_probe_") as _probe_dir:
                        _probe_result = compile_new_pyc(
                            _probe_source,
                            os.path.join(_probe_dir, "probe.py"),
                            os.path.join(_probe_dir, "probe.pyc"),
                            bytecode_version,
                        )
                    initial_error_description = choose_initial_error(initial_error_description, _probe_result)

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

            # Window-budget discard: if the repair window AT THE ERROR already exceeds the LLM
            # generation budget (2048 tokens), the model can't regenerate it in one shot --
            # discard the file, do not attempt it, and EXCLUDE it from the analyzability
            # denominator (compiled_success=None). Gated by SYNTACTIC_MAX_WINDOW_TOKENS (0=off).
            _win_cap = syntactic_max_window_tokens()
            _win_err_line = extract_line_number(initial_error_description)
            # Payloads hidden from the model for this file, restored by finalize_and_compile before
            # any output is written. Empty for every file that was already windowable.
            elision_mapping = {}
            if _win_cap > 0 and _win_err_line is not None:
                if syntactic_codeobject_window_enabled():
                    _win_seg = codeobject_window_syntax_context(copy_dir, _win_err_line, initial_error_description, 0)
                else:
                    _win_seg = minimal_window_syntax_context(copy_dir, _win_err_line, initial_error_description, 0)
                # An over-budget window is a reason to look at LESS of the file, not to abandon it:
                # clamp around the error line and let the retry loop slide the window across the
                # file. Only a file whose ERROR LINE ALONE busts the budget (minified/packer
                # one-liner) is genuinely un-windowable and therefore out of scope.
                if _win_seg is not None and window_exceeds_token_budget(_win_seg.text, _win_cap, count_window_tokens):
                    if clamp_syntax_segment(_win_seg, _win_err_line, _win_cap, count_window_tokens) is None:
                        # LAST CHANCE before discarding: the error line is almost always oversized
                        # because it carries an embedded payload, not because it is complex. Hide
                        # the payload and re-check; the exact bytes are restored by
                        # finalize_and_compile before anything is written. Fires only here, so a
                        # file that was already in scope never takes this path.
                        _elided_source, elision_mapping = elide_to_fit_window(
                            read_file(copy_dir) or "", _win_err_line, _win_cap, count_window_tokens
                        )
                        if elision_mapping:
                            with open(copy_dir, "w", encoding="utf-8") as f:
                                f.write(_elided_source)
                            log_rec.update(
                                {
                                    "payload_elision_used": True,
                                    "payload_elision_count": len(elision_mapping),
                                }
                            )
                        else:
                            _win_tok = count_window_tokens(_win_seg.text)
                            log_rec.update(
                                {
                                    "skipped_due_to_blob_window": True,
                                    "unwindowable_error_line": True,
                                    "blob_window_tokens": _win_tok if _win_tok < 10 ** 9 else None,
                                    "blob_window_chars": len(_win_seg.text),
                                    "compiled_success": None,
                                }
                            )
                            append_log(log_file, log_rec)
                            return

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
                    if _run_delete_only_mode():
                        break

                # --- Task 6, bullet 2: deterministic mechanical pre-pass, tried
                # BEFORE the LLM. If it makes the file compile, write it via the
                # same compile_new_pyc path used everywhere else in this loop and
                # break WITHOUT calling the LLM. Gated entirely behind
                # SYNTACTIC_DETERMINISTIC_PREPASS (default "1"); when the flag is
                # "0" this whole block is skipped and the loop below is untouched.
                if deterministic_prepass_enabled():
                    if _try_deterministic_prepass():
                        break

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

                # Window-budget discard (in-loop): the window grows as expansion widens to outer
                # code objects across retries. Whenever the window the LLM would actually see
                # exceeds the generation budget (2048 tokens), discard the file -- stop
                # attempting it and EXCLUDE it from the analyzability denominator
                # (compiled_success=None). Gated by SYNTACTIC_MAX_WINDOW_TOKENS (0=off).
                _win_cap = syntactic_max_window_tokens()
                if _win_cap > 0:
                    _win_err_line = extract_line_number(initial_error_description)
                    _win_seg = (
                        syntax_context_provider(copy_dir, _win_err_line, initial_error_description, total_attempts_completed)
                        if _win_err_line is not None
                        else None
                    )
                    if _win_seg is not None and window_exceeds_token_budget(_win_seg.text, _win_cap, count_window_tokens):
                        if clamp_syntax_segment(_win_seg, _win_err_line, _win_cap, count_window_tokens) is None:
                            _win_tok = count_window_tokens(_win_seg.text)
                            log_rec.update(
                                {
                                    "skipped_due_to_blob_window": True,
                                    "unwindowable_error_line": True,
                                    "blob_window_tokens": _win_tok if _win_tok < 10 ** 9 else None,
                                    "blob_window_chars": len(_win_seg.text),
                                    "compiled_success": None,
                                }
                            )
                            append_log(log_file, log_rec)
                            break
                        # Windowable: the LLM sees a budget-clamped view of the window. Successive
                        # retries re-localize to the next error, sliding the window over the file.
                        syntax_context_provider = _clamped_context_provider(
                            syntax_context_provider, _win_cap, count_window_tokens
                        )

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

                        compilation_result = finalize_and_compile(candidate_compilation_candidate, out_py_path, out_pyc_path, version, elision_mapping)
                        candidate_compile_ms = int((time.perf_counter() - t_candidate) * 1000)
                        candidate_is_compiled = compilation_result["is_compiled"]
                        candidate_error_description = compilation_result["error_description"]

                        # POST-LLM deterministic pass: the model's output is frequently ALMOST
                        # parseable -- one unclosed bracket, a handler-less `try:`, a literal
                        # assignment target. The same compile-gated operators that clean the input
                        # can finish the job here, turning a near-miss into a compiling file
                        # instead of burning another retry (or falling through to deletion).
                        # Compile-gated end to end: only adopted if it actually compiles.
                        if not candidate_is_compiled and candidate_compilation_candidate:
                            # The EXHAUSTED deterministic layer, after this generation. Measured
                            # on real artifacts: rescues 13.3% of LLM outputs that would otherwise
                            # fall to the delete-only fallback, turning each into a GENUINE repair.
                            # The weak `maybe_prepass` used to run here; it is kept as a fallback
                            # because it carries the GT literal splice, which the search does not.
                            # `compile_version` takes (py_path, out_path, version) -- FILE PATHS.
                            # Passing it as a (source, version) probe raises TypeError on EVERY
                            # call, and the blanket try/except swallows it, so the layer silently
                            # reported "no fix" for every file. This defect predates the exhaustive
                            # layer: the original maybe_prepass call here had it too, which means
                            # the historical "post-LLM prepass 0/57" result measured a broken call
                            # rather than a useless idea. Probe source TEXT, like the pre-LLM site.
                            def _post_llm_probe(candidate_source: str, candidate_version: str) -> None:
                                with tempfile.TemporaryDirectory(prefix="post_llm_probe_") as _d:
                                    _r = compile_new_pyc(candidate_source,
                                                         os.path.join(_d, "probe.py"),
                                                         os.path.join(_d, "probe.pyc"),
                                                         candidate_version)
                                if not _r["is_compiled"]:
                                    raise CompileError(_r.get("error_description") or "post-LLM probe: compile failed")

                            post_source, post_compiled, post_ops = maybe_post_llm_repair(
                                candidate_compilation_candidate, version, _post_llm_probe
                            )
                            if not post_compiled:
                                post_source, post_compiled, post_ops = maybe_prepass(
                                    candidate_compilation_candidate, version, _post_llm_probe,
                                    gt_sequences=maybe_gt_sequences(
                                        gt_pyc_value, candidate_compilation_candidate
                                    ),
                                )
                            if post_compiled and post_source:
                                create_file_from_response(copy_dir, post_source)
                                post_result = finalize_and_compile(post_source, out_py_path, out_pyc_path, version, elision_mapping)
                                if post_result["is_compiled"]:
                                    candidate_compilation_candidate = post_source
                                    candidate_is_compiled = True
                                    candidate_error_description = None
                                    log_rec.setdefault("post_llm_prepass_operations", []).extend(post_ops or [])
                                    log_rec["post_llm_prepass_fixed"] = True
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
                    if _finalise_with_fallback():
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

    for idx, row in df.iterrows():
        _process_row(idx, row)
