"""END-TO-END integration test for the syntactic-repair runner loop
(``pipeline.runner.run_experiment``'s ``while not is_compiled:`` cycle),
with the LLM seam mocked -- no real model, no GPU.

The individual seams (``minimal_window_syntax_context``, ``maybe_prepass``,
``reattach_window``) are already unit-tested elsewhere (see
tests/test_syntactic_runner_prepass.py). What is NOT covered anywhere else
is the wiring between them inside ``run_experiment`` itself:

    LLM returns a fix
        -> the reattach-decision branch (recompute the minimal window from
           the CURRENT file state and compare to the fixer's start/end --
           runner.py's ``reattached_via_minimal_window`` block, ~450-472)
        -> ``compile_new_pyc``
        -> ``is_compiled``
        -> retry-with-widening (or give up)

Mock point: ``pipeline.repair_engine.attempt_repair`` (imported by
``run_experiment`` via a fresh ``from pipeline.repair_engine import
attempt_repair`` on every call -- see runner.py:154 -- so monkeypatching the
module attribute before calling ``run_experiment`` is picked up every time).
A fake returns the exact 7-tuple production returns:
``(final_code, llm_metrics, with_pin_point, start_ln, end_ln, base_indent,
anchor_indent)``. Everything downstream of that return value -- the
recompute-and-compare, the real ``reattach_window``/``reattach_block``
splice, the real ``compile_new_pyc`` (native py_compile, host is 3.12) --
runs for real. No dataset/env plumbing is touched: ``RuntimeConfig`` is a
plain dataclass built directly in-process, and ``pipeline.dataset`` /
``pipeline.runner``'s ``BASE_DATASET_PATH`` are monkeypatched to a one-row
CSV written into ``tmp_path``, with ``file_path``/``file`` columns pointing
straight at a tmp source file (bypassing the on-disk PyPi/pylingual dataset
tree entirely). ``run_experiment``'s relative ``results/...`` output tree is
confined to ``tmp_path`` via ``monkeypatch.chdir``.

No non-test file was modified to make this possible.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

import pipeline.dataset as dataset_mod
import pipeline.repair_engine as repair_engine_mod
import pipeline.runner as runner_mod
from pipeline.config import RuntimeConfig

HOST_VERSION = "3.12"  # must match the running interpreter -> compile_version's native path


def _make_runtime_config(**overrides: Any) -> RuntimeConfig:
    base: Dict[str, Any] = dict(
        previous_run_log_path=None,
        delete_only_mode=False,
        enable_delete_only_fallback=False,
        enable_syntax_explanation=False,
        enable_whole_file_repair=False,
        use_local_llm=True,
        delete_only_infinite_iters=False,
        delete_only_max_iters=5000,
        delete_only_base_window=1,
        delete_only_max_deleted_ratio=0.95,
        max_whole_file_bytes=1024 * 1024,
        max_retries_default=5,
        local_llm_idx=0,
        run_timestamp="20260101T000000Z",
    )
    base.update(overrides)
    return RuntimeConfig(**base)


def _row(
    *,
    file_hash: str,
    file_path: Path,
    error_description: str,
    file_name: str = "sample.py",
) -> Dict[str, Any]:
    """A single synthetic dataset row. ``file_path``/``file`` are read by
    runner.py BEFORE it would fall back to resolving a path from the on-disk
    PyPi/pylingual dataset tree (pipeline.dataset.resolve_syntax_dataset_source_path),
    so supplying them here sidesteps that resolution machinery entirely --
    a legitimate row shape (row.get("file_path") is checked first), just not
    what a real production CSV row currently carries (see dataset/*.csv)."""
    return {
        "file_hash": file_hash,
        "file_path": str(file_path),
        "file": file_name,
        "source": "TestSource",
        "decompiler": "TestDecompiler",
        "dataset": "test_dataset",
        "bytecode_version": HOST_VERSION,
        "error_type": "syntactic_error",
        "error_description": error_description,
        "error_message": "",
        "error": "",
    }


def _drive_run_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row: Dict[str, Any],
    fake_attempt_repair,
    *,
    config: RuntimeConfig | None = None,
) -> List[Dict[str, Any]]:
    """Write `row` as a one-row dataset CSV, point pipeline.dataset /
    pipeline.runner's BASE_DATASET_PATH at it, monkeypatch
    pipeline.repair_engine.attempt_repair, confine all of run_experiment's
    relative results/ output to tmp_path, and drive the REAL
    pipeline.runner.run_experiment loop. Returns the list of log records
    passed to pipeline.runner.append_log (captured, not written to disk),
    with a resolved absolute "_abs_out" path spliced in when "path_out" is
    present, since run_experiment writes that path relative to cwd."""
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    monkeypatch.setattr(dataset_mod, "BASE_DATASET_PATH", csv_path)
    monkeypatch.setattr(runner_mod, "BASE_DATASET_PATH", csv_path)
    monkeypatch.setattr(repair_engine_mod, "attempt_repair", fake_attempt_repair)

    logged: List[Dict[str, Any]] = []
    monkeypatch.setattr(runner_mod, "append_log", lambda path, record: logged.append(dict(record)))

    monkeypatch.chdir(tmp_path)
    runner_mod.run_experiment(config or _make_runtime_config(), source=None, limit=None)

    if logged and logged[-1].get("path_out"):
        logged[-1]["_abs_out"] = str((tmp_path / logged[-1]["path_out"]).resolve())
    return logged


def _fake_metrics() -> Dict[str, Any]:
    return {
        "fits_single_run": True,
        "avg_chunk_tokens": 5,
        "max_chunk_tokens": 5,
        "llm_calls": 1,
        "llm_latency_ms_total": 3,
    }


# ---------------------------------------------------------------------------
# 1. Prepass fixes it -> LLM (attempt_repair) is never called.
# ---------------------------------------------------------------------------

def test_prepass_fixes_file_llm_never_called(tmp_path, monkeypatch):
    src = tmp_path / "sample.py"
    src.write_text("x = f(1, 2\n")  # mechanically closable by balance_delimiters

    row = _row(
        file_hash="syntit_prepass_only",
        file_path=src,
        error_description="SyntaxError: '(' was never closed (sample.py, line 1)",
    )

    def fake_attempt_repair(**kwargs):
        raise AssertionError("attempt_repair must not be called when the deterministic prepass already compiled the file")

    logged = _drive_run_experiment(tmp_path, monkeypatch, row, fake_attempt_repair)

    assert len(logged) == 1
    rec = logged[0]
    assert rec["compiled_success"] is True
    assert rec.get("llm_calls") == 0
    assert rec.get("deterministic_prepass_used") is True
    # The sweep cascade reaches this file before the shipped error-driven prepass does and reports
    # the same operator under its own namespace ("legacy:balance_delimiters"), so match on the
    # operator rather than the namespace -- what this test asserts is that the fix was mechanical
    # and attributed, not which of the two deterministic drivers got there first.
    operations = rec.get("deterministic_prepass_operations", [])
    assert any(op.endswith("balance_delimiters") for op in operations), operations
    # Stage 1 is the code-preserving stage; a bracket close must never be attributed to synthesis.
    assert rec.get("deterministic_cascade_stage") == 1
    assert rec.get("deterministic_control_flow_rewrites") == []

    out_content = Path(rec["_abs_out"]).read_text()
    assert out_content == "x = f(1, 2)\n"
    compile(out_content, "<t>", "exec")  # genuinely compiles, not just claimed to

    # Never mutate the original decompiled input.
    assert src.read_text() == "x = f(1, 2\n"


# ---------------------------------------------------------------------------
# 2. LLM fix reattached via the minimal-window path and compiles.
# ---------------------------------------------------------------------------

def test_llm_fix_reattached_via_minimal_window_and_compiles(tmp_path, monkeypatch):
    src = tmp_path / "sample.py"
    original = "def f()\n    return 1\n"  # missing ':' -- balance_delimiters/dedent/etc. cannot fix this
    src.write_text(original)

    row = _row(
        file_hash="syntit_minimal_window",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        # minimal_repair_window(original, lineno=1, expansion=0) isolates exactly
        # line 1 ("def f()\n") with indent "" -- verified against the real
        # utils.syntactic_prepass.minimal_repair_window before writing this test.
        return ("def f():\n", _fake_metrics(), True, 1, 1, "", "")

    logged = _drive_run_experiment(tmp_path, monkeypatch, row, fake_attempt_repair)

    assert len(calls) == 1
    assert calls[0]["expansion_level"] == 0

    assert len(logged) == 1
    rec = logged[0]
    assert rec["compiled_success"] is True
    assert rec.get("llm_calls") == 1

    out_content = Path(rec["_abs_out"]).read_text()
    # The fixed first line, byte-preserved second line + indentation.
    assert out_content == "def f():\n    return 1\n"
    compile(out_content, "<t>", "exec")

    # Original input untouched.
    assert src.read_text() == original


# ---------------------------------------------------------------------------
# 3. Bad LLM fix fails the compile gate -> loop retries -> good fix accepted.
# ---------------------------------------------------------------------------

def test_bad_fix_rejected_by_compile_gate_then_retry_succeeds(tmp_path, monkeypatch):
    src = tmp_path / "sample.py"
    original = "def f()\n    return 1\n"
    src.write_text(original)

    row = _row(
        file_hash="syntit_retry",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # Still-broken candidate (bracket-balanced so the window stays a
            # single self-contained line -- verified against the real
            # compile_new_pyc + minimal_repair_window(expansion=1) before
            # writing this test: the recomputed window is still (1, 1)).
            return ("def f(x y):\n", _fake_metrics(), True, 1, 1, "", "")
        return ("def f():\n", _fake_metrics(), True, 1, 1, "", "")

    logged = _drive_run_experiment(
        tmp_path, monkeypatch, row, fake_attempt_repair,
        config=_make_runtime_config(max_retries_default=5),
    )

    assert len(calls) == 2, "the loop must retry after the first (bad) fix fails to compile"
    assert calls[0]["expansion_level"] == 0
    assert calls[1]["expansion_level"] == 1  # widened on the retry

    # Only ONE append_log call: the mid-loop failed attempt does not log on
    # its own (only success or final give-up do) -- so a single, successful
    # record proves the compile gate genuinely rejected attempt 1 and the
    # loop kept going rather than accepting a non-compiling splice.
    assert len(logged) == 1
    rec = logged[0]
    assert rec["compiled_success"] is True
    assert rec["total_attempts_completed"] == 2

    out_content = Path(rec["_abs_out"]).read_text()
    assert out_content == "def f():\n    return 1\n"
    compile(out_content, "<t>", "exec")


# ---------------------------------------------------------------------------
# 4. Flag-off regression: SYNTACTIC_DETERMINISTIC_PREPASS="0" skips the
#    prepass block entirely and preserves the pre-prepass legacy behavior
#    (LLM always called, full-file reattach when with_pin_point=False).
# ---------------------------------------------------------------------------

def test_prepass_flag_off_skips_prepass_and_still_calls_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTACTIC_DETERMINISTIC_PREPASS", "0")

    src = tmp_path / "sample.py"
    original = "x = f(1, 2\n"  # mechanically fixable -- but the flag is off
    src.write_text(original)

    row = _row(
        file_hash="syntit_flagoff",
        file_path=src,
        error_description="SyntaxError: '(' was never closed (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        return ("x = f(1, 2)\n", _fake_metrics(), False, None, None, None, None)

    def _raise_if_prepass_called(*args, **kwargs):
        raise AssertionError("maybe_prepass must not be called when SYNTACTIC_DETERMINISTIC_PREPASS=0")

    monkeypatch.setattr(runner_mod, "maybe_prepass", _raise_if_prepass_called)

    logged = _drive_run_experiment(tmp_path, monkeypatch, row, fake_attempt_repair)

    assert len(calls) == 1, "flag off must still route through attempt_repair (the LLM path)"

    assert len(logged) == 1
    rec = logged[0]
    assert rec["compiled_success"] is True
    assert "deterministic_prepass_used" not in rec  # prepass block never ran at all
    assert rec.get("llm_calls") == 1

    out_content = Path(rec["_abs_out"]).read_text()
    assert out_content == "x = f(1, 2)\n"
    compile(out_content, "<t>", "exec")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
