"""TDD for compile-gated best-of-N sampling on the SYNTACTIC-repair LLM path.

``SYNTACTIC_BEST_OF_N`` (default 1) == today's single-greedy-call behavior,
byte-identical (see ``pipeline.runner.syntactic_best_of_n``). N>1: try the
greedy candidate (temperature 0, the EXACT SAME call production already
makes) first; only when it fails to compile are up to N-1 additional
SAMPLED candidates (temperature ~0.7 / top_p ~0.9) generated and
compile-checked, one at a time, accepting the first that compiles -- see
``pipeline.repair_engine.select_best_of_n_candidate`` for the pure
selection logic, and ``pipeline.runner.run_experiment``'s
``while not is_compiled:`` loop for how it is wired to the real
reattach/compile machinery.

Mocks all LLM/server calls -- no GPU, no real model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

import pipeline.dataset as dataset_mod
import pipeline.repair_engine as repair_engine_mod
import pipeline.runner as runner_mod
from pipeline.config import RuntimeConfig
from pipeline.repair_engine import select_best_of_n_candidate
from pipeline.runner import syntactic_best_of_n

HOST_VERSION = "3.12"  # must match the running interpreter -> compile_version's native path


# ---------------------------------------------------------------------------
# syntactic_best_of_n()
# ---------------------------------------------------------------------------


def test_best_of_n_default_is_one_when_env_unset(monkeypatch):
    monkeypatch.delenv("SYNTACTIC_BEST_OF_N", raising=False)
    assert syntactic_best_of_n() == 1


def test_best_of_n_reads_env_int(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_BEST_OF_N", "4")
    assert syntactic_best_of_n() == 4


@pytest.mark.parametrize("raw", ["0", "-1", "-5"])
def test_best_of_n_clamps_nonpositive_to_one(monkeypatch, raw):
    monkeypatch.setenv("SYNTACTIC_BEST_OF_N", raw)
    assert syntactic_best_of_n() == 1


def test_best_of_n_clamps_garbage_to_one(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_BEST_OF_N", "not-a-number")
    assert syntactic_best_of_n() == 1


# ---------------------------------------------------------------------------
# select_best_of_n_candidate: pure compile-gated candidate selection
# ---------------------------------------------------------------------------


def test_selects_first_compiling_sampled_candidate_after_greedy_fails():
    generate_sampled = MagicMock(side_effect=["sampled1", "sampled2"])
    compiles = MagicMock(side_effect=[False, False, True])  # greedy, sampled1, sampled2

    candidate, compiled = select_best_of_n_candidate(
        best_of_n=4,
        generate_greedy=lambda: "greedy",
        generate_sampled=generate_sampled,
        compiles=compiles,
    )

    assert candidate == "sampled2"
    assert compiled is True
    assert generate_sampled.call_count == 2
    assert compiles.call_count == 3


def test_greedy_success_short_circuits_without_generating_samples():
    generate_sampled = MagicMock()
    compiles = MagicMock(return_value=True)

    candidate, compiled = select_best_of_n_candidate(
        best_of_n=5,
        generate_greedy=lambda: "greedy",
        generate_sampled=generate_sampled,
        compiles=compiles,
    )

    assert candidate == "greedy"
    assert compiled is True
    generate_sampled.assert_not_called()
    assert compiles.call_count == 1


def test_n_equals_one_calls_generator_exactly_once_no_sampling():
    generate_greedy = MagicMock(return_value="greedy")
    generate_sampled = MagicMock()
    compiles = MagicMock(return_value=False)  # even on failure, N=1 must not sample

    candidate, compiled = select_best_of_n_candidate(
        best_of_n=1,
        generate_greedy=generate_greedy,
        generate_sampled=generate_sampled,
        compiles=compiles,
    )

    assert candidate == "greedy"
    assert compiled is False
    assert generate_greedy.call_count == 1
    generate_sampled.assert_not_called()


def test_all_candidates_fail_falls_back_to_greedy():
    generate_sampled = MagicMock(side_effect=["sampled1", "sampled2", "sampled3"])
    compiles = MagicMock(return_value=False)

    candidate, compiled = select_best_of_n_candidate(
        best_of_n=4,
        generate_greedy=lambda: "greedy",
        generate_sampled=generate_sampled,
        compiles=compiles,
    )

    assert candidate == "greedy"
    assert compiled is False
    assert generate_sampled.call_count == 3  # best_of_n - 1


# ---------------------------------------------------------------------------
# Runner integration: real reattach + real compile_new_pyc, LLM (attempt_repair)
# mocked. Mirrors tests/test_syntactic_runner_integration.py's harness.
# ---------------------------------------------------------------------------


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


def _row(*, file_hash: str, file_path: Path, error_description: str, file_name: str = "sample.py") -> Dict[str, Any]:
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


def test_best_of_n_default_one_never_samples_even_when_greedy_fails(tmp_path, monkeypatch):
    """SYNTACTIC_BEST_OF_N unset (default 1): a failing greedy candidate must NOT trigger
    any extra sampled candidate generation -- exactly one attempt_repair call per while-loop
    iteration, identical to pre-best-of-N behavior."""
    monkeypatch.delenv("SYNTACTIC_BEST_OF_N", raising=False)

    src = tmp_path / "sample.py"
    original = "def f()\n    return 1\n"
    src.write_text(original)

    row = _row(
        file_hash="bon_default_one",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        # always broken -> exhausts retries -> no successful compile.
        return ("def f(x y):\n", _fake_metrics(), True, 1, 1, "", "")

    logged = _drive_run_experiment(
        tmp_path, monkeypatch, row, fake_attempt_repair,
        config=_make_runtime_config(max_retries_default=1, enable_delete_only_fallback=False),
    )

    # max_retries_default=1 -> 2 while-loop iterations (attempts 1 and 2), ONE
    # attempt_repair call per iteration since best-of-n is 1 -- no extra sampled calls.
    assert len(calls) == 2
    for c in calls:
        assert c.get("generation_config_override") is None


def test_best_of_n_greedy_fails_sampled_candidate_compiles(tmp_path, monkeypatch):
    """SYNTACTIC_BEST_OF_N=3: greedy fails, first sampled candidate fails, second sampled
    candidate compiles -> accepted. All three candidates are generated within the SAME
    while-loop attempt (expansion_level stays 0 for all three -- this is NOT a retry), the
    greedy call carries no override, and the two sampled calls carry a do_sample=True
    generation_config override. The compiled output matches the ACCEPTED (sampled) candidate,
    not a stale mutation left behind by an earlier rejected candidate."""
    monkeypatch.setenv("SYNTACTIC_BEST_OF_N", "3")

    src = tmp_path / "sample.py"
    original = "def f()\n    return 1\n"
    src.write_text(original)

    row = _row(
        file_hash="bon_sampled_wins",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        idx = len(calls)
        if idx == 1:
            return ("def f(x y):\n", _fake_metrics(), True, 1, 1, "", "")  # greedy: broken
        if idx == 2:
            return ("def f(a b):\n", _fake_metrics(), True, 1, 1, "", "")  # sampled1: broken
        return ("def f():\n", _fake_metrics(), True, 1, 1, "", "")  # sampled2: compiles

    logged = _drive_run_experiment(tmp_path, monkeypatch, row, fake_attempt_repair)

    assert len(calls) == 3, "greedy + 2 sampled candidates, all within ONE attempt"
    assert [c["expansion_level"] for c in calls] == [0, 0, 0], "best-of-N candidates share one attempt, not retries"
    assert calls[0].get("generation_config_override") is None, "candidate 0 must be the untouched greedy call"
    for sampled_call in calls[1:]:
        cfg = sampled_call.get("generation_config_override")
        assert isinstance(cfg, dict)
        assert cfg.get("do_sample") is True
        assert cfg.get("temperature") not in (0, 0.0, None)

    assert len(logged) == 1
    rec = logged[0]
    assert rec["compiled_success"] is True
    assert rec["total_attempts_completed"] == 1, "best-of-N candidate search happens within ONE outer attempt"

    out_content = Path(rec["_abs_out"]).read_text()
    assert out_content == "def f():\n    return 1\n"
    compile(out_content, "<t>", "exec")

    # Never mutate the original decompiled input.
    assert src.read_text() == original


def test_best_of_n_all_candidates_fail_behaves_like_single_greedy_failure(tmp_path, monkeypatch):
    """SYNTACTIC_BEST_OF_N=3, every candidate (greedy + 2 sampled) fails to compile: the
    attempt is treated as failed exactly like a plain greedy failure -- the loop retries on
    the next while-loop iteration (widened window), NOT an early abort."""
    monkeypatch.setenv("SYNTACTIC_BEST_OF_N", "3")

    src = tmp_path / "sample.py"
    original = "def f()\n    return 1\n"
    src.write_text(original)

    row = _row(
        file_hash="bon_all_fail",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        n = len(calls)
        if n <= 3:
            # attempt 1: greedy + 2 sampled, all broken but bracket-balanced (window
            # recompute stays a single self-contained line, as in the existing
            # bad-fix-then-retry test).
            return (f"def f(x{n} y{n}):\n", _fake_metrics(), True, 1, 1, "", "")
        return ("def f():\n", _fake_metrics(), True, 1, 1, "", "")  # attempt 2, greedy: fixed

    logged = _drive_run_experiment(
        tmp_path, monkeypatch, row, fake_attempt_repair,
        config=_make_runtime_config(max_retries_default=5),
    )

    assert len(calls) == 4, "3 failed candidates in attempt 1, then 1 greedy candidate in attempt 2"
    assert [c["expansion_level"] for c in calls] == [0, 0, 0, 1]

    assert len(logged) == 1
    rec = logged[0]
    assert rec["compiled_success"] is True
    assert rec["total_attempts_completed"] == 2

    out_content = Path(rec["_abs_out"]).read_text()
    assert out_content == "def f():\n    return 1\n"
    compile(out_content, "<t>", "exec")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
