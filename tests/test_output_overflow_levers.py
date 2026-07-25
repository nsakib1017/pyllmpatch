"""TDD for the OUTPUT_OVERFLOW anti-repetition decode guard + prompt-taxonomy additions on
the SYNTACTIC-repair path.

Residual diagnosis: the single biggest remaining fallback mode (13/25 remaining fallbacks,
3.13/3.15-heavy) is OUTPUT_OVERFLOW -- the local model blows its output-token budget via
REPETITION LOOPS (re-emitting the same lines verbatim) and gratuitous reformatting, so it
never emits a valid completion. This is a DECODE/HARNESS fault, not a knowledge gap.

Two levers, both env-gated and OFF by default (byte-identical to pre-lever behavior):

- ``pipeline.runner.syntactic_repetition_penalty()`` (SYNTACTIC_REPETITION_PENALTY, default
  1.0 == off) and ``pipeline.runner.syntactic_generation_config(base)`` -- the composition
  helper that merges repetition_penalty on TOP of whatever generation_config is already being
  built (the model's own greedy config, or best-of-N's ``sampling_generation_config(...)``
  output), never replacing the sampling settings already present.
- ``pipeline.runner.syntactic_max_tokens_override()`` (SYNTACTIC_MAX_TOKENS, default unset ==
  None == off) -- overrides the max_tokens budget passed to the syntactic LLM call in
  ``pipeline.repair_engine.process_file_in_single_run``.

Plus three new prompt-taxonomy rules in ``utils.token_helpers.SYSTEM_PROMPT_FOR_LOCAL``:
ANTI-REPETITION, MULTI-LINE STRING LITERAL, and STRAY TOP-OF-MODULE STATEMENT.

Mocks all LLM/server calls -- no GPU, no real model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

import pipeline.dataset as dataset_mod
import pipeline.repair_engine as repair_engine_mod
import pipeline.runner as runner_mod
from pipeline.config import RuntimeConfig
from pipeline.repair_engine import process_file_for_syntax_error_patching
from pipeline.runner import (
    sampling_generation_config,
    syntactic_generation_config,
    syntactic_max_tokens_override,
    syntactic_repetition_penalty,
)
from utils.providers import OPEN_LLM_MODELS
from utils.token_helpers import SYSTEM_PROMPT_FOR_LOCAL

HOST_VERSION = "3.12"  # must match the running interpreter -> compile_version's native path


# ---------------------------------------------------------------------------
# syntactic_repetition_penalty()
# ---------------------------------------------------------------------------


def test_repetition_penalty_default_is_one_when_env_unset(monkeypatch):
    monkeypatch.delenv("SYNTACTIC_REPETITION_PENALTY", raising=False)
    assert syntactic_repetition_penalty() == 1.0


def test_repetition_penalty_reads_env_float(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_REPETITION_PENALTY", "1.15")
    assert syntactic_repetition_penalty() == 1.15


def test_repetition_penalty_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_REPETITION_PENALTY", "not-a-float")
    assert syntactic_repetition_penalty() == 1.0


# ---------------------------------------------------------------------------
# syntactic_max_tokens_override()
# ---------------------------------------------------------------------------


def test_max_tokens_override_default_is_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("SYNTACTIC_MAX_TOKENS", raising=False)
    assert syntactic_max_tokens_override() is None


def test_max_tokens_override_reads_env_int(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_MAX_TOKENS", "8000")
    assert syntactic_max_tokens_override() == 8000


def test_max_tokens_override_garbage_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_MAX_TOKENS", "not-an-int")
    assert syntactic_max_tokens_override() is None


def test_max_tokens_override_blank_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_MAX_TOKENS", "   ")
    assert syntactic_max_tokens_override() is None


# ---------------------------------------------------------------------------
# syntactic_generation_config(): the repetition-penalty composition helper
# ---------------------------------------------------------------------------


def test_syntactic_generation_config_default_is_noop_on_none(monkeypatch):
    monkeypatch.delenv("SYNTACTIC_REPETITION_PENALTY", raising=False)
    assert syntactic_generation_config(None) is None


def test_syntactic_generation_config_default_is_byte_identical_on_dict(monkeypatch):
    monkeypatch.delenv("SYNTACTIC_REPETITION_PENALTY", raising=False)
    base = {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 2048}
    result = syntactic_generation_config(base)
    # Off by default: same object back, not a copy -- proves zero mutation, zero surprise.
    assert result is base
    assert result == {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 2048}


def test_syntactic_generation_config_composes_with_best_of_n_sampling_config(monkeypatch):
    """The composition test: apply the repetition-penalty merge on top of best-of-N's
    sampling generation_config and confirm BOTH survive -- sampling temperature/top_p
    untouched, repetition_penalty newly present."""
    monkeypatch.setenv("SYNTACTIC_REPETITION_PENALTY", "1.15")

    base_model_generation_config = {"max_new_tokens": 2048, "do_sample": False, "temperature": 0.0, "top_p": 1.0}
    sampled_cfg = sampling_generation_config(base_model_generation_config)
    assert sampled_cfg["do_sample"] is True
    assert sampled_cfg["temperature"] == 0.7
    assert sampled_cfg["top_p"] == 0.9

    merged = syntactic_generation_config(sampled_cfg)

    assert merged["do_sample"] is True
    assert merged["temperature"] == 0.7
    assert merged["top_p"] == 0.9
    assert merged["max_new_tokens"] == 2048
    assert merged["repetition_penalty"] == 1.15


def test_syntactic_generation_config_applies_to_greedy_config_too(monkeypatch):
    """Same helper, applied directly to a GREEDY (do_sample=False) config -- repetition_penalty
    must merge in WITHOUT flipping do_sample/temperature away from greedy."""
    monkeypatch.setenv("SYNTACTIC_REPETITION_PENALTY", "1.15")

    greedy_cfg = {"max_new_tokens": 2048, "do_sample": False, "temperature": 0.0, "top_p": 1.0}
    merged = syntactic_generation_config(greedy_cfg)

    assert merged["do_sample"] is False
    assert merged["temperature"] == 0.0
    assert merged["repetition_penalty"] == 1.15


def test_syntactic_generation_config_handles_none_base_when_penalty_on(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_REPETITION_PENALTY", "1.2")
    merged = syntactic_generation_config(None)
    assert merged == {"repetition_penalty": 1.2}


# ---------------------------------------------------------------------------
# Prompt taxonomy additions in SYSTEM_PROMPT_FOR_LOCAL
# ---------------------------------------------------------------------------


class TestPromptTaxonomyAdditions:
    def setup_method(self):
        self.prompt = SYSTEM_PROMPT_FOR_LOCAL.lower()

    def test_anti_repetition_rule_present(self):
        assert "anti-repetition" in self.prompt
        assert "emit each line of the snippet at most once" in self.prompt
        assert "never repeat, duplicate, or loop lines" in self.prompt
        assert "never re-output unchanged regions more than once" in self.prompt

    def test_multi_line_string_literal_rule_present(self):
        assert "multi-line string literal" in self.prompt
        assert "split across physical lines by a raw newline" in self.prompt
        assert "close it on its own line" in self.prompt
        assert "do not merge unrelated code into the string" in self.prompt

    def test_stray_top_of_module_statement_rule_present(self):
        assert "stray top-of-module statement" in self.prompt
        assert "stray orphaned statement at module level" in self.prompt
        assert "should be removed" in self.prompt

    def test_existing_rules_not_regressed(self):
        # Don't clobber the pre-existing surgical/never-invent/truncated-literal language
        # while adding the new taxonomy entries.
        assert "surgical" in self.prompt
        assert "never invent" in self.prompt
        assert "truncated" in self.prompt
        assert "close it as-is" in self.prompt
        assert "minimal changes" in self.prompt


# ---------------------------------------------------------------------------
# max_tokens_override wiring: pipeline.repair_engine.process_file_for_syntax_error_patching
# -> process_file_in_single_run -> make_call_to_local_llm's max_tokens argument.
# ---------------------------------------------------------------------------


def test_max_tokens_override_reaches_local_llm_call(tmp_path):
    llm = dict(OPEN_LLM_MODELS[0])  # qwen2.5-coder-7b: no generation_config of its own
    llm["token_for_completion"] = 8192

    with mock.patch.object(repair_engine_mod, "make_call_to_local_llm", return_value="fixed = 1\n") as fake_call:
        result = process_file_for_syntax_error_patching(
            "broken = 1\n",
            "SyntaxError: bad",
            tmp_path,
            llm=llm,
            enable_syntax_explanation=False,
            max_tokens_override=8000,
        )

    assert result is not None
    assert fake_call.call_args.args[5] == 8000  # positional max_tokens arg


def test_max_tokens_override_none_keeps_model_token_for_completion(tmp_path):
    llm = dict(OPEN_LLM_MODELS[0])
    llm["token_for_completion"] = 8192

    with mock.patch.object(repair_engine_mod, "make_call_to_local_llm", return_value="fixed = 1\n") as fake_call:
        result = process_file_for_syntax_error_patching(
            "broken = 1\n",
            "SyntaxError: bad",
            tmp_path,
            llm=llm,
            enable_syntax_explanation=False,
            max_tokens_override=None,
        )

    assert result is not None
    assert fake_call.call_args.args[5] == 8192  # unchanged: byte-identical default


# ---------------------------------------------------------------------------
# Runner integration: repetition_penalty reaches BOTH the greedy attempt_repair call and any
# sampled best-of-N attempt_repair call, via generation_config_override.
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
        local_llm_idx=2,  # qwen3-coder-30b: HAS a generation_config (do_sample=False greedy)
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
    return logged


def _fake_metrics() -> Dict[str, Any]:
    return {
        "fits_single_run": True,
        "avg_chunk_tokens": 5,
        "max_chunk_tokens": 5,
        "llm_calls": 1,
        "llm_latency_ms_total": 3,
    }


def test_repetition_penalty_off_by_default_greedy_override_unchanged(tmp_path, monkeypatch):
    """Default (SYNTACTIC_REPETITION_PENALTY unset): the greedy call's generation_config_override
    must carry exactly the model's own generation_config -- no repetition_penalty key, proving
    the lever is a true no-op end to end when off."""
    monkeypatch.delenv("SYNTACTIC_REPETITION_PENALTY", raising=False)
    monkeypatch.delenv("SYNTACTIC_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SYNTACTIC_BEST_OF_N", raising=False)

    src = tmp_path / "sample.py"
    original = "def f()\n    return 1\n"
    src.write_text(original)

    row = _row(
        file_hash="overflow_default_off",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        return ("def f():\n", _fake_metrics(), True, 1, 1, "", "")

    _drive_run_experiment(
        tmp_path, monkeypatch, row, fake_attempt_repair,
        config=_make_runtime_config(max_retries_default=1),
    )

    assert len(calls) == 1
    cfg = calls[0].get("generation_config_override")
    # qwen3-coder-30b's own generation_config, untouched -- no repetition_penalty key at all.
    assert cfg == {"max_new_tokens": 2048, "do_sample": False, "temperature": 0.0, "top_p": 1.0}
    assert "repetition_penalty" not in cfg


def test_repetition_penalty_applies_to_greedy_and_sampled_calls(tmp_path, monkeypatch):
    """SYNTACTIC_REPETITION_PENALTY=1.15 + SYNTACTIC_BEST_OF_N=2: repetition_penalty must show
    up in BOTH the greedy call's generation_config_override AND the sampled candidate's,
    composed on top of (not replacing) each one's own do_sample/temperature settings."""
    monkeypatch.setenv("SYNTACTIC_REPETITION_PENALTY", "1.15")
    monkeypatch.setenv("SYNTACTIC_BEST_OF_N", "2")

    src = tmp_path / "sample.py"
    original = "def f()\n    return 1\n"
    src.write_text(original)

    row = _row(
        file_hash="overflow_penalty_on",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        idx = len(calls)
        if idx == 1:
            return ("def f(x y):\n", _fake_metrics(), True, 1, 1, "", "")  # greedy: broken
        return ("def f():\n", _fake_metrics(), True, 1, 1, "", "")  # sampled: compiles

    logged = _drive_run_experiment(
        tmp_path, monkeypatch, row, fake_attempt_repair,
        config=_make_runtime_config(max_retries_default=5),
    )

    assert len(calls) == 2, "greedy + 1 sampled candidate"

    greedy_cfg = calls[0].get("generation_config_override")
    assert greedy_cfg["repetition_penalty"] == 1.15
    assert greedy_cfg["do_sample"] is False  # still greedy, not flipped to sampling
    assert greedy_cfg["temperature"] == 0.0

    sampled_cfg = calls[1].get("generation_config_override")
    assert sampled_cfg["repetition_penalty"] == 1.15
    assert sampled_cfg["do_sample"] is True
    assert sampled_cfg["temperature"] == 0.7
    assert sampled_cfg["top_p"] == 0.9

    assert len(logged) == 1
    assert logged[0]["compiled_success"] is True


def test_max_tokens_override_reaches_run_experiment_attempt_repair_calls(tmp_path, monkeypatch):
    """SYNTACTIC_MAX_TOKENS=8000: the value reaches attempt_repair as max_tokens_override on
    both the greedy call and (when triggered) sampled calls."""
    monkeypatch.setenv("SYNTACTIC_MAX_TOKENS", "8000")
    monkeypatch.delenv("SYNTACTIC_REPETITION_PENALTY", raising=False)
    monkeypatch.delenv("SYNTACTIC_BEST_OF_N", raising=False)

    src = tmp_path / "sample.py"
    src.write_text("def f()\n    return 1\n")

    row = _row(
        file_hash="overflow_max_tokens",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        return ("def f():\n", _fake_metrics(), True, 1, 1, "", "")

    _drive_run_experiment(
        tmp_path, monkeypatch, row, fake_attempt_repair,
        config=_make_runtime_config(max_retries_default=1),
    )

    assert len(calls) == 1
    assert calls[0].get("max_tokens_override") == 8000


def test_max_tokens_override_unset_passes_none_to_attempt_repair(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTACTIC_MAX_TOKENS", raising=False)

    src = tmp_path / "sample.py"
    src.write_text("def f()\n    return 1\n")

    row = _row(
        file_hash="overflow_max_tokens_unset",
        file_path=src,
        error_description="SyntaxError: expected ':' (sample.py, line 1)",
    )

    calls: List[Dict[str, Any]] = []

    def fake_attempt_repair(**kwargs):
        calls.append(kwargs)
        return ("def f():\n", _fake_metrics(), True, 1, 1, "", "")

    _drive_run_experiment(
        tmp_path, monkeypatch, row, fake_attempt_repair,
        config=_make_runtime_config(max_retries_default=1),
    )

    assert len(calls) == 1
    assert calls[0].get("max_tokens_override") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
