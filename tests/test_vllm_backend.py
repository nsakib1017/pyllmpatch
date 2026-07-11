"""Lever B seam: a feature-flagged vLLM inference backend that serves the SAME local merged
Qwen model (offline, in-process — no external API, compliant with the local-only rule) to
speed the dominant generation step with no quality drop. These tests pin the parts that are
pure and testable without vLLM installed or a GPU: backend selection from env, and the
HF->vLLM sampling-parameter mapping (greedy must map to temperature 0.0 so decoding matches
the HF greedy path). The actual LLM load/generate is import-guarded and covered by the
deferred GPU smoke-test."""
import os
import unittest
from unittest import mock

from model.vllm_backend import build_sampling_params, select_inference_backend


class SelectInferenceBackendTest(unittest.TestCase):
    # NB: distinct from loader.py's SEMANTIC_INFERENCE_BACKEND (unsloth|transformers) — the
    # engine selector (vllm|hf) is orthogonal to how the HF loader loads a model.
    def test_defaults_to_hf_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEMANTIC_LLM_ENGINE", None)
            self.assertEqual(select_inference_backend(), "hf")

    def test_vllm_when_set(self):
        with mock.patch.dict(os.environ, {"SEMANTIC_LLM_ENGINE": "vLLM"}):
            self.assertEqual(select_inference_backend(), "vllm")

    def test_unknown_value_falls_back_to_hf(self):
        with mock.patch.dict(os.environ, {"SEMANTIC_LLM_ENGINE": "sglang-typo"}):
            self.assertEqual(select_inference_backend(), "hf")


class BuildSamplingParamsTest(unittest.TestCase):
    def test_greedy_maps_to_temperature_zero(self):
        params = build_sampling_params({}, max_tokens=4096)
        self.assertEqual(params["temperature"], 0.0, "greedy must be temperature 0.0 to match HF do_sample=False")
        # default new-token cap mirrors the HF path: min(max_tokens, 2048)
        self.assertEqual(params["max_tokens"], 2048)

    def test_max_new_tokens_is_capped_by_max_tokens(self):
        params = build_sampling_params({"max_new_tokens": 100000}, max_tokens=512)
        self.assertEqual(params["max_tokens"], 512)

    def test_explicit_max_new_tokens_honoured(self):
        params = build_sampling_params({"max_new_tokens": 256}, max_tokens=4096)
        self.assertEqual(params["max_tokens"], 256)

    def test_sampling_passes_through_decoding_params(self):
        cfg = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.1,
        }
        params = build_sampling_params(cfg, max_tokens=4096)
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_p"], 0.95)
        self.assertEqual(params["top_k"], 20)
        self.assertEqual(params["repetition_penalty"], 1.1)

    def test_absent_top_k_is_disabled(self):
        params = build_sampling_params({"do_sample": True, "temperature": 0.8}, max_tokens=4096)
        # vLLM disables top_k with -1; never emit a bogus 0/None
        self.assertEqual(params.get("top_k", -1), -1)


if __name__ == "__main__":
    unittest.main()
