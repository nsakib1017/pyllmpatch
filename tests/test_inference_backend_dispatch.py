"""call_llm_with_message dispatches to the selected inference backend: vLLM when enabled,
HF otherwise, and it MUST fall back to HF if vLLM is unavailable — so turning the flag on can
never break a run, only speed it up. Routing is tested with the two backend implementations
mocked so no model is loaded."""
import unittest
from unittest import mock

from model import inference
from model.vllm_backend import VllmUnavailable

_ARGS = dict(messages=[{"role": "user", "content": "hi"}], model_path="/m", max_tokens=1024)


class InferenceBackendDispatchTest(unittest.TestCase):
    def test_hf_backend_used_by_default(self):
        with mock.patch.object(inference, "select_inference_backend", return_value="hf"), \
             mock.patch.object(inference, "_hf_generate", return_value="HF") as hf, \
             mock.patch.object(inference, "vllm_generate") as vllm:
            self.assertEqual(inference.call_llm_with_message(**_ARGS), "HF")
        hf.assert_called_once()
        vllm.assert_not_called()

    def test_vllm_backend_used_when_selected(self):
        with mock.patch.object(inference, "select_inference_backend", return_value="vllm"), \
             mock.patch.object(inference, "_hf_generate") as hf, \
             mock.patch.object(inference, "vllm_generate", return_value="VLLM") as vllm:
            self.assertEqual(inference.call_llm_with_message(**_ARGS), "VLLM")
        vllm.assert_called_once()
        hf.assert_not_called()

    def test_vllm_unavailable_falls_back_to_hf(self):
        with mock.patch.object(inference, "select_inference_backend", return_value="vllm"), \
             mock.patch.object(inference, "_hf_generate", return_value="HF") as hf, \
             mock.patch.object(inference, "vllm_generate", side_effect=VllmUnavailable("no vllm")) as vllm:
            self.assertEqual(inference.call_llm_with_message(**_ARGS), "HF")
        vllm.assert_called_once()
        hf.assert_called_once()


if __name__ == "__main__":
    unittest.main()
