"""Syntactic-repair pipeline routing to a shared local ``vllm serve`` endpoint, mirroring the
semantic pipeline's transport swap (utils/providers.py::make_llm_call_from_config /
call_local_vllm_server). pipeline/repair_engine.py::make_call_to_local_llm currently calls
model/inference.py::call_llm_with_message DIRECTLY (in-process HF/vLLM engine), which requires
``vllm`` installed in the current env; the main pyllmpatch env does not have it. When
SEMANTIC_VLLM_SERVER_URL is set, the syntactic path must route through
utils/providers.py::call_local_vllm_server (an OpenAI-compatible HTTP call to a LOCAL vllm-serve
endpoint) instead, so it can share the same running server as the semantic pipeline. Unset env ==
the old in-process path, unchanged (all pure, no GPU / no server; both branches are monkeypatched).
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import utils.providers as providers
from pipeline.repair_engine import make_call_to_local_llm


MODEL_PATH = "/models/qwen"
TOKENIZER_PATH = "/models/qwen-tok"
MAX_TOKENS = 512
GEN_CFG = {"do_sample": False, "temperature": 0.0}


class SyntacticLocalLlmServerRoutingTest(unittest.TestCase):
    def test_routes_to_local_server_when_url_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            affected_file_path = Path(tmp) / "case"
            with mock.patch.dict(os.environ, {"SEMANTIC_VLLM_SERVER_URL": "http://127.0.0.1:8001/v1"}):
                with mock.patch.object(
                    providers,
                    "call_local_vllm_server",
                    return_value={"content": "REPAIRED_VIA_SERVER", "usage": None},
                ) as srv, mock.patch("model.inference.call_llm_with_message") as inproc:
                    out = make_call_to_local_llm(
                        "def f(:\n    pass",
                        "SyntaxError: invalid syntax",
                        "",
                        affected_file_path,
                        MODEL_PATH,
                        MAX_TOKENS,
                        TOKENIZER_PATH,
                        GEN_CFG,
                    )

        self.assertEqual(out, "REPAIRED_VIA_SERVER")
        srv.assert_called_once()
        _, kw = srv.call_args
        self.assertEqual(kw["base_url"], "http://127.0.0.1:8001/v1")
        inproc.assert_not_called()  # in-process engine must NOT be built when the server is used

    def test_in_process_when_url_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            affected_file_path = Path(tmp) / "case"
            env = dict(os.environ)
            env.pop("SEMANTIC_VLLM_SERVER_URL", None)
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch(
                    "model.inference.call_llm_with_message", return_value="REPAIRED_IN_PROCESS"
                ) as inproc, mock.patch.object(providers, "call_local_vllm_server") as srv:
                    out = make_call_to_local_llm(
                        "def f(:\n    pass",
                        "SyntaxError: invalid syntax",
                        "",
                        affected_file_path,
                        MODEL_PATH,
                        MAX_TOKENS,
                        TOKENIZER_PATH,
                    )

        self.assertEqual(out, "REPAIRED_IN_PROCESS")
        inproc.assert_called_once()
        srv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
