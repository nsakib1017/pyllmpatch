"""Server-client seam (speedup step 3): when SEMANTIC_VLLM_SERVER_URL is set, a local-Qwen
config (one with a ``model_path``) routes its generate call to a LOCAL ``vllm serve`` OpenAI-
compatible endpoint on ``localhost`` — nothing leaves the box, so the local-Qwen-only rule holds
— instead of building the in-process vLLM engine. This is the transport swap that lets N worker
processes share ONE server, so vLLM continuous-batches independent files on the one GPU (the
cross-file batching that the current batch-1 loop leaves on the table).

Invariants pinned here (all pure, no GPU / no server):
  * Unset ``SEMANTIC_VLLM_SERVER_URL`` == the old in-process path (fully reversible).
  * Cloud providers (OpenAI/DeepSeek/Google) are unaffected by the server URL (local-only intact).
  * Greedy (do_sample False) maps to ``temperature=0.0`` so server decoding matches the offline/
    HF greedy path; sampling extras (top_p / top_k / repetition_penalty) pass through.
  * The OpenAI client is built once per base_url and reused (not the Azure/DeepSeek clients).
"""
import os
import unittest
from unittest import mock

import utils.providers as providers


LOCAL_CFG = {
    "provider": "Alibaba",
    "name": "qwen3-coder-30b",
    "model_path": "/models/qwen",
    "token_for_completion": 512,
    "generation_config": {"do_sample": False, "temperature": 0.0},
}
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def _fake_completion(text="hello"):
    msg = mock.Mock()
    msg.content = text
    choice = mock.Mock()
    choice.message = msg
    comp = mock.Mock()
    comp.choices = [choice]
    comp.usage = {"total_tokens": 3}
    return comp


class ServerSeamRoutingTest(unittest.TestCase):
    def test_routes_to_local_server_when_url_set(self):
        with mock.patch.dict(os.environ, {"SEMANTIC_VLLM_SERVER_URL": "http://localhost:8001/v1"}):
            with mock.patch.object(
                providers, "call_local_vllm_server", return_value={"content": "X", "usage": None}
            ) as srv, mock.patch("model.inference.call_llm_with_message") as inproc:
                out = providers.make_llm_call_from_config(MESSAGES, LOCAL_CFG)
        self.assertEqual(out["content"], "X")
        srv.assert_called_once()
        _, kw = srv.call_args
        self.assertEqual(kw["base_url"], "http://localhost:8001/v1")
        inproc.assert_not_called()  # in-process engine must NOT be built when the server is used

    def test_in_process_when_url_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEMANTIC_VLLM_SERVER_URL", None)
            with mock.patch("model.inference.call_llm_with_message", return_value="Y") as inproc, \
                    mock.patch.object(providers, "call_local_vllm_server") as srv:
                out = providers.make_llm_call_from_config(MESSAGES, LOCAL_CFG)
        self.assertEqual(out["content"], "Y")
        inproc.assert_called_once()
        srv.assert_not_called()

    def test_cloud_provider_ignores_server_url(self):
        cloud = {"provider": "OpenAI", "name": "gpt-5.5-1", "token_for_completion": 512}
        with mock.patch.dict(os.environ, {"SEMANTIC_VLLM_SERVER_URL": "http://localhost:8001/v1"}):
            with mock.patch.object(providers, "make_llm_call", return_value={"content": "Z"}) as cloudcall, \
                    mock.patch.object(providers, "call_local_vllm_server") as srv:
                out = providers.make_llm_call_from_config(MESSAGES, cloud)
        self.assertEqual(out["content"], "Z")
        cloudcall.assert_called_once()
        srv.assert_not_called()  # never route a cloud provider to the local server


class CallLocalVllmServerTest(unittest.TestCase):
    def test_greedy_maps_to_temperature_zero_localhost(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = _fake_completion("RESULT")
        with mock.patch.object(providers, "_local_vllm_client", return_value=fake_client) as factory:
            out = providers.call_local_vllm_server(
                MESSAGES,
                base_url="http://localhost:8001/v1",
                model="repair-model",
                max_tokens=512,
                generation_config={"do_sample": False},
            )
        self.assertEqual(out["content"], "RESULT")
        factory.assert_called_once_with("http://localhost:8001/v1")
        _, kw = fake_client.chat.completions.create.call_args
        self.assertEqual(kw["model"], "repair-model")
        self.assertEqual(kw["temperature"], 0.0)
        self.assertEqual(kw["max_tokens"], 512)
        self.assertEqual(kw["messages"], MESSAGES)
        self.assertFalse(kw.get("stream", False))

    def test_sampling_passes_top_p_and_extra_body(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = _fake_completion("S")
        with mock.patch.object(providers, "_local_vllm_client", return_value=fake_client):
            providers.call_local_vllm_server(
                MESSAGES,
                base_url="http://localhost:8001/v1",
                model="m",
                max_tokens=256,
                generation_config={
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 20,
                    "repetition_penalty": 1.1,
                },
            )
        _, kw = fake_client.chat.completions.create.call_args
        self.assertEqual(kw["temperature"], 0.7)
        self.assertEqual(kw["top_p"], 0.95)
        self.assertEqual(kw["extra_body"]["top_k"], 20)
        self.assertEqual(kw["extra_body"]["repetition_penalty"], 1.1)

    def test_client_cached_by_base_url(self):
        providers._LOCAL_VLLM_CLIENTS.clear()
        with mock.patch.object(providers, "OpenAI", return_value=mock.Mock()) as ctor:
            c1 = providers._local_vllm_client("http://localhost:9/v1")
            c2 = providers._local_vllm_client("http://localhost:9/v1")
        self.assertIs(c1, c2)
        ctor.assert_called_once()
        _, kw = ctor.call_args
        self.assertEqual(kw["base_url"], "http://localhost:9/v1")


if __name__ == "__main__":
    unittest.main()
