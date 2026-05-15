from __future__ import annotations

import contextlib
import io
import signal
import unittest
from unittest.mock import patch

from finetuning.model_finetuner_qwen_codeobject_semantic import filter_samples_by_token_budget
from pipeline.code_object_repair_loop import LLMFragmentFixer, SemanticSampleTimeoutError, _semantic_sample_timeout


class CodeObj:
    co_name = "sample"
    co_qualname = "sample"
    co_argcount = 0
    co_posonlyargcount = 0
    co_kwonlyargcount = 0
    co_flags = 0
    co_varnames = ()
    co_names = ()
    co_freevars = ()
    co_cellvars = ()
    co_consts = (None,)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(str(message["content"]) for message in messages)

    def __call__(self, text, add_special_tokens=False, truncation=False):
        del add_special_tokens, truncation
        return {"input_ids": text.split()}


class SemanticTokenLimitTest(unittest.TestCase):
    def test_llm_fragment_fixer_skips_oversized_prompt_before_provider_call(self) -> None:
        fixer = LLMFragmentFixer(provider="Google", model="gemini-2.5-flash-lite")
        source = "def sample():\n    return 1\n"

        output = io.StringIO()
        with patch("pipeline.code_object_repair_loop.count_tokens_safe", return_value=70000), patch(
            "pipeline.code_object_repair_loop.make_llm_call_from_config"
        ) as mocked_call:
            with contextlib.redirect_stdout(output):
                candidate = fixer.generate_candidate(
                    qualname="<module>.sample",
                    gt_code_object=CodeObj(),
                    derived_code_object=CodeObj(),
                    derived_source_fragment=source,
                    repair_context={},
                )

        mocked_call.assert_not_called()
        self.assertEqual(candidate, source)
        self.assertTrue(fixer.calls[-1]["skipped_due_to_token_limit"])
        self.assertEqual(fixer.calls[-1]["prompt_token_count"], 70000)
        self.assertIn("skipping LLM call for <module>.sample", output.getvalue())
        self.assertIn("prompt token count 70000 exceeds threshold", output.getvalue())

    def test_missing_target_token_skip_returns_empty_fragment(self) -> None:
        fixer = LLMFragmentFixer(provider="Google", model="gemini-2.5-flash-lite")

        with patch("pipeline.code_object_repair_loop.count_tokens_safe", return_value=70000), patch(
            "pipeline.code_object_repair_loop.make_llm_call_from_config"
        ) as mocked_call:
            candidate = fixer.generate_candidate(
                qualname="<module>.missing",
                gt_code_object=CodeObj(),
                derived_code_object=None,
                derived_source_fragment="def parent():\n    pass\n",
                repair_context={},
            )

        mocked_call.assert_not_called()
        self.assertEqual(candidate, "")
        self.assertTrue(fixer.calls[-1]["skipped_due_to_token_limit"])

    def test_finetuning_filter_drops_chat_templated_samples_over_max_seq_length(self) -> None:
        samples = [
            {"messages": [{"role": "user", "content": "short prompt"}, {"role": "assistant", "content": "ok"}]},
            {"messages": [{"role": "user", "content": "this prompt has too many tokens"}, {"role": "assistant", "content": "ok"}]},
        ]

        kept = filter_samples_by_token_budget(
            samples,
            FakeTokenizer(),
            max_seq_length=4,
            enable_thinking_template=False,
        )

        self.assertEqual(kept, [samples[0]])

    @unittest.skipUnless(hasattr(signal, "SIGALRM"), "SIGALRM is required for sample timeout tests")
    def test_semantic_sample_timeout_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(SemanticSampleTimeoutError, "sample exceeded timeout of 10 seconds"):
            with _semantic_sample_timeout(10):
                signal.raise_signal(signal.SIGALRM)


if __name__ == "__main__":
    unittest.main()
