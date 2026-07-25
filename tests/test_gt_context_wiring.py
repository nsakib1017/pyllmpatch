"""GT-context wiring for the SYNTACTIC repair prompt.

``utils/gt_syntactic_context.py::build_gt_object_context`` already extracts a short brief
of the TRUE structure (from the ground-truth ``.pyc``) of the code object enclosing a
syntax error. This file tests threading that brief -- as an optional ``gt_context`` string
-- into the local-LLM prompt-building chain:

    pipeline.runner.run_experiment (reads row["gt_pyc"], gated by
    SYNTACTIC_GT_CONTEXT)
        -> pipeline.repair_engine.attempt_repair(gt_context=...)
        -> pipeline.repair_engine.make_call_to_local_llm(gt_context=...)
        -> utils.token_helpers.build_chat_messages(gt_context=...)
        -> USER_PROMPT_TEMPLATE_LOCAL / _WITHOUT_EXPLANATION / _RETRY

The flag defaults OFF (``syntactic_gt_context_enabled()`` is False when the env var is
unset) and an empty ``gt_context`` must render a user prompt byte-identical to today's
(no dangling header, no empty section) -- both asserted below.

No GPU / no real model: the LLM call is mocked at ``model.inference.call_llm_with_message``.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.repair_engine import make_call_to_local_llm
from pipeline.runner import syntactic_gt_context_enabled
from utils.token_helpers import (
    SYSTEM_PROMPT_FOR_LOCAL,
    USER_PROMPT_TEMPLATE_LOCAL,
    USER_PROMPT_TEMPLATE_LOCAL_RETRY,
    USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION,
    build_chat_messages,
)

MODEL_PATH = "/models/qwen"
TOKENIZER_PATH = "/models/qwen-tok"
MAX_TOKENS = 512

GT_BRIEF = "def cc_zip(*, strict):"
FRAMING = "reconstruct to match"


class BuildChatMessagesGtContextTest(unittest.TestCase):
    """gt_context, when non-empty, must appear in the rendered user message along with
    the "reconstruct to match" authoritative framing; when empty, the rendering must be
    byte-identical to not passing gt_context at all (no header, no dangling section)."""

    TEMPLATES = {
        "LOCAL_WITHOUT_EXPLANATION": USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION,
        "LOCAL_RETRY": USER_PROMPT_TEMPLATE_LOCAL_RETRY,
    }

    def test_nonempty_gt_context_present_with_framing(self):
        for name, template in self.TEMPLATES.items():
            with self.subTest(template=name):
                messages = build_chat_messages(
                    code_snippet="def <generic parameters of cc_zip>(.defaults):\n    pass",
                    error_message="SyntaxError: invalid syntax",
                    system_prompt=SYSTEM_PROMPT_FOR_LOCAL,
                    user_prompt_template=template,
                    gt_context=GT_BRIEF,
                )
                user_content = messages[1]["content"]
                self.assertIn(GT_BRIEF, user_content)
                self.assertIn(FRAMING, user_content.lower())

    def test_empty_gt_context_identical_to_no_context(self):
        for name, template in self.TEMPLATES.items():
            with self.subTest(template=name):
                kwargs = dict(
                    code_snippet="def <generic parameters of cc_zip>(.defaults):\n    pass",
                    error_message="SyntaxError: invalid syntax",
                    system_prompt=SYSTEM_PROMPT_FOR_LOCAL,
                    user_prompt_template=template,
                )
                with_empty = build_chat_messages(gt_context="", **kwargs)
                without_arg = build_chat_messages(**kwargs)

                self.assertEqual(with_empty, without_arg)

                user_content = with_empty[1]["content"]
                self.assertNotIn(FRAMING, user_content.lower())
                self.assertNotIn("ground-truth structure", user_content.lower())
                self.assertNotIn(GT_BRIEF, user_content)

    def test_explanation_template_also_supports_gt_context(self):
        # USER_PROMPT_TEMPLATE_LOCAL requires a non-empty current_explanation (it has a
        # dedicated {current_explanation} slot); verify gt_context composes with it too.
        messages = build_chat_messages(
            code_snippet="def <generic parameters of cc_zip>(.defaults):\n    pass",
            error_message="SyntaxError: invalid syntax",
            system_prompt=SYSTEM_PROMPT_FOR_LOCAL,
            user_prompt_template=USER_PROMPT_TEMPLATE_LOCAL,
            current_explanation="The generic-parameter scaffolding is decompiler leakage.",
            gt_context=GT_BRIEF,
        )
        user_content = messages[1]["content"]
        self.assertIn(GT_BRIEF, user_content)
        self.assertIn(FRAMING, user_content.lower())
        self.assertIn("decompiler leakage", user_content)


class SyntacticGtContextEnabledFlagTest(unittest.TestCase):
    def test_false_when_env_unset(self):
        env = dict(os.environ)
        env.pop("SYNTACTIC_GT_CONTEXT", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(syntactic_gt_context_enabled())

    def test_true_when_env_set_to_1(self):
        with mock.patch.dict(os.environ, {"SYNTACTIC_GT_CONTEXT": "1"}):
            self.assertTrue(syntactic_gt_context_enabled())

    def test_false_when_env_set_to_0(self):
        with mock.patch.dict(os.environ, {"SYNTACTIC_GT_CONTEXT": "0"}):
            self.assertFalse(syntactic_gt_context_enabled())


class MakeCallToLocalLlmGtContextWiringTest(unittest.TestCase):
    """make_call_to_local_llm must accept an optional gt_context and forward it all the
    way into the messages built for the LLM call (in-process path; no GPU/model)."""

    def _capture_messages(self, **call_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            affected_file_path = Path(tmp) / "case"
            env = dict(os.environ)
            env.pop("SEMANTIC_VLLM_SERVER_URL", None)
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch(
                    "model.inference.call_llm_with_message", return_value="FIXED"
                ) as inproc:
                    make_call_to_local_llm(
                        call_kwargs.pop("content", "def f(:\n    pass"),
                        call_kwargs.pop("error", "SyntaxError: invalid syntax"),
                        call_kwargs.pop("current_explanation", ""),
                        affected_file_path,
                        MODEL_PATH,
                        MAX_TOKENS,
                        TOKENIZER_PATH,
                        call_kwargs.pop("generation_config", None),
                        **call_kwargs,
                    )
                    messages = inproc.call_args.kwargs["messages"]
        return messages

    def test_forwards_nonempty_gt_context_into_messages(self):
        messages = self._capture_messages(gt_context=GT_BRIEF)
        user_content = messages[1]["content"]
        self.assertIn(GT_BRIEF, user_content)
        self.assertIn(FRAMING, user_content.lower())

    def test_default_gt_context_is_empty_and_unchanged(self):
        # gt_context omitted entirely -> must behave exactly like the pre-existing,
        # no-gt-context rendering (no framing string leaked in).
        messages = self._capture_messages()
        user_content = messages[1]["content"]
        self.assertNotIn(FRAMING, user_content.lower())
        self.assertNotIn("ground-truth structure", user_content.lower())


if __name__ == "__main__":
    unittest.main()
