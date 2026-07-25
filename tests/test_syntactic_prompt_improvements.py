"""Prompt-lever improvements for the SYNTACTIC-repair local-LLM path
(pipeline/repair_engine.py::make_call_to_local_llm + utils/token_helpers.py).

Data-driven from a 48-failure diagnosis of the local model (Qwen3-Coder-30B):
two dominant, prompt-fixable failure modes.

- BETTER_PROMPT (16/48): the model overflows its output-token budget by
  reformatting compact decompiler output into verbose PEP-8, or invents new
  functionality / completes truncated string literals with imagined content,
  or faithfully preserves decompiler garbage instead of repairing it. This
  file asserts SYSTEM_PROMPT_FOR_LOCAL carries explicit, forceful language
  against each of those failure shapes, including a named taxonomy of
  decompiler artifacts (orphan continue/break, dangling except/finally,
  leaked PEP-695 scaffolding, corrupted identifiers/literals).

- ERROR_FEEDBACK (6/48): on retries, the model repeats the same omission
  because the previous compile error is de-emphasized as "for reference
  only". This file asserts a dedicated retry template
  (USER_PROMPT_TEMPLATE_LOCAL_RETRY) presents the latest compile error as
  AUTHORITATIVE feedback, and that pipeline.repair_engine.make_call_to_local_llm
  selects it only when is_retry=True -- first-attempt behavior (and the
  existing WITH/WITHOUT-explanation template split) is unchanged.

These prompts (SYSTEM_PROMPT_FOR_LOCAL / USER_PROMPT_TEMPLATE_LOCAL*) are
used ONLY by the syntactic-repair local path (pipeline/repair_engine.py);
the semantic engine (pipeline/code_object_repair_loop.py) defines its own
separate SEMANTIC_REPAIR_SYSTEM_PROMPT and does not import these symbols --
verified by grep before writing this test, and re-asserted below so a future
refactor that accidentally shares them gets caught.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.repair_engine import make_call_to_local_llm
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


class SystemPromptSurgicalLanguageTest(unittest.TestCase):
    """BETTER_PROMPT lever: SYSTEM_PROMPT_FOR_LOCAL must forcefully state the
    surgical / never-invent / close-literal-as-is / artifact-taxonomy rules."""

    def setUp(self):
        self.prompt = SYSTEM_PROMPT_FOR_LOCAL.lower()

    def test_surgical_formatting_preservation(self):
        self.assertIn("surgical", self.prompt)
        self.assertIn("compact", self.prompt)
        self.assertIn("pep-8", self.prompt)

    def test_never_invent_functionality(self):
        self.assertIn("never invent", self.prompt)
        self.assertIn("docstring", self.prompt)

    def test_truncated_literal_close_as_is(self):
        self.assertIn("truncated", self.prompt)
        self.assertIn("close it as-is", self.prompt)

    def test_decompiler_artifact_taxonomy_present(self):
        # orphan continue/break under a bare `if` -> restore the `while`
        self.assertIn("orphan", self.prompt)
        self.assertIn("while", self.prompt)
        # dangling except/finally -> needs a try wrapper
        self.assertIn("dangling", self.prompt)
        self.assertIn("try", self.prompt)
        # leaked PEP-695 scaffolding
        self.assertIn("pep-695", self.prompt)
        self.assertIn(".type_params", self.prompt)
        self.assertIn("generic parameters of", self.prompt)
        self.assertIn(".defaults", self.prompt)
        # corrupted identifier / raw newline inside a string literal
        self.assertIn("corrupted", self.prompt)
        self.assertIn("newline", self.prompt)

    def test_original_minimal_change_language_preserved(self):
        # Don't regress the existing "minimal changes" instruction while
        # strengthening the prompt.
        self.assertIn("minimal changes", self.prompt)


class RetryTemplateContentTest(unittest.TestCase):
    """ERROR_FEEDBACK lever: the retry template must carry the exact error
    text and frame it as authoritative feedback from a previous attempt."""

    def test_retry_template_contains_exact_error_and_feedback_framing(self):
        messages = build_chat_messages(
            code_snippet="def f(:\n    pass",
            error_message="SyntaxError: invalid syntax: bad",
            system_prompt=SYSTEM_PROMPT_FOR_LOCAL,
            user_prompt_template=USER_PROMPT_TEMPLATE_LOCAL_RETRY,
            current_explanation="",
        )
        user_content = messages[1]["content"]
        self.assertIn("SyntaxError: invalid syntax: bad", user_content)
        lowered = user_content.lower()
        self.assertIn("previous attempt", lowered)
        self.assertIn("retry", lowered)

    def test_retry_template_is_distinct_from_first_attempt_templates(self):
        self.assertNotEqual(USER_PROMPT_TEMPLATE_LOCAL_RETRY, USER_PROMPT_TEMPLATE_LOCAL)
        self.assertNotEqual(
            USER_PROMPT_TEMPLATE_LOCAL_RETRY, USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION
        )


class MakeCallToLocalLlmRetryWiringTest(unittest.TestCase):
    """First attempt keeps the current template unchanged; only is_retry=True
    switches to the retry template."""

    def _capture_messages(self, **call_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            affected_file_path = Path(tmp) / "case"
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
            written = json.loads((affected_file_path / "last_message_to_llm_for_file.json").read_text())
        return messages, written

    def test_first_attempt_uses_non_retry_template(self):
        messages, _ = self._capture_messages(is_retry=False)
        user_content = messages[1]["content"]
        # The retry-only framing phrase must NOT leak into the first-attempt prompt.
        self.assertNotIn("RETRY", user_content)
        self.assertNotIn("previous attempt", user_content.lower())

    def test_first_attempt_default_is_non_retry(self):
        # is_retry omitted entirely -> must behave exactly like is_retry=False.
        messages, _ = self._capture_messages()
        user_content = messages[1]["content"]
        self.assertNotIn("RETRY", user_content)

    def test_is_retry_true_selects_retry_template(self):
        messages, _ = self._capture_messages(
            is_retry=True, error="SyntaxError: unexpected EOF while parsing: xyz"
        )
        user_content = messages[1]["content"]
        self.assertIn("SyntaxError: unexpected EOF while parsing: xyz", user_content)
        self.assertIn("previous attempt", user_content.lower())

    def test_is_retry_true_overrides_explanation_selection(self):
        # Even with a non-empty current_explanation, is_retry=True must pick
        # the retry template, not USER_PROMPT_TEMPLATE_LOCAL.
        messages, _ = self._capture_messages(
            is_retry=True,
            current_explanation="some root-cause explanation",
            error="SyntaxError: bad thing: qrs",
        )
        user_content = messages[1]["content"]
        self.assertIn("SyntaxError: bad thing: qrs", user_content)
        self.assertIn("previous attempt", user_content.lower())


if __name__ == "__main__":
    unittest.main()
