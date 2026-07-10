"""Tests for the prompt-side gate/formatter of the GT reconstruction brief
(pipeline/code_object_repair_loop.py), added in TDD step 1 before the formatter exists.

These use PLAIN DICT repair_contexts only — no reattach, no LLM, no GPU. They pin:
  * missing targets: brief is shown UNCONDITIONALLY (missing objects carry a
    'Missing bytecode'/None failure message that the structural gate would otherwise drop);
  * matched (present) targets: the brief obeys the SAME structural gate as the existing
    _format_gt_control_flow_skeleton — shown on 'Different control flow' and on
    'Different bytecode' with no failed_offset, suppressed on 'Different bytecode' WITH a
    failed_offset;
  * build_semantic_prompt_summaries splices the brief text into its assembled output, so it
    lands in the messages the token gate counts.
"""
import os
import unittest

from pipeline.code_object_repair_loop import (
    _format_gt_reconstruction_brief,
    build_semantic_prompt_summaries,
)


class _BriefEnabledMixin:
    """Enable the brief for tests that exercise its GATING logic (independent of the master
    default-OFF toggle)."""

    def setUp(self):
        self._prev = os.environ.get("SEMANTIC_RECONSTRUCTION_BRIEF")
        os.environ["SEMANTIC_RECONSTRUCTION_BRIEF"] = "true"

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("SEMANTIC_RECONSTRUCTION_BRIEF", None)
        else:
            os.environ["SEMANTIC_RECONSTRUCTION_BRIEF"] = self._prev


class FormatGateTest(_BriefEnabledMixin, unittest.TestCase):
    def test_format_brief_shown_for_missing_target(self):
        repair_context = {
            "gt_reconstruction_brief": "BRIEF",
            "target_is_missing": True,
            "pylingual_failed_result": {"message": "Missing bytecode"},
            "failed_offset": None,
        }
        self.assertEqual(_format_gt_reconstruction_brief(repair_context), "BRIEF")

    def test_format_brief_gated_by_structural_message_for_matched(self):
        shown_control_flow = {
            "gt_reconstruction_brief": "BRIEF",
            "target_is_missing": False,
            "pylingual_failed_result": {"message": "Different control flow"},
            "failed_offset": None,
        }
        shown_bytecode_no_offset = {
            "gt_reconstruction_brief": "BRIEF",
            "target_is_missing": False,
            "pylingual_failed_result": {"message": "Different bytecode"},
            "failed_offset": None,
        }
        suppressed_bytecode_with_offset = {
            "gt_reconstruction_brief": "BRIEF",
            "target_is_missing": False,
            "pylingual_failed_result": {"message": "Different bytecode"},
            "failed_offset": 42,
        }
        self.assertEqual(_format_gt_reconstruction_brief(shown_control_flow), "BRIEF")
        self.assertEqual(_format_gt_reconstruction_brief(shown_bytecode_no_offset), "BRIEF")
        self.assertEqual(_format_gt_reconstruction_brief(suppressed_bytecode_with_offset), "")


class AssembledSummariesTest(_BriefEnabledMixin, unittest.TestCase):
    def test_brief_appears_in_assembled_summaries(self):
        repair_context = {
            "gt_reconstruction_brief": "BRIEFTEXT",
            "pylingual_failed_result": {"message": "Different control flow"},
            "failed_offset": None,
        }
        summaries = build_semantic_prompt_summaries(
            qualname="<module>.foo",
            gt_code_object=None,
            derived_code_object=object(),
            repair_context=repair_context,
        )
        self.assertIn("BRIEFTEXT", summaries)


class ReconstructionBriefToggleTest(unittest.TestCase):
    """The SEMANTIC_RECONSTRUCTION_BRIEF env toggle. Default OFF (an A/B showed no file-perfect
    gain); the brief is emitted only when the env var is explicitly truthy."""

    def _ctx(self):
        return {"gt_reconstruction_brief": "BRIEF", "target_is_missing": True}

    def test_toggle_off_and_default_suppress_brief(self):
        prev = os.environ.get("SEMANTIC_RECONSTRUCTION_BRIEF")
        try:
            # explicit false-y values -> suppressed
            for val in ("false", "0", "no", "off", ""):
                os.environ["SEMANTIC_RECONSTRUCTION_BRIEF"] = val
                self.assertEqual(_format_gt_reconstruction_brief(self._ctx()), "")
            # DEFAULT (unset) -> suppressed (default OFF)
            os.environ.pop("SEMANTIC_RECONSTRUCTION_BRIEF", None)
            self.assertEqual(_format_gt_reconstruction_brief(self._ctx()), "")
        finally:
            if prev is None:
                os.environ.pop("SEMANTIC_RECONSTRUCTION_BRIEF", None)
            else:
                os.environ["SEMANTIC_RECONSTRUCTION_BRIEF"] = prev

    def test_toggle_on_shows_brief(self):
        prev = os.environ.get("SEMANTIC_RECONSTRUCTION_BRIEF")
        try:
            for val in ("1", "true", "yes", "on"):
                os.environ["SEMANTIC_RECONSTRUCTION_BRIEF"] = val
                self.assertEqual(_format_gt_reconstruction_brief(self._ctx()), "BRIEF")
        finally:
            if prev is None:
                os.environ.pop("SEMANTIC_RECONSTRUCTION_BRIEF", None)
            else:
                os.environ["SEMANTIC_RECONSTRUCTION_BRIEF"] = prev


if __name__ == "__main__":
    unittest.main()
