"""Tests for the deterministic-signal prompt enrichment (pipeline contract part 2):
_format_structural_repair_directive + _format_diff_chunk_diagnosis, gated behind the
SEMANTIC_STRUCTURAL_DIRECTIVE flag (default off for A/B)."""
import os
import unittest
from unittest import mock

from pipeline.code_object_repair_loop import (
    _format_structural_repair_directive,
    _format_diff_chunk_diagnosis,
)

_STRUCT_CTX = {
    "structural_repair_directive": "Structural fix: wrap the body in a for-loop.",
    "diff_chunk_diagnosis": "Bytecode divergence diagnosis (1 localized change): "
                            "the ground truth has a call arguments your version dropped "
                            "[GT bytecode: LOAD_GLOBAL UObject, KW_NAMES ('sdk',)].",
    "pylingual_failed_result": {"message": "Different control flow"},
    "failed_offset": None,
}


def _flag(on: bool):
    return mock.patch.dict(os.environ, {"SEMANTIC_STRUCTURAL_DIRECTIVE": "true" if on else "false"})


class StructuralDirectivePromptTest(unittest.TestCase):
    def test_flag_off_by_default_returns_empty(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEMANTIC_STRUCTURAL_DIRECTIVE", None)
            self.assertEqual(_format_structural_repair_directive(_STRUCT_CTX), "")
            self.assertEqual(_format_diff_chunk_diagnosis(_STRUCT_CTX), "")

    def test_flag_off_returns_empty(self):
        with _flag(False):
            self.assertEqual(_format_structural_repair_directive(_STRUCT_CTX), "")
            self.assertEqual(_format_diff_chunk_diagnosis(_STRUCT_CTX), "")

    def test_directive_present_structural_message_flag_on(self):
        with _flag(True):
            self.assertEqual(_format_structural_repair_directive(_STRUCT_CTX),
                             "Structural fix: wrap the body in a for-loop.")

    def test_directive_gated_off_on_localized_operand_failure(self):
        # "Different bytecode" WITH a single failed_offset = an operand swap, not structural;
        # the structural directive must NOT fire (mirrors the skeleton gate).
        ctx = {**_STRUCT_CTX, "pylingual_failed_result": {"message": "Different bytecode"},
               "failed_offset": 42}
        with _flag(True):
            self.assertEqual(_format_structural_repair_directive(ctx), "")

    def test_directive_absent_returns_empty(self):
        ctx = {**_STRUCT_CTX, "structural_repair_directive": None}
        with _flag(True):
            self.assertEqual(_format_structural_repair_directive(ctx), "")

    def test_diagnosis_present_flag_on_any_message(self):
        with _flag(True):
            out = _format_diff_chunk_diagnosis(_STRUCT_CTX)
            self.assertIn("call arguments your version dropped", out)
        # also fires for a plain "Different bytecode" operand case (diagnosis is always useful)
        ctx = {**_STRUCT_CTX, "pylingual_failed_result": {"message": "Different bytecode"},
               "failed_offset": 42}
        with _flag(True):
            self.assertTrue(_format_diff_chunk_diagnosis(ctx))

    def test_diagnosis_absent_returns_empty(self):
        ctx = {**_STRUCT_CTX, "diff_chunk_diagnosis": ""}
        with _flag(True):
            self.assertEqual(_format_diff_chunk_diagnosis(ctx), "")

    def test_none_context_returns_empty(self):
        with _flag(True):
            self.assertEqual(_format_structural_repair_directive(None), "")
            self.assertEqual(_format_diff_chunk_diagnosis(None), "")


if __name__ == "__main__":
    unittest.main()
