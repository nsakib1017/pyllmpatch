"""TDD spec for the CONTROL-FLOW/STRUCTURE enrichment appended to
utils.gt_syntactic_context.build_gt_object_context.

A residual diagnosis of syntactic-repair failures found cases (3.13/3.15-heavy, PEP-695
generics) where the bare signature brief (`def cc_zip(*, strict):  # defaults/annotations
unknown`) was NOT enough for the LLM to re-nest a decompiler-flattened body -- it also needs
the object's TRUE control-flow structure. This suite pins the enriched shape using two real
rows from dataset/syntactic_corpus_gt.csv:

- `790eeb56fd40...` (3.13): its `smooth` function has real GT loop back-edges (the
  decompiler flattened an if/assert/if/else chain into one dead-straight block --
  literally the "unexpected indent" residual case this enrichment targets).
- `42539768a344...` (3.13): the existing cc_zip row. Its `cc_zip` GT code object is
  straight-line (no branches/loops/try) -- the graceful-degradation case where enrichment
  must add nothing while leaving the signature-only brief untouched.
"""
from __future__ import annotations

import csv
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.gt_syntactic_context import build_gt_object_context

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_CSV = REPO_ROOT / "dataset" / "syntactic_corpus_gt.csv"

CONTROL_FLOW_HASH_PREFIX = "790eeb56fd40"
CONTROL_FLOW_ERROR_LINE = 26  # the row's own recorded SyntaxError line, inside `smooth`

TRIVIAL_HASH_PREFIX = "42539768a344"
TRIVIAL_ERROR_LINE = 23  # same error line used by tests/test_gt_syntactic_context.py


def _find_row(hash_prefix: str) -> dict:
    with open(CORPUS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["file_hash"].startswith(hash_prefix):
                return row
    raise AssertionError(f"no row in {CORPUS_CSV} with file_hash prefix {hash_prefix}")


def _row_available(hash_prefix: str):
    """Returns (row, reason_unavailable); reason is None if usable."""
    try:
        row = _find_row(hash_prefix)
    except AssertionError as exc:
        return None, str(exc)
    src_path = Path(row["file_path"])
    gt_path = Path(row["gt_pyc"])
    if not src_path.is_file():
        return None, f"broken source missing on disk: {src_path}"
    if not gt_path.is_file():
        return None, f"gt pyc missing on disk: {gt_path}"
    return row, None


_CF_ROW, _CF_UNAVAILABLE = _row_available(CONTROL_FLOW_HASH_PREFIX)
_TRIVIAL_ROW, _TRIVIAL_UNAVAILABLE = _row_available(TRIVIAL_HASH_PREFIX)


class ControlFlowEnrichmentTest(unittest.TestCase):
    """The positive case: a GT object with real control flow gets BOTH the signature and the
    new structure section."""

    @unittest.skipIf(_CF_ROW is None, f"control-flow corpus row unavailable: {_CF_UNAVAILABLE}")
    def test_brief_contains_signature_and_control_flow_section(self):
        row = _CF_ROW
        broken_source = Path(row["file_path"]).read_text(encoding="utf-8")

        result = build_gt_object_context(broken_source, CONTROL_FLOW_ERROR_LINE, row["gt_pyc"])

        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        # Signature line is still present -- enrichment is additive, never a replacement.
        self.assertIn("def smooth(", result)
        # The new control-flow/structure section: header + at least one structural token.
        self.assertIn("Control flow", result)
        self.assertTrue(
            any(tok in result for tok in ("loop", "branch", "try", "for-loop")),
            f"expected a structural token in the control-flow section, got:\n{result}",
        )
        # The signature line must come BEFORE the control-flow section (additive, "below").
        self.assertLess(result.index("def smooth("), result.index("Control flow"))

    @unittest.skipIf(_CF_ROW is None, f"control-flow corpus row unavailable: {_CF_UNAVAILABLE}")
    def test_control_flow_section_is_bounded(self):
        row = _CF_ROW
        broken_source = Path(row["file_path"]).read_text(encoding="utf-8")

        result = build_gt_object_context(broken_source, CONTROL_FLOW_ERROR_LINE, row["gt_pyc"])

        # Never let the enrichment bloat the prompt unboundedly.
        self.assertLess(len(result), 2000)
        cf_section = result.split("Control flow", 1)[1]
        self.assertLessEqual(cf_section.count("\n- "), 24)


class GracefulDegradationTest(unittest.TestCase):
    """No/empty control flow (or a misbehaving generator) must never cost the caller the
    signature-only brief, and must never raise."""

    @unittest.skipIf(_TRIVIAL_ROW is None, f"trivial corpus row unavailable: {_TRIVIAL_UNAVAILABLE}")
    def test_trivial_control_flow_still_yields_signature_only_brief(self):
        row = _TRIVIAL_ROW
        broken_source = Path(row["file_path"]).read_text(encoding="utf-8")

        result = build_gt_object_context(broken_source, TRIVIAL_ERROR_LINE, row["gt_pyc"])

        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        self.assertIn("cc_zip", result)
        # cc_zip's GT body is straight-line: no control-flow section should be appended.
        self.assertNotIn("Control flow", result)

    @unittest.skipIf(_CF_ROW is None, f"control-flow corpus row unavailable: {_CF_UNAVAILABLE}")
    def test_control_flow_generator_raising_falls_back_to_signature_only(self):
        row = _CF_ROW
        broken_source = Path(row["file_path"]).read_text(encoding="utf-8")

        with patch(
            "utils.reattach_source_code_object._gt_control_flow_brief",
            side_effect=RuntimeError("boom"),
        ):
            result = build_gt_object_context(broken_source, CONTROL_FLOW_ERROR_LINE, row["gt_pyc"])

        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        self.assertIn("def smooth(", result)
        self.assertNotIn("Control flow", result)

    @unittest.skipIf(_CF_ROW is None, f"control-flow corpus row unavailable: {_CF_UNAVAILABLE}")
    def test_control_flow_generator_returning_empty_falls_back_to_signature_only(self):
        row = _CF_ROW
        broken_source = Path(row["file_path"]).read_text(encoding="utf-8")

        with patch(
            "utils.reattach_source_code_object._gt_control_flow_brief",
            return_value="",
        ):
            result = build_gt_object_context(broken_source, CONTROL_FLOW_ERROR_LINE, row["gt_pyc"])

        self.assertNotEqual(result, "")
        self.assertIn("def smooth(", result)
        self.assertNotIn("Control flow", result)


class UnchangedFailureModesTest(unittest.TestCase):
    """Enrichment must not touch any of the pre-existing failure-mode contracts."""

    @unittest.skipIf(_TRIVIAL_ROW is None, f"trivial corpus row unavailable: {_TRIVIAL_UNAVAILABLE}")
    def test_empty_gt_pyc_path_still_returns_empty_string(self):
        broken_source = Path(_TRIVIAL_ROW["file_path"]).read_text(encoding="utf-8")
        self.assertEqual(build_gt_object_context(broken_source, TRIVIAL_ERROR_LINE, ""), "")

    @unittest.skipIf(_TRIVIAL_ROW is None, f"trivial corpus row unavailable: {_TRIVIAL_UNAVAILABLE}")
    def test_nonexistent_gt_pyc_path_still_returns_empty_string(self):
        broken_source = Path(_TRIVIAL_ROW["file_path"]).read_text(encoding="utf-8")
        result = build_gt_object_context(broken_source, TRIVIAL_ERROR_LINE, "/nonexistent/path/x.pyc")
        self.assertEqual(result, "")

    def test_module_scope_error_still_returns_empty_string(self):
        result = build_gt_object_context("x = f(1, 2\n", 1, "/tmp/whatever-does-not-exist.pyc")
        self.assertEqual(result, "")

    def test_never_raises_on_garbage_inputs(self):
        try:
            result = build_gt_object_context("", 0, "")
            self.assertEqual(result, "")
            result = build_gt_object_context("not python (((", 1, "/tmp/does-not-exist.pyc")
            self.assertEqual(result, "")
        except Exception as exc:  # pragma: no cover - the whole point of the test
            self.fail(f"build_gt_object_context raised {exc!r} instead of returning \"\"")


if __name__ == "__main__":
    unittest.main()
