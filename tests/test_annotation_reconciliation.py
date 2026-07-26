"""The gt-source annotation-reconcile operator was REMOVED so that NO ground-truth SOURCE is
read in the deployable LLM path — that path uses only gt/derived BYTECODE + derived source.

Retained (and covered by tests/test_annotate_target_skip.py): the gt-source-FREE `__annotate__`
target *skip* (`_is_synthetic_annotation_qualname`), which only filters synthetic annotate objects
so one can't abort the whole file. It reads no source.

Removed: the flag `SEMANTIC_ANNOTATION_RECONCILE` and the two gt-source helpers it drove
(`_annotate_enclosing_qualname`, `_gt_def_source_by_qualname`) plus the reconcile branch that
spliced the verbatim GT `def` source for a diverged/missing `__annotate__` object.
"""
import inspect
import tempfile
import unittest
from pathlib import Path

import utils.reattach_source_code_object as R
from utils.reattach_source_code_object import repair_mismatching_code_objects


class AnnotationReconcileOperatorRemovedTest(unittest.TestCase):
    def test_gt_source_reconcile_symbols_removed(self):
        self.assertFalse(hasattr(R, "SEMANTIC_ANNOTATION_RECONCILE"))
        self.assertFalse(hasattr(R, "_annotate_enclosing_qualname"))
        self.assertFalse(hasattr(R, "_gt_def_source_by_qualname"))

    def test_gt_source_free_skip_retained(self):
        self.assertTrue(hasattr(R, "_is_synthetic_annotation_qualname"))
        self.assertTrue(R._is_synthetic_annotation_qualname("<module>.Config.__annotate__"))
        self.assertFalse(R._is_synthetic_annotation_qualname("<module>.Config"))

    def test_no_reconcile_operator_left_in_source(self):
        src = inspect.getsource(R)
        self.assertNotIn("annotation_reconcile", src)
        self.assertNotIn("_gt_def_source_by_qualname", src)
        self.assertNotIn("_annotate_enclosing_qualname", src)
        self.assertNotIn("SEMANTIC_ANNOTATION_RECONCILE", src)


FX_MISSING = Path(__file__).parent / "fixtures" / "annotation" / "missing"


def _no_candidates_fixer(*_args, **_kwargs):
    """A non-None fragment_fixer (i.e. LLM deployment) that always abstains."""
    return []


class LlmPathNeverReadsGtSourceTest(unittest.TestCase):
    """Behavioral guard for the deployed LLM path: with a real (non-None) fragment_fixer,
    gt_source must never be opened. Point gt_source at a non-existent path — any live gt-source
    read in the LLM path would raise FileNotFoundError; completing proves it is never read."""

    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE

    def tearDown(self):
        R.ACCEPTANCE_MODE = self._prev_mode

    def test_missing_gt_source_never_read_with_llm_fixer(self):
        missing = Path("/nonexistent/gt_source_must_not_be_read.py")
        with tempfile.TemporaryDirectory() as td:
            res = repair_mismatching_code_objects(
                FX_MISSING / "gt.pyc", FX_MISSING / "derived.pyc", FX_MISSING / "derived.py",
                gt_source=missing, output_dir=Path(td),
                fragment_fixer=_no_candidates_fixer, acceptance_mode="distance")
        self.assertGreater(res["final_summary"]["combined_distance"], 0)
