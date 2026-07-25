import tempfile
import unittest
from pathlib import Path
from unittest import mock

import utils.reattach_source_code_object as R
from utils.reattach_source_code_object import (
    _annotate_enclosing_qualname,
    _gt_def_source_by_qualname,
    repair_mismatching_code_objects,
)

class AnnotateEnclosingQualnameTest(unittest.TestCase):
    def test_maps_annotate_child_to_enclosing_def(self):
        self.assertEqual(_annotate_enclosing_qualname("<module>.Config.__annotate__"), "<module>.Config")
    def test_maps_nested(self):
        self.assertEqual(_annotate_enclosing_qualname("<module>.A.B.__annotate__"), "<module>.A.B")
    def test_module_level_annotate_has_no_enclosing_def(self):
        self.assertIsNone(_annotate_enclosing_qualname("<module>.__annotate__"))
    def test_non_annotate_returns_none(self):
        self.assertIsNone(_annotate_enclosing_qualname("<module>.Config"))
        self.assertIsNone(_annotate_enclosing_qualname(None))


SRC = (
    "class Outer:\n"
    "    x: int = 1\n"
    "    class Inner:\n"
    "        y: str = 'a'\n"
    "def top(a: int) -> str:\n"
    "    return str(a)\n"
)


class GtDefSourceTest(unittest.TestCase):
    def test_top_level_class(self):
        seg = _gt_def_source_by_qualname(SRC, "<module>.Outer")
        self.assertTrue(seg.startswith("class Outer:")); self.assertIn("class Inner:", seg)
    def test_nested_class(self):
        seg = _gt_def_source_by_qualname(SRC, "<module>.Outer.Inner")
        self.assertTrue(seg.startswith("class Inner:")); self.assertIn("y: str", seg)
    def test_function(self):
        seg = _gt_def_source_by_qualname(SRC, "<module>.top")
        self.assertTrue(seg.startswith("def top(")); self.assertIn("-> str", seg)
    def test_missing_returns_none(self):
        self.assertIsNone(_gt_def_source_by_qualname(SRC, "<module>.Nope"))


FX = Path(__file__).parent / "fixtures" / "annotation" / "settings"


class AnnotationReconcileEngineTest(unittest.TestCase):
    """Two adjustments vs. the plan's literal pseudocode, both "verify against real code" fixes
    (the plan itself flags the result-key name as needing this kind of check):

    1. `SEMANTIC_ANNOTATION_RECONCILE` (like its sibling `SEMANTIC_*` flags in this module) is
       read from `os.environ` once at IMPORT time into a module-level constant, so mutating
       `os.environ` from inside a test has no effect on an already-imported module. The
       established pattern for exercising these flags at test time (see
       `tests/test_post_llm_deterministic.py::_ModeIsolatedTest`) is
       `mock.patch.object(R, "FLAG_NAME", True/False)`, used below.
    2. `acceptance_mode="distance"` (not "oracle"): the deterministic-prepass accept gate this
       feature reuses (`run_deterministic_prepass`'s `compare_verifications` check) is ALWAYS
       oracle-verification-based, regardless of the global `ACCEPTANCE_MODE` -- so it validates
       the feature identically either way. `acceptance_mode="oracle"` additionally activates
       `_oracle_primary_fragment_targets` in the main iteration loop, which has a pre-existing gap
       (unrelated to this feature: it lacks the `_is_synthetic_annotation_qualname` guard its
       sibling selectors `select_repair_targets` / `select_missing_repair_targets` already carry)
       that raises an uncaught `ReattachError` for any unresolved `__annotate__` divergence.
       Pinning "distance" avoids that orthogonal, pre-existing bug so this test isolates Task 3's
       behavior. Restoring the module global in `tearDown` follows the existing
       `_ModeIsolatedTest` pattern in tests/test_prepass_reject_instrumentation.py."""

    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE

    def tearDown(self):
        R.ACCEPTANCE_MODE = self._prev_mode

    def test_flag_on_converts_pure_annotation_file(self):
        with mock.patch.object(R, "SEMANTIC_ANNOTATION_RECONCILE", True):
            with tempfile.TemporaryDirectory() as td:
                res = repair_mismatching_code_objects(
                    FX/"gt.pyc", FX/"derived.pyc", FX/"derived.py",
                    gt_source=FX/"gt.py", output_dir=Path(td),
                    fragment_fixer=None, acceptance_mode="distance",
                    verify_with_pylingual=True, verify_each_step_with_pylingual=True)
        self.assertEqual(res["final_summary"]["combined_distance"], 0)

    def test_flag_off_leaves_distance_unchanged(self):
        with mock.patch.object(R, "SEMANTIC_ANNOTATION_RECONCILE", False):
            with tempfile.TemporaryDirectory() as td:
                res = repair_mismatching_code_objects(
                    FX/"gt.pyc", FX/"derived.pyc", FX/"derived.py",
                    gt_source=FX/"gt.py", output_dir=Path(td),
                    fragment_fixer=None, acceptance_mode="distance")
        self.assertGreater(res["final_summary"]["combined_distance"], 0)  # unchanged: still diverges


FX_MISSING = Path(__file__).parent / "fixtures" / "annotation" / "missing"


def _no_candidates_fixer(*_args, **_kwargs):
    """A `fragment_fixer` stand-in that never proposes a candidate.

    `<module>.Config` also diverges independently in this fixture (see class docstring), and
    `repair_mismatching_code_objects`'s MAIN iteration loop has its own pre-existing, unrelated
    GT-verbatim fallback that activates whenever `fragment_fixer is None` (oracle mode) -- which
    would fix `<module>.Config` (and, as a side effect, its now-matching `__annotate__` child) all
    by itself, whether or not the deterministic-prepass annotate-reconcile branch under test does
    anything. Passing a real (non-None) fixer that always abstains keeps that main-loop fallback
    out of the picture, so these tests isolate the PREPASS branch these findings are about.
    """
    return []


class AnnotationReconcileMissingBytecodeTest(unittest.TestCase):
    """Covers the code-review gap: a `__annotate__` code object entirely ABSENT from
    derived.pyc surfaces from `compare_pyc` as `message == "Missing bytecode"` (bc_b is
    None), not `"Different bytecode"` / `"Different control flow"`. `derived.py` here drops
    the `z: int` bare annotation on `Config` outright (not just reorders it), so PyLingual's
    "recompile" never synthesizes a `__annotate__` code object for `Config` at all -- the
    object is missing, not diverged. Because CPython only emits the class-body machinery that
    installs `__annotate__` when the class has ANY annotation, `<module>.Config` itself also
    independently shows up as "Different bytecode" (no single offset) here -- so
    `fragment_fixer` is a real no-op callable (see `_no_candidates_fixer`), not `None`, to keep
    the main loop's separate GT-verbatim fallback from masking whether the prepass fix works."""

    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE

    def tearDown(self):
        R.ACCEPTANCE_MODE = self._prev_mode

    def test_flag_on_reconciles_missing_annotate_object(self):
        with mock.patch.object(R, "SEMANTIC_ANNOTATION_RECONCILE", True):
            with tempfile.TemporaryDirectory() as td:
                res = repair_mismatching_code_objects(
                    FX_MISSING/"gt.pyc", FX_MISSING/"derived.pyc", FX_MISSING/"derived.py",
                    gt_source=FX_MISSING/"gt.py", output_dir=Path(td),
                    fragment_fixer=_no_candidates_fixer, acceptance_mode="distance",
                    verify_with_pylingual=True, verify_each_step_with_pylingual=True)
        self.assertEqual(res["final_summary"]["combined_distance"], 0)

    def test_flag_off_leaves_missing_annotate_object_unfixed(self):
        with mock.patch.object(R, "SEMANTIC_ANNOTATION_RECONCILE", False):
            with tempfile.TemporaryDirectory() as td:
                res = repair_mismatching_code_objects(
                    FX_MISSING/"gt.pyc", FX_MISSING/"derived.pyc", FX_MISSING/"derived.py",
                    gt_source=FX_MISSING/"gt.py", output_dir=Path(td),
                    fragment_fixer=_no_candidates_fixer, acceptance_mode="distance")
        self.assertGreater(res["final_summary"]["combined_distance"], 0)  # unchanged: still diverges


FX_SPACE_DEBUG = Path(__file__).parent / "fixtures" / "annotation" / "space_debug"
FX_DTO = Path(__file__).parent / "fixtures" / "annotation" / "dto"


class AnnotationReconcileBroaderFixturesTest(unittest.TestCase):
    """Task 4: broaden coverage beyond the single `settings` fixture with two more real, proven
    pure-annotation 3.15 files from the oracle-ceiling corpus (file_hash prefixes `d42177d3`
    "space_debug_draw_options.py" and `670c67a8` "dto.py" -- see
    docs/superpowers/plans/2026-07-25-annotation-reconciliation-stage1.md Task 4 and
    $JT/poc_prove_all.py, which proved both convert to distance 0 under GT-def splicing). Same
    flag-on/distance-0 assertion and `fragment_fixer=None` / `acceptance_mode="distance"` pattern
    as `AnnotationReconcileEngineTest` above -- these are end-to-end convergence checks (not
    isolated to the prepass branch the way the "missing" fixture's tests are), run against files
    the deterministic-prepass branch was not tuned against."""

    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE

    def tearDown(self):
        R.ACCEPTANCE_MODE = self._prev_mode

    def test_flag_on_converts_space_debug(self):
        with mock.patch.object(R, "SEMANTIC_ANNOTATION_RECONCILE", True):
            with tempfile.TemporaryDirectory() as td:
                res = repair_mismatching_code_objects(
                    FX_SPACE_DEBUG/"gt.pyc", FX_SPACE_DEBUG/"derived.pyc", FX_SPACE_DEBUG/"derived.py",
                    gt_source=FX_SPACE_DEBUG/"gt.py", output_dir=Path(td),
                    fragment_fixer=None, acceptance_mode="distance",
                    verify_with_pylingual=True, verify_each_step_with_pylingual=True)
        self.assertEqual(res["final_summary"]["combined_distance"], 0)

    def test_flag_on_converts_dto(self):
        with mock.patch.object(R, "SEMANTIC_ANNOTATION_RECONCILE", True):
            with tempfile.TemporaryDirectory() as td:
                res = repair_mismatching_code_objects(
                    FX_DTO/"gt.pyc", FX_DTO/"derived.pyc", FX_DTO/"derived.py",
                    gt_source=FX_DTO/"gt.py", output_dir=Path(td),
                    fragment_fixer=None, acceptance_mode="distance",
                    verify_with_pylingual=True, verify_each_step_with_pylingual=True)
        self.assertEqual(res["final_summary"]["combined_distance"], 0)
