"""Deterministic-prepass REJECT instrumentation.

`_store_semantic_step` is only reached AFTER the oracle gate passes
(`reattach_source_code_object.py:4144`). Both rejection paths -- the compile failure at :4131 and
the gate rejection at :4138 -- `continue` silently after `step_index -= 1`, so an operator that
FIRES and is then rejected leaves ZERO trace in result.json.

That blind spot makes "did the operator fire?" unanswerable: accepted-step counts cannot
distinguish "the builder deferred (returned None)" from "the builder fired, was correct, and was
thrown away". Two separate repair designs were built on that ambiguity and both were wrong.
`tools/deterministic_firing_audit.py:100` inherits the same blind spot.

These tests pin the rejected-attempt record: every operator candidate that reaches the gate and
loses must be recorded with an attributable cause. No behavioural change -- acceptance is
untouched; we only stop discarding the evidence.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual"))

import utils.reattach_source_code_object as R
from utils.generate_bytecode import compile_version

HOST = f"{sys.version_info[0]}.{sys.version_info[1]}"

# A leaf divergence: same instruction count, one differing operand -> the oracle reports
# "Different bytecode" WITH a failed_offset, which routes to the is_leaf branch and calls
# `leaf_value_candidate` first (reattach:4058). Patching that one symbol is therefore enough
# to drive a chosen candidate into the gate.
GT_SRC = "def f(x):\n    return x + 1\n"
DER_SRC = "def f(x):\n    return x - 1\n"


def _scenario(gt_src: str = GT_SRC, der_src: str = DER_SRC) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "gt.py").write_text(gt_src)
    (d / "der.py").write_text(der_src)
    compile_version(str(d / "gt.py"), str(d / "gt.pyc"), HOST)
    compile_version(str(d / "der.py"), str(d / "der.pyc"), HOST)
    return d


def _repair(d: Path) -> dict:
    return R.repair_mismatching_code_objects(
        d / "gt.pyc", d / "der.pyc", d / "der.py",
        output_dir=d / "out", fragment_fixer=None, acceptance_mode="oracle",
    )


def _fake_operator(text: str, operator: str = "fake_op"):
    """Stand in for a wired operator that FIRES (returns a candidate) rather than deferring."""
    return lambda bc_a, bc_b, fragment, ctx: {
        "text": text, "operator": operator, "confidence": "unique",
    }


class _ModeIsolatedTest(unittest.TestCase):
    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE  # repair_* mutates the global; restore for isolation

    def tearDown(self):
        R.ACCEPTANCE_MODE = self._prev_mode


class RejectRecordShapeTest(_ModeIsolatedTest):
    def test_result_exposes_rejected_attempts_list(self):
        # The run-level attribution key must always exist, so consumers can rely on it.
        d = _scenario()
        res = _repair(d)
        self.assertIn("prepass_rejected_attempts", res)
        self.assertIsInstance(res["prepass_rejected_attempts"], list)

    def test_no_rejection_recorded_when_operators_defer(self):
        # Real operators deferring (returning None) must NOT be logged as rejections -- a deferral
        # is not a rejection, and conflating them recreates the very blind spot this closes.
        d = _scenario()
        with mock.patch.object(R, "leaf_value_candidate", lambda *a, **k: None), \
             mock.patch.object(R, "collapse_dead_pass_else_candidate", lambda *a, **k: None):
            res = _repair(d)
        self.assertEqual(
            [r for r in res["prepass_rejected_attempts"] if r.get("operator") == "fake_op"], []
        )


class RejectCauseAttributionTest(_ModeIsolatedTest):
    def test_non_compiling_candidate_is_recorded_as_compile_error(self):
        # Fires, but the replacement is not valid Python -> compile fails at :4128-4131.
        d = _scenario()
        with mock.patch.object(
            R, "leaf_value_candidate", _fake_operator("def f(x):\n    return (((\n")
        ):
            res = _repair(d)
        rec = [r for r in res["prepass_rejected_attempts"] if r.get("operator") == "fake_op"]
        self.assertTrue(rec, "a firing operator whose candidate cannot compile must be recorded")
        self.assertEqual(rec[0]["reason"], "compile_error")
        # `tr.names()` is the dotted scope path, so a module-level def is `<module>.f`.
        self.assertEqual(rec[0]["qualname"], "<module>.f")
        self.assertEqual(rec[0]["strategy"], "deterministic_leaf")

    def test_regressing_candidate_is_recorded_as_regressed(self):
        # `return 0` drops the instruction count, so the oracle falls from
        # RANK_DIFF_BYTECODE_OFFSET (3) to RANK_DIFF_BYTECODE_NO_OFFSET (2) -- a strict
        # regression, rejected at :4138 via delta.regressed.
        d = _scenario()
        with mock.patch.object(R, "leaf_value_candidate", _fake_operator("def f(x):\n    return 0\n")):
            res = _repair(d)
        rec = [r for r in res["prepass_rejected_attempts"] if r.get("operator") == "fake_op"]
        self.assertTrue(rec, "a firing operator whose candidate regresses must be recorded")
        self.assertEqual(rec[0]["reason"], "regressed")

    def test_rejected_record_carries_attribution_fields(self):
        d = _scenario()
        with mock.patch.object(R, "leaf_value_candidate", _fake_operator("def f(x):\n    return 0\n")):
            res = _repair(d)
        rec = [r for r in res["prepass_rejected_attempts"] if r.get("operator") == "fake_op"][0]
        for key in ("qualname", "phase", "iteration", "repair_operation", "strategy",
                    "operator", "reason"):
            self.assertIn(key, rec, f"rejected-attempt record must carry {key!r} for attribution")
        self.assertEqual(rec["repair_operation"], "deterministic_prepass_leaf")
        self.assertEqual(rec["phase"], "pre_llm")


class RejectDoesNotAlterBehaviourTest(_ModeIsolatedTest):
    def test_accepted_steps_are_unchanged_by_instrumentation(self):
        # The real operators are untouched here; instrumentation must not perturb acceptance.
        # A correct GT-matching candidate must still be ACCEPTED and produce no rejection record.
        d = _scenario()
        with mock.patch.object(R, "leaf_value_candidate", _fake_operator(GT_SRC)):
            res = _repair(d)
        accepted = [s for s in res.get("steps", []) if s.get("accepted")]
        self.assertTrue(accepted, "a GT-correct candidate must still be accepted")
        self.assertEqual(
            [r for r in res["prepass_rejected_attempts"] if r.get("operator") == "fake_op"], [],
            "an accepted candidate must not also be recorded as rejected",
        )


if __name__ == "__main__":
    unittest.main()
