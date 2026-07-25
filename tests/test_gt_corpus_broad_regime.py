"""Broaden the corpus capture gate to ALL failing regimes, not just brief-rendering ones.

`_should_capture_target` was written to mirror the brief RENDER gate: it keeps missing,
"Different control flow", and "Different bytecode" with no failed offset -- and silently drops
"Different bytecode" WITH a failed offset, i.e. the LEAF regime (8% of production failures).

That is correct for deciding whether a brief renders, but wrong for deciding what to TRAIN on.
A fine-tune from stock base on a leaf-free corpus never sees a leaf example and starts at zero on
that regime, so it can regress the overall rate while improving control flow. The training set must
cover every regime the model must handle at inference.

Broadening is safe because the PROMPT build is independent of capture: the brief still renders (or
not) by the same engine gate, so a leaf row simply gets a brief-less prompt -- exactly what
inference sends for a leaf target. These tests pin the broadened gate behind `capture_all`, keep the
narrow default (existing pins in test_gt_reconstruction_corpus.py stay green), and require the
regime label to distinguish leaf so coverage is auditable.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.build_gt_reconstruction_corpus import (
    _should_capture_target,
    _repair_operation_for_target,
)


def _ctx(message, failed_offset, *, missing=False):
    if missing:
        return {"target_is_missing": True}
    return {"pylingual_failed_result": {"message": message}, "failed_offset": failed_offset}


class NarrowGateUnchangedTest(unittest.TestCase):
    """Default behaviour must be byte-identical to before -- other pins depend on it."""

    def test_leaf_still_excluded_by_default(self):
        self.assertFalse(_should_capture_target(_ctx("Different bytecode", 42)))

    def test_controlflow_still_included_by_default(self):
        self.assertTrue(_should_capture_target(_ctx("Different control flow", None)))

    def test_missing_still_included_by_default(self):
        self.assertTrue(_should_capture_target(_ctx(None, None, missing=True)))


class BroadGateTest(unittest.TestCase):
    def test_leaf_is_captured_when_capture_all(self):
        # The whole point: "Different bytecode" WITH a failed offset is the leaf regime.
        self.assertTrue(_should_capture_target(_ctx("Different bytecode", 42), capture_all=True))

    def test_controlflow_still_captured_when_capture_all(self):
        self.assertTrue(_should_capture_target(_ctx("Different control flow", None), capture_all=True))

    def test_diffbc_no_offset_still_captured_when_capture_all(self):
        self.assertTrue(_should_capture_target(_ctx("Different bytecode", None), capture_all=True))

    def test_missing_still_captured_when_capture_all(self):
        self.assertTrue(_should_capture_target(_ctx(None, None, missing=True), capture_all=True))

    def test_missing_extra_bytecode_captured_when_capture_all(self):
        self.assertTrue(_should_capture_target(_ctx("Missing bytecode", None), capture_all=True))
        self.assertTrue(_should_capture_target(_ctx("Extra bytecode", None), capture_all=True))

    def test_a_non_failing_target_is_never_captured(self):
        # No failure message and not missing => nothing to learn. Broadening must not sweep in
        # already-correct objects; that would train the model to rewrite passing code.
        self.assertFalse(_should_capture_target({"failed_offset": None}, capture_all=True))
        self.assertFalse(_should_capture_target(None, capture_all=True))


class RegimeLabelTest(unittest.TestCase):
    """The label must distinguish leaf, or corpus composition is unauditable."""

    def test_leaf_is_labelled_leaf(self):
        self.assertEqual(_repair_operation_for_target(_ctx("Different bytecode", 42)), "leaf")

    def test_missing_is_labelled_insert_missing(self):
        self.assertEqual(
            _repair_operation_for_target(_ctx(None, None, missing=True)), "insert_missing"
        )

    def test_control_flow_is_labelled_control_flow(self):
        self.assertEqual(
            _repair_operation_for_target(_ctx("Different control flow", None)), "control_flow"
        )


if __name__ == "__main__":
    unittest.main()
