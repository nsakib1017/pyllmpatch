"""Speedup Lever B: tail early-abort. A file whose best state has not improved for
SEMANTIC_TAIL_DEADLINE seconds is stopped early — treated as an EARLIER hard-timeout, so it
inherits the exact established semantics (restore best-state, or raise the no-improvement error
if best == initial). This reclaims the hard tail (files grinding to the 1200s cap without
converging) at ~zero quality cost: the engine always rolls back to the best state found.

Pure predicate pinned here (no engine / no GPU). Flag off (deadline <= 0) => never aborts =>
byte-identical to today, so the 722-test suite and every oracle gate are unaffected.
"""
import os
import unittest
from unittest import mock

import utils.reattach_source_code_object as RS


class TailDeadlineSecondsTest(unittest.TestCase):
    def test_unset_is_zero_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEMANTIC_TAIL_DEADLINE", None)
            self.assertEqual(RS._tail_deadline_seconds(), 0.0)

    def test_reads_env(self):
        with mock.patch.dict(os.environ, {"SEMANTIC_TAIL_DEADLINE": "300"}):
            self.assertEqual(RS._tail_deadline_seconds(), 300.0)

    def test_garbage_is_zero(self):
        with mock.patch.dict(os.environ, {"SEMANTIC_TAIL_DEADLINE": "abc"}):
            self.assertEqual(RS._tail_deadline_seconds(), 0.0)


class ShouldTailAbortTest(unittest.TestCase):
    def test_disabled_when_deadline_zero(self):
        # even after a long stall, deadline 0 => never abort
        self.assertFalse(RS._should_tail_abort(now=1000.0, last_improvement=0.0, tail_deadline=0.0))

    def test_disabled_when_deadline_negative(self):
        self.assertFalse(RS._should_tail_abort(now=1000.0, last_improvement=0.0, tail_deadline=-5.0))

    def test_aborts_after_stall_exceeds_deadline(self):
        self.assertTrue(RS._should_tail_abort(now=400.0, last_improvement=50.0, tail_deadline=300.0))

    def test_no_abort_within_deadline(self):
        self.assertFalse(RS._should_tail_abort(now=200.0, last_improvement=50.0, tail_deadline=300.0))

    def test_boundary_exactly_at_deadline_aborts(self):
        self.assertTrue(RS._should_tail_abort(now=350.0, last_improvement=50.0, tail_deadline=300.0))


if __name__ == "__main__":
    unittest.main()
