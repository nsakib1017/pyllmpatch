"""SEMANTIC_DISABLE_CORPUS_WRITE: opt-in kill switch for the fine-tune training-corpus append.

`_append_accepted_code_object_dataset` (~reattach_source_code_object.py:3611) appends every
accepted repair step to the training corpus used by `mine_semantic_repair_actions.py` /
`rebuild_semantic_prompt_refresh_dataset.py`. Contaminated or unwanted rows currently have no way
to be excluded from that append without also silencing `_append_accepted_case_telemetry` (a
separate, unrelated log this flag must NOT touch).

Like its sibling `SEMANTIC_*` flags, this one is read from `os.environ` once at IMPORT time into a
module-level bool, so mutating `os.environ` from inside a test has no effect on an already-imported
module. The established pattern for exercising these flags at test time is
`mock.patch.object(R, "FLAG_NAME", True/False)` (see tests/test_annotation_reconciliation.py,
tests/test_prepass_reject_instrumentation.py::_ModeIsolatedTest).
"""
from __future__ import annotations

import unittest
from unittest import mock

import utils.reattach_source_code_object as R

ACCEPTED_STEP_RECORD = {
    "accepted": True,
    "qualname": "<module>.f",
    "repair_operation": "deterministic_leaf",
    "iteration": 0,
    "step": 0,
}


class CorpusWriteGateTest(unittest.TestCase):
    def test_flag_true_skips_corpus_write(self):
        with mock.patch.object(R, "SEMANTIC_DISABLE_CORPUS_WRITE", True), \
             mock.patch.object(R, "append_log") as fake_append_log:
            R._append_accepted_code_object_dataset(None, "run1", "hash1", ACCEPTED_STEP_RECORD)
        fake_append_log.assert_not_called()

    def test_flag_false_still_writes_corpus_entry(self):
        with mock.patch.object(R, "SEMANTIC_DISABLE_CORPUS_WRITE", False), \
             mock.patch.object(R, "append_log") as fake_append_log:
            R._append_accepted_code_object_dataset(None, "run1", "hash1", ACCEPTED_STEP_RECORD)
        fake_append_log.assert_called_once()

    def test_flag_true_leaves_case_telemetry_untouched(self):
        # Only the training corpus is gated; case telemetry (a separate log) must still write.
        with mock.patch.object(R, "SEMANTIC_DISABLE_CORPUS_WRITE", True), \
             mock.patch.object(R, "append_log") as fake_append_log:
            R._append_accepted_case_telemetry(None, "run1", "hash1", ACCEPTED_STEP_RECORD)
        fake_append_log.assert_called_once()

    def test_flag_defaults_to_off(self):
        self.assertFalse(R.SEMANTIC_DISABLE_CORPUS_WRITE)


if __name__ == "__main__":
    unittest.main()
