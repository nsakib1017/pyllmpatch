"""The runner seam that makes the GT constant splice reachable in production.

Harvesting constant sequences means loading the ground-truth `.pyc` through xdis and the vendored
pylingual package -- seconds per file. Paying that on every row to serve the minority that carry a
truncated literal would be a large, silent throughput regression, so the harvest is gated behind a
pure text check first: no line matching the truncation signature, no load.
"""
import unittest
from unittest import mock

from pipeline import runner as runner_mod
from pipeline.runner import maybe_gt_sequences

FULL = tuple("e%02d" % i for i in range(60))
TRUNCATED = "PAYLOAD = [" + ", ".join(repr(x) for x in FULL[:51]) + ", 'e5\n"
CLEAN = "import os\nx = 1\n"


class MaybeGtSequencesTest(unittest.TestCase):
    def test_no_gt_path_harvests_nothing(self):
        with mock.patch.object(runner_mod, "harvest_constant_sequences") as harvest:
            self.assertEqual(maybe_gt_sequences(None, TRUNCATED), [])
            self.assertEqual(maybe_gt_sequences("", TRUNCATED), [])
            harvest.assert_not_called()

    def test_source_without_a_truncated_literal_never_loads_the_pyc(self):
        with mock.patch.object(runner_mod, "harvest_constant_sequences") as harvest:
            self.assertEqual(maybe_gt_sequences("/gt.pyc", CLEAN), [])
            harvest.assert_not_called()

    def test_truncated_literal_triggers_the_harvest(self):
        with mock.patch.object(runner_mod, "harvest_constant_sequences",
                               return_value=[FULL]) as harvest:
            self.assertEqual(maybe_gt_sequences("/gt.pyc", TRUNCATED), [FULL])
            harvest.assert_called_once_with("/gt.pyc")

    def test_harvest_failure_degrades_to_no_ground_truth(self):
        with mock.patch.object(runner_mod, "harvest_constant_sequences",
                               side_effect=RuntimeError("bad pyc")):
            self.assertEqual(maybe_gt_sequences("/gt.pyc", TRUNCATED), [])

    def test_empty_source_is_safe(self):
        self.assertEqual(maybe_gt_sequences("/gt.pyc", ""), [])
        self.assertEqual(maybe_gt_sequences("/gt.pyc", None), [])

    def test_kill_switch_disables_the_harvest(self):
        with mock.patch.object(runner_mod, "harvest_constant_sequences") as harvest:
            with mock.patch.dict("os.environ", {"SYNTACTIC_GT_LITERAL_SPLICE": "0"}):
                self.assertEqual(maybe_gt_sequences("/gt.pyc", TRUNCATED), [])
            harvest.assert_not_called()

    def test_enabled_by_default(self):
        with mock.patch.object(runner_mod, "harvest_constant_sequences",
                               return_value=[FULL]) as harvest:
            with mock.patch.dict("os.environ", {}, clear=False):
                import os

                os.environ.pop("SYNTACTIC_GT_LITERAL_SPLICE", None)
                self.assertEqual(maybe_gt_sequences("/gt.pyc", TRUNCATED), [FULL])
            harvest.assert_called_once()


if __name__ == "__main__":
    unittest.main()
