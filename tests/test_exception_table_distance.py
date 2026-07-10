"""Tests for the exception-table-aware distance term (M2, #7)."""
import unittest
from pathlib import Path

import utils.pyc_code_object_distance as D
from utils.file_helpers import fetch_pyllmpatch_repair_paths


class _FakeEntry:
    def __init__(self, start, end, target, depth):
        self.start, self.end, self.target, self.depth = start, end, target, depth


class _FakeBytecode:
    def __init__(self, table):
        self.named_exception_table = table


class HelpersTest(unittest.TestCase):
    def test_serialize_empty(self):
        self.assertEqual(D._serialize_exception_table(_FakeBytecode(None)), ())
        self.assertEqual(D._serialize_exception_table(_FakeBytecode([])), ())

    def test_serialize_sorted_tuples(self):
        bc = _FakeBytecode([_FakeEntry(10, 20, 30, 0), _FakeEntry(2, 4, 6, 1)])
        sig = D._serialize_exception_table(bc)
        self.assertEqual(sig, ((2, 4, 6, 1), (10, 20, 30, 0)))

    def test_distance_symmetric_difference(self):
        a = ((2, 4, 6, 1), (10, 20, 30, 0))
        b = ((2, 4, 6, 1),)  # candidate missing one handler
        self.assertEqual(D.exception_table_distance(a, b), 1)
        self.assertEqual(D.exception_table_distance(a, a), 0)
        self.assertEqual(D.exception_table_distance((), ()), 0)
        self.assertEqual(D.exception_table_distance(a, ()), 2)

    def test_toggle_flag(self):
        prev = D._EXC_TABLE_DISTANCE_ENABLED
        try:
            D.set_exception_table_distance_enabled(True)
            self.assertTrue(D._EXC_TABLE_DISTANCE_ENABLED)
            D.set_exception_table_distance_enabled(False)
            self.assertFalse(D._EXC_TABLE_DISTANCE_ENABLED)
        finally:
            D.set_exception_table_distance_enabled(prev)


class IntegrationTest(unittest.TestCase):
    """Real prepared structures + a forced exception-table mutation prove the
    gating arithmetic: with the flag on, combined_distance gains exactly
    EXCEPTION_TABLE_WEIGHT per differing table entry (and nothing when off).

    NOTE: the 3.10-3.13 benchmark set has *no* genuine exception-table-only
    divergences (its no-offset "Different bytecode" cases are length mismatches,
    already captured by instruction distance), so we inject one here.
    """
    ROW = "797018b9bbf3ee3564083585f9cdbb3de7f785fe71b37448137dfb9ecb232c8b"  # 3.11

    def setUp(self):
        # Isolate the pickle cache so we don't disturb a concurrently-running loop.
        self._orig_cache = D.DISTANCE_CACHE_DIR
        D.DISTANCE_CACHE_DIR = Path("/home/mxs220189/.claude/jobs/c42cbf21/tmp/exc_test_cache")
        self._orig_flag = D._EXC_TABLE_DISTANCE_ENABLED

    def tearDown(self):
        D.DISTANCE_CACHE_DIR = self._orig_cache
        D.set_exception_table_distance_enabled(self._orig_flag)

    def test_gating_arithmetic(self):
        import copy
        gt, der, _ = fetch_pyllmpatch_repair_paths(self.ROW, "PyLingual.IO")
        if not (gt and der):
            self.skipTest("row artifacts not resolvable")

        prepared_gt = D.prepare_pyc(Path(gt))
        prepared_der = copy.deepcopy(D.prepare_pyc(Path(der)))

        # Inject a differing exception-table entry on one matched object.
        gt_names = {o["name"] for o in prepared_gt["code_objects"]}
        target = next(o for o in prepared_der["code_objects"] if o["name"] in gt_names)
        target["exception_table"] = tuple(target.get("exception_table", ())) + ((9999, 9999, 9999, 9),)

        D.set_exception_table_distance_enabled(False)
        off = D.compare_code_object_distances(prepared_gt, prepared_der)
        D.set_exception_table_distance_enabled(True)
        on = D.compare_code_object_distances(prepared_gt, prepared_der)

        for r in on:
            if r["status"] == "matched":
                self.assertIn("exception_table_distance", r)

        delta = sum(r["combined_distance"] for r in on) - sum(r["combined_distance"] for r in off)
        expected = D.EXCEPTION_TABLE_WEIGHT * sum(
            r["exception_table_distance"] for r in on if r["status"] == "matched"
        )
        self.assertEqual(delta, expected)
        self.assertGreater(expected, 0)  # the injected entry must register


if __name__ == "__main__":
    unittest.main()
