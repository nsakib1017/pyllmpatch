"""Unit tests for the oracle-state lattice (utils/oracle_state.py, M2)."""
import unittest

from utils.oracle_state import (
    RANK_EQUAL,
    RANK_DIFF_BYTECODE_OFFSET,
    RANK_DIFF_BYTECODE_NO_OFFSET,
    RANK_DIFF_CONTROL_FLOW,
    RANK_MISSING_EXTRA,
    all_equal,
    classify_regime,
    compare_verifications,
    count_successes,
    object_state,
)


def _eq(name):
    return {"names": name, "success": True, "message": "Equal"}


def _diff_bc(name, offset):
    return {"names": name, "success": False, "message": "Different bytecode",
            "failed_offset": offset}


def _diff_cf(name):
    return {"names": name, "success": False, "message": "Different control flow",
            "failed_offset": None}


def _missing(name):
    return {"names": name, "success": False, "message": "Missing bytecode",
            "failed_offset": None}


class ObjectStateTest(unittest.TestCase):
    def test_rank_ordering(self):
        self.assertEqual(object_state(_eq("f"))[0], RANK_EQUAL)
        self.assertEqual(object_state(_diff_bc("f", 40))[0], RANK_DIFF_BYTECODE_OFFSET)
        self.assertEqual(object_state(_diff_bc("f", None))[0], RANK_DIFF_BYTECODE_NO_OFFSET)
        self.assertEqual(object_state(_diff_cf("f"))[0], RANK_DIFF_CONTROL_FLOW)
        self.assertEqual(object_state(_missing("f"))[0], RANK_MISSING_EXTRA)

    def test_full_lattice_strictly_decreasing(self):
        states = [
            object_state(_eq("f")),
            object_state(_diff_bc("f", 40)),
            object_state(_diff_bc("f", None)),
            object_state(_diff_cf("f")),
            object_state(_missing("f")),
        ]
        self.assertEqual(states, sorted(states, reverse=True))
        self.assertEqual(len(set(states)), len(states))

    def test_deeper_offset_is_better(self):
        # Same regime; deeper failed_offset = more leading instructions matched.
        self.assertGreater(object_state(_diff_bc("f", 200)), object_state(_diff_bc("f", 40)))

    def test_exception_table_no_offset_below_offset(self):
        self.assertGreater(object_state(_diff_bc("f", 4)), object_state(_diff_bc("f", None)))


class CompareVerificationsTest(unittest.TestCase):
    def test_rank_improvement(self):
        prev = [_diff_cf("a")]
        cand = [_diff_bc("a", 12)]  # control-flow -> bytecode = up the lattice
        d = compare_verifications(prev, cand)
        self.assertEqual(d.improved, ["a"])
        self.assertEqual(d.regressed, [])
        self.assertTrue(d.strict_improvement)
        self.assertTrue(d.non_regressing)

    def test_regime_regression_detected(self):
        prev = [_diff_bc("a", 12)]
        cand = [_diff_cf("a")]  # bytecode -> control-flow = down the lattice
        d = compare_verifications(prev, cand)
        self.assertEqual(d.regressed, ["a"])
        self.assertFalse(d.strict_improvement)
        self.assertFalse(d.non_regressing)

    def test_offset_deepening_is_improvement(self):
        prev = [_diff_bc("a", 20)]
        cand = [_diff_bc("a", 80)]
        d = compare_verifications(prev, cand)
        self.assertEqual(d.improved, ["a"])
        self.assertTrue(d.strict_improvement)

    def test_flip_to_equal_is_improvement(self):
        prev = [_diff_bc("a", 20), _eq("b")]
        cand = [_eq("a"), _eq("b")]
        d = compare_verifications(prev, cand)
        self.assertEqual(d.improved, ["a"])
        self.assertEqual(d.successes_prev, 1)
        self.assertEqual(d.successes_cand, 2)
        self.assertTrue(d.non_regressing)

    def test_lost_passing_object_is_regression(self):
        prev = [_eq("a")]
        cand = []  # object 'a' disappeared while passing
        d = compare_verifications(prev, cand)
        self.assertEqual(d.regressed, ["a"])
        self.assertFalse(d.non_regressing)

    def test_new_passing_object_is_improvement(self):
        prev = [_eq("a")]
        cand = [_eq("a"), _eq("b")]  # inserted a passing object
        d = compare_verifications(prev, cand)
        self.assertEqual(d.improved, ["b"])
        self.assertTrue(d.non_regressing)

    def test_no_change_holds(self):
        prev = [_diff_cf("a"), _eq("b")]
        cand = [_diff_cf("a"), _eq("b")]
        d = compare_verifications(prev, cand)
        self.assertEqual(d.improved, [])
        self.assertEqual(d.regressed, [])
        self.assertTrue(d.non_regressing)
        self.assertFalse(d.strict_improvement)

    def test_duplicate_names_keep_worst(self):
        # Two objects share a dotted name; the worse one must drive comparison.
        prev = [_eq("dup"), _eq("dup")]
        cand = [_eq("dup"), _diff_cf("dup")]  # one occurrence regressed
        d = compare_verifications(prev, cand)
        self.assertEqual(d.regressed, ["dup"])
        self.assertFalse(d.non_regressing)


class HelpersTest(unittest.TestCase):
    def test_count_and_all_equal(self):
        results = [_eq("a"), _diff_cf("b")]
        self.assertEqual(count_successes(results), 1)
        self.assertFalse(all_equal(results))
        self.assertTrue(all_equal([_eq("a"), _eq("b")]))
        self.assertFalse(all_equal([]))

    def test_classify_regime_labels(self):
        self.assertEqual(classify_regime(_eq("a")), "Equal")
        self.assertEqual(classify_regime(_diff_bc("a", 4)), "Different bytecode (offset)")
        self.assertEqual(classify_regime(_diff_bc("a", None)),
                         "Different bytecode (no-offset / exc-table)")
        self.assertEqual(classify_regime(_diff_cf("a")), "Different control flow")
        self.assertEqual(classify_regime(_missing("a")), "Missing bytecode")


if __name__ == "__main__":
    unittest.main()
