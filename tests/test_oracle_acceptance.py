"""Tests for oracle-primary acceptance + target selection (M2).

Exercises the flag-gated behaviour added to reattach_source_code_object.py:
distance mode (default) must be unchanged; oracle mode uses the lattice.
"""
import unittest

import utils.reattach_source_code_object as R


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


def _verif(results):
    return {"all_equal": all(r["success"] for r in results), "results": results}


def _summary(cd):
    return {"combined_distance": cd}


class _OracleModeCtx:
    """Temporarily force ACCEPTANCE_MODE."""
    def __init__(self, mode):
        self.mode = mode

    def __enter__(self):
        self._prev = R.ACCEPTANCE_MODE
        R.ACCEPTANCE_MODE = self.mode

    def __exit__(self, *exc):
        R.ACCEPTANCE_MODE = self._prev


class DistanceModeUnchangedTest(unittest.TestCase):
    """Default 'distance' mode must behave exactly as before."""
    def test_accept_on_distance_drop(self):
        ok, reason = R._should_accept_candidate(_summary(10), _summary(5), None, None)
        self.assertTrue(ok)
        self.assertEqual(reason, "combined distance improved")

    def test_reject_when_no_distance_drop_and_no_verif(self):
        ok, _ = R._should_accept_candidate(_summary(5), _summary(5), None, None)
        self.assertFalse(ok)

    def test_distance_drop_but_success_regressed(self):
        prev_v = _verif([_eq("a"), _eq("b")])
        cand_v = _verif([_eq("a"), _diff_cf("b")])
        ok, reason = R._should_accept_candidate(_summary(10), _summary(5), prev_v, cand_v)
        self.assertFalse(ok)
        self.assertIn("regressed", reason)

    def test_success_count_improved_without_distance_drop(self):
        prev_v = _verif([_diff_cf("a")])
        cand_v = _verif([_eq("a")])
        ok, reason = R._should_accept_candidate(_summary(5), _summary(5), prev_v, cand_v)
        self.assertTrue(ok)
        self.assertEqual(reason, "PyLingual success count improved")


class OracleModeAcceptanceTest(unittest.TestCase):
    def test_accept_on_lattice_improvement_even_without_distance_drop(self):
        with _OracleModeCtx("oracle"):
            prev_v = _verif([_diff_cf("a")])
            cand_v = _verif([_diff_bc("a", 12)])  # control-flow -> bytecode = up
            ok, reason = R._should_accept_candidate(_summary(5), _summary(5), prev_v, cand_v)
        self.assertTrue(ok)
        self.assertIn("improved", reason)

    def test_reject_on_regime_regression_even_with_distance_drop(self):
        with _OracleModeCtx("oracle"):
            prev_v = _verif([_diff_bc("a", 12)])
            cand_v = _verif([_diff_cf("a")])  # down the lattice
            ok, reason = R._should_accept_candidate(_summary(10), _summary(1), prev_v, cand_v)
        self.assertFalse(ok)
        self.assertIn("regressed", reason)

    def test_held_state_uses_distance_gradient(self):
        with _OracleModeCtx("oracle"):
            v = _verif([_diff_cf("a")])
            ok_drop, _ = R._should_accept_candidate(_summary(10), _summary(5), v, _verif([_diff_cf("a")]))
            ok_flat, _ = R._should_accept_candidate(_summary(5), _summary(5), v, _verif([_diff_cf("a")]))
        self.assertTrue(ok_drop)
        self.assertFalse(ok_flat)

    def test_exception_table_flip_accepted_at_zero_distance(self):
        # The headline case: combined_distance stays 0 but the oracle flips to Equal.
        with _OracleModeCtx("oracle"):
            prev_v = _verif([_diff_bc("a", None)])  # exc-table, no offset
            cand_v = _verif([_eq("a")])
            ok, reason = R._should_accept_candidate(_summary(0), _summary(0), prev_v, cand_v)
        self.assertTrue(ok)
        self.assertIn("improved", reason)


class OraclePrimaryTargetSelectionTest(unittest.TestCase):
    def _rows(self, *matched_names):
        return [{"status": "matched", "gt_name": n, "derived_name": n,
                 "combined_distance": 0} for n in matched_names]

    def test_picks_zero_distance_oracle_failure(self):
        rows = self._rows("f")
        verif = _verif([_diff_bc("f", None)])  # oracle rejects, distance 0
        got = R._oracle_primary_fragment_targets(verif, rows, {}, 1, include_module=True)
        self.assertEqual(got, ["f"])

    def test_skips_missing_and_extra(self):
        rows = self._rows("f")
        verif = _verif([_eq("f"), _missing("g")])  # g not a matched row
        got = R._oracle_primary_fragment_targets(verif, rows, {}, 1, include_module=True)
        self.assertEqual(got, [])

    def test_module_gated_by_include_flag(self):
        rows = [{"status": "matched", "gt_name": "<module>", "derived_name": "<module>",
                 "combined_distance": 0}]
        verif = _verif([_diff_bc("<module>", None)])
        self.assertEqual(
            R._oracle_primary_fragment_targets(verif, rows, {}, 1, include_module=False), [])
        self.assertEqual(
            R._oracle_primary_fragment_targets(verif, rows, {}, 1, include_module=True), ["<module>"])

    def test_respects_attempt_cap(self):
        rows = self._rows("f")
        verif = _verif([_diff_cf("f")])
        got = R._oracle_primary_fragment_targets(verif, rows, {"f": 1}, 1, include_module=True)
        self.assertEqual(got, [])


if __name__ == "__main__":
    unittest.main()
