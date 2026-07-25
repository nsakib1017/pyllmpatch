"""An audit that measured nothing must not report success.

Run against the winnable CSV without `ROOT_FOR_FILES` / `BASE_DIR_PYTHON_FILES_PYLINGUAL`, the tool
resolved 0 of 16 rows, produced empty per-regime tallies, and printed:

    OK: every deterministic regime with matchable objects saw operators fire and land.

Every regime check is vacuously true over an empty set, so total failure to resolve renders as a
clean bill of health. This is the silent-zero failure mode that has repeatedly produced false
conclusions in this project (a zero result read as a passing result). `resolved == 0` means the
audit is INVALID, not passing, and the two must never look alike.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.deterministic_firing_audit import audit_verdict


class AuditVerdictTest(unittest.TestCase):
    def test_zero_resolved_rows_is_invalid_not_ok(self):
        ok, msg = audit_verdict(totals={"rows": 16, "resolved": 0}, zero_fire=[],
                                fires_but_never_lands=[])
        self.assertFalse(ok, "an audit that resolved nothing must not report OK")
        self.assertIn("INVALID", msg.upper())

    def test_all_rows_unresolved_mentions_the_cause(self):
        # The operator is meant to act on this: it is an environment problem, not an operator one.
        _, msg = audit_verdict(totals={"rows": 400, "resolved": 0}, zero_fire=[],
                               fires_but_never_lands=[])
        self.assertIn("ROOT_FOR_FILES", msg)

    def test_resolved_rows_with_clean_regimes_is_ok(self):
        ok, msg = audit_verdict(totals={"rows": 16, "resolved": 16}, zero_fire=[],
                                fires_but_never_lands=[])
        self.assertTrue(ok)
        self.assertIn("OK", msg)

    def test_coverage_gap_is_not_ok(self):
        ok, msg = audit_verdict(totals={"rows": 16, "resolved": 16},
                                zero_fire=["control_flow"], fires_but_never_lands=[])
        self.assertFalse(ok)
        self.assertIn("control_flow", msg)

    def test_defective_operator_is_not_ok(self):
        ok, msg = audit_verdict(totals={"rows": 16, "resolved": 16},
                                zero_fire=[], fires_but_never_lands=["leaf"])
        self.assertFalse(ok)
        self.assertIn("leaf", msg)


if __name__ == "__main__":
    unittest.main()
