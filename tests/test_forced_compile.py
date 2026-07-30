"""REVISE spec for utils/forced_compile.py — encodes the 3 risks the adversarial verify found.

The 10 smoke files cannot surface these; a pathological input can. Each test must fail on the winner
as-copied, then pass after the fix.
"""
import ast
import time
import unittest

from utils.forced_compile import force_compile


def _compiles(text):
    try:
        compile(text, "<t>", "exec")
        return True
    except SyntaxError:
        return False


class GuaranteeTest(unittest.TestCase):
    """RISK 1 (CRITICAL): output must actually COMPILE, and the ledger must not lie."""

    def test_ledger_reports_a_real_compile_result(self):
        _text, led = force_compile("x = 1\n", None, "3.12")
        self.assertIn("compiles", led)
        self.assertIs(led["compiles"], True)

    def test_pathological_file_still_compiles(self):
        # thousands of `return` outside any function -> compile-only errors beyond the old 2000 cap
        src = "".join(f"return {i}\n" for i in range(6000))
        text, led = force_compile(src, None, "3.12")
        self.assertTrue(_compiles(text), "output must compile even past the iteration cap")
        self.assertTrue(led["compiles"])

    def test_ledger_never_claims_compile_without_it_being_true(self):
        src = "".join(f"break\n" for _ in range(6000))
        text, led = force_compile(src, None, "3.12")
        self.assertEqual(led["compiles"], _compiles(text))


class PerformanceTest(unittest.TestCase):
    """RISK 2 (HIGH): no O(n^2) stall on large pathological input."""

    def test_large_pathological_file_is_not_quadratic(self):
        src = "".join(f"return {i}\n" for i in range(5000))
        t0 = time.time()
        text, led = force_compile(src, None, "3.12")
        dt = time.time() - t0
        self.assertTrue(_compiles(text))
        self.assertLess(dt, 10.0, f"5000-line pathological file took {dt:.1f}s (was ~50s at O(n^2))")


class BytePreservationTest(unittest.TestCase):
    """RISK 3 (MEDIUM): match/case and try rewrites must not drop bytes."""

    def test_match_case_pattern_bytes_survive(self):
        src = (
            "def f(x):\n"
            "    match x:\n"
            "    case 'SECRET_PATTERN_XYZ':\n"
            "        return 1\n"
            "    case 2:\n"
            "        return 2\n"
        )
        text, led = force_compile(src, None, "3.12")
        self.assertTrue(_compiles(text))
        self.assertIn("SECRET_PATTERN_XYZ", text, "case pattern literal must survive in the output")

    def test_no_physical_line_is_deleted(self):
        src = "def f(x):\n    match x:\n    case 1:\n        return 1\n"
        text, _ = force_compile(src, None, "3.12")
        self.assertGreaterEqual(len(text.splitlines()), len(src.splitlines()))


if __name__ == "__main__":
    unittest.main()
