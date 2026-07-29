"""extract_line_number must be None-safe.

Regression: the SYNTACTIC_GT_CONTEXT path (pipeline.runner.maybe_gt_context) calls
extract_line_number on the current compile-error description, which can legitimately be
None (a row whose initial error description isn't populated yet). maybe_gt_context's own
docstring promises it returns "" when "the current compile error doesn't carry a resolvable
line number" -- i.e. it relies on extract_line_number returning None, NOT raising. Before the
fix, re.search(pattern, None) raised `TypeError: expected string or bytes-like object, got
'NoneType'`, which the run loop caught as an "Unexpected exception" for EVERY file, producing
zero results on the malware syntactic run. The sibling clean_error_message already guards None.
"""
import unittest

from pipeline.logging_utils import extract_line_number


class ExtractLineNumberNoneSafeTest(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(extract_line_number(None))

    def test_empty_returns_none(self):
        self.assertIsNone(extract_line_number(""))

    def test_no_line_token_returns_none(self):
        self.assertIsNone(extract_line_number("SyntaxError: invalid syntax"))

    def test_extracts_line_number(self):
        self.assertEqual(extract_line_number("  File x, line 42\n    bad"), 42)


if __name__ == "__main__":
    unittest.main()
