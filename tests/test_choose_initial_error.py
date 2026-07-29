"""choose_initial_error must yield a LINE-NUMBERED error description.

Root cause of the malware syntactic run producing 0 LLM calls / 3.2% compile: the runner set
initial_error_description from row columns only (`error`/`error_message`/`error_description`),
but the malware manifest carries the error under `syntactic_error_message` with a value like
"unexpected indent" that has NO line number. attempt_repair -> extract_line_number(None-or-
lineless) -> None -> empty window -> returns None -> repair_not_attempted_or_failed_precheck.

The fix derives the error from the FILE itself (a fresh compile probe) whenever the row error
lacks a resolvable line number, since the decompiled file is the source of truth for its current
syntax error. choose_initial_error encodes that selection purely (probe result injected).
"""
import unittest

from pipeline.logging_utils import choose_initial_error


class ChooseInitialErrorTest(unittest.TestCase):
    def test_row_error_with_lineno_is_kept_no_probe_needed(self):
        row = "  File x, line 178\n    unexpected indent"
        self.assertEqual(choose_initial_error(row, probe_result=None), row)

    def test_lineless_row_error_falls_back_to_probe(self):
        # "unexpected indent" has no line number -> must use the probe's line-numbered error
        probe = {"is_compiled": False, "error_description": "IndentationError: line 178 unexpected indent"}
        self.assertEqual(choose_initial_error("unexpected indent", probe_result=probe), probe["error_description"])

    def test_none_row_error_falls_back_to_probe(self):
        probe = {"is_compiled": False, "error_description": "SyntaxError: invalid syntax (x, line 42)"}
        self.assertEqual(choose_initial_error(None, probe_result=probe), probe["error_description"])

    def test_probe_says_compiled_returns_row_error(self):
        # nothing to fix per the probe -> return whatever the row had (may be None)
        self.assertIsNone(choose_initial_error(None, probe_result={"is_compiled": True, "error_description": None}))

    def test_no_probe_and_lineless_row_returns_row(self):
        self.assertEqual(choose_initial_error("unexpected indent", probe_result=None), "unexpected indent")


if __name__ == "__main__":
    unittest.main()
