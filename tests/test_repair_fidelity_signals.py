"""Fidelity signals for syntactic repair.

Two measured problems these address:

1. METRIC INTEGRITY. `balance_delimiters` closes an unterminated literal by appending a quote and
   the missing brackets. When the literal was TRUNCATED by the decompiler (PyLingual caps constant
   sequences at 51 elements), that "repair" compiles while silently DISCARDING elements 52+, yet
   still scores as a no-deletion "genuine" repair. It fires 131 times on the malware delete-only
   set, so the headline genuine rate is partly gameable by data loss. `is_truncated_literal_line`
   lets the runner label those repairs lossy instead of counting them as faithful.

2. WASTED BUDGET. 16.8% of delete-only files carry `<mask_N>` / `<CodeNNN code object>` markers --
   PyLingual explicitly reporting it never recovered that content. Nothing can reconstruct it, yet
   these files consume the full 9-iteration x 4-candidate budget before falling back anyway.
   `has_decompiler_surrender_markers` lets the runner route them straight to the fallback.

Both are pure predicates so they can be trusted at the decision sites without a model or GPU.
"""
import unittest

from utils.syntactic_prepass import has_decompiler_surrender_markers, is_truncated_literal_line


class TruncatedLiteralLineTest(unittest.TestCase):
    def test_truncated_constant_sequence_is_detected(self):
        # The measured signature: long line, odd quote count, still-open bracket.
        line = "USERS = (" + ", ".join(f"'user{i}'" for i in range(51)) + ", '"
        self.assertTrue(is_truncated_literal_line(line))

    def test_complete_literal_is_not_flagged(self):
        line = "USERS = (" + ", ".join(f"'user{i}'" for i in range(51)) + ")"
        self.assertFalse(is_truncated_literal_line(line))

    def test_short_line_is_not_flagged(self):
        self.assertFalse(is_truncated_literal_line("x = ('a'"))

    def test_balanced_brackets_with_odd_quotes_is_not_flagged(self):
        # An apostrophe inside a comment: odd quotes, but nothing left open.
        line = "y = 1  # " + "z" * 90 + " don't"
        self.assertFalse(is_truncated_literal_line(line))

    def test_empty_and_none_are_safe(self):
        self.assertFalse(is_truncated_literal_line(""))
        self.assertFalse(is_truncated_literal_line(None))


class SurrenderMarkerTest(unittest.TestCase):
    def test_mask_marker_is_detected(self):
        self.assertTrue(has_decompiler_surrender_markers("a = <mask_3>\n"))

    def test_code_object_repr_is_detected(self):
        self.assertTrue(has_decompiler_surrender_markers("f = <Code123 code object>\n"))

    def test_clean_source_is_not_flagged(self):
        self.assertFalse(has_decompiler_surrender_markers("def f():\n    return 1\n"))

    def test_similar_looking_text_is_not_flagged(self):
        self.assertFalse(has_decompiler_surrender_markers("x = a < mask and b > c\n"))
        self.assertFalse(has_decompiler_surrender_markers("# mask_3 is a variable name\n"))

    def test_empty_and_none_are_safe(self):
        self.assertFalse(has_decompiler_surrender_markers(""))
        self.assertFalse(has_decompiler_surrender_markers(None))


if __name__ == "__main__":
    unittest.main()
