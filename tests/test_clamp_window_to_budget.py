"""clamp_window_to_budget shrinks an oversized repair window around the error line instead of
DISCARDING the whole file.

Directive: syntactic repair operates inside a <=N-token window and SLIDES to cover the file. A file
is only out of scope when the ERROR LINE ITSELF cannot fit the budget (a minified/packer one-liner).
Previously any window over budget discarded the file outright, silently dropping ~29% of the malware
syntactic corpus -- 69% of which were ordinary multi-line files whose window was merely too wide.

The counter here is injected so these tests stay hermetic (no tokenizer/model load): each test uses a
simple deterministic count so the arithmetic of the clamp is what is under test, not the tokenizer.
"""
import unittest

from pipeline.logging_utils import clamp_window_to_budget


def chars_over_4(text: str) -> int:
    """Deterministic stand-in for a tokenizer: 4 characters == 1 token."""
    return (len(text) + 3) // 4


class ClampWindowToBudgetTest(unittest.TestCase):
    def test_window_already_within_budget_is_returned_whole(self):
        lines = ["a = 1\n", "b = 2\n", "c = 3\n"]
        self.assertEqual(
            clamp_window_to_budget(lines, 1, 1000, chars_over_4), (0, 3)
        )

    def test_oversized_window_is_shrunk_and_fits_budget(self):
        # 40 lines of 40 chars = 10 tokens each; a 25-token budget cannot hold them all.
        lines = ["x" * 39 + "\n" for _ in range(40)]
        span = clamp_window_to_budget(lines, 20, 25, chars_over_4)
        self.assertIsNotNone(span)
        start, end = span
        self.assertLess(end - start, 40)
        self.assertLessEqual(chars_over_4("".join(lines[start:end])), 25)

    def test_clamped_window_always_contains_the_error_line(self):
        lines = ["y" * 39 + "\n" for _ in range(40)]
        for err in (0, 7, 20, 39):
            start, end = clamp_window_to_budget(lines, err, 25, chars_over_4)
            self.assertTrue(start <= err < end, f"error line {err} fell outside [{start},{end})")

    def test_error_line_alone_over_budget_is_unwindowable(self):
        # The packer/minified one-liner case -- genuinely out of scope.
        lines = ["a = 1\n", "z" * 100_000 + "\n", "b = 2\n"]
        self.assertIsNone(clamp_window_to_budget(lines, 1, 2048, chars_over_4))

    def test_long_line_elsewhere_does_not_disqualify_the_file(self):
        # 17% of IN-SCOPE files carry a huge line away from the error; they must stay repairable.
        lines = ["a = 1\n", "b = 2\n", "z" * 100_000 + "\n"]
        span = clamp_window_to_budget(lines, 0, 50, chars_over_4)
        self.assertIsNotNone(span)
        start, end = span
        self.assertTrue(start <= 0 < end)
        self.assertLessEqual(chars_over_4("".join(lines[start:end])), 50)

    def test_disabled_budget_returns_whole_window(self):
        lines = ["a = 1\n", "b = 2\n"]
        self.assertEqual(clamp_window_to_budget(lines, 0, 0, chars_over_4), (0, 2))

    def test_empty_or_out_of_range_never_raises(self):
        self.assertIsNone(clamp_window_to_budget([], 0, 100, chars_over_4))
        span = clamp_window_to_budget(["a = 1\n"], 99, 100, chars_over_4)
        self.assertIsNotNone(span)


if __name__ == "__main__":
    unittest.main()
