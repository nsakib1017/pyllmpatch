"""clamp_syntax_segment narrows an oversized SyntaxSegment around the error line.

The segment carries start_line/end_line, and the repaired window is spliced back into the file at
exactly that span -- so a clamp that trims `text` without moving the bounds would write the repair
to the WRONG lines and corrupt the file. These tests pin the bounds/text/roles staying consistent,
which is the property that makes clamping safe to enable.
"""
import unittest

from utils.file_helpers import SyntaxSegment, clamp_syntax_segment


def chars_over_4(text: str) -> int:
    return (len(text) + 3) // 4


def seg(lines, start_line=10, kind="window"):
    return SyntaxSegment(
        text="".join(lines),
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        base_indent="",
        anchor_indent="    ",
        segment_kind=kind,
        line_roles=tuple("I" for _ in lines),
    )


class ClampSyntaxSegmentTest(unittest.TestCase):
    def test_segment_within_budget_is_unchanged(self):
        s = seg(["a = 1\n", "b = 2\n"])
        self.assertIs(clamp_syntax_segment(s, 10, 1000, chars_over_4), s)

    def test_oversized_segment_is_narrowed_with_consistent_bounds(self):
        lines = ["x" * 39 + "\n" for _ in range(40)]  # 10 tokens each
        s = seg(lines, start_line=100)
        out = clamp_syntax_segment(s, 120, 25, chars_over_4)
        self.assertIsNotNone(out)
        self.assertLess(len(out.text.splitlines()), 40)
        # bounds must describe exactly the retained text
        self.assertEqual(out.end_line - out.start_line + 1, len(out.text.splitlines()))
        self.assertEqual(len(out.line_roles), len(out.text.splitlines()))
        self.assertLessEqual(chars_over_4(out.text), 25)
        # retained text must be a contiguous slice of the original at the reported offset
        self.assertEqual(out.text, "".join(lines[out.start_line - 100:out.end_line - 100 + 1]))

    def test_clamped_span_still_contains_the_error_line(self):
        lines = ["y" * 39 + "\n" for _ in range(40)]
        s = seg(lines, start_line=100)
        out = clamp_syntax_segment(s, 130, 25, chars_over_4)
        self.assertTrue(out.start_line <= 130 <= out.end_line)

    def test_unwindowable_error_line_returns_none(self):
        s = seg(["a = 1\n", "z" * 100_000 + "\n"], start_line=5)
        self.assertIsNone(clamp_syntax_segment(s, 6, 2048, chars_over_4))

    def test_preserves_indent_and_kind_metadata(self):
        lines = ["q" * 39 + "\n" for _ in range(40)]
        s = seg(lines, start_line=1, kind="codeobject")
        out = clamp_syntax_segment(s, 20, 25, chars_over_4)
        self.assertEqual(out.base_indent, s.base_indent)
        self.assertEqual(out.anchor_indent, s.anchor_indent)
        self.assertEqual(out.segment_kind, s.segment_kind)

    def test_none_segment_and_disabled_budget_are_safe(self):
        self.assertIsNone(clamp_syntax_segment(None, 1, 100, chars_over_4))
        s = seg(["a = 1\n"])
        self.assertIs(clamp_syntax_segment(s, 10, 0, chars_over_4), s)


if __name__ == "__main__":
    unittest.main()
