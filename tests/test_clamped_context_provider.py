"""_clamped_context_provider wraps a syntax-context provider so the window handed to the LLM is
budget-clamped around the error line rather than the file being discarded.

This is the seam that actually makes the sliding window reach the model: the runner passes the
wrapped provider as `segment_syntax_context`, so every retry re-localizes to the current error and
receives a window that fits the generation budget.
"""
import unittest

from pipeline.runner import _clamped_context_provider
from utils.file_helpers import SyntaxSegment


def chars_over_4(text: str) -> int:
    return (len(text) + 3) // 4


def make_segment(lines, start_line=1):
    return SyntaxSegment(
        text="".join(lines),
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        base_indent="",
        anchor_indent="",
        segment_kind="window",
        line_roles=tuple("I" for _ in lines),
    )


class ClampedContextProviderTest(unittest.TestCase):
    def test_disabled_budget_returns_original_provider(self):
        def provider(path, err_line, err_desc, expansion):
            return make_segment(["a = 1\n"])

        self.assertIs(_clamped_context_provider(provider, 0, chars_over_4), provider)

    def test_within_budget_window_passes_through_unchanged(self):
        seg = make_segment(["a = 1\n", "b = 2\n"])
        wrapped = _clamped_context_provider(lambda *a: seg, 1000, chars_over_4)
        self.assertIs(wrapped("p", 1, "err", 0), seg)

    def test_oversized_window_is_clamped_to_budget(self):
        lines = ["x" * 39 + "\n" for _ in range(40)]
        wrapped = _clamped_context_provider(lambda *a: make_segment(lines, 1), 25, chars_over_4)
        out = wrapped("p", 20, "err", 0)
        self.assertIsNotNone(out)
        self.assertLessEqual(chars_over_4(out.text), 25)
        self.assertTrue(out.start_line <= 20 <= out.end_line)

    def test_unwindowable_window_yields_none(self):
        lines = ["a = 1\n", "z" * 100_000 + "\n"]
        wrapped = _clamped_context_provider(lambda *a: make_segment(lines, 1), 2048, chars_over_4)
        self.assertIsNone(wrapped("p", 2, "err", 0))

    def test_none_segment_from_inner_provider_is_passed_through(self):
        wrapped = _clamped_context_provider(lambda *a: None, 2048, chars_over_4)
        self.assertIsNone(wrapped("p", 1, "err", 0))

    def test_provider_arguments_are_forwarded(self):
        seen = {}

        def provider(path, err_line, err_desc, expansion):
            seen.update(path=path, err_line=err_line, err_desc=err_desc, expansion=expansion)
            return make_segment(["a = 1\n"])

        _clamped_context_provider(provider, 100, chars_over_4)("f.py", 7, "boom", 3)
        self.assertEqual(seen, {"path": "f.py", "err_line": 7, "err_desc": "boom", "expansion": 3})


if __name__ == "__main__":
    unittest.main()
