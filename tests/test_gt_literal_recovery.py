"""Tests for mechanical truncated-literal recovery (utils/gt_literal_recovery.py).

Written before the implementation: every test below must fail on an empty module first.
"""
import ast
import unittest

from utils.gt_literal_recovery import (
    LiteralIndex,
    find_truncated_lines,
    repair_source,
)


class FindTruncatedLinesTest(unittest.TestCase):
    def test_flags_line_with_unclosed_string(self):
        src = "x = 1\ny = {'a', 'b', 'c\nz = 2\n"
        self.assertEqual([n for n, _ in find_truncated_lines(src)], [2])

    def test_ignores_balanced_lines(self):
        src = "x = {'a', 'b'}\ny = \"it's fine\"\n"
        self.assertEqual(find_truncated_lines(src), [])

    def test_ignores_escaped_quote(self):
        src = "x = 'it\\'s ok'\n"
        self.assertEqual(find_truncated_lines(src), [])

    def test_ignores_comment_apostrophe(self):
        src = "x = 1  # don't count this\n"
        self.assertEqual(find_truncated_lines(src), [])


class LiteralIndexTest(unittest.TestCase):
    def test_indexes_container_by_every_element(self):
        idx = LiteralIndex([frozenset({"alpha", "beta"}), ("gamma",)])
        self.assertEqual(len(idx.lookup("alpha")), 1)
        self.assertEqual(len(idx.lookup("beta")), 1)
        self.assertEqual(len(idx.lookup("gamma")), 1)
        self.assertEqual(idx.lookup("missing"), [])

    def test_best_match_prefers_superset_of_all_visible(self):
        small = frozenset({"a", "b"})
        big = frozenset({"a", "b", "c", "d"})
        idx = LiteralIndex([small, big])
        self.assertIs(idx.best_match(["a", "b"]), big)

    def test_best_match_rejects_container_missing_a_visible_element(self):
        idx = LiteralIndex([frozenset({"a", "x"})])
        self.assertIsNone(idx.best_match(["a", "b"]))


class RepairSourceTest(unittest.TestCase):
    def test_rebuilds_truncated_set_from_ground_truth(self):
        src = "if name in {'a', 'b', 'c\n    pass\n"
        idx = LiteralIndex([frozenset({"a", "b", "c", "d", "e"})])
        out, stats = repair_source(src, idx)
        ast.parse(out)
        self.assertEqual(stats["recovered"], 1)
        self.assertIn("'e'", out)

    def test_lossy_close_when_ground_truth_has_no_match(self):
        src = "if name in {'a', 'b', 'c\n    pass\n"
        out, stats = repair_source(src, LiteralIndex([]))
        ast.parse(out)
        self.assertEqual(stats["lossy_closed"], 1)
        self.assertEqual(stats["recovered"], 0)

    def test_repairs_every_truncated_line_in_one_pass(self):
        src = "a = {'p', 'q\nb = {'r', 's\n"
        idx = LiteralIndex([frozenset({"p", "q", "Q"}), frozenset({"r", "s", "S"})])
        out, stats = repair_source(src, idx)
        ast.parse(out)
        self.assertEqual(stats["recovered"], 2)

    def test_leaves_clean_source_untouched(self):
        src = "x = {'a', 'b'}\n"
        out, stats = repair_source(src, LiteralIndex([]))
        self.assertEqual(out, src)
        self.assertEqual(stats["recovered"] + stats["lossy_closed"], 0)

    def test_preserves_indentation_and_statement_prefix(self):
        src = "def f(v):\n    if v.lower() in {'x', 'y\n        return 1\n"
        idx = LiteralIndex([frozenset({"x", "y", "z"})])
        out, _ = repair_source(src, idx)
        ast.parse(out)
        self.assertTrue(out.splitlines()[1].startswith("    if v.lower() in {"))

    def test_closes_an_enclosing_open_call(self):
        src = "def f(v):\n    return any(v in {'x', 'y\n"
        idx = LiteralIndex([frozenset({"x", "y", "z"})])
        out, _ = repair_source(src, idx)
        ast.parse(out)  # the open `any(` must be closed too, or this raises


if __name__ == "__main__":
    unittest.main()
