"""Deterministic operators for the two dominant structural defects in decompiled malware.

Measured on 593 delete-only malware files (files that only compiled by deleting a median of 243
lines of real code):
  * 400 (67%) contain a `try:` with NO except/finally  -> complete_orphan_try
  * 81  (13%) contain a literal on the left of `=`     -> literal_lhs_rename

Both trade a MINIMAL, explicit edit for the whole body that deletion would otherwise destroy:
completing a try block synthesizes two scaffold lines instead of discarding the entire suite.
Because they add scaffolding they are NOT "genuine" repairs in the strict sense -- callers must
report them in their own bucket (ranked below genuine, above deletion) rather than folding them in,
which is exactly the fidelity mistake balance_delimiters already makes.

Both are pure source->source transforms so the prepass keeps its compile gate: a candidate that
does not actually improve the parse is rejected upstream.
"""
import unittest

from utils.syntactic_prepass import complete_orphan_try, literal_lhs_rename


def compiles(src):
    try:
        compile(src, "<t>", "exec")
        return True
    except SyntaxError:
        return False


class CompleteOrphanTryTest(unittest.TestCase):
    def test_completes_a_try_with_no_handler(self):
        src = "def f():\n    try:\n        x = 1\n"
        out = complete_orphan_try(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(compiles(out), out)
        self.assertIn("x = 1", out)          # the real code survives
        self.assertIn("except", out)

    def test_leaves_a_complete_try_alone(self):
        src = "try:\n    x = 1\nexcept ValueError:\n    pass\n"
        self.assertIsNone(complete_orphan_try(src, None))

    def test_leaves_try_finally_alone(self):
        src = "try:\n    x = 1\nfinally:\n    y = 2\n"
        self.assertIsNone(complete_orphan_try(src, None))

    def test_handler_is_inserted_at_the_try_indent(self):
        src = "class C:\n    def m(self):\n        try:\n            a = 1\n"
        out = complete_orphan_try(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(compiles(out), out)
        handler = [l for l in out.splitlines() if l.lstrip().startswith("except")][0]
        self.assertEqual(len(handler) - len(handler.lstrip()), 8)  # same indent as `try:`

    def test_trailing_code_after_the_try_block_is_preserved(self):
        src = "try:\n    a = 1\nb = 2\n"
        out = complete_orphan_try(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(compiles(out), out)
        self.assertIn("b = 2", out)
        self.assertLess(out.index("except"), out.index("b = 2"))

    def test_no_try_at_all_returns_none(self):
        self.assertIsNone(complete_orphan_try("x = 1\ny = 2\n", None))

    def test_never_raises(self):
        for src in ("", None, "try:", "    "):
            complete_orphan_try(src, None)


class LiteralLhsRenameTest(unittest.TestCase):
    def test_renames_numeric_left_hand_side(self):
        src = "72 = compute(1, 2)\n"
        out = literal_lhs_rename(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(compiles(out), out)
        self.assertIn("compute(1, 2)", out)   # the analysable expression survives

    def test_preserves_indentation(self):
        src = "def f():\n    5 = g()\n"
        out = literal_lhs_rename(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(compiles(out), out)
        renamed = [l for l in out.splitlines() if "g()" in l][0]
        self.assertTrue(renamed.startswith("    "))

    def test_ignores_comparisons_and_valid_code(self):
        self.assertIsNone(literal_lhs_rename("if x == 72:\n    pass\n", None))
        self.assertIsNone(literal_lhs_rename("a = 72\n", None))

    def test_ignores_keyword_arguments(self):
        self.assertIsNone(literal_lhs_rename("f(timeout=72)\n", None))

    def test_never_raises(self):
        for src in ("", None, "== =", "72 ="):
            literal_lhs_rename(src, None)


if __name__ == "__main__":
    unittest.main()
