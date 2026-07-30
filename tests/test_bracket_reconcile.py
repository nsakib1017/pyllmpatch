"""Tests for statement-local bracket reconciliation + invalid-identifier renaming."""
import ast
import unittest

from utils.bracket_reconcile import reconcile_brackets, rename_invalid_targets


def _parses(t):
    try:
        ast.parse(t)
        return True
    except SyntaxError:
        return False


class ReconcileBracketsTest(unittest.TestCase):
    def test_deletes_surplus_closer(self):
        out, led = reconcile_brackets("x = f(a))\n")
        self.assertTrue(_parses(out))
        self.assertEqual(led["surplus_deleted"], 1)

    def test_rewrites_mismatched_closer(self):
        out, led = reconcile_brackets("x = [1, 2)\n")
        self.assertTrue(_parses(out))
        self.assertEqual(led["mismatched_rewritten"], 1)

    def test_appends_missing_closer_on_its_own_statement(self):
        out, led = reconcile_brackets("x = f(a\ny = 2\n")
        self.assertTrue(_parses(out))
        self.assertEqual(led["closers_appended"], 1)
        self.assertIn("y = 2", out)

    def test_leaves_balanced_source_untouched(self):
        src = "x = f(a)\ny = [1, 2]\n"
        out, led = reconcile_brackets(src)
        self.assertEqual(out, src)
        self.assertEqual(led["total_edits"], 0)

    def test_respects_multi_line_statement(self):
        src = "x = f(\n    1,\n    2,\n)\n"
        out, led = reconcile_brackets(src)
        self.assertEqual(out, src)
        self.assertEqual(led["total_edits"], 0)

    def test_ignores_brackets_inside_strings_and_comments(self):
        src = "x = ')('  # )(\n"
        out, led = reconcile_brackets(src)
        self.assertEqual(out, src)
        self.assertEqual(led["total_edits"], 0)

    def test_never_deletes_a_line(self):
        src = "a = 1))\nb = 2\n"
        out, _ = reconcile_brackets(src)
        self.assertIn("b = 2", out)
        self.assertEqual(len(out.splitlines()), len(src.splitlines()))


class RenameInvalidTargetsTest(unittest.TestCase):
    def test_renames_mac_address_target(self):
        out, led = rename_invalid_targets("1e:6c:34:93:68:64 = ['a', 'b']\n")
        self.assertTrue(_parses(out))
        self.assertEqual(led["renamed"], 1)
        self.assertIn("'a', 'b'", out)          # payload preserved verbatim

    def test_renames_guid_attribute_target(self):
        src = "7AB5C494-39F5-4941-9163-47F54D6D5016.False = [1]\n"
        out, led = rename_invalid_targets(src)
        self.assertTrue(_parses(out))
        self.assertEqual(led["renamed"], 1)

    def test_same_token_gets_the_same_new_name(self):
        src = "1e:6c = [1]\nx = 1e:6c\n"
        out, _ = rename_invalid_targets(src)
        names = {ln.split("=")[0].strip() for ln in out.splitlines() if ln.startswith("_")}
        self.assertEqual(len(names), 1)

    def test_valid_targets_untouched(self):
        src = "x = [1]\ny.z = 2\n"
        out, led = rename_invalid_targets(src)
        self.assertEqual(out, src)
        self.assertEqual(led["renamed"], 0)


if __name__ == "__main__":
    unittest.main()
