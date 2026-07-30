"""Tests for guarantee-by-construction closure (utils/structural_closure.py)."""
import ast
import unittest

from utils.structural_closure import close_source


def _parses(text):
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


class ClosesUnfinishedConstructsTest(unittest.TestCase):
    def test_closes_unclosed_bracket_at_eof(self):
        out, led = close_source("x = [1, 2, 3\n")
        self.assertTrue(_parses(out))
        self.assertEqual(led["brackets_closed"], 1)

    def test_closes_unterminated_string(self):
        out, led = close_source("x = 'abc\n")
        self.assertTrue(_parses(out))
        self.assertEqual(led["strings_closed"], 1)

    def test_synthesises_missing_except(self):
        out, led = close_source("try:\n    f()\n")
        self.assertTrue(_parses(out))
        self.assertEqual(led["handlers_synthesised"], 1)

    def test_gives_headerless_block_a_body(self):
        out, led = close_source("def f():\nx = 1\n")
        self.assertTrue(_parses(out))
        self.assertGreaterEqual(led["bodies_inserted"], 1)

    def test_handles_several_defects_in_one_file(self):
        src = "def f():\n    try:\n        y = {'a', 'b\n"
        out, led = close_source(src)
        self.assertTrue(_parses(out))
        self.assertGreaterEqual(led["strings_closed"] + led["brackets_closed"], 1)
        self.assertEqual(led["handlers_synthesised"], 1)


class OrphanedHandlerTest(unittest.TestCase):
    def test_inserts_try_for_orphaned_except(self):
        src = "def f():\n    return g()\n    except:\n        pass\n"
        out, led = close_source(src)
        self.assertTrue(_parses(out))
        self.assertEqual(led["tries_inserted"], 1)

    def test_orphaned_finally_also_gets_a_try(self):
        src = "def f():\n    x = 1\n    finally:\n        pass\n"
        out, led = close_source(src)
        self.assertTrue(_parses(out))
        self.assertEqual(led["tries_inserted"], 1)


class HonestyTest(unittest.TestCase):
    def test_valid_source_is_returned_untouched(self):
        src = "def f():\n    return {'a', 'b'}\n"
        out, led = close_source(src)
        self.assertEqual(out, src)
        self.assertEqual(led["total_edits"], 0)
        self.assertTrue(led["faithful"])

    def test_any_edit_marks_the_result_unfaithful(self):
        out, led = close_source("x = [1, 2\n")
        self.assertFalse(led["faithful"])
        self.assertGreater(led["total_edits"], 0)

    def test_never_deletes_a_line(self):
        src = "def f():\n    try:\n        y = [1, 2\n"
        out, led = close_source(src)
        for line in src.splitlines():
            if line.strip():
                self.assertIn(line.rstrip("\n").rstrip(), out)
        self.assertEqual(led["lines_deleted"], 0)

    def test_reports_failure_rather_than_looping_forever(self):
        out, led = close_source("def f(:\n", max_rounds=3)
        self.assertIn("rounds", led)
        self.assertLessEqual(led["rounds"], 3)


if __name__ == "__main__":
    unittest.main()


class ExtendedHandlerTest(unittest.TestCase):
    """Handlers added when the goal became compilation at any cost (fidelity tracked separately)."""

    def test_neutralises_line_no_handler_can_fix(self):
        out, led = close_source("def f():\n    @tag @nonce garbage ===\n")
        self.assertTrue(_parses(out))
        self.assertGreaterEqual(led["lines_neutralised"], 1)

    def test_neutralised_line_keeps_its_bytes_as_data(self):
        src = "x ===== !!! @@@\n"
        out, led = close_source(src)
        self.assertTrue(_parses(out))
        self.assertIn("!!!", out)          # content retained, as a string
        self.assertEqual(led["lines_deleted"], 0)

    def test_renames_literal_assignment_target(self):
        out, led = close_source("108 = ['a', 'b']\n")
        self.assertTrue(_parses(out))
        self.assertIn("'a', 'b'", out)     # payload preserved

    def test_fixes_assignment_used_as_comparison(self):
        out, led = close_source("if x = 1:\n    pass\n")
        self.assertTrue(_parses(out))

    def test_strips_stray_line_continuation(self):
        out, led = close_source("x = 1 \\ 2\n")
        self.assertTrue(_parses(out))

    def test_always_reaches_a_parsing_file(self):
        nasty = (
            "def f():\n"
            "    try:\n"
            "        y = {'a', 'b\n"
            "    except:\n"
            "        pass\n"
            "108 = [1, 2\n"
            "Some prose that was a docstring = [{'x': 1}]\n"
            "@deco @deco2 broken))\n"
        )
        out, led = close_source(nasty)
        self.assertTrue(_parses(out))
        self.assertEqual(led["lines_deleted"], 0)
        self.assertFalse(led["faithful"])


class NeutralisationMarkerTest(unittest.TestCase):
    """A neutralised line must be machine-findable and cost no bytecode."""

    def test_marker_is_appended(self):
        out, _ = close_source("def f():\n    @a @b broken))\n")
        self.assertTrue(_parses(out))
        self.assertIn("# pyllmpatch: unparsed", out)

    def test_marker_line_still_emits_no_bytecode(self):
        import dis
        plain = compile("def f():\n    'x = 1'\n    g()\n", "<t>", "exec")
        marked = compile("def f():\n    'x = 1'  # pyllmpatch: unparsed\n    g()\n", "<t>", "exec")
        fp = [c for c in plain.co_consts if hasattr(c, "co_code")][0]
        fm = [c for c in marked.co_consts if hasattr(c, "co_code")][0]
        self.assertEqual(
            [i.opname for i in dis.get_instructions(fp)],
            [i.opname for i in dis.get_instructions(fm)],
        )

    def test_original_text_is_recoverable_from_the_marked_line(self):
        out, _ = close_source("def f():\n    @a @b broken))\n")
        marked = [ln for ln in out.splitlines() if "pyllmpatch: unparsed" in ln]
        self.assertEqual(len(marked), 1)
        self.assertIn("@a @b broken", marked[0])

    def test_clean_source_gets_no_marker(self):
        out, _ = close_source("def f():\n    return 1\n")
        self.assertNotIn("pyllmpatch", out)
