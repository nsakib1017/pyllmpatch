"""Tests for chunk-granularity forced parsing (utils/chunk_neutralize.py)."""
import ast
import unittest

from utils.chunk_neutralize import force_parseable, top_level_chunks


def _parses(t):
    try:
        ast.parse(t)
        return True
    except SyntaxError:
        return False


class TopLevelChunksTest(unittest.TestCase):
    def test_splits_on_column_zero_statements(self):
        src = "import os\ndef f():\n    return 1\nx = 2\n"
        self.assertEqual(len(top_level_chunks(src.splitlines())), 3)

    def test_keeps_indented_body_with_its_header(self):
        src = "def f():\n    a = 1\n    b = 2\n"
        chunks = top_level_chunks(src.splitlines())
        self.assertEqual(chunks, [(0, 2)])

    def test_does_not_split_inside_open_brackets(self):
        src = "x = [\n1,\n2,\n]\n"
        self.assertEqual(top_level_chunks(src.splitlines()), [(0, 3)])


class ForceParseableTest(unittest.TestCase):
    def test_valid_source_unchanged(self):
        src = "import os\n\n\ndef f():\n    return 1\n"
        out, led = force_parseable(src)
        self.assertEqual(out, src)
        self.assertEqual(led["chunks_neutralised"], 0)

    def test_neutralises_only_the_broken_chunk(self):
        src = "def good():\n    return 1\ndef bad(:\n    @@@ ???\nx = 3\n"
        out, led = force_parseable(src)
        self.assertTrue(_parses(out))
        self.assertEqual(led["chunks_neutralised"], 1)
        self.assertIn("def good():", out)      # the good chunk survives verbatim
        self.assertIn("x = 3", out)

    def test_block_level_defect_is_handled(self):
        src = "def f():\n    x = 1\n        stray_indent = 2\n    return x\n"
        out, led = force_parseable(src)
        self.assertTrue(_parses(out))

    def test_missing_handler_block_is_handled(self):
        src = "def f():\n    try:\n        g()\ny = 1\n"
        out, led = force_parseable(src)
        self.assertTrue(_parses(out))
        self.assertIn("y = 1", out)

    def test_preserves_every_byte_of_a_neutralised_chunk(self):
        src = "def bad(:\n    @@@ secret_payload ???\n"
        out, _ = force_parseable(src)
        self.assertTrue(_parses(out))
        self.assertIn("secret_payload", out)

    def test_never_deletes_a_line(self):
        src = "def bad(:\n    @@@\nz = 9\n"
        out, led = force_parseable(src)
        self.assertEqual(led["lines_deleted"], 0)
        self.assertEqual(len(out.splitlines()), len(src.splitlines()))

    def test_always_parses_however_broken(self):
        src = (
            "def a(:\n"
            "    try:\n"
            "        x = {'p', 'q\n"
            "    except:\n"
            "  bad_indent\n"
            "108 = [1, 2\n"
            "@deco @deco2 broken))\n"
            "prose that was a docstring = [{'k': 1}]\n"
        )
        out, led = force_parseable(src)
        self.assertTrue(_parses(out))
        self.assertEqual(led["lines_deleted"], 0)


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
