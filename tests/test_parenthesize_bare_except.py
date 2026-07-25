"""TDD spec for ``utils.reattach_source_code_object.parenthesize_bare_except``.

DIAGNOSIS: 9 of 184 oracle failures are all
``SyntaxError: multiple exception types must be parenthesized``. Some ground-truth sources
under ``pyllm_py_dataset`` (used by the oracle fixer AND the annotation-reconciliation
operator) contain the decompiler artifact ``except A, B:`` -- Python-2-style multiple
exception types WITHOUT parentheses, invalid in Python 3. When a GT fragment containing this
is reattached and recompiled, it raises ``SyntaxError``, so the oracle can't convert the
file. Confirmed: GT source ``headers.py`` line 239 ==
``except json.JSONDecodeError, ValueError:``.

``parenthesize_bare_except`` is a deterministic, idempotent, semantics-preserving
normalizer: it rewrites unparenthesized multi-type except clauses to the parenthesized
Python-3 form and leaves everything else -- including single-type excepts, ``as``-bound
excepts, already-parenthesized excepts, and commas anywhere else in the file (call args,
subscripts, comments, strings) -- untouched. It must tolerate input that does not compile
(that's the whole point -- the input contains the very SyntaxError being fixed), so it must
not use ``ast.parse``.
"""
from __future__ import annotations

import unittest

from utils.reattach_source_code_object import parenthesize_bare_except


class SingleStatementCasesTest(unittest.TestCase):
    """One except-clause header per case, exercising each shape named in the spec."""

    def test_single_type_unchanged(self):
        src = "try:\n    pass\nexcept X:\n    pass\n"
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_single_type_as_binding_unchanged(self):
        src = "try:\n    pass\nexcept X as e:\n    pass\n"
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_already_parenthesized_unchanged(self):
        src = "try:\n    pass\nexcept (X, Y):\n    pass\n"
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_already_parenthesized_with_as_unchanged(self):
        src = "try:\n    pass\nexcept (X, Y) as e:\n    pass\n"
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_bare_two_types_parenthesized(self):
        src = "try:\n    pass\nexcept X, Y:\n    pass\n"
        expected = "try:\n    pass\nexcept (X, Y):\n    pass\n"
        self.assertEqual(parenthesize_bare_except(src), expected)

    def test_bare_two_types_with_as_parenthesized(self):
        src = "try:\n    pass\nexcept X, Y as e:\n    pass\n"
        expected = "try:\n    pass\nexcept (X, Y) as e:\n    pass\n"
        self.assertEqual(parenthesize_bare_except(src), expected)

    def test_dotted_three_types_parenthesized(self):
        src = "try:\n    pass\nexcept a.B, c.D, E:\n    pass\n"
        expected = "try:\n    pass\nexcept (a.B, c.D, E):\n    pass\n"
        self.assertEqual(parenthesize_bare_except(src), expected)

    def test_confirmed_real_world_case(self):
        # headers.py line 239
        src = "def f():\n    try:\n        pass\n    except json.JSONDecodeError, ValueError:\n        pass\n"
        expected = (
            "def f():\n    try:\n        pass\n"
            "    except (json.JSONDecodeError, ValueError):\n        pass\n"
        )
        self.assertEqual(parenthesize_bare_except(src), expected)


class IdempotenceTest(unittest.TestCase):
    def test_running_twice_equals_running_once(self):
        cases = [
            "try:\n    pass\nexcept X, Y:\n    pass\n",
            "try:\n    pass\nexcept X, Y as e:\n    pass\n",
            "try:\n    pass\nexcept a.B, c.D, E:\n    pass\n",
            "try:\n    pass\nexcept X:\n    pass\n",
            "try:\n    pass\nexcept (X, Y):\n    pass\n",
        ]
        for src in cases:
            once = parenthesize_bare_except(src)
            twice = parenthesize_bare_except(once)
            self.assertEqual(once, twice, msg=f"not idempotent for: {src!r}")


class NoOverMatchTest(unittest.TestCase):
    def test_comment_comma_on_except_line_untouched(self):
        src = "try:\n    pass\nexcept SomeError:  # a, b\n    pass\n"
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_call_args_comma_untouched(self):
        src = "x = f(a, b)\n"
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_subscript_comma_untouched(self):
        src = "x = d[a, b]\n"
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_string_with_except_word_untouched(self):
        src = 'msg = "except X, Y:"\n'
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_mixed_file_only_bare_except_touched(self):
        src = (
            "def f(a, b):\n"
            "    d = {a: b}\n"
            "    try:\n"
            "        pass\n"
            "    except X, Y:\n"
            "        pass\n"
            "    except Z:\n"
            "        pass\n"
            "    return d[a, b]\n"
        )
        expected = (
            "def f(a, b):\n"
            "    d = {a: b}\n"
            "    try:\n"
            "        pass\n"
            "    except (X, Y):\n"
            "        pass\n"
            "    except Z:\n"
            "        pass\n"
            "    return d[a, b]\n"
        )
        self.assertEqual(parenthesize_bare_except(src), expected)


class ResultCompilesTest(unittest.TestCase):
    """The normalized output must actually compile where the original did not."""

    def test_original_fails_to_compile(self):
        src = "def f():\n    try:\n        pass\n    except X, Y:\n        pass\n"
        with self.assertRaises(SyntaxError):
            compile(src, "<test>", "exec")

    def test_normalized_compiles(self):
        src = "def f():\n    try:\n        pass\n    except X, Y:\n        pass\n"
        normalized = parenthesize_bare_except(src)
        compile(normalized, "<test>", "exec")  # must not raise


class RobustnessTest(unittest.TestCase):
    def test_unrelated_tokenize_error_returns_source_unchanged(self):
        # Unterminated string -- tokenize itself cannot lex this; must not raise.
        src = "x = 'unterminated\n"
        self.assertEqual(parenthesize_bare_except(src), src)

    def test_empty_source_unchanged(self):
        self.assertEqual(parenthesize_bare_except(""), "")

    def test_no_except_clauses_unchanged(self):
        src = "x = 1\ny = 2\n"
        self.assertEqual(parenthesize_bare_except(src), src)


if __name__ == "__main__":
    unittest.main()
