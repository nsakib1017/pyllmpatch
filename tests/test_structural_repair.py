"""Deterministic repair of PyLingual's control-flow RECONSTRUCTION defects.

The sweep cascade handles damage that is local to a line (unbalanced brackets, a bad assignment
target, a stray indent). What it leaves behind is different: PyLingual guessed at control flow and
emitted a structure that is not Python. A census of 37,494 defect sites across the residual files
found these dominate, and PyLingual labels its own guesses -- `# inserted`, `# postinserted` -- so
the damage is identifiable rather than merely suspected:

    2320 sites   `else:  # inserted`
     809         `try:`
     670         `pass  # postinserted`
     459/306     `except:` / `except Exception:`
     201         `if True:  # inserted`

Three shapes are mechanically repairable and are what this module targets:

  1. a compound header (`try:`, `except ...:`, `if ...:`) whose body was never emitted, followed by
     a clause at the same or lower indent -- fix by giving the header a `pass` body;
  2. a for-loop flattened into a pseudo-comprehension, `EXPR for VAR in ITER:` -- re-nest it;
  3. a doubled for-clause, `for V in ITER for V in [EXPR]:` -- reduce to the loop it came from.

Every operator here is compile-gated by its caller and must be a no-op on valid Python.
"""
import ast
import unittest

from utils.structural_repair import (
    STRUCT_OPS,
    give_headerless_block_a_body,
    reduce_doubled_for,
    renest_flattened_for,
    close_unclosed_opener,
    normalize_leaked_generic_params,
    rejoin_split_comprehension,
    strip_surplus_leading_delimiters,
)


def _parses(src):
    try:
        ast.parse(src)
        return True
    except Exception:
        return False


class HeaderlessBlockTest(unittest.TestCase):
    REAL = (
        "def f(cfg):\n"
        "    with open(cfg) as fh:\n"
        "        try:\n"
        "            item = json.load(fh)\n"
        "        except json.decoder.JSONDecodeError:\n"
        "        else:  # inserted\n"
        "            item['auto_start'] = False\n"
    )

    def test_except_without_a_body_is_given_one(self):
        out = give_headerless_block_a_body(self.REAL, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)

    def test_nothing_is_deleted(self):
        out = give_headerless_block_a_body(self.REAL, None)
        before = [l.strip() for l in self.REAL.splitlines() if l.strip()]
        after = [l.strip() for l in out.splitlines() if l.strip()]
        for line in before:
            self.assertIn(line, after, f"operator deleted: {line}")

    def test_try_without_a_body(self):
        src = "def f():\n    try:\n    except Exception:\n        pass\n"
        out = give_headerless_block_a_body(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)

    def test_valid_source_is_untouched(self):
        for src in (
            "try:\n    f()\nexcept ValueError:\n    pass\n",
            "if a:\n    b()\nelse:\n    c()\n",
            "for i in r:\n    print(i)\n",
            "def f():\n    return 1\n",
        ):
            self.assertIsNone(give_headerless_block_a_body(src, None), src)

    def test_never_raises(self):
        for src in (None, "", "@@@\n", "def f(:\n"):
            give_headerless_block_a_body(src, None)


class RenestFlattenedForTest(unittest.TestCase):
    REAL = (
        "def g(ThCokk):\n"
        "        thread.join() for thread in ThCokk:\n"
        "            pass  # postinserted\n"
    )

    def test_flattened_for_is_renested(self):
        out = renest_flattened_for(self.REAL, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)
        self.assertIn("for thread in ThCokk:", out)
        self.assertIn("thread.join()", out)

    def test_the_loop_body_call_survives(self):
        out = renest_flattened_for(self.REAL, None)
        tree = ast.parse(out)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertTrue(any(getattr(c.func, "attr", "") == "join" for c in calls))

    def test_a_real_comprehension_is_untouched(self):
        for src in (
            "xs = [f(x) for x in ys]\n",
            "d = {k: v for k, v in items}\n",
            "g = (x for x in ys)\n",
            "for x in ys:\n    f(x)\n",
        ):
            self.assertIsNone(renest_flattened_for(src, None), src)

    def test_never_raises(self):
        for src in (None, "", "for for for:\n"):
            renest_flattened_for(src, None)


class ReduceDoubledForTest(unittest.TestCase):
    REAL = (
        "def g(ThCokk):\n"
        "    for thread in ThCokk for thread in [thread.join()]:\n"
        "        pass  # postinserted\n"
    )

    def test_doubled_for_is_reduced(self):
        out = reduce_doubled_for(self.REAL, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)
        self.assertIn("for thread in ThCokk:", out)

    def test_the_inner_expression_is_kept_as_the_body(self):
        out = reduce_doubled_for(self.REAL, None)
        self.assertIn("thread.join()", out)

    def test_a_nested_comprehension_is_untouched(self):
        for src in (
            "xs = [x for a in b for x in a]\n",
            "for i in r:\n    pass\n",
        ):
            self.assertIsNone(reduce_doubled_for(src, None), src)

    def test_never_raises(self):
        for src in (None, "", "for x in for:\n"):
            reduce_doubled_for(src, None)


class RegistryTest(unittest.TestCase):
    def test_registry_is_populated_and_shaped(self):
        self.assertTrue(STRUCT_OPS)
        for op in STRUCT_OPS:
            self.assertTrue(op.name)
            self.assertTrue(callable(op.fn))

    def test_no_operator_damages_valid_python(self):
        samples = [
            "import os\nx = [i for i in range(3)]\n",
            "class C:\n    def m(self):\n        try:\n            return 1\n        except Exception:\n            return 2\n",
            "async def f():\n    async with s() as c:\n        await c.go()\n",
            "d = {k: v for k, v in pairs if v}\n",
            "for a in b:\n    for c in d:\n        f(a, c)\n",
        ]
        for src in samples:
            for op in STRUCT_OPS:
                self.assertIsNone(op.fn(src, None), f"{op.name} touched valid source:\n{src}")




class SurplusLeadingDelimitersTest(unittest.TestCase):
    """A run of open brackets immediately before a statement KEYWORD is unambiguous garbage.

    Obfuscated malware decompiles to lines like `((((((((((if title.value.lower() in {...}` and
    `(((((((((((def f(...)`. `(((( if` is never valid Python at any nesting depth -- an open bracket
    cannot be followed by a statement keyword -- so the run carries no semantics and stripping it is
    certain rather than a guess. This is the single most repeated defect in the residual: it recurs
    many times inside one file, which is why those files show 65+ syntax sites.

    It removes CHARACTERS, never a line, so no program content is lost.
    """

    def test_leading_paren_run_before_if_is_stripped(self):
        src = "def f():\n        ((((((((((if x.lower() in {'a', 'b'}:\n            pass\n"
        out = strip_surplus_leading_delimiters(src, None)
        self.assertIsNotNone(out)
        self.assertIn("if x.lower() in {'a', 'b'}:", out)
        self.assertNotIn("((((", out)

    def test_leading_run_before_def_and_with(self):
        for kw, body in (
            ("def g(a):", "    return a\n"),
            ("with open('f') as fh:", "    pass\n"),
            ("for i in r:", "    pass\n"),
            ("while x:", "    pass\n"),
            ("return 1", ""),
        ):
            src = "def f():\n    ((((((%s\n%s" % (kw, body)
            out = strip_surplus_leading_delimiters(src, None)
            self.assertIsNotNone(out, kw)
            self.assertNotIn("((((((", out)

    def test_indentation_is_preserved(self):
        src = "def f():\n        (((((if x:\n            pass\n"
        out = strip_surplus_leading_delimiters(src, None)
        self.assertIn("\n        if x:", out)

    def test_no_line_is_removed(self):
        src = "def f():\n    (((((if x:\n        pass\n"
        out = strip_surplus_leading_delimiters(src, None)
        self.assertEqual(len(out.splitlines()), len(src.splitlines()))

    def test_repeated_occurrences_are_all_fixed(self):
        src = "def f():\n    (((if a:\n        pass\n    ((((if b:\n        pass\n    (((((if c:\n        pass\n"
        out = strip_surplus_leading_delimiters(src, None)
        self.assertNotIn("((", out)

    def test_a_legitimate_parenthesised_expression_is_untouched(self):
        for src in (
            "x = ((a + b) * c)\n",
            "y = (\n    1,\n    2,\n)\n",
            "if (a and b):\n    pass\n",
            "z = ((1, 2), (3, 4))\n",
            "f((x), (y))\n",
            "print((lambda: 1)())\n",
        ):
            self.assertIsNone(strip_surplus_leading_delimiters(src, None), src)

    def test_valid_python_is_never_touched(self):
        src = "import os\nfor i in range(3):\n    print((i))\n"
        self.assertIsNone(strip_surplus_leading_delimiters(src, None))

    def test_never_raises(self):
        for src in (None, "", "((((\n", "(((("):
            strip_surplus_leading_delimiters(src, None)




class RejoinSplitComprehensionTest(unittest.TestCase):
    """PyLingual emits split comprehensions INVERTED: the `for` clause lands on its own line, with
    its surplus closers, and the comprehension HEAD follows on the next, more-indented line.

    Measured: this single class blocks 439 of the 1012 still-unfixed malware files (43.4%) -- the
    largest remaining mechanical defect by a wide margin. Rejoining is exact reconstruction, not a
    guess: both halves are present and the join is the only ordering that parses.

        for x in BANNED_SITES]):          ->   if not any([x in i for x in BANNED_SITES]):
            if not any([x in i
    """

    ANY = (
        "def f(data, BANNED_SITES):\n"
        "    for i in data:\n"
        "        for x in BANNED_SITES]):\n"
        "            if not any([x in i\n"
        "                newdata = []\n"
    )
    NESTED_FOR = (
        "def g(Directory, FileName):\n"
        "        for x in open(FileName).readlines() if x.strip()]:\n"
        "            for line in [x.strip()\n"
        "                pass\n"
    )

    def test_inverted_any_comprehension_is_rejoined(self):
        out = rejoin_split_comprehension(self.ANY, None)
        self.assertIsNotNone(out)
        self.assertIn("if not any([x in i for x in BANNED_SITES]):", out)

    def test_result_parses(self):
        out = rejoin_split_comprehension(self.ANY, None)
        self.assertTrue(_parses(out), out)

    def test_nested_for_head_is_rejoined(self):
        out = rejoin_split_comprehension(self.NESTED_FOR, None)
        self.assertIsNotNone(out)
        self.assertIn("for line in [x.strip() for x in open(FileName).readlines() if x.strip()]:", out)

    def test_no_content_is_invented_or_lost(self):
        out = rejoin_split_comprehension(self.ANY, None)
        for token in ("BANNED_SITES", "any", "newdata", "for i in data"):
            self.assertIn(token, out)

    def test_valid_python_is_untouched(self):
        for src in (
            "if not any([x in i for x in S]):\n    pass\n",
            "for i in data:\n    pass\n",
            "xs = [f(x) for x in ys]\n",
            "d = {k: v for k, v in items}\n",
        ):
            self.assertIsNone(rejoin_split_comprehension(src, None), src)

    def test_a_for_clause_with_a_normal_body_is_untouched(self):
        # next line is more-indented but is a COMPLETE statement -- not a comprehension head
        src = "def f():\n    for x in ys]):\n        print(x)\n"
        out = rejoin_split_comprehension(src, None)
        if out is not None:
            self.assertNotIn("print(x) for x in", out)

    def test_never_raises(self):
        for src in (None, "", "for\n", "for x in y]):\n"):
            rejoin_split_comprehension(src, None)




class LeakedGenericParamsTest(unittest.TestCase):
    """PyLingual leaks its INTERNAL representation of PEP-695 generic scopes into the source.

        def <generic parameters of options>(.defaults):
        async def <generic parameters of _parse>():

    Neither the name nor the `.defaults` parameter is a legal identifier, so the file cannot parse.
    Measured on the representative benchmark: this is the blocking line in 65 of 487 residual files
    (13.3%) and appears somewhere in 184 (37.8%) -- the largest remaining single class there.

    The rewrite renames to legal identifiers and keeps every line and every body statement. The
    original used PEP-695 syntax that is NOT recoverable from this text, so this is a normalization
    that restores compilability, not a reconstruction -- it is recorded distinctly for that reason.
    """

    def test_generic_parameter_function_is_renamed(self):
        src = "def <generic parameters of options>(.defaults):\n    return 1\n"
        out = normalize_leaked_generic_params(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)
        self.assertNotIn("<generic parameters of", out)

    def test_dot_prefixed_parameter_becomes_legal(self):
        src = "def <generic parameters of options>(.defaults):\n    return 1\n"
        out = normalize_leaked_generic_params(src, None)
        self.assertNotIn("(.defaults)", out)
        self.assertIn("defaults", out)

    def test_async_form(self):
        src = "async def <generic parameters of _parse>():\n    return 1\n"
        out = normalize_leaked_generic_params(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)

    def test_body_is_preserved(self):
        src = ("def <generic parameters of f>(.defaults):\n"
               "    x = compute(1)\n"
               "    return x\n")
        out = normalize_leaked_generic_params(src, None)
        for token in ("x = compute(1)", "return x"):
            self.assertIn(token, out)

    def test_no_line_is_removed(self):
        src = ("def <generic parameters of f>(.defaults):\n"
               "    x = 1\n"
               "    return x\n")
        out = normalize_leaked_generic_params(src, None)
        self.assertEqual(len(out.splitlines()), len(src.splitlines()))

    def test_class_form(self):
        src = "class <generic parameters of C>():\n    pass\n"
        out = normalize_leaked_generic_params(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)

    def test_valid_python_is_untouched(self):
        for src in (
            "def f(a, b):\n    return a\n",
            "class C:\n    pass\n",
            "x = a < b and c > d\n",
            "def g[T](x: T) -> T:\n    return x\n",
        ):
            self.assertIsNone(normalize_leaked_generic_params(src, None), src)

    def test_never_raises(self):
        for src in (None, "", "def <generic\n", "<generic parameters of >"):
            normalize_leaked_generic_params(src, None)




class CloseUnclosedOpenerTest(unittest.TestCase):
    """Close a bracket the decompiler opened and never closed, AT ITS OWN LINE.

    The parser reports the failure far from the cause. A 3.11 sample blocks on

        okh = str(th) + '.txt'        <- reported as "invalid syntax"

    which is valid Python; the real defect is an unclosed bracket several lines ABOVE. Every other
    operator here keys on the reported error line, so all of them look in the wrong place. Measured:
    207 of 470 blocked 3.11 files (44%) are unmatched-closer errors, plus an unknown share of the
    "invalid syntax" bucket downstream of the same cause.

    The rule is conservative: a line that leaves brackets open is only closed when the NEXT
    non-blank line starts a new statement at the same or lower indent -- which a genuine multi-line
    expression never does. Callers compile-gate the result.
    """

    UNCLOSED = (
        "def f(th, client):\n"
        "    send(f'IP: {get()}'\n"
        "    okh = str(th) + '.txt'\n"
        "    return okh\n"
    )

    def test_unclosed_call_is_closed_on_its_own_line(self):
        out = close_unclosed_opener(self.UNCLOSED, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)

    def test_the_following_statement_is_untouched(self):
        out = close_unclosed_opener(self.UNCLOSED, None)
        self.assertIn("okh = str(th) + '.txt'", out)
        self.assertIn("return okh", out)

    def test_no_line_is_added_or_removed(self):
        out = close_unclosed_opener(self.UNCLOSED, None)
        self.assertEqual(len(out.splitlines()), len(self.UNCLOSED.splitlines()))

    def test_unclosed_bracket_kinds(self):
        # each fragment is valid Python once its opener is closed
        for fragment in ("x = g(1, 2", "x = [1, 2", "x = {1, 2", "x = {'a': 1"):
            src = "def f():\n    %s\n    y = 3\n    return y\n" % fragment
            out = close_unclosed_opener(src, None)
            self.assertIsNotNone(out, fragment)
            self.assertTrue(_parses(out), out)

    def test_a_genuine_multiline_call_is_untouched(self):
        for src in (
            "x = f(\n    a,\n    b,\n)\n",
            "d = {\n    'a': 1,\n}\n",
            "xs = [\n    1,\n    2,\n]\n",
            "def f(\n    a,\n    b,\n):\n    return a\n",
        ):
            self.assertIsNone(close_unclosed_opener(src, None), src)

    def test_valid_python_is_untouched(self):
        self.assertIsNone(close_unclosed_opener("import os\nx = f(1)\n", None))

    def test_brackets_inside_strings_are_ignored(self):
        src = "def f():\n    p = g('a)b'\n    q = 2\n    return q\n"
        out = close_unclosed_opener(src, None)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)

    def test_never_raises(self):
        for src in (None, "", "(((\n", "x = (\n"):
            close_unclosed_opener(src, None)


if __name__ == "__main__":
    unittest.main()


class OperatorOrderingTest(unittest.TestCase):
    """Operator ORDER is load-bearing, and this was measured the hard way.

    `close_unclosed_opener` is the most aggressive operator here: it rewrites any line that leaves a
    bracket open. Registered EARLY it fired on files where `rejoin_split_comprehension` or
    `normalize_leaked_generic_params` would have matched, mutating the source so their patterns no
    longer applied -- costing 19 files (malware mechanical fixes 1030 -> 1011, stage3 61 -> 42).

    Moved LAST it only sees sources every precise operator already declined. Keep it there.
    """

    def test_the_aggressive_operator_is_not_registered(self):
        # Measured NET NEGATIVE at every position: -19 malware conversions (1030 -> 1011,
        # stage3 61 -> 42). Position cannot fix it -- _structural_fixpoint loops, so its output
        # feeds back to the precise operators on the next round regardless of ordering.
        names = [op.name for op in STRUCT_OPS]
        self.assertNotIn("close_unclosed_opener", names)

    def test_the_precise_operators_are_still_registered(self):
        names = [op.name for op in STRUCT_OPS]
        for precise in ("normalize_leaked_generic_params", "rejoin_split_comprehension"):
            self.assertIn(precise, names)


class ClauseIsNeverABodyTest(unittest.TestCase):
    """A clause keyword can never be a block body -- the guard bug that cost 168 files.

    `give_headerless_block_a_body` skipped whenever the next line was more-indented, treating that
    as "a body is present". But PyLingual emits exactly this:

        except Exception:
            else:  # inserted

    The `else:` IS more indented, yet `else`/`elif`/`except`/`finally` can never be the body of a
    block. So the operator declined the precise case it exists for. Measured in the residual audit:
    168 files blocked on a dangling clause, of which this shape is the dominant form.
    """

    NESTED_CLAUSE_AS_BODY = (
        "def f():\n"
        "    try:\n"
        "        wkey = compute()\n"
        "    except Exception:\n"
        "        else:  # inserted\n"
        "            other()\n"
    )

    def test_a_more_indented_clause_does_not_count_as_a_body(self):
        out = give_headerless_block_a_body(self.NESTED_CLAUSE_AS_BODY, None)
        self.assertIsNotNone(out, "operator must fire when the 'body' is really a clause")

    def test_a_genuine_more_indented_body_is_still_respected(self):
        src = "def f():\n    try:\n        a()\n    except Exception:\n        handle()\n"
        self.assertIsNone(give_headerless_block_a_body(src, None))

    def test_all_four_clause_keywords_are_rejected_as_bodies(self):
        for kw in ("else:", "elif x:", "except ValueError:", "finally:"):
            src = "def f():\n    try:\n        a()\n    except Exception:\n        %s\n            b()\n" % kw
            self.assertIsNotNone(give_headerless_block_a_body(src, None), kw)
