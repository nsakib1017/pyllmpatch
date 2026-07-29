"""Splicing decompiler-truncated constant sequences back from the ground-truth `.pyc`.

PyLingual stops emitting a constant sequence after 51 elements, leaving a dangling opening quote
inside an unclosed bracket. `balance_delimiters` closes that bracket, so the file compiles and the
repair scores "genuine" -- while every element past the 51st is gone. The complete sequence is
still in the GT `.pyc`, which the syntactic pipeline already has on disk, so the faithful repair is
available and purely mechanical.

`recover_truncated_literal` does the per-line reconstruction and is already tested. What is tested
here is the wiring that makes it reachable: harvesting the candidate sequences out of a code-object
tree, applying them across a whole file, and threading them through the prepass driver AHEAD of
`balance_delimiters` -- because whichever of the two fires first decides whether the repair is
faithful or lossy.
"""
import unittest

from utils.gt_syntactic_context import constant_sequences_from_code, harvest_constant_sequences
from utils.syntactic_prepass import (
    SyntaxErrorInfo,
    run_syntactic_prepass,
    splice_truncated_literals,
)

# 60 elements: past PyLingual's 51-element cap, so the decompiled form is truncated.
FULL = tuple("e%02d" % i for i in range(60))
TRUNCATED_LINE = "PAYLOAD = [" + ", ".join(repr(x) for x in FULL[:51]) + ", 'e5"


def _module_with_sequence():
    return compile("PAYLOAD = %r\n" % (FULL,), "<gt>", "exec")


class HarvestConstantSequencesTest(unittest.TestCase):
    def test_finds_a_tuple_constant_in_the_module(self):
        seqs = constant_sequences_from_code(_module_with_sequence())
        self.assertIn(FULL, seqs)

    def test_walks_into_nested_code_objects(self):
        code = compile("def f():\n    return %r\n" % (FULL,), "<gt>", "exec")
        self.assertIn(FULL, constant_sequences_from_code(code))

    def test_frozenset_constants_are_collected(self):
        code = compile("x = 1 in {%s}\n" % ", ".join(str(i) for i in range(60)), "<gt>", "exec")
        seqs = constant_sequences_from_code(code)
        self.assertTrue(any(isinstance(s, frozenset) for s in seqs))

    def test_short_sequences_are_dropped(self):
        # A 1-element tuple can never be the truncation case and only adds false candidates.
        seqs = constant_sequences_from_code(compile("x = (1,)\n", "<gt>", "exec"))
        self.assertNotIn((1,), seqs)

    def test_sequence_count_is_capped(self):
        src = "\n".join("v%d = (%s)" % (i, ", ".join(str(j) for j in range(3))) for i in range(50))
        seqs = constant_sequences_from_code(compile(src, "<gt>", "exec"), max_sequences=5)
        self.assertLessEqual(len(seqs), 5)

    def test_non_code_input_is_safe(self):
        self.assertEqual(constant_sequences_from_code(None), [])
        self.assertEqual(constant_sequences_from_code("not a code object"), [])

    def test_missing_pyc_path_yields_nothing(self):
        self.assertEqual(harvest_constant_sequences("/nonexistent/gt.pyc"), [])
        self.assertEqual(harvest_constant_sequences(None), [])
        self.assertEqual(harvest_constant_sequences(""), [])


class SpliceTruncatedLiteralsTest(unittest.TestCase):
    def test_truncated_line_is_completed_from_the_ground_truth(self):
        src = TRUNCATED_LINE + "\n"
        out = splice_truncated_literals(src, SyntaxErrorInfo(1, 1, "unterminated string literal"),
                                        [FULL])
        self.assertIsNotNone(out)
        self.assertIn(repr(FULL[59]), out)
        import ast

        ast.parse(out)

    def test_surrounding_lines_are_preserved(self):
        src = "import os\n" + TRUNCATED_LINE + "\nx = 1\n"
        out = splice_truncated_literals(src, SyntaxErrorInfo(2, 1, "unterminated string literal"),
                                        [FULL])
        self.assertIsNotNone(out)
        self.assertTrue(out.startswith("import os\n"))
        self.assertTrue(out.rstrip("\n").endswith("x = 1"))

    def test_no_ground_truth_sequences_is_a_no_op(self):
        src = TRUNCATED_LINE + "\n"
        self.assertIsNone(splice_truncated_literals(src, SyntaxErrorInfo(1, 1, ""), []))
        self.assertIsNone(splice_truncated_literals(src, SyntaxErrorInfo(1, 1, ""), None))

    def test_untruncated_source_is_a_no_op(self):
        src = "x = (1, 2, 3)\n"
        self.assertIsNone(splice_truncated_literals(src, SyntaxErrorInfo(1, 1, ""), [FULL]))

    def test_unrelated_ground_truth_does_not_fabricate_content(self):
        src = TRUNCATED_LINE + "\n"
        self.assertIsNone(
            splice_truncated_literals(src, SyntaxErrorInfo(1, 1, ""), [("zzz", "yyy", "xxx")])
        )

    def test_never_raises_on_junk(self):
        self.assertIsNone(splice_truncated_literals(None, None, [FULL]))
        self.assertIsNone(splice_truncated_literals("", None, [FULL]))


class PrepassOrderingTest(unittest.TestCase):
    """The splice must be tried BEFORE `balance_delimiters`.

    Both make the file compile. `balance_delimiters` does it by closing the bracket, discarding
    every element the decompiler never emitted; the splice restores them. Whichever runs first
    wins, so ordering -- not availability -- is what decides fidelity.
    """

    def test_splice_wins_over_balance_when_ground_truth_is_available(self):
        result = run_syntactic_prepass(TRUNCATED_LINE + "\n", gt_sequences=[FULL])
        self.assertTrue(result.compiled)
        self.assertIn(repr(FULL[59]), result.source)
        self.assertEqual(result.operations[0], "splice_truncated_literals")

    def test_without_ground_truth_the_shipped_behaviour_is_unchanged(self):
        result = run_syntactic_prepass(TRUNCATED_LINE + "\n")
        self.assertNotIn("splice_truncated_literals", result.operations)

    def test_ground_truth_does_not_disturb_files_with_no_truncation(self):
        src = "x = f(1, 2\n"
        with_gt = run_syntactic_prepass(src, gt_sequences=[FULL])
        without_gt = run_syntactic_prepass(src)
        self.assertEqual(with_gt.source, without_gt.source)
        self.assertEqual(with_gt.operations, without_gt.operations)


class MaybePrepassThreadingTest(unittest.TestCase):
    def test_runner_wrapper_forwards_ground_truth_sequences(self):
        from pipeline.runner import maybe_prepass

        source, compiled, ops = maybe_prepass(
            TRUNCATED_LINE + "\n",
            "3.12",
            compile_version_fn=lambda s, v: compile(s, "<t>", "exec"),
            gt_sequences=[FULL],
        )
        self.assertTrue(compiled)
        self.assertIn("splice_truncated_literals", ops)
        self.assertIn(repr(FULL[59]), source)

    def test_default_call_is_unchanged(self):
        from pipeline.runner import maybe_prepass

        source, compiled, ops = maybe_prepass(
            "x = f(1, 2\n", "3.12", compile_version_fn=lambda s, v: compile(s, "<t>", "exec")
        )
        self.assertTrue(compiled)
        self.assertNotIn("splice_truncated_literals", ops)


class HarvestCachingTest(unittest.TestCase):
    """A file is probed by the prepass once before the LLM and again after each LLM candidate.
    Re-loading the same ground-truth `.pyc` through xdis on every one of those would multiply a
    seconds-long load by the retry budget, so identical paths must resolve from cache."""

    def test_repeated_paths_load_the_pyc_once(self):
        from unittest import mock

        from utils import gt_syntactic_context as gtc

        gtc.harvest_constant_sequences.cache_clear()
        with mock.patch.object(gtc, "_harvest_uncached", return_value=(FULL,)) as loader:
            first = gtc.harvest_constant_sequences("/gt.pyc")
            second = gtc.harvest_constant_sequences("/gt.pyc")
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(list(first), list(second))

    def test_distinct_paths_are_not_confused(self):
        from unittest import mock

        from utils import gt_syntactic_context as gtc

        gtc.harvest_constant_sequences.cache_clear()
        with mock.patch.object(gtc, "_harvest_uncached", side_effect=[(FULL,), ()]) as loader:
            self.assertEqual(list(gtc.harvest_constant_sequences("/a.pyc")), [FULL])
            self.assertEqual(list(gtc.harvest_constant_sequences("/b.pyc")), [])
        self.assertEqual(loader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
