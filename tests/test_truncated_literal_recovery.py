"""recover_truncated_literal rebuilds a decompiler-truncated constant sequence from ground truth.

PyLingual caps emitted constant sequences at 51 elements and stops mid-literal, leaving a dangling
opening quote inside an unclosed bracket. Today `balance_delimiters` "repairs" that by appending a
quote and closers: the file compiles and no lines are deleted, so it scores as a GENUINE repair --
while silently discarding elements 52+. That is the measured fidelity hole (26% of genuine malware
repairs carry a truncated literal).

The full sequence survives verbatim in the ground-truth .pyc's co_consts, so recovery is EXACT
rather than invented. Verified by hand on three malware files (51->60, 51->99, 51->62 elements).

Safety is the priority over coverage: the function returns None -- declining to guess -- whenever
the match is ambiguous or unsupported, because emitting the WRONG constants would be worse than the
status quo (it would look faithful while being wrong).
"""
import unittest

from utils.syntactic_prepass import recover_truncated_literal


class RecoverTruncatedLiteralTest(unittest.TestCase):
    def test_recovers_truncated_tuple_from_ground_truth(self):
        line = "USERS = ('alice', 'bob', '"
        gt = [("alice", "bob", "carol", "dave")]
        out = recover_truncated_literal(line, gt)
        self.assertIsNotNone(out)
        self.assertIn("'carol'", out)
        self.assertIn("'dave'", out)
        # the recovered line must be valid Python and evaluate to the GT sequence
        ns = {}
        exec(out, {}, ns)
        self.assertEqual(tuple(ns["USERS"]), ("alice", "bob", "carol", "dave"))

    def test_preserves_the_prefix_and_nested_call(self):
        line = "BLOCKED = frozenset(('a', 'b', '"
        gt = [("a", "b", "c")]
        out = recover_truncated_literal(line, gt)
        self.assertIsNotNone(out)
        ns = {}
        exec(out, {}, ns)
        self.assertEqual(ns["BLOCKED"], frozenset(("a", "b", "c")))

    def test_declines_when_no_gt_sequence_matches(self):
        self.assertIsNone(
            recover_truncated_literal("X = ('zzz', '", [("alice", "bob", "carol")])
        )

    def test_declines_when_the_prefix_is_ambiguous(self):
        # Two GT sequences share the parsed prefix -- guessing could emit the wrong constants.
        gt = [("a", "b", "c"), ("a", "b", "d")]
        self.assertIsNone(recover_truncated_literal("X = ('a', 'b', '", gt))

    def test_declines_when_gt_sequence_is_not_longer(self):
        # Nothing was actually lost; there is nothing to recover.
        self.assertIsNone(recover_truncated_literal("X = ('a', 'b', '", [("a", "b")]))

    def test_declines_on_a_line_with_no_dangling_quote(self):
        self.assertIsNone(recover_truncated_literal("X = ('a', 'b')", [("a", "b", "c")]))

    def test_never_raises_on_junk_input(self):
        for line in ("", None, "= = =", "X = ('unterminated"):
            self.assertIsNone(recover_truncated_literal(line, [("a",)]))
        self.assertIsNone(recover_truncated_literal("X = ('a', '", None))
        self.assertIsNone(recover_truncated_literal("X = ('a', '", []))


if __name__ == "__main__":
    unittest.main()
