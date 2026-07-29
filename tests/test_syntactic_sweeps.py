"""The two-stage deterministic sweep cascade.

`utils.syntactic_prepass` runs error-driven operators against ONE localized syntax error and gates
each step behind `advanced()`. Measured over the full malware population that fixes 12/1362 files
(0.9%). The cascade here is the architecture that measured ~30 points instead: whole-file sweeps
driven to a fixpoint, split into two ORDERED stages.

  STAGE 1 (`PURE_OPS`) -- code-preserving rewrites only: nothing is added, nothing is removed.
  STAGE 2 (`SYNTH_OPS`) -- scaffolding, run ONLY after stage 1 has failed.

The ordering is load-bearing: merging the two stacks into one converts 193 files where running
them in sequence converts 211. A synthesis operator that fires early can satisfy the parser in a
way that masks the pure fix a later sweep would have found.

Two operators measured in the research stack are deliberately ABSENT: `close_unclosed_lines` and
`split_flattened_for` each break valid Python outside this distribution (5/602 and 2/602). They
cost 211 -> 197 conversions; the registries must not carry them back in.
"""
import ast
import unittest

from utils.syntactic_sweeps import (
    PURE_OPS,
    SYNTH_OPS,
    CONTROL_FLOW_REWRITES,
    SweepOp,
    parse_error,
    run_stack,
    audit_fidelity,
)


def _parses(src):
    try:
        ast.parse(src)
        return True
    except Exception:
        return False


class RegistryShapeTest(unittest.TestCase):
    def test_both_registries_are_populated(self):
        self.assertTrue(PURE_OPS)
        self.assertTrue(SYNTH_OPS)

    def test_entries_are_sweep_ops(self):
        for op in list(PURE_OPS) + list(SYNTH_OPS):
            self.assertIsInstance(op, SweepOp)
            self.assertTrue(callable(op.fn))
            self.assertTrue(op.name)
            self.assertTrue(op.family)

    def test_names_are_unique_across_both_stages(self):
        names = [op.name for op in list(PURE_OPS) + list(SYNTH_OPS)]
        self.assertEqual(len(names), len(set(names)))

    def test_pure_ops_claim_preservation_and_synth_ops_do_not(self):
        self.assertTrue(all(op.preserves_code for op in PURE_OPS))
        self.assertFalse(any(op.preserves_code for op in SYNTH_OPS))

    def test_unsafe_operators_are_excluded(self):
        # Both break valid Python outside the malware distribution -- see module docstring.
        names = {op.name for op in list(PURE_OPS) + list(SYNTH_OPS)}
        self.assertNotIn("close_unclosed_lines", names)
        self.assertNotIn("split_flattened_for", names)

    def test_control_flow_rewrites_are_declared_and_are_pure_ops(self):
        # `clause_to_if_true` rewrites `else:` -> `if True:`. It deletes nothing, so it is genuine
        # under the project definition, but it CHANGES CONTROL FLOW and must stay separable in
        # reporting rather than hiding inside an undifferentiated "deterministic fix" count.
        self.assertIn("clause_to_if_true", CONTROL_FLOW_REWRITES)
        pure_names = {op.name for op in PURE_OPS}
        for name in CONTROL_FLOW_REWRITES:
            self.assertIn(name, pure_names)


class ParseErrorTest(unittest.TestCase):
    def test_valid_source_has_no_error(self):
        self.assertIsNone(parse_error("x = 1\n"))

    def test_invalid_source_reports_a_line(self):
        err = parse_error("def f(:\n    pass\n")
        self.assertIsNotNone(err)
        self.assertEqual(err.lineno, 1)

    def test_null_bytes_do_not_raise(self):
        # ast.parse raises ValueError, not SyntaxError, on embedded nulls.
        err = parse_error("x = 1\x00\n")
        self.assertIsNotNone(err)


class RunStackTest(unittest.TestCase):
    def test_already_parsing_source_is_returned_untouched(self):
        src = "def f():\n    return 1\n"
        out, fired = run_stack(src, PURE_OPS)
        self.assertEqual(out, src)
        self.assertEqual(fired, [])

    def test_unfixable_source_is_returned_unchanged_and_still_fails(self):
        src = "@@@ this is not python @@@\n"
        out, fired = run_stack(src, PURE_OPS)
        self.assertFalse(_parses(out))
        self.assertEqual(out, src)

    def test_records_which_operators_fired(self):
        src = "try:\n    x = 1\n"  # orphan try -- stage 2 territory
        self.assertFalse(_parses(src))
        out, fired = run_stack(src, SYNTH_OPS)
        self.assertTrue(_parses(out))
        self.assertTrue(fired)

    def test_stage_one_does_not_scaffold_an_orphan_try(self):
        # The whole point of the split: stage 1 must LEAVE this for stage 2.
        src = "try:\n    x = 1\n"
        out, _ = run_stack(src, PURE_OPS)
        self.assertFalse(_parses(out))

    def test_round_budget_is_honoured(self):
        src = "def f(:\n    pass\n"
        out, fired = run_stack(src, PURE_OPS, max_rounds=1)
        self.assertIsInstance(out, str)

    def test_empty_source_is_safe(self):
        out, fired = run_stack("", PURE_OPS)
        self.assertEqual(out, "")
        self.assertEqual(fired, [])


class AuditFidelityTest(unittest.TestCase):
    def test_identical_sources_report_no_change(self):
        a = audit_fidelity("a = 1\nb = 2\n", "a = 1\nb = 2\n")
        self.assertEqual(a["deleted"], 0)
        self.assertEqual(a["inserted"], 0)

    def test_deletion_is_counted(self):
        a = audit_fidelity("a = 1\nb = 2\nc = 3\n", "a = 1\nc = 3\n")
        self.assertEqual(a["deleted"], 1)

    def test_insertion_is_counted(self):
        a = audit_fidelity("try:\n    a = 1\n", "try:\n    a = 1\nexcept Exception:\n    pass\n")
        self.assertEqual(a["inserted"], 2)

    def test_reindentation_is_neither_deletion_nor_insertion(self):
        a = audit_fidelity("if x:\n    a\n", "if x:\n        a\n")
        self.assertEqual(a["deleted"], 0)
        self.assertEqual(a["inserted"], 0)


class NoValidPythonRegressionTest(unittest.TestCase):
    """The cascade runs on files the pipeline has already failed to compile, but a wiring mistake
    could route valid source through it. Neither stage may damage source that already parses."""

    SAMPLES = [
        "for i in range(3):\n    print(i)\n",
        "try:\n    f()\nexcept ValueError:\n    pass\n",
        "d = {'a': [1, 2, 3], 'b': (4, 5)}\n",
        "class C:\n    def m(self):\n        return [x for x in range(3)]\n",
        "with open('f') as fh:\n    data = fh.read()\n",
        "if a:\n    b()\nelse:\n    c()\n",
        "x = 1 if y else 2\n",
        "async def g():\n    await h()\n",
    ]

    def test_valid_sources_are_returned_byte_identical(self):
        for src in self.SAMPLES:
            for ops in (PURE_OPS, SYNTH_OPS):
                out, fired = run_stack(src, ops)
                self.assertEqual(out, src, msg=src)
                self.assertEqual(fired, [])


if __name__ == "__main__":
    unittest.main()
