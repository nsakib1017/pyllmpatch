"""The runner seam that runs the two-stage sweep cascade ahead of the LLM.

Measured over the full non-parsing malware population (974 files): stage 1 converts 10.0% with a
line-level audit of 0 deletions and 0 insertions on all 97, and stage 2 converts a further 20.2% --
30.2% of files that would otherwise reach the LLM, or fall through it to the delete-only fallback.

Two properties this seam must hold that the sweep module alone cannot:

  * The cascade decides with `ast.parse` on the HOST interpreter, but the file targets a specific
    CPython version. A rewrite is adopted only after the version-specific compiler accepts it.
  * When the cascade cannot finish a file, the ORIGINAL is forwarded, never the partial rewrite.
    Measured: on files the stack does not finish it reduces the error-site count on only 19/250,
    so a partial pass is not an LLM aid -- and deploying one regresses 7/233 currently-genuine
    files. A rewrite that does not reach a parse is not worth that.
"""
import unittest
from unittest import mock

from pipeline import runner as runner_mod
from pipeline.runner import maybe_sweep_cascade

STAGE1 = "def f():\n    'literal' = 1\n    return 1\n"
STAGE1_CONTROL_FLOW = "def f():\n    x = 1\nelse:\n    x = 2\n"
STAGE2 = "def f():\n    try:\n        x = 1\n"
UNFIXABLE = "@@@ this is not python @@@\n"
VALID = "def f():\n    return 1\n"

FULL = tuple("e%02d" % i for i in range(60))
TRUNCATED = "PAYLOAD = [" + ", ".join(repr(x) for x in FULL[:51]) + ", 'e5\n"


def _host_compile(source, version):
    compile(source, "<t>", "exec")


class StageOneTest(unittest.TestCase):
    def test_pure_stage_fixes_and_reports_stage_one(self):
        source, compiled, ops, stage = maybe_sweep_cascade(STAGE1, "3.12", _host_compile)
        self.assertTrue(compiled)
        self.assertEqual(stage, 1)
        self.assertIn("bad_lhs_rename", ops)

    def test_stage_one_does_not_delete_or_insert_lines(self):
        source, compiled, ops, stage = maybe_sweep_cascade(STAGE1, "3.12", _host_compile)
        from utils.syntactic_sweeps import audit_fidelity

        audit = audit_fidelity(STAGE1, source)
        self.assertEqual(audit["deleted"], 0)
        self.assertEqual(audit["inserted"], 0)


class StageTwoTest(unittest.TestCase):
    def test_scaffolding_stage_runs_only_after_stage_one_fails(self):
        source, compiled, ops, stage = maybe_sweep_cascade(STAGE2, "3.12", _host_compile)
        self.assertTrue(compiled)
        self.assertEqual(stage, 2)
        self.assertIn("complete_all_orphan_try", ops)

    def test_stage_two_synthesis_is_an_insertion_not_a_deletion(self):
        source, compiled, ops, stage = maybe_sweep_cascade(STAGE2, "3.12", _host_compile)
        from utils.syntactic_sweeps import audit_fidelity

        audit = audit_fidelity(STAGE2, source)
        self.assertEqual(audit["deleted"], 0)
        self.assertGreater(audit["inserted"], 0)


class ForwardTheOriginalTest(unittest.TestCase):
    def test_unfinished_cascade_forwards_the_original_untouched(self):
        source, compiled, ops, stage = maybe_sweep_cascade(UNFIXABLE, "3.12", _host_compile)
        self.assertFalse(compiled)
        self.assertEqual(source, UNFIXABLE)
        self.assertEqual(stage, 0)

    def test_source_that_already_parses_is_left_alone(self):
        source, compiled, ops, stage = maybe_sweep_cascade(VALID, "3.12", _host_compile)
        self.assertEqual(source, VALID)
        self.assertEqual(ops, [])
        self.assertEqual(stage, 0)

    def test_empty_source_is_safe(self):
        self.assertEqual(maybe_sweep_cascade("", "3.12", _host_compile)[0], "")
        self.assertEqual(maybe_sweep_cascade(None, "3.12", _host_compile)[0], None)


class VersionCompileGateTest(unittest.TestCase):
    """`ast.parse` runs on the host; the file targets another CPython. A rewrite the host accepts
    and the target rejects must not be adopted -- that is how a "fix" becomes a broken output."""

    def test_rewrite_rejected_by_the_target_version_is_not_adopted(self):
        def _always_fails(source, version):
            raise SyntaxError("target version rejects this")

        source, compiled, ops, stage = maybe_sweep_cascade(STAGE1, "3.12", _always_fails)
        self.assertFalse(compiled)
        self.assertEqual(source, STAGE1)
        self.assertEqual(stage, 0)

    def test_target_compiler_is_consulted_before_adopting(self):
        calls = []

        def _spy(source, version):
            calls.append((source, version))
            compile(source, "<t>", "exec")

        maybe_sweep_cascade(STAGE1, "3.11", _spy)
        self.assertTrue(calls)
        self.assertEqual(calls[-1][1], "3.11")


class SubReportingTest(unittest.TestCase):
    """`clause_to_if_true` preserves every line but rewrites `else:` into `if True:` -- genuine by
    the project definition, yet the reconstructed control flow is not the original's. It is the
    largest single stage-1 contributor, so it must be countable on its own rather than absorbed
    into an undifferentiated deterministic-fix total."""

    def test_control_flow_rewrites_are_reported_separately(self):
        source, compiled, ops, stage = maybe_sweep_cascade(
            STAGE1_CONTROL_FLOW, "3.12", _host_compile
        )
        self.assertTrue(compiled)
        self.assertIn("clause_to_if_true", ops)
        self.assertEqual(runner_mod.control_flow_rewrites_in(ops), ["clause_to_if_true"])

    def test_a_plain_stage_one_fix_reports_no_control_flow_rewrite(self):
        _, _, ops, _ = maybe_sweep_cascade(STAGE1, "3.12", _host_compile)
        self.assertEqual(runner_mod.control_flow_rewrites_in(ops), [])


class FidelityOrderingTest(unittest.TestCase):
    """`balance_delimiters` sits inside the cascade's legacy sweep and will happily close a
    truncated literal, discarding everything past the 51st element. When the ground truth is
    available the splice must therefore run BEFORE the cascade, not after it."""

    def test_ground_truth_splice_precedes_the_cascade(self):
        source, compiled, ops, stage = maybe_sweep_cascade(
            TRUNCATED, "3.12", _host_compile, gt_sequences=[FULL]
        )
        self.assertTrue(compiled)
        self.assertIn(repr(FULL[59]), source)
        self.assertEqual(ops[0], "splice_truncated_literals")

    def test_without_ground_truth_the_cascade_still_closes_the_literal(self):
        source, compiled, ops, stage = maybe_sweep_cascade(TRUNCATED, "3.12", _host_compile)
        self.assertTrue(compiled)
        self.assertNotIn("splice_truncated_literals", ops)


class FlagTest(unittest.TestCase):
    def test_kill_switch_makes_it_a_passthrough(self):
        with mock.patch.dict("os.environ", {"SYNTACTIC_SWEEP_CASCADE": "0"}):
            source, compiled, ops, stage = maybe_sweep_cascade(STAGE1, "3.12", _host_compile)
        self.assertEqual(source, STAGE1)
        self.assertFalse(compiled)
        self.assertEqual(ops, [])
        self.assertEqual(stage, 0)

    def test_enabled_by_default(self):
        import os

        os.environ.pop("SYNTACTIC_SWEEP_CASCADE", None)
        _, compiled, _, stage = maybe_sweep_cascade(STAGE1, "3.12", _host_compile)
        self.assertTrue(compiled)
        self.assertEqual(stage, 1)

    def test_never_raises_when_an_operator_explodes(self):
        with mock.patch.object(runner_mod, "run_stack", side_effect=RuntimeError("boom")):
            source, compiled, ops, stage = maybe_sweep_cascade(STAGE1, "3.12", _host_compile)
        self.assertEqual(source, STAGE1)
        self.assertFalse(compiled)


if __name__ == "__main__":
    unittest.main()


class StageThreeStructuralTest(unittest.TestCase):
    """Stage 3 repairs PyLingual control-flow RECONSTRUCTION defects, and is adopt-only-if-it-parses.

    That guard is not defensive styling -- it is required by measurement. On the 410-file residual
    the structural pass rescues 10 files (2.4%, zero lines deleted) but makes files WORSE on
    average: median defects 61 -> 74, with 76 of 150 files worsened against 4 improved and none
    crossing into LLM-winnable range. A partial structural rewrite must therefore never be
    forwarded; only a complete fix is taken.
    """

    HEADERLESS = (
        "def f(cfg):\n"
        "    with open(cfg) as fh:\n"
        "        try:\n"
        "            item = json.load(fh)\n"
        "        except ValueError:\n"
        "        else:  # inserted\n"
        "            item['x'] = False\n"
    )

    def test_structural_defect_is_repaired_and_reported_as_stage_three(self):
        source, compiled, ops, stage = maybe_sweep_cascade(self.HEADERLESS, "3.12", _host_compile)
        self.assertTrue(compiled)
        self.assertEqual(stage, 3)
        self.assertIn("give_headerless_block_a_body", ops)

    def test_stage_three_deletes_nothing(self):
        source, compiled, _, _ = maybe_sweep_cascade(self.HEADERLESS, "3.12", _host_compile)
        from utils.syntactic_sweeps import audit_fidelity

        self.assertEqual(audit_fidelity(self.HEADERLESS, source)["deleted"], 0)

    def test_partial_structural_rewrite_is_never_forwarded(self):
        # Structural ops fire here (a flattened-for shape) but cannot finish the file; the caller
        # must receive the ORIGINAL, not the half-rewritten source.
        src = "def g(xs):\n    f(x) for x in xs:\n        pass\n    @@@ broken @@@\n"
        source, compiled, ops, stage = maybe_sweep_cascade(src, "3.12", _host_compile)
        self.assertFalse(compiled)
        self.assertEqual(source, src)
        self.assertEqual(stage, 0)
