"""Tests for corrective residual feedback (#54): after a rejected LLM candidate, the next
prompt is told what the candidate's OWN bytecode still got wrong vs ground truth, instead of
only an abstract distance number. See
docs/superpowers/specs/2026-07-12-corrective-feedback-prompting-design.md.

Two units:
  * ``_candidate_residual_snapshot`` (utils/reattach_source_code_object.py) — pure: given the
    GT and candidate bytecode for the target object plus the failure message/offset, produce a
    compact GT-vs-candidate instruction window (or a largest-divergence diff when there is no
    single offset). Never raises; None when bytecode is unavailable.
  * ``_format_rejected_attempt_summary`` (pipeline/code_object_repair_loop.py) — renders that
    residual into the retry-feedback prompt when present, and is unchanged when absent.
"""
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.reattach_source_code_object import _candidate_residual_snapshot  # noqa: E402
from pipeline.code_object_repair_loop import _format_rejected_attempt_summary  # noqa: E402


class _Inst:
    def __init__(self, opname, offset, arg=None, argrepr="", argval=None, starts_line=None):
        self.opname = opname
        self.offset = offset
        self.arg = arg
        self.argrepr = argrepr
        self.argval = argval
        self.starts_line = starts_line


class _BC:
    """Editable-bytecode shape: has ``.instructions`` and is iterable (matches what
    ``_instruction_records`` consumes)."""
    def __init__(self, insts):
        self.instructions = insts

    def __iter__(self):
        return iter(self.instructions)


def _stream(pairs):
    # pairs: list of (opname, argrepr); offsets are 2 apart like real bytecode.
    return _BC([_Inst(op, offset=i * 2, argrepr=arg) for i, (op, arg) in enumerate(pairs)])


class CandidateResidualSnapshotTest(unittest.TestCase):
    def test_windowed_diff_around_failed_offset(self):
        # GT loads const 5; candidate loads const 7 at the same offset -> the window must show
        # both so the model sees exactly what its edit still gets wrong.
        gt = _stream([("RESUME", ""), ("LOAD_CONST", "5"), ("RETURN_VALUE", "")])
        cand = _stream([("RESUME", ""), ("LOAD_CONST", "7"), ("RETURN_VALUE", "")])
        snap = _candidate_residual_snapshot(gt, cand, message="Different bytecode", failed_offset=2)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["failed_offset"], 2)
        self.assertEqual(snap["message"], "Different bytecode")
        # window mentions both the GT operand and the candidate operand
        self.assertIn("5", snap["window"])
        self.assertIn("7", snap["window"])

    def test_no_single_offset_uses_divergence_diff(self):
        # failed_offset=None (the dominant "Different bytecode, no single offset" case): still
        # surface a compact GT-vs-candidate divergence rather than nothing.
        gt = _stream([("RESUME", ""), ("LOAD_FAST", "a"), ("LOAD_FAST", "b"), ("BINARY_OP", "+"), ("RETURN_VALUE", "")])
        cand = _stream([("RESUME", ""), ("LOAD_FAST", "a"), ("LOAD_FAST", "b"), ("BINARY_OP", "-"), ("RETURN_VALUE", "")])
        snap = _candidate_residual_snapshot(gt, cand, message="Different bytecode", failed_offset=None)
        self.assertIsNotNone(snap)
        self.assertIsNone(snap["failed_offset"])
        self.assertTrue(snap["window"].strip(), "must render a non-empty divergence window")

    def test_none_when_bytecode_unavailable(self):
        self.assertIsNone(_candidate_residual_snapshot(None, _stream([("RESUME", "")]),
                                                       message="Different bytecode", failed_offset=0))
        self.assertIsNone(_candidate_residual_snapshot(_stream([("RESUME", "")]), None,
                                                       message="Different bytecode", failed_offset=0))

    def test_never_raises_on_garbage(self):
        # Malformed bytecode objects must defer to None, never raise.
        self.assertIsNone(_candidate_residual_snapshot(object(), object(),
                                                       message="x", failed_offset=None))


class FormatRejectedAttemptSummaryTest(unittest.TestCase):
    def _ctx(self, residual):
        return {"rejected_attempts": [{
            "attempt": 1,
            "acceptance_reason": "combined distance did not improve",
            "selected_action_types": ["leaf_value"],
            "candidate_residual": residual,
        }]}

    def test_renders_corrective_residual_when_present(self):
        residual = {"message": "Different bytecode", "failed_offset": 2,
                    "window": "GT:\n=> offset=2 LOAD_CONST 5\nYours:\n=> offset=2 LOAD_CONST 7"}
        out = _format_rejected_attempt_summary(self._ctx(residual))
        self.assertIn("offset 2", out)
        self.assertIn("LOAD_CONST 7", out)  # the model's own wrong output is surfaced
        self.assertIn("LOAD_CONST 5", out)  # ...next to ground truth

    def test_unchanged_when_no_residual(self):
        # An attempt without candidate_residual renders exactly as before (no corrective block).
        out = _format_rejected_attempt_summary(self._ctx(None))
        self.assertNotIn("still differs from ground truth", out)

    def test_empty_when_no_attempts(self):
        self.assertEqual(_format_rejected_attempt_summary({"rejected_attempts": []}), "")


if __name__ == "__main__":
    unittest.main()
