"""Unit tests for the GT-bytecode prompt-enrichment renderers (control-flow skeletons +
statement-level diff). These are READ-ONLY context strings shown to the LLM; the tests pin
that they are CORRECT and non-misleading (a wrong skeleton/diff would steer the LLM astray)."""
import unittest

from utils.reattach_source_code_object import (
    _derived_control_flow_skeleton,
    _gt_control_flow_skeleton,
    _missing_extra_statement_summary,
)


class _Ins:
    def __init__(self, offset, opname, argrepr="", argval=None, starts_line=None):
        self.offset = offset
        self.opname = opname
        self.argrepr = argrepr
        self.argval = argval
        self.starts_line = starts_line


class _BC:
    """Minimal stand-in for a pylingual EditableBytecode: has .instructions and is iterable."""
    def __init__(self, insts):
        self.instructions = insts
        self._i = insts

    def __iter__(self):
        return iter(self._i)


class StatementDiffTest(unittest.TestCase):
    def test_missing_run_flagged(self):
        gt = _BC([_Ins(0, "LOAD_CONST", "1"), _Ins(2, "STORE_FAST", "x"),
                  _Ins(4, "LOAD_FAST", "x"), _Ins(6, "RETURN_VALUE")])
        der = _BC([_Ins(0, "LOAD_CONST", "1"), _Ins(2, "RETURN_VALUE")])
        out = _missing_extra_statement_summary(gt, der)
        self.assertIn("MISSING", out)
        self.assertIn("STORE_FAST", out)

    def test_no_gt_source_line_anchor(self):
        # review finding: a GT source-line number would point at a file the LLM can't see
        gt = _BC([_Ins(0, "LOAD_CONST", "1"), _Ins(2, "STORE_FAST", "x", starts_line=217),
                  _Ins(4, "LOAD_FAST", "x"), _Ins(6, "RETURN_VALUE")])
        der = _BC([_Ins(0, "LOAD_CONST", "1"), _Ins(2, "RETURN_VALUE")])
        out = _missing_extra_statement_summary(gt, der)
        self.assertNotIn("line", out.lower())
        self.assertNotIn("217", out)

    def test_spurious_run_flagged(self):
        gt = _BC([_Ins(0, "LOAD_CONST", "1"), _Ins(2, "RETURN_VALUE")])
        der = _BC([_Ins(0, "LOAD_CONST", "1"), _Ins(2, "POP_TOP"), _Ins(4, "RETURN_VALUE")])
        out = _missing_extra_statement_summary(gt, der)
        self.assertIn("SPURIOUS", out)
        self.assertIn("POP_TOP", out)

    def test_changed_region_not_framed_as_statement(self):
        # review finding: a one-opcode change must NOT be framed as add/remove a whole statement
        gt = _BC([_Ins(0, "LOAD_FAST", "x"), _Ins(2, "LOAD_METHOD", "foo"), _Ins(4, "RETURN_VALUE")])
        der = _BC([_Ins(0, "LOAD_FAST", "x"), _Ins(2, "LOAD_ATTR", "foo"), _Ins(4, "RETURN_VALUE")])
        out = _missing_extra_statement_summary(gt, der)
        bullets = [ln for ln in out.splitlines() if ln.startswith("- ")]
        self.assertTrue(bullets and all(ln.startswith("- CHANGED") for ln in bullets), bullets)

    def test_identical_streams_empty(self):
        gt = _BC([_Ins(0, "LOAD_FAST", "x"), _Ins(2, "RETURN_VALUE")])
        self.assertEqual(_missing_extra_statement_summary(gt, gt), "")


class ControlFlowSkeletonTest(unittest.TestCase):
    def test_renders_branch_and_target(self):
        gt = _BC([_Ins(0, "LOAD_FAST", "x"), _Ins(2, "POP_JUMP_IF_FALSE", "to 8", 8),
                  _Ins(4, "LOAD_CONST", "1"), _Ins(6, "RETURN_VALUE"),
                  _Ins(8, "LOAD_CONST", "2", starts_line=3), _Ins(10, "RETURN_VALUE")])
        out = _gt_control_flow_skeleton(gt)
        self.assertIn("POP_JUMP_IF_FALSE", out)
        self.assertIn("-> @8", out)   # jump destination rendered explicitly
        self.assertIn(">>", out)      # @8 is a jump target

    def test_straight_line_returns_empty(self):
        gt = _BC([_Ins(0, "LOAD_FAST", "x"), _Ins(2, "RETURN_VALUE")])
        self.assertEqual(_gt_control_flow_skeleton(gt), "")

    def test_derived_skeleton_has_reshape_header(self):
        der = _BC([_Ins(0, "LOAD_FAST", "x"), _Ins(2, "POP_JUMP_IF_FALSE", "to 8", 8),
                   _Ins(4, "LOAD_CONST", "1"), _Ins(6, "RETURN_VALUE"),
                   _Ins(8, "LOAD_CONST", "2"), _Ins(10, "RETURN_VALUE")])
        out = _derived_control_flow_skeleton(der)
        self.assertTrue(out.lower().startswith("current"))
        self.assertIn("-> @8", out)


if __name__ == "__main__":
    unittest.main()
