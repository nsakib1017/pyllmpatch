"""Unit tests for the jump-sense operator (utils/semantic_operators/jump.py).

collapse_dead_pass_else_candidate recovers a branch-sense inversion the decompiler
emits as ``if COND: pass else: BODY``. The GT bytecode differs from the derived
bytecode by EXACTLY one POP_JUMP polarity (TRUE<->FALSE) at the same target — a
pure single flip. The fix drops the dead ``pass`` then-branch and promotes the
``else`` body under the SAME condition (``if COND: BODY``), which flips only the
jump and leaves every other instruction (incl. any COMPARE/CONTAINS operand)
untouched, byte-matching GT. Runs no LLM.
"""
import ast
import unittest

from utils.semantic_operators.jump import collapse_dead_pass_else_candidate


class _Inst:
    """Fake instruction carrying offset/opname/argval (jump target lives in argval)."""

    def __init__(self, offset, opname, argval=None, argrepr=""):
        self.offset, self.opname, self.argval, self.argrepr = offset, opname, argval, argrepr


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


# A realistic for-loop fragment: `if value not in d: pass else: BODY`. In a loop the
# dead `pass` then-branch and the `else` body both fall through to the loop continue,
# so GT (`if value not in d: BODY`) and this derived form differ by ONLY the POP_JUMP.
FRAGMENT = (
    "def f(d, value):\n"
    "    for value in d:\n"
    "        if value not in d:\n"
    "            pass\n"
    "        else:\n"
    "            d[value] = 1\n"
)

_OFF = 24


def _streams(der_op, gt_op, der_tgt=28, gt_tgt=28, extra_der=None, extra_gt=None):
    """Two streams identical except the POP_JUMP at _OFF (a pure single flip)."""
    def base(jump_op, tgt):
        return [
            _Inst(18, "LOAD_FAST", "value"),
            _Inst(20, "LOAD_FAST", "d"),
            _Inst(22, "CONTAINS_OP", 1),
            _Inst(_OFF, jump_op, tgt),
            _Inst(26, "JUMP_BACKWARD", 14),
            _Inst(28, "LOAD_CONST", 1),
            _Inst(30, "STORE_SUBSCR", None),
        ]
    der = base(der_op, der_tgt) + (extra_der or [])
    gt = base(gt_op, gt_tgt) + (extra_gt or [])
    return _BC(der), _BC(gt)


def _ctx(off=_OFF, message="Different bytecode"):
    return {"pylingual_failed_result": {"message": message}, "failed_offset": off}


class FiresTest(unittest.TestCase):
    def test_fires_and_collapses(self):
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE")
        out = collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx())
        self.assertIsNotNone(out)
        text = out["text"]
        tree = ast.parse(text)  # round-trips
        self.assertNotIn("pass", text)
        ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
        self.assertEqual(len(ifs), 1)
        node = ifs[0]
        self.assertEqual(node.orelse, [])                 # else removed
        self.assertEqual(len(node.body), 1)               # body sits directly under the if
        self.assertIsInstance(node.body[0], ast.Assign)   # the promoted assignment
        self.assertFalse(any(isinstance(s, ast.Pass) for s in node.body))
        self.assertEqual(out["kind"], "jump_sense")
        self.assertEqual(out["opname"], "POP_JUMP_IF_FALSE")

    def test_collapsed_form_matches_gt_jump_polarity(self):
        """The collapsed source compiles to the GT polarity (…_IF_TRUE), flipping ONLY
        the jump while keeping the CONTAINS_OP operand intact (byte-equivalent to GT)."""
        import dis
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE")
        out = collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx())
        self.assertIsNotNone(out)
        fn = [k for k in compile(out["text"], "<c>", "exec").co_consts if hasattr(k, "co_name")][0]
        ops = [(i.opname, i.argval) for i in dis.get_instructions(fn)]
        self.assertEqual([a for op, a in ops if op == "CONTAINS_OP"], [1])  # operand preserved
        self.assertTrue(any(op.startswith("POP_JUMP") and "TRUE" in op for op, _ in ops))
        # the derived (uncollapsed) form uses the OPPOSITE (FALSE) polarity
        der_fn = [k for k in compile(FRAGMENT, "<d>", "exec").co_consts if hasattr(k, "co_name")][0]
        der_jumps = [i.opname for i in dis.get_instructions(der_fn) if i.opname.startswith("POP_JUMP")]
        self.assertTrue(any("FALSE" in j for j in der_jumps))

    def test_fires_on_311_forward_variant_names(self):
        der, gt = _streams("POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_FORWARD_IF_TRUE")
        out = collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx())
        self.assertIsNotNone(out)
        self.assertNotIn("pass", out["text"])


class DeclineTest(unittest.TestCase):
    def test_declines_same_opcode(self):
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_FALSE")
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx()))

    def test_declines_wrong_message(self):
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE")
        self.assertIsNone(collapse_dead_pass_else_candidate(
            gt, der, FRAGMENT, _ctx(message="Different control flow")))

    def test_declines_no_failed_offset(self):
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE")
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": None}
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, FRAGMENT, ctx))

    def test_declines_then_branch_not_pass(self):
        # `if C: x=1 else: y=2` — the then-branch is real work, not a dead pass.
        frag = (
            "def f(d, value):\n"
            "    for value in d:\n"
            "        if value not in d:\n"
            "            x = 1\n"
            "        else:\n"
            "            y = 2\n"
        )
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE")
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, frag, _ctx()))

    def test_declines_two_dead_pass_else_blocks(self):
        frag = (
            "def f(a, b):\n"
            "    if a:\n"
            "        pass\n"
            "    else:\n"
            "        x = 1\n"
            "    if b:\n"
            "        pass\n"
            "    else:\n"
            "        y = 2\n"
        )
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE")
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, frag, _ctx()))

    def test_declines_second_instruction_differs(self):
        # A second opname difference (CONTAINS vs COMPARE) means it is not a pure flip.
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE")
        list(gt)[2].opname = "COMPARE_OP"
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx()))

    def test_declines_length_mismatch(self):
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
                           extra_gt=[_Inst(32, "RETURN_CONST", None)])
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx()))

    def test_declines_target_mismatch(self):
        # Same polarity flip but different jump targets = a restructure, not a flip.
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", der_tgt=28, gt_tgt=40)
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx()))

    def test_declines_offset_not_found(self):
        der, gt = _streams("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE")
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx(off=999)))

    def test_declines_forward_backward_same_sense(self):
        # FORWARD vs BACKWARD with the SAME sense (both TRUE) is not a polarity flip.
        der, gt = _streams("POP_JUMP_FORWARD_IF_TRUE", "POP_JUMP_BACKWARD_IF_TRUE")
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, FRAGMENT, _ctx()))


class EndToEndPrepassTest(unittest.TestCase):
    """Drive the real deterministic prepass (oracle acceptance, NO LLM) over a
    compiled GT/DER pair and assert the collapse candidate is ACCEPTED."""

    def test_prepass_accepts_collapse(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        import utils.reattach_source_code_object as R

        gt_src = (
            "def f(self, idx):\n"
            "    for value in self:\n"
            "        if value not in self._dict:\n"
            "            self._dict[value] = idx\n"
            "            idx += 1\n"
        )
        der_src = (
            "def f(self, idx):\n"
            "    for value in self:\n"
            "        if value not in self._dict:\n"
            "            pass\n"
            "        else:\n"
            "            self._dict[value] = idx\n"
            "            idx += 1\n"
        )
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            gt_py = d / "gt.py"; gt_py.write_text(gt_src)
            der_py = d / "der.py"; der_py.write_text(der_src)
            version = (3, sys_minor())
            gt_pyc = R.compile_source_to_pyc(gt_py, d / "gt.pyc", version)
            der_pyc = R.compile_source_to_pyc(der_py, d / "der.pyc", version)

            # Sanity: the pair must be a genuine pure jump-sense flip.
            from pylingual.equivalence_check import compare_pyc
            diverged = [t for t in compare_pyc(gt_pyc, der_pyc)
                        if not t.success and t.failed_offset is not None]
            self.assertTrue(diverged, "expected a divergent code object")

            with mock.patch.object(R, "SEMANTIC_DETERMINISTIC_PREPASS", True):
                result = R.repair_mismatching_code_objects(
                    gt_pyc, der_pyc, der_py,
                    output_dir=d / "out",
                    fragment_fixer=None,
                    verify_with_pylingual=True,
                    acceptance_mode="oracle",
                )
        steps = result.get("steps", [])
        jump_steps = [s for s in steps
                      if s.get("selected_candidate_strategy") == "deterministic_jump"
                      and s.get("accepted")]
        self.assertTrue(jump_steps, f"expected an accepted deterministic_jump step; got {steps}")


def sys_minor():
    import sys
    return sys.version_info[1]


if __name__ == "__main__":
    unittest.main()
