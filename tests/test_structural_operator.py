"""Unit tests for the try/except structural operator (utils/semantic_operators/structural.py)."""
import ast
import unittest

from utils.semantic_operators.structural import (
    _handler_skeleton,
    _match_indent_convention,
    _only_pass_or_dead,
    _protected_call_parts,
    _real_try_blocks,
    try_except_candidate,
)


class _Inst:
    def __init__(self, offset, opname, argrepr=""):
        self.offset, self.opname, self.argrepr = offset, opname, argrepr


class _Entry:
    def __init__(self, start, end, target, depth, lasti):
        self.start, self.end, self.target, self.depth, self.lasti = start, end, target, depth, lasti


class _BC:
    def __init__(self, insts, table=None):
        self._insts = insts
        self.named_exception_table = table

    def __iter__(self):
        return iter(self._insts)


# GT for a function whose body is `try: os.remove(path) except: pass`.
GT_INSTS = [
    _Inst(12, "LOAD_GLOBAL", "NULL + os"), _Inst(14, "LOAD_ATTR", "remove"),
    _Inst(16, "LOAD_FAST", "path"), _Inst(18, "CALL"), _Inst(20, "POP_TOP"),
    _Inst(22, "JUMP_FORWARD"), _Inst(24, "PUSH_EXC_INFO"), _Inst(26, "POP_TOP"),
    _Inst(28, "POP_EXCEPT"), _Inst(30, "JUMP_FORWARD"),
    _Inst(38, "LOAD_CONST", "None"), _Inst(40, "RETURN_VALUE"),
]
GT_BC = _BC(GT_INSTS, table=[_Entry(12, 22, 24, 0, False), _Entry(24, 28, 32, 1, True)])
DER_BC = _BC([], table=[])


class HelpersTest(unittest.TestCase):
    def test_real_try_blocks_excludes_cleanup(self):
        # Only the lasti=False entry is a real try body.
        self.assertEqual(_real_try_blocks(GT_BC), [(12, 22, 24, 0)])
        self.assertEqual(_real_try_blocks(DER_BC), [])

    def test_protected_call_parts_handles_null_prefix(self):
        self.assertEqual(_protected_call_parts(GT_INSTS, 12, 22), ["os", "remove"])

    def test_protected_call_parts_declines_multi_call(self):
        insts = [_Inst(0, "LOAD_GLOBAL", "a"), _Inst(2, "CALL"), _Inst(4, "CALL")]
        self.assertIsNone(_protected_call_parts(insts, 0, 6))

    def test_handler_skeleton_bare_except(self):
        trivial, exc = _handler_skeleton(GT_INSTS, 24)
        self.assertTrue(trivial)
        self.assertIsNone(exc)

    def test_handler_skeleton_typed_except(self):
        insts = [
            _Inst(24, "PUSH_EXC_INFO"), _Inst(26, "LOAD_GLOBAL", "ValueError"),
            _Inst(28, "CHECK_EXC_MATCH"), _Inst(30, "POP_JUMP_FORWARD_IF_FALSE"),
            _Inst(32, "POP_TOP"), _Inst(34, "POP_EXCEPT"), _Inst(36, "JUMP_FORWARD"),
        ]
        trivial, exc = _handler_skeleton(insts, 24)
        self.assertTrue(trivial)
        self.assertEqual(exc, "ValueError")

    def test_handler_skeleton_nontrivial_body_declines(self):
        insts = [_Inst(24, "PUSH_EXC_INFO"), _Inst(26, "LOAD_FAST", "x"),
                 _Inst(28, "STORE_FAST", "y"), _Inst(30, "JUMP_FORWARD")]
        trivial, _ = _handler_skeleton(insts, 24)
        self.assertFalse(trivial)

    def test_match_indent_hanging_convention(self):
        original = "def f(path):\n        pass\n"  # first line col 0, body col 8
        unparsed = "def f(path):\n    pass"          # unparse: body col 4
        out = _match_indent_convention(original, unparsed)
        self.assertEqual(out, "def f(path):\n        pass\n")

    # --- regression tests for adversarial-review findings ---

    def test_handler_with_return_is_not_trivial(self):
        # `except: return None` must NOT be reduced to `except: pass`.
        insts = [_Inst(24, "PUSH_EXC_INFO"), _Inst(26, "POP_TOP"),
                 _Inst(28, "POP_EXCEPT"), _Inst(30, "RETURN_CONST", "None")]
        trivial, _ = _handler_skeleton(insts, 24)
        self.assertFalse(trivial)

    def test_protected_call_parts_ignores_argument_attributes(self):
        # os.remove(self.path): callee is os.remove; self.path is an ARGUMENT.
        insts = [_Inst(0, "LOAD_GLOBAL", "NULL + os"), _Inst(2, "LOAD_ATTR", "remove"),
                 _Inst(4, "LOAD_FAST", "self"), _Inst(6, "LOAD_ATTR", "path"), _Inst(8, "CALL")]
        self.assertEqual(_protected_call_parts(insts, 0, 10), ["os", "remove"])

    def test_only_pass_or_dead_respects_nested_else(self):
        live = ast.parse("try:\n pass\nexcept:\n pass\nelse:\n x = 1\n").body
        self.assertFalse(_only_pass_or_dead(live))
        dead = ast.parse("try:\n pass\nexcept:\n pass\n").body
        self.assertTrue(_only_pass_or_dead(dead))

    def test_decorated_function_does_not_crash(self):
        # Hanging decorated fragment: must produce parseable output or decline (never crash).
        frag = "@contextmanager\ndef f():\n    shutil.rmtree(d)\n"
        out = _match_indent_convention(frag, "@contextmanager\ndef f():\n    try:\n        shutil.rmtree(d)\n    except:\n        pass")
        # Whatever it returns, the operator's re-parse guard would decline invalid output;
        # here we just assert _match_indent_convention itself returns a string without error.
        self.assertIsInstance(out, str)


class OperatorTest(unittest.TestCase):
    FRAGMENT = (
        "def delete_file(path):\n"
        "        if os.path.exists(path):\n"
        "            os.remove(path)\n"
        "        else:\n"
        "            try:\n"
        "                pass\n"
        "            except:\n"
        "                pass\n"
        "            pass\n"
        "        return None\n"
    )

    def test_fires_and_wraps(self):
        ctx = {"pylingual_failed_result": {"message": "Different control flow"}, "failed_offset": None}
        out = try_except_candidate(GT_BC, DER_BC, self.FRAGMENT, ctx)
        self.assertIsNotNone(out)
        text = out["text"]
        # os.remove wrapped in try/except; dead placeholder + else removed.
        self.assertIn("try:", text)
        self.assertIn("os.remove(path)", text)
        self.assertNotIn("else:", text)
        # the wrapped call is now inside a try under the if
        self.assertEqual(text.count("try:"), 1)
        self.assertEqual(text.count("except"), 1)

    def test_declines_non_control_flow(self):
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 4}
        self.assertIsNone(try_except_candidate(GT_BC, DER_BC, self.FRAGMENT, ctx))

    def test_declines_when_counts_match(self):
        # Candidate already has the same number of real try blocks as GT.
        der = _BC([], table=[_Entry(0, 4, 6, 0, False)])
        ctx = {"pylingual_failed_result": {"message": "Different control flow"}, "failed_offset": None}
        self.assertIsNone(try_except_candidate(GT_BC, der, self.FRAGMENT, ctx))


if __name__ == "__main__":
    unittest.main()
