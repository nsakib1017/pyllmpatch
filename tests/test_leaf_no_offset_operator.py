"""Unit tests for the offset-independent leaf swap (leaf_value_no_offset_candidate):
the "Different bytecode" + no-failed_offset regime (~84% of residual objects), where the
lone diverging operand is inferred from the GT/derived instruction-stream diff."""
import unittest

from utils.semantic_operators.leaf import (
    _infer_single_operand_offset,
    leaf_value_no_offset_candidate,
)


class _Inst:
    def __init__(self, offset, opname, argrepr, argval=None):
        self.offset, self.opname, self.argrepr = offset, opname, argrepr
        self.argval = argval


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


def _stream(const_val):
    """LOAD_FAST x; LOAD_CONST <const_val>; BINARY_OP +; RETURN_VALUE."""
    return _BC([
        _Inst(0, "LOAD_FAST", "x", "x"),
        _Inst(2, "LOAD_CONST", repr(const_val), const_val),
        _Inst(4, "BINARY_OP", "+", "+"),
        _Inst(6, "RETURN_VALUE", "", None),
    ])


NO_OFFSET_CTX = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": None}


class InferSingleOperandOffsetTest(unittest.TestCase):
    def test_single_literal_divergence_returns_derived_offset(self):
        der, gt = _stream(1), _stream(2)
        self.assertEqual(_infer_single_operand_offset(gt, der), 2)

    def test_identical_streams_return_none(self):
        self.assertIsNone(_infer_single_operand_offset(_stream(1), _stream(1)))

    def test_length_mismatch_defers(self):
        der = _BC([_Inst(0, "LOAD_FAST", "x", "x"), _Inst(2, "RETURN_VALUE", "", None)])
        gt = _stream(2)
        self.assertIsNone(_infer_single_operand_offset(gt, der))

    def test_opcode_mismatch_defers(self):
        der = _BC([
            _Inst(0, "LOAD_FAST", "x", "x"),
            _Inst(2, "LOAD_CONST", "1", 1),      # derived loads a const
            _Inst(4, "BINARY_OP", "+", "+"),
            _Inst(6, "RETURN_VALUE", "", None),
        ])
        gt = _BC([
            _Inst(0, "LOAD_FAST", "x", "x"),
            _Inst(2, "LOAD_GLOBAL", "y", "y"),   # GT loads a name -> structural, defer
            _Inst(4, "BINARY_OP", "+", "+"),
            _Inst(6, "RETURN_VALUE", "", None),
        ])
        self.assertIsNone(_infer_single_operand_offset(gt, der))

    def test_two_operand_divergences_defer(self):
        der = _BC([
            _Inst(0, "LOAD_FAST", "x", "x"),
            _Inst(2, "LOAD_CONST", "1", 1),
            _Inst(4, "COMPARE_OP", "<", "<"),
            _Inst(6, "RETURN_VALUE", "", None),
        ])
        gt = _BC([
            _Inst(0, "LOAD_FAST", "x", "x"),
            _Inst(2, "LOAD_CONST", "2", 2),      # divergence 1
            _Inst(4, "COMPARE_OP", ">", ">"),    # divergence 2 -> compound, defer
            _Inst(6, "RETURN_VALUE", "", None),
        ])
        self.assertIsNone(_infer_single_operand_offset(gt, der))


class LeafNoOffsetCandidateTest(unittest.TestCase):
    def test_literal_swap_when_no_offset(self):
        frag = "def f(x):\n    return x + 1\n"
        out = leaf_value_no_offset_candidate(_stream(2), _stream(1), frag, NO_OFFSET_CTX)
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n    return x + 2\n")
        self.assertEqual(out["kind"], "literal")

    def test_defers_when_failed_offset_present(self):
        # the offset path (leaf_value_candidate) owns this case; the no-offset variant declines
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 2}
        frag = "def f(x):\n    return x + 1\n"
        self.assertIsNone(leaf_value_no_offset_candidate(_stream(2), _stream(1), frag, ctx))

    def test_defers_on_wrong_message(self):
        ctx = {"pylingual_failed_result": {"message": "Different control flow"}, "failed_offset": None}
        frag = "def f(x):\n    return x + 1\n"
        self.assertIsNone(leaf_value_no_offset_candidate(_stream(2), _stream(1), frag, ctx))

    def test_defers_when_identical(self):
        frag = "def f(x):\n    return x + 1\n"
        self.assertIsNone(leaf_value_no_offset_candidate(_stream(1), _stream(1), frag, NO_OFFSET_CTX))

    def test_none_context_defers(self):
        self.assertIsNone(leaf_value_no_offset_candidate(_stream(2), _stream(1), "def f(x):\n    return x\n", None))


if __name__ == "__main__":
    unittest.main()
