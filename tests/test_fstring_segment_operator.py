"""Unit tests for the f-string spurious-segment operator
(utils/semantic_operators/fstring.py).

Pattern targeted: PyLingual reconstructs an f-string with one EXTRA trailing
literal segment the ground truth does not have — classically ``f"_{x}0"`` for a
ground-truth ``f"_{x}"`` (a stray ``LOAD_CONST '0'`` before ``BUILD_STRING``,
whose count is one higher than GT's). The operator reads the surplus from the
bytecode, localizes the unique matching JoinedStr in the source, and drops the
trailing literal segment. Every ambiguity defers (returns None); the loop's
recompile + oracle accept gate is the final backstop.
"""
import os
import tempfile
import unittest
from pathlib import Path

from utils.semantic_operators.fstring import fstring_segment_candidate


class _Inst:
    """Minimal instruction stub exposing the attributes ``_instructions`` reads."""

    def __init__(self, offset, opname, argrepr, argval=None):
        self.offset, self.opname, self.argrepr, self.argval = offset, opname, argrepr, argval


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


def _ctx(message="Different bytecode", failed_offset=None):
    return {"pylingual_failed_result": {"message": message}, "failed_offset": failed_offset}


class FStringSegmentHappyPathTest(unittest.TestCase):
    def test_drops_trailing_zero_segment(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "LOAD_CONST", "'0'", "0"),
            _Inst(8, "BUILD_STRING", "3", 3),
        ])
        frag = "def f(x):\n    return f'a{x}0'\n"
        out = fstring_segment_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n    return f'a{x}'\n")
        self.assertEqual(out["opname"], "BUILD_STRING")
        self.assertEqual(out["kind"], "fstring_segment")
        self.assertEqual(out["confidence"], "unique")

    def test_drops_trailing_nonzero_segment(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "LOAD_CONST", "'2'", "2"),
            _Inst(8, "BUILD_STRING", "3", 3),
        ])
        frag = "def f(x):\n    return f'a{x}2'\n"
        out = fstring_segment_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n    return f'a{x}'\n")


class FStringSegmentDeclineTest(unittest.TestCase):
    def test_declines_interior_surplus(self):
        # GT pieces [Const 'a', FMT, Const 'b']; DER inserts a surplus INTERIOR const.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "LOAD_CONST", "'b'", "b"),
            _Inst(8, "BUILD_STRING", "3", 3),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_CONST", "'X'", "X"),
            _Inst(4, "LOAD_FAST", "x", "x"),
            _Inst(6, "FORMAT_VALUE", "", None),
            _Inst(8, "LOAD_CONST", "'b'", "b"),
            _Inst(10, "BUILD_STRING", "4", 4),
        ])
        frag = "def f(x):\n    return f'aX{x}b'\n"
        self.assertIsNone(fstring_segment_candidate(gt, der, frag, _ctx()))

    def test_declines_two_surplus_pieces(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "LOAD_CONST", "'0'", "0"),
            _Inst(8, "LOAD_CONST", "'1'", "1"),
            _Inst(10, "BUILD_STRING", "4", 4),
        ])
        frag = "def f(x):\n    return f'a{x}01'\n"
        self.assertIsNone(fstring_segment_candidate(gt, der, frag, _ctx()))

    def test_declines_ambiguous_two_joinedstr(self):
        # Two f-strings in the fragment, both der_count pieces ending in Constant '0'.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "LOAD_CONST", "'0'", "0"),
            _Inst(8, "BUILD_STRING", "3", 3),
        ])
        frag = "def f(x):\n    a = f'a{x}0'\n    b = f'b{x}0'\n    return a + b\n"
        self.assertIsNone(fstring_segment_candidate(gt, der, frag, _ctx()))

    def test_declines_surplus_is_format_value(self):
        # der count one higher, but the trailing extra op is a FORMAT_VALUE (non-Constant).
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "LOAD_FAST", "y", "y"),
            _Inst(8, "FORMAT_VALUE", "", None),
            _Inst(10, "BUILD_STRING", "3", 3),
        ])
        frag = "def f(x, y):\n    return f'a{x}{y}'\n"
        self.assertIsNone(fstring_segment_candidate(gt, der, frag, _ctx()))

    def test_declines_equal_counts(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        frag = "def f(x):\n    return f'a{x}'\n"
        self.assertIsNone(fstring_segment_candidate(gt, der, frag, _ctx()))

    def test_declines_gt_contains_surplus_operand(self):
        # GT itself has a LOAD_CONST '0' segment -> never delete a segment GT keeps.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'0'", "0"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'0'", "0"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "LOAD_CONST", "'0'", "0"),
            _Inst(8, "BUILD_STRING", "3", 3),
        ])
        frag = "def f(x):\n    return f'0{x}0'\n"
        self.assertIsNone(fstring_segment_candidate(gt, der, frag, _ctx()))

    def test_declines_wrong_regime_or_identical(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "BUILD_STRING", "2", 2),
        ])
        der = _BC([
            _Inst(0, "LOAD_CONST", "'a'", "a"),
            _Inst(2, "LOAD_FAST", "x", "x"),
            _Inst(4, "FORMAT_VALUE", "", None),
            _Inst(6, "LOAD_CONST", "'0'", "0"),
            _Inst(8, "BUILD_STRING", "3", 3),
        ])
        frag = "def f(x):\n    return f'a{x}0'\n"
        # wrong message regime
        self.assertIsNone(fstring_segment_candidate(gt, der, frag, _ctx("Different control flow")))
        self.assertIsNone(fstring_segment_candidate(gt, der, frag, None))
        # GT == DER (no divergence at all)
        self.assertIsNone(fstring_segment_candidate(gt, gt, "def f(x):\n    return f'a{x}'\n", _ctx()))


class FStringSegmentIntegrationTest(unittest.TestCase):
    """End-to-end: compile a real GT (``f"a{x}"``) and derived (``f"a{x}0"``) to
    .pyc, run the real pylingual compare_pyc, apply the operator to the derived
    fragment, recompile, and confirm every code object is now Equal (oracle
    accept, no PyLingual regression) with no LLM involved."""

    def test_recompiled_candidate_matches_gt(self):
        try:
            from pylingual.equivalence_check import compare_pyc  # noqa: F401
        except Exception as exc:  # pragma: no cover - pylingual missing
            self.skipTest(f"pylingual unavailable: {exc}")
        import py_compile

        from pylingual.equivalence_check import compare_pyc

        gt_src = 'def f(x):\n    return f"a{x}"\n'
        der_src = 'def f(x):\n    return f"a{x}0"\n'
        with tempfile.TemporaryDirectory() as d:
            gt_py = os.path.join(d, "gt.py")
            der_py = os.path.join(d, "der.py")
            Path(gt_py).write_text(gt_src, encoding="utf-8")
            Path(der_py).write_text(der_src, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc")))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc")))

            failures = [tr for tr in compare_pyc(gt_pyc, der_pyc) if not tr.success]
            self.assertTrue(failures, "expected a bytecode divergence for the extra segment")
            tr = next(t for t in failures if t.message == "Different bytecode")

            ctx = {
                "pylingual_failed_result": {"message": tr.message},
                "failed_offset": tr.failed_offset,
            }
            out = fstring_segment_candidate(tr.bc_a, tr.bc_b, der_src, ctx)
            self.assertIsNotNone(out)
            # ast.unparse normalizes quote style; the surplus segment must be gone and
            # the f-string preserved (bytecode-equal to GT is asserted below).
            self.assertEqual(out["text"], "def f(x):\n    return f'a{x}'\n")
            self.assertNotIn("0", out["text"])

            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc")))
            self.assertTrue(all(tr2.success for tr2 in compare_pyc(gt_pyc, fixed_pyc)))


if __name__ == "__main__":
    unittest.main()
