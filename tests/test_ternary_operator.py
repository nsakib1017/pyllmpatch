"""Unit + end-to-end tests for the ternary (conditional-expression) operator
(utils/semantic_operators/ternary_op.py).

The operator deterministically recovers a ``X if COND else Y`` value expression from
GROUND-TRUTH bytecode and splices it into a derived fragment that rendered it wrong
(kept only one arm, stranded it, or wrote an ``if`` statement). A ternary compiles to a
bare ``POP_JUMP_IF_*`` on the condition and two value-producing arms sharing one consumer
(a duplicated terminating tail, or a convergence point). NO ``STORE`` sits inside an arm's
value — that is what separates a ternary EXPRESSION from a multi-statement ``if``.

Pure recovery + detection are exercised with mock instruction streams (offsets/opcodes
copied from the real 3.12 disassembly via pylingual's editable bytecode); the end-to-end
tests drive the real pylingual ``compare_pyc`` on the 3.12 host and assert the recompiled
output reaches combined_distance 0 against GT. Mirrors tests/test_bool_shortcircuit.py.
"""
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# <repo>/pylingual (the importable package root) and <repo> must both be on sys.path so
# `import pylingual.*` and `import utils.*` resolve regardless of launch cwd. Mirrors the
# repo's other operator tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.ternary_op import (
    _detect_single_ternary,
    _recover_value_expr,
    ternary_op_candidate,
)


class _Inst:
    """Mock instruction matching the shape pylingual's editable bytecode yields."""

    def __init__(self, opname, argrepr="", arg=None, argval=None, starts_line=None, offset=0):
        self.opname = opname
        self.argrepr = argrepr
        self.arg = arg
        self.argval = argval
        self.starts_line = starts_line
        self.offset = offset


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


def _ctx(message="Different control flow"):
    return {"pylingual_failed_result": {"message": message}}


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import (
        compare_code_object_distances,
        summarize_results,
    )
    return summarize_results(compare_code_object_distances(gt_pyc, derived_pyc)).get(
        "combined_distance"
    )


def _lf(name, off, line=None):
    return _Inst("LOAD_FAST", name, argval=name, starts_line=line, offset=off)


def _ret(off):
    return _Inst("RETURN_VALUE", offset=off)


# ---------------------------------------------------------------------------
# Canonical mock instruction streams (offsets/opcodes from the real 3.12
# disassembly through pylingual — argval carries names, jump targets, argcounts).
# ---------------------------------------------------------------------------
def _bc_return_ternary():
    """``return a if c else b`` — duplicated-tail form, RETURN consumer."""
    return [
        _lf("c", 0, line=2),
        _Inst("POP_JUMP_IF_FALSE", "to 8", argval=8, offset=2),
        _lf("a", 4), _ret(6),
        _lf("b", 8), _ret(10),
    ]


def _bc_assign_ternary():
    """``x = p if cond else q`` (then ``return x``) — duplicated-tail, STORE consumer."""
    return [
        _lf("cond", 0, line=2),
        _Inst("POP_JUMP_IF_FALSE", "to 12", argval=12, offset=2),
        _lf("p", 4), _Inst("STORE_FAST", "x", argval="x", offset=6), _lf("x", 8), _ret(10),
        _lf("q", 12), _Inst("STORE_FAST", "x", argval="x", offset=14), _lf("x", 16), _ret(18),
    ]


def _bc_call_arg_ternary():
    """``g(a if c else b)`` — duplicated-tail, CALL consumer (ternary as a call argument)."""
    return [
        _Inst("PUSH_NULL", offset=0),
        _lf("g", 2, line=2), _lf("c", 4),
        _Inst("POP_JUMP_IF_FALSE", "to 16", argval=16, offset=6),
        _lf("a", 8), _Inst("CALL", arg=1, argval=1, offset=10),
        _Inst("POP_TOP", offset=12), _Inst("RETURN_CONST", "None", argval=None, offset=14),
        _lf("b", 16), _Inst("CALL", arg=1, argval=1, offset=18),
        _Inst("POP_TOP", offset=20), _Inst("RETURN_CONST", "None", argval=None, offset=22),
    ]


def _bc_call_branch_ternary():
    """``file = io.BytesIO() if mode == 'rb' else io.StringIO()`` (real 642dfddb shape):
    compare condition, both arms are simple calls, STORE consumer."""
    return [
        _lf("mode", 0, line=2),
        _Inst("LOAD_CONST", '"rb"', argval="rb", offset=2),
        _Inst("COMPARE_OP", "==", argval="==", offset=4),
        _Inst("POP_JUMP_IF_FALSE", "to 20", argval=20, offset=6),
        _lf("io", 8), _Inst("LOAD_ATTR", "NULL|self + BytesIO", argval="BytesIO", offset=10),
        _Inst("CALL", arg=0, argval=0, offset=12),
        _Inst("STORE_FAST", "file", argval="file", offset=14), _lf("file", 16), _ret(18),
        _lf("io", 20), _Inst("LOAD_ATTR", "NULL|self + StringIO", argval="StringIO", offset=22),
        _Inst("CALL", arg=0, argval=0, offset=24),
        _Inst("STORE_FAST", "file", argval="file", offset=26), _lf("file", 28), _ret(30),
    ]


def _bc_convergent_assign():
    """``x = p if cond else q`` with trailing ``z = x + 1`` — convergent (JUMP_FORWARD) form."""
    return [
        _lf("cond", 0, line=2),
        _Inst("POP_JUMP_IF_FALSE", "to 8", argval=8, offset=2),
        _lf("p", 4), _Inst("JUMP_FORWARD", "to 10", argval=10, offset=6),
        _lf("q", 8),
        _Inst("STORE_FAST", "x", argval="x", offset=10),
        _lf("x", 12), _Inst("LOAD_CONST", "1", argval=1, offset=14),
        _Inst("BINARY_OP", "+", argval=0, offset=16),
        _Inst("STORE_FAST", "z", argval="z", offset=18), _lf("z", 20), _ret(22),
    ]


def _bc_compound_condition():
    """``return a if c and d else b`` — two POP_JUMPs (compound condition) -> defer."""
    return [
        _lf("c", 0, line=2),
        _Inst("POP_JUMP_IF_FALSE", "to 12", argval=12, offset=2),
        _lf("d", 4),
        _Inst("POP_JUMP_IF_FALSE", "to 12", argval=12, offset=6),
        _lf("a", 8), _ret(10),
        _lf("b", 12), _ret(14),
    ]


def _bc_if_statement_multistore():
    """A genuine multi-statement ``if`` (stores to two names per arm) -> defer."""
    return [
        _lf("cond", 0, line=2),
        _Inst("POP_JUMP_IF_FALSE", "to 20", argval=20, offset=2),
        _lf("a", 4), _Inst("STORE_FAST", "x", argval="x", offset=6),
        _Inst("LOAD_CONST", "1", argval=1, offset=8), _Inst("STORE_FAST", "y", argval="y", offset=10),
        _lf("x", 12), _lf("y", 14), _Inst("BINARY_OP", "+", argval=0, offset=16), _ret(18),
        _lf("b", 20), _Inst("STORE_FAST", "x", argval="x", offset=22),
        _Inst("LOAD_CONST", "2", argval=2, offset=24), _Inst("STORE_FAST", "y", argval="y", offset=26),
        _lf("x", 28), _lf("y", 30), _Inst("BINARY_OP", "+", argval=0, offset=32), _ret(34),
    ]


# ---------------------------------------------------------------------------
# Pure value-recovery helper tests (mock instructions)
# ---------------------------------------------------------------------------
class RecoverValueExprTest(unittest.TestCase):
    def test_single_name(self):
        self.assertEqual(_recover_value_expr([_lf("a", 0)]), "a")

    def test_attribute_chain(self):
        insts = [_lf("self", 0), _Inst("LOAD_ATTR", argval="mode")]
        self.assertEqual(_recover_value_expr(insts), "self.mode")

    def test_const(self):
        self.assertEqual(_recover_value_expr([_Inst("LOAD_CONST", argval="rb")]), "'rb'")

    def test_compare(self):
        insts = [_lf("mode", 0), _Inst("LOAD_CONST", argval="rb"), _Inst("COMPARE_OP", "==", argval="==")]
        self.assertEqual(_recover_value_expr(insts), "mode == 'rb'")

    def test_simple_call_zero_args(self):
        insts = [_lf("io", 0), _Inst("LOAD_ATTR", argval="BytesIO"), _Inst("CALL", arg=0, argval=0)]
        self.assertEqual(_recover_value_expr(insts), "io.BytesIO()")

    def test_simple_call_one_arg(self):
        insts = [_lf("f", 0), _lf("a", 2), _Inst("CALL", arg=1, argval=1)]
        self.assertEqual(_recover_value_expr(insts), "f(a)")

    def test_store_defers(self):
        self.assertIsNone(_recover_value_expr([_lf("a", 0), _Inst("STORE_FAST", argval="x")]))

    def test_empty_defers(self):
        self.assertIsNone(_recover_value_expr([]))

    def test_two_values_defer(self):
        self.assertIsNone(_recover_value_expr([_lf("a", 0), _lf("b", 2)]))

    def test_binary_op_defers(self):
        insts = [_lf("a", 0), _lf("b", 2), _Inst("BINARY_OP", "+", argval=0)]
        self.assertIsNone(_recover_value_expr(insts))


# ---------------------------------------------------------------------------
# Detection tests (mock instructions)
# ---------------------------------------------------------------------------
class DetectTernaryTest(unittest.TestCase):
    def test_return_ternary(self):
        u = _detect_single_ternary(_BC(_bc_return_ternary()))
        self.assertIsNotNone(u)
        self.assertEqual((u["then"], u["cond"], u["else"]), ("a", "c", "b"))
        self.assertEqual(u["consumer"], ("return", None))

    def test_assign_ternary(self):
        u = _detect_single_ternary(_BC(_bc_assign_ternary()))
        self.assertIsNotNone(u)
        self.assertEqual((u["then"], u["cond"], u["else"]), ("p", "cond", "q"))
        self.assertEqual(u["consumer"], ("assign", "x"))

    def test_call_arg_ternary(self):
        u = _detect_single_ternary(_BC(_bc_call_arg_ternary()))
        self.assertIsNotNone(u)
        self.assertEqual((u["then"], u["cond"], u["else"]), ("a", "c", "b"))
        self.assertEqual(u["consumer"], ("call", None))

    def test_call_branch_ternary(self):
        u = _detect_single_ternary(_BC(_bc_call_branch_ternary()))
        self.assertIsNotNone(u)
        self.assertEqual(u["then"], "io.BytesIO()")
        self.assertEqual(u["else"], "io.StringIO()")
        self.assertEqual(u["cond"], "mode == 'rb'")
        self.assertEqual(u["consumer"], ("assign", "file"))

    def test_convergent_assign(self):
        u = _detect_single_ternary(_BC(_bc_convergent_assign()))
        self.assertIsNotNone(u)
        self.assertEqual((u["then"], u["cond"], u["else"]), ("p", "cond", "q"))
        self.assertEqual(u["consumer"], ("assign", "x"))

    def test_compound_condition_defers(self):
        self.assertIsNone(_detect_single_ternary(_BC(_bc_compound_condition())))

    def test_if_statement_multistore_defers(self):
        self.assertIsNone(_detect_single_ternary(_BC(_bc_if_statement_multistore())))

    def test_no_branch_defers(self):
        self.assertIsNone(_detect_single_ternary(_BC([_lf("a", 0), _ret(2)])))


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests through the public entry point (mock bytecode)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_wrong_regime_defers(self):
        self.assertIsNone(ternary_op_candidate(
            _BC(_bc_return_ternary()), _BC([_lf("a", 0), _ret(2)]),
            "def f(a, b, c):\n    return a\n", _ctx("Different names")))

    def test_missing_context_defers(self):
        self.assertIsNone(ternary_op_candidate(
            _BC(_bc_return_ternary()), _BC([_lf("a", 0), _ret(2)]),
            "def f(a, b, c):\n    return a\n", None))

    def test_if_statement_defers(self):
        # (D) a genuine multi-statement if-STATEMENT -> control_flow's job -> defer.
        der = "def f(a, b, cond):\n    x = a\n    y = 1\n    return x + y\n"
        self.assertIsNone(ternary_op_candidate(
            _BC(_bc_if_statement_multistore()), _BC([_lf("a", 0)]), der, _ctx()))

    def test_compound_condition_defers(self):
        # (D) compound `a and b` condition -> second POP_JUMP -> defer.
        self.assertIsNone(ternary_op_candidate(
            _BC(_bc_compound_condition()), _BC([_lf("a", 0), _ret(2)]),
            "def f(a, b, c, d):\n    return a\n", _ctx()))

    def test_unparseable_fragment_defers(self):
        self.assertIsNone(ternary_op_candidate(
            _BC(_bc_return_ternary()), _BC([_lf("a", 0)]), "def (:\n", _ctx()))

    def test_no_matching_position_defers(self):
        # GT ternary is on `a`/`b`, but derived has neither -> nowhere to splice -> defer.
        self.assertIsNone(ternary_op_candidate(
            _BC(_bc_return_ternary()), _BC([_lf("z", 0), _ret(2)]),
            "def f(a, b, c):\n    return z\n", _ctx()))

    def test_noop_when_already_correct_defers(self):
        # Derived already renders the ternary -> span replace is a no-op -> defer.
        self.assertIsNone(ternary_op_candidate(
            _BC(_bc_return_ternary()), _BC(_bc_return_ternary()),
            "def f(a, b, c):\n    return a if c else b\n", _ctx()))

    def test_never_raises_on_garbage(self):
        gt = _BC([_Inst("WEIRD_UNKNOWN_OP", arg=3), _Inst("MAP_ADD", arg=5)])
        der = _BC([None])  # attribute access on None inside -> swallowed by try/except
        self.assertIsNone(ternary_op_candidate(gt, der, "x = 1\n", _ctx()))


# ---------------------------------------------------------------------------
# Positive splice tests through the public entry point (mock bytecode)
# ---------------------------------------------------------------------------
class PositiveMockTest(unittest.TestCase):
    def test_return_recovers_dropped_arm(self):
        out = ternary_op_candidate(_BC(_bc_return_ternary()), _BC([_lf("a", 0), _ret(2)]),
                                   "def f(a, b, c):\n    return a\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b, c):\n    return a if c else b\n")
        self.assertEqual(out["kind"], "ternary")
        self.assertEqual(out["confidence"], "unique")
        compile(out["text"], "<m>", "exec")

    def test_return_recovers_from_other_arm(self):
        # Derived rendered the ELSE arm (`b`); still uniquely located and rewritten.
        out = ternary_op_candidate(_BC(_bc_return_ternary()), _BC([_lf("b", 0), _ret(2)]),
                                   "def f(a, b, c):\n    return b\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b, c):\n    return a if c else b\n")

    def test_assign_recovers(self):
        out = ternary_op_candidate(_BC(_bc_assign_ternary()), _BC([_lf("p", 0)]),
                                   "def f(p, q, cond):\n    x = p\n    return x\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(p, q, cond):\n    x = p if cond else q\n    return x\n")

    def test_call_arg_recovers(self):
        out = ternary_op_candidate(_BC(_bc_call_arg_ternary()), _BC([_lf("a", 0)]),
                                   "def f(a, b, c, g):\n    g(a)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b, c, g):\n    g(a if c else b)\n")

    def test_preserves_indentation(self):
        frag = "def f(a, b, c):\n        return a\n"
        out = ternary_op_candidate(_BC(_bc_return_ternary()), _BC([_lf("a", 0)]), frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b, c):\n        return a if c else b\n")


# ---------------------------------------------------------------------------
# End-to-end tests through the real pylingual compare_pyc + oracle distance
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


def _run_end_to_end(testcase, gt_src, der_src, expect_operator_substr):
    from pylingual.equivalence_check import compare_pyc

    with tempfile.TemporaryDirectory() as d:
        gt_py = os.path.join(d, "gt.py")
        der_py = os.path.join(d, "der.py")
        Path(gt_py).write_text(gt_src, encoding="utf-8")
        Path(der_py).write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
        der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))

        failures = [tr for tr in compare_pyc(gt_pyc, der_pyc)
                    if not tr.success and tr.bc_a is not None and tr.bc_b is not None]
        testcase.assertTrue(failures, "expected a bytecode divergence")
        tr = failures[0]
        testcase.assertIn(tr.message, ("Different bytecode", "Different control flow"))

        out = ternary_op_candidate(
            tr.bc_a, tr.bc_b, der_src,
            {"pylingual_failed_result": {"message": tr.message}},
        )
        testcase.assertIsNotNone(out, "operator should fire on this case")
        testcase.assertIn(expect_operator_substr, out["operator"])

        fixed_py = os.path.join(d, "fixed.py")
        Path(fixed_py).write_text(out["text"], encoding="utf-8")
        fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
        testcase.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)


def _defers_end_to_end(testcase, gt_src, der_src):
    """Assert the operator DEFERS (returns None) on a real divergence — never fires wrong."""
    from pylingual.equivalence_check import compare_pyc

    with tempfile.TemporaryDirectory() as d:
        gt_py = os.path.join(d, "gt.py")
        der_py = os.path.join(d, "der.py")
        Path(gt_py).write_text(gt_src, encoding="utf-8")
        Path(der_py).write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
        der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
        failures = [tr for tr in compare_pyc(gt_pyc, der_pyc)
                    if not tr.success and tr.bc_a is not None and tr.bc_b is not None]
        testcase.assertTrue(failures, "expected a bytecode divergence")
        tr = failures[0]
        out = ternary_op_candidate(
            tr.bc_a, tr.bc_b, der_src,
            {"pylingual_failed_result": {"message": tr.message}},
        )
        testcase.assertIsNone(out, "operator should DEFER on this case")


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class EndToEndTest(unittest.TestCase):
    def test_A_return_ternary(self):
        # (A) GT `return a if c else b`; derived kept only `a`.
        _run_end_to_end(self, "def f(a, b, c):\n    return a if c else b\n",
                        "def f(a, b, c):\n    return a\n", "if c else")

    def test_A_return_ternary_other_arm(self):
        # (A) variant: derived kept the ELSE arm `b`.
        _run_end_to_end(self, "def f(a, b, c):\n    return a if c else b\n",
                        "def f(a, b, c):\n    return b\n", "a if c else b")

    def test_B_assign_ternary(self):
        # (B) GT `x = p if cond else q`; derived kept only `p`.
        _run_end_to_end(self, "def f(p, q, cond):\n    x = p if cond else q\n    return x\n",
                        "def f(p, q, cond):\n    x = p\n    return x\n", "p if cond else q")

    def test_B_assign_ternary_with_trailing(self):
        # (B) convergent form: a statement follows the ternary assignment.
        _run_end_to_end(self,
                        "def f(p, q, cond):\n    x = p if cond else q\n    z = x + 1\n    return z\n",
                        "def f(p, q, cond):\n    x = p\n    z = x + 1\n    return z\n",
                        "p if cond else q")

    def test_C_call_arg_ternary(self):
        # (C) GT `g(a if c else b)` — ternary as a call argument; derived kept only `a`.
        _run_end_to_end(self, "def f(a, b, c, g):\n    g(a if c else b)\n",
                        "def f(a, b, c, g):\n    g(a)\n", "a if c else b")

    def test_call_branch_values_642_shape(self):
        # Real 642dfddb shape: both arms are simple calls, compare condition.
        _run_end_to_end(
            self,
            "def f(mode, io):\n    file = io.BytesIO() if mode == 'rb' else io.StringIO()\n    return file\n",
            "def f(mode, io):\n    file = io.StringIO()\n    return file\n",
            "io.BytesIO() if mode == 'rb' else io.StringIO()")

    def test_compare_condition(self):
        # Condition is a comparison; both arms simple names.
        _run_end_to_end(self, "def f(a, b, n):\n    return a if n == 3 else b\n",
                        "def f(a, b, n):\n    return a\n", "a if n == 3 else b")

    # --- adversarial: must DEFER on a real divergence, never fire wrong ---
    def test_D_if_statement_multistore_defers(self):
        _defers_end_to_end(
            self,
            "def f(a, b, cond):\n    if cond:\n        x = a\n        y = 1\n    else:\n        x = b\n        y = 2\n    return x + y\n",
            "def f(a, b, cond):\n    x = a\n    y = 1\n    return x + y\n")

    def test_D_compound_condition_defers(self):
        _defers_end_to_end(self, "def f(a, b, c, d):\n    return a if c and d else b\n",
                           "def f(a, b, c, d):\n    return a\n")

    def test_D_nested_ternary_defers(self):
        # Two ternaries (nested) -> more than one conditional jump -> defer.
        _defers_end_to_end(self, "def f(a, b, c, p, q):\n    return (a if p else b) if q else c\n",
                           "def f(a, b, c, p, q):\n    return c\n")


if __name__ == "__main__":
    unittest.main()
