"""Unit + end-to-end tests for the boolean short-circuit operator
(utils/semantic_operators/bool_shortcircuit.py).

The operator deterministically recovers a short-circuit boolean expression
(``a and b`` / ``p or q`` and single-operator chains) from GROUND-TRUTH bytecode and
splices it into a derived fragment that rendered it wrong (dropped the tail operand,
kept a single operand, or used the wrong operator). On 3.11 the value-producing
short-circuit compiles to ``JUMP_IF_FALSE_OR_POP`` (``and``) / ``JUMP_IF_TRUE_OR_POP``
(``or``); on 3.12+ it is the ``COPY 1; POP_JUMP_IF_FALSE/TRUE; POP_TOP`` triple with
every branch targeting the value's consumer offset. Both forms are recognised.

Pure recovery is exercised with mock instruction streams (both encodings); the
end-to-end tests drive the real pylingual ``compare_pyc`` on the 3.12 host and assert the
recompiled output reaches combined_distance 0 against GT. Mirrors
tests/test_control_flow_operator.py and tests/test_literal_form_operator.py.
"""
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# The importable pylingual package lives at <repo>/pylingual/pylingual, so <repo>/pylingual
# must be on sys.path (alongside the repo root) for `import pylingual.*` and for `utils` to
# resolve regardless of launch cwd. Mirrors the repo's other operator tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.bool_shortcircuit import (
    _find_units,
    _recover_expr,
    bool_shortcircuit_candidate,
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


# ---------------------------------------------------------------------------
# Canonical mock instruction streams (offsets/opcodes copied from the real 3.12
# and 3.11 disassemblies).
# ---------------------------------------------------------------------------
def _copy(off):
    return _Inst("COPY", arg=1, argval=1, offset=off)


def _pop_top(off):
    return _Inst("POP_TOP", offset=off)


def _bc_return_and(off_line=2):
    """3.12 ``return a and b``: COPY/POP_JUMP_IF_FALSE/POP_TOP -> RETURN_VALUE."""
    return [
        _Inst("LOAD_FAST", "a", argval="a", starts_line=off_line, offset=0),
        _copy(2),
        _Inst("POP_JUMP_IF_FALSE", "to 10", argval=10, offset=4),
        _pop_top(6),
        _Inst("LOAD_FAST", "b", argval="b", offset=8),
        _Inst("RETURN_VALUE", offset=10),
    ]


def _bc_assign_or():
    """3.12 ``x = p or q``: COPY/POP_JUMP_IF_TRUE/POP_TOP -> STORE_FAST x."""
    return [
        _Inst("LOAD_FAST", "p", argval="p", starts_line=2, offset=0),
        _copy(2),
        _Inst("POP_JUMP_IF_TRUE", "to 10", argval=10, offset=4),
        _pop_top(6),
        _Inst("LOAD_FAST", "q", argval="q", offset=8),
        _Inst("STORE_FAST", "x", argval="x", offset=10),
    ]


def _bc_return_and_chain():
    """3.12 ``return a and b and c``: two FALSE boundaries, same target."""
    return [
        _Inst("LOAD_FAST", "a", argval="a", starts_line=2, offset=0),
        _copy(2),
        _Inst("POP_JUMP_IF_FALSE", "to 18", argval=18, offset=4),
        _pop_top(6),
        _Inst("LOAD_FAST", "b", argval="b", offset=8),
        _copy(10),
        _Inst("POP_JUMP_IF_FALSE", "to 18", argval=18, offset=12),
        _pop_top(14),
        _Inst("LOAD_FAST", "c", argval="c", offset=16),
        _Inst("RETURN_VALUE", offset=18),
    ]


def _bc_return_and_orpop_3_11():
    """3.11 form of ``return a and b`` using JUMP_IF_FALSE_OR_POP."""
    return [
        _Inst("LOAD_FAST", "a", argval="a", starts_line=2, offset=0),
        _Inst("JUMP_IF_FALSE_OR_POP", "to 6", argval=6, offset=2),
        _Inst("LOAD_FAST", "b", argval="b", offset=4),
        _Inst("RETURN_VALUE", offset=6),
    ]


def _bc_mixed_and_or():
    """3.12 ``return a and b or c`` -> one connected region, mixed kind/target."""
    return [
        _Inst("LOAD_FAST", "a", argval="a", starts_line=2, offset=2),
        _copy(4),
        _Inst("POP_JUMP_IF_FALSE", "to 12", argval=12, offset=6),
        _pop_top(8),
        _Inst("LOAD_FAST", "b", argval="b", offset=10),
        _copy(12),
        _Inst("POP_JUMP_IF_TRUE", "to 20", argval=20, offset=14),
        _pop_top(16),
        _Inst("LOAD_FAST", "c", argval="c", offset=18),
        _Inst("RETURN_VALUE", offset=20),
    ]


def _bc_call_operand():
    """3.12 ``return g() and b`` -> first operand is a call -> unrecoverable."""
    return [
        _Inst("LOAD_GLOBAL", "g", argval="g", starts_line=2, offset=0),
        _Inst("CALL", argval=0, offset=2),
        _copy(4),
        _Inst("POP_JUMP_IF_FALSE", "to 12", argval=12, offset=6),
        _pop_top(8),
        _Inst("LOAD_FAST", "b", argval="b", offset=10),
        _Inst("RETURN_VALUE", offset=12),
    ]


def _bc_two_booleans():
    """3.12 ``x = a and b`` / ``return c and d`` -> two separate regions."""
    return [
        _Inst("LOAD_FAST", "a", argval="a", starts_line=2, offset=0),
        _copy(2),
        _Inst("POP_JUMP_IF_FALSE", "to 10", argval=10, offset=4),
        _pop_top(6),
        _Inst("LOAD_FAST", "b", argval="b", offset=8),
        _Inst("STORE_FAST", "x", argval="x", offset=10),
        _Inst("LOAD_FAST", "c", argval="c", starts_line=3, offset=12),
        _copy(14),
        _Inst("POP_JUMP_IF_FALSE", "to 22", argval=22, offset=16),
        _pop_top(18),
        _Inst("LOAD_FAST", "d", argval="d", offset=20),
        _Inst("RETURN_VALUE", offset=22),
    ]


# ---------------------------------------------------------------------------
# Pure operand-recovery helper tests (mock instructions)
# ---------------------------------------------------------------------------
class RecoverExprTest(unittest.TestCase):
    def test_single_name(self):
        self.assertEqual(_recover_expr([_Inst("LOAD_FAST", "a", argval="a")]), "a")

    def test_attribute_chain(self):
        insts = [_Inst("LOAD_FAST", "self", argval="self"), _Inst("LOAD_ATTR", "mode", argval="mode")]
        self.assertEqual(_recover_expr(insts), "self.mode")

    def test_compare_const(self):
        insts = [_Inst("LOAD_FAST", "a", argval="a"), _Inst("LOAD_CONST", argval=1),
                 _Inst("COMPARE_OP", "==")]
        self.assertEqual(_recover_expr(insts), "a == 1")

    def test_const_string(self):
        self.assertEqual(_recover_expr([_Inst("LOAD_CONST", argval="rb")]), "'rb'")

    def test_empty_defers(self):
        self.assertIsNone(_recover_expr([]))

    def test_call_defers(self):
        insts = [_Inst("LOAD_GLOBAL", "g", argval="g"), _Inst("CALL", argval=0)]
        self.assertIsNone(_recover_expr(insts))

    def test_underflow_defers(self):
        self.assertIsNone(_recover_expr([_Inst("COMPARE_OP", "==")]))


# ---------------------------------------------------------------------------
# Unit / region detection tests (mock instructions)
# ---------------------------------------------------------------------------
class FindUnitsTest(unittest.TestCase):
    def test_return_and(self):
        units = _find_units(_bc_return_and())
        self.assertEqual(len(units), 1)
        u = units[0]
        self.assertTrue(u["valid"])
        self.assertEqual(u["kind"], "and")
        self.assertEqual(u["operands"], ["a", "b"])
        self.assertEqual(u["expr"], "a and b")
        self.assertEqual(u["consumer"], ("return", None))

    def test_assign_or(self):
        units = _find_units(_bc_assign_or())
        self.assertEqual(len(units), 1)
        u = units[0]
        self.assertTrue(u["valid"])
        self.assertEqual(u["kind"], "or")
        self.assertEqual(u["expr"], "p or q")
        self.assertEqual(u["consumer"], ("assign", "x"))

    def test_and_chain(self):
        units = _find_units(_bc_return_and_chain())
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0]["valid"])
        self.assertEqual(units[0]["expr"], "a and b and c")
        self.assertEqual(units[0]["operands"], ["a", "b", "c"])

    def test_orpop_3_11_form(self):
        # Cross-version: the 3.10-3.11 JUMP_IF_FALSE_OR_POP encoding recovers identically.
        units = _find_units(_bc_return_and_orpop_3_11())
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0]["valid"])
        self.assertEqual(units[0]["expr"], "a and b")
        self.assertEqual(units[0]["consumer"], ("return", None))

    def test_mixed_and_or_is_invalid(self):
        # `a and b or c`: one connected region with mixed kind + target -> not fireable.
        units = _find_units(_bc_mixed_and_or())
        self.assertEqual(len(units), 1)
        self.assertFalse(units[0]["valid"])

    def test_call_operand_is_invalid(self):
        units = _find_units(_bc_call_operand())
        self.assertEqual(len(units), 1)
        self.assertFalse(units[0]["valid"])

    def test_two_booleans_two_regions(self):
        units = _find_units(_bc_two_booleans())
        self.assertEqual(len(units), 2)
        self.assertTrue(all(u["valid"] for u in units))

    def test_no_boolean_no_units(self):
        insts = [_Inst("LOAD_FAST", "a", argval="a"), _Inst("RETURN_VALUE", offset=2)]
        self.assertEqual(_find_units(insts), [])


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests through the public entry point (mock bytecode)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_wrong_regime_defers(self):
        # (D) message not in the guarded set -> None.
        self.assertIsNone(bool_shortcircuit_candidate(
            _BC(_bc_return_and()), _BC([_Inst("LOAD_FAST", "a", argval="a")]),
            "def f(a, b):\n    return a\n", _ctx("Different names")))

    def test_missing_context_defers(self):
        self.assertIsNone(bool_shortcircuit_candidate(
            _BC(_bc_return_and()), _BC([_Inst("LOAD_FAST", "a", argval="a")]),
            "def f(a, b):\n    return a\n", None))

    def test_call_operand_defers(self):
        # (D) first operand is a call -> unrecoverable -> defer.
        der = _BC([_Inst("LOAD_GLOBAL", "g", argval="g"), _Inst("CALL", argval=0),
                   _Inst("RETURN_VALUE", offset=4)])
        self.assertIsNone(bool_shortcircuit_candidate(
            _BC(_bc_call_operand()), der, "def f(a, b):\n    return g()\n", _ctx()))

    def test_two_boolean_positions_defer(self):
        # (D) more than one short-circuit position -> ambiguous -> defer.
        der = _BC([_Inst("LOAD_FAST", "a", argval="a"), _Inst("STORE_FAST", "x", argval="x"),
                   _Inst("LOAD_FAST", "c", argval="c"), _Inst("RETURN_VALUE", offset=6)])
        self.assertIsNone(bool_shortcircuit_candidate(
            _BC(_bc_two_booleans()), der,
            "def f(a, b, c, d):\n    x = a\n    return c\n", _ctx()))

    def test_mixed_and_or_defers(self):
        der = _BC([_Inst("LOAD_FAST", "a", argval="a"), _Inst("RETURN_VALUE", offset=2)])
        self.assertIsNone(bool_shortcircuit_candidate(
            _BC(_bc_mixed_and_or()), der, "def f(a, b, c):\n    return a\n", _ctx()))

    def test_unparseable_fragment_defers(self):
        self.assertIsNone(bool_shortcircuit_candidate(
            _BC(_bc_return_and()), _BC([_Inst("LOAD_FAST", "a", argval="a")]),
            "def (:\n", _ctx()))

    def test_no_boolean_in_gt_defers(self):
        # GT has no short-circuit boolean at all -> None.
        gt = _BC([_Inst("LOAD_FAST", "a", argval="a"), _Inst("RETURN_VALUE", offset=2)])
        self.assertIsNone(bool_shortcircuit_candidate(
            gt, gt, "def f(a):\n    return a\n", _ctx()))

    def test_noop_when_already_correct_defers(self):
        # GT boolean == derived rendering -> span replace is a no-op -> None.
        self.assertIsNone(bool_shortcircuit_candidate(
            _BC(_bc_return_and()), _BC(_bc_return_and()),
            "def f(a, b):\n    return a and b\n", _ctx()))

    def test_never_raises_on_garbage(self):
        gt = _BC([_Inst("WEIRD_UNKNOWN_OP", arg=3), _Inst("MAP_ADD", arg=5)])
        der = _BC([None])  # attribute access on None inside -> swallowed by try/except
        self.assertIsNone(bool_shortcircuit_candidate(gt, der, "x = 1\n", _ctx()))


# ---------------------------------------------------------------------------
# Positive splice tests through the public entry point (mock bytecode)
# ---------------------------------------------------------------------------
class PositiveMockTest(unittest.TestCase):
    def test_return_recovers_dropped_and(self):
        der = _BC([_Inst("LOAD_FAST", "a", argval="a"), _Inst("RETURN_VALUE", offset=2)])
        out = bool_shortcircuit_candidate(_BC(_bc_return_and()), der,
                                          "def f(a, b):\n    return a\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b):\n    return a and b\n")
        self.assertEqual(out["kind"], "bool_shortcircuit")
        self.assertEqual(out["opname"], "POP_JUMP_IF_FALSE")
        self.assertEqual(out["confidence"], "unique")
        compile(out["text"], "<m>", "exec")

    def test_assign_recovers_dropped_or(self):
        der = _BC([_Inst("LOAD_FAST", "p", argval="p"), _Inst("STORE_FAST", "x", argval="x")])
        out = bool_shortcircuit_candidate(_BC(_bc_assign_or()), der,
                                          "def f(p, q):\n    x = p\n    return x\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(p, q):\n    x = p or q\n    return x\n")
        self.assertEqual(out["opname"], "POP_JUMP_IF_TRUE")

    def test_preserves_indentation(self):
        der = _BC([_Inst("LOAD_FAST", "a", argval="a"), _Inst("RETURN_VALUE", offset=2)])
        frag = "def f(a, b):\n        return a\n"
        out = bool_shortcircuit_candidate(_BC(_bc_return_and()), der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b):\n        return a and b\n")


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

        out = bool_shortcircuit_candidate(
            tr.bc_a, tr.bc_b, der_src,
            {"pylingual_failed_result": {"message": tr.message}},
        )
        testcase.assertIsNotNone(out, "operator should fire on this case")
        testcase.assertIn(expect_operator_substr, out["operator"])

        fixed_py = os.path.join(d, "fixed.py")
        Path(fixed_py).write_text(out["text"], encoding="utf-8")
        fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
        testcase.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class EndToEndTest(unittest.TestCase):
    def test_A_return_dropped_and(self):
        # (A) GT `return a and b`; derived dropped `and b`.
        _run_end_to_end(self, "def f(a, b):\n    return a and b\n",
                        "def f(a, b):\n    return a\n", "a and b")

    def test_B_assign_dropped_or(self):
        # (B) GT `x = p or q`; derived dropped `or q`.
        _run_end_to_end(self, "def f(p, q):\n    x = p or q\n    return x\n",
                        "def f(p, q):\n    x = p\n    return x\n", "p or q")

    def test_C_return_and_chain(self):
        # (C) GT `return a and b and c` chain; derived kept only `a`.
        _run_end_to_end(self, "def f(a, b, c):\n    return a and b and c\n",
                        "def f(a, b, c):\n    return a\n", "a and b and c")

    def test_wrong_operator_and_vs_or(self):
        # Bonus: derived used the WRONG operator (`or` for `and`) -> recover from GT.
        _run_end_to_end(self, "def f(a, b):\n    return a and b\n",
                        "def f(a, b):\n    return a or b\n", "a and b")

    def test_attribute_and_compare_operands(self):
        # Bonus: operands are an attribute chain and a comparison (still simple loads).
        _run_end_to_end(self, "def f(self, n):\n    return self.ready and n == 3\n",
                        "def f(self, n):\n    return self.ready\n", "self.ready and n == 3")


if __name__ == "__main__":
    unittest.main()
