"""Unit + end-to-end tests for the deterministic comprehension operator
(utils/semantic_operators/comprehension_op.py).

The operator reconstructs a SIMPLE list/set/dict comprehension or generator
expression from ground-truth bytecode when the decompiler rendered it wrong, and
splices it over the unique matching comprehension display in the derived source.

Two bytecode regimes are covered, both exercised on this CPython 3.12 host:

  * INLINED (PEP 709): list/set/dict comprehensions compile *inline* into the
    enclosing code object (LOAD_FAST_AND_CLEAR + an inline FOR_ITER loop). The
    whole comprehension (element, target, iterable, optional filter) is recovered
    from the enclosing object's bytecode. Tests A, B, D(list) use this path.
  * NESTED code object: generator expressions ALWAYS compile to a separate
    ``<genexpr>`` code object (even on 3.12); pre-3.12 list/set/dict comprehensions
    do too. When such a body diverges, pylingual reports the failure on the nested
    record and hands the body directly as ``bc_a`` (its eager outermost iterable
    lives in the — Equal — enclosing frame, so the derived source's iterable is
    reused verbatim). Test C uses this path.

Every emitted fragment is asserted to recompile, and the happy-path tests assert
the recompiled output reaches combined_distance 0 (or a strict drop) against GT
through the real pylingual compare_pyc + oracle distance.
"""
import ast
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# <repo>/pylingual (importable package root) and <repo> both on sys.path, mirroring
# the repo's other tests, so `import pylingual.*` and `import utils.*` resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.comprehension_op import (  # noqa: E402
    _recover_stack,
    comprehension_op_candidate,
)


# ---------------------------------------------------------------------------
# Lightweight instruction / bytecode doubles (shape matches pylingual's editable
# bytecode: .opname, .arg, .argval, .argrepr, .offset, iterable).
# ---------------------------------------------------------------------------
class _Inst:
    def __init__(self, opname, arg=None, argval=None, argrepr="", offset=0):
        self.opname = opname
        self.arg = arg
        self.argval = argval
        self.argrepr = argrepr
        self.offset = offset


class _BC:
    def __init__(self, insts):
        self._insts = list(insts)

    def __iter__(self):
        return iter(self._insts)


def _ctx(message="Different bytecode"):
    return {"pylingual_failed_result": {"message": message}}


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import (
        compare_code_object_distances,
        summarize_results,
    )
    return summarize_results(compare_code_object_distances(gt_pyc, derived_pyc)).get(
        "combined_distance"
    )


def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


def _failing(gt_src, der_src, d):
    """Compile gt/der sources, return (gt_pyc, der_pyc, [failing traces])."""
    from pylingual.equivalence_check import compare_pyc

    gt_py = os.path.join(d, "gt.py")
    der_py = os.path.join(d, "der.py")
    Path(gt_py).write_text(gt_src, encoding="utf-8")
    Path(der_py).write_text(der_src, encoding="utf-8")
    gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
    der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
    fails = [tr for tr in compare_pyc(gt_pyc, der_pyc) if not tr.success]
    return gt_pyc, der_pyc, fails


# ===========================================================================
# Helper-level unit tests: the expression stack machine.
# ===========================================================================
class RecoverStackTest(unittest.TestCase):
    def test_name(self):
        st = _recover_stack([_Inst("LOAD_FAST", 0, "x", "x")])
        self.assertEqual([e.text for e in st], ["x"])

    def test_binop_is_parenthesised_in_eval_order(self):
        # (a + b) * c  ->  built as ((a + b)) * (c): parens force the parse tree to
        # match the bytecode-implied tree (parens emit no bytecode).
        st = _recover_stack([
            _Inst("LOAD_FAST", 0, "a", "a"), _Inst("LOAD_FAST", 1, "b", "b"),
            _Inst("BINARY_OP", 0, 0, "+"),
            _Inst("LOAD_FAST", 2, "c", "c"),
            _Inst("BINARY_OP", 5, 5, "*"),
        ])
        self.assertEqual([e.text for e in st], ["(a + b) * c"])

    def test_attribute_chain(self):
        st = _recover_stack([
            _Inst("LOAD_FAST", 0, "v", "v"),
            _Inst("LOAD_ATTR", 2, "name", "name"),
        ])
        self.assertEqual([e.text for e in st], ["v.name"])

    def test_simple_call(self):
        st = _recover_stack([
            _Inst("PUSH_NULL"),
            _Inst("LOAD_NAME", 0, "range", "range"),
            _Inst("LOAD_NAME", 1, "n", "n"),
            _Inst("CALL", 1, 1, ""),
        ])
        self.assertEqual([e.text for e in st], ["range(n)"])

    def test_dict_element_leaves_two(self):
        st = _recover_stack([
            _Inst("LOAD_FAST", 0, "k", "k"), _Inst("LOAD_FAST", 1, "v", "v"),
        ])
        self.assertEqual([e.text for e in st], ["k", "v"])

    def test_call_chain_over_one_call_defers(self):
        # a.b().c() -> two CALLs -> not a "simple call" -> None.
        st = _recover_stack([
            _Inst("LOAD_FAST", 0, "a", "a"),
            _Inst("LOAD_ATTR", 3, "b", "NULL|self + b"), _Inst("CALL", 0, 0, ""),
            _Inst("LOAD_ATTR", 5, "c", "NULL|self + c"), _Inst("CALL", 0, 0, ""),
        ])
        self.assertIsNone(st)

    def test_unknown_opcode_defers(self):
        self.assertIsNone(_recover_stack([_Inst("WEIRD_OP", 1, 1, "")]))


# ===========================================================================
# A/B: inlined list & dict comprehensions -> combined_distance 0.
# ===========================================================================
@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 12), "inlined-comprehension path needs 3.12+")
class InlinedComprehensionTest(unittest.TestCase):
    def _fix_to_zero(self, gt_src, der_src, needle):
        with tempfile.TemporaryDirectory() as d:
            gt_pyc, der_pyc, fails = _failing(gt_src, der_src, d)
            self.assertTrue(fails, "expected a divergence")
            tr = next(t for t in fails if t.names() == "<module>")
            self.assertEqual(tr.message, "Different bytecode")
            out = comprehension_op_candidate(tr.bc_a, tr.bc_b, der_src, _ctx(tr.message))
            self.assertIsNotNone(out, "operator should reconstruct the comprehension")
            self.assertEqual(out["kind"], "comprehension")
            self.assertIn(needle, out["text"])
            compile(out["text"], "<m>", "exec")  # must parse+compile

            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            self.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)

    def test_A_listcomp_wrong_element(self):
        # GT [x*2 for x in items]; derived dropped the "*2".
        self._fix_to_zero(
            "r = [x*2 for x in items]\n", "r = [x for x in items]\n", "for x in items",
        )

    def test_A_listcomp_wrong_iterable(self):
        # GT iterates range(n); derived iterates the wrong name -> iterable recovered too.
        self._fix_to_zero(
            "r = [x for x in range(n)]\n", "r = [x for x in wrong]\n", "range(n)",
        )

    def test_B_dictcomp_wrong_value(self):
        # GT {k: v for k, v in pairs}; derived mangled the value (tuple target too).
        self._fix_to_zero(
            "r = {k: v for k, v in pairs}\n", "r = {k: k for k, v in pairs}\n", "for k, v in pairs",
        )

    def test_setcomp_wrong_element(self):
        self._fix_to_zero(
            "r = {x*2 for x in items}\n", "r = {x for x in items}\n", "for x in items",
        )

    def test_listcomp_with_filter(self):
        # optional single if-filter recovered (keep-if-true). The derived keeps a filter
        # (so the divergence stays "Different bytecode", not "Different control flow") but
        # mangles the element; the operator must reconstruct BOTH element and filter.
        self._fix_to_zero(
            "r = [x*2 for x in items if x > 0]\n", "r = [x for x in items if x > 0]\n", "if x > 0",
        )


# ===========================================================================
# C: generator expression (nested <genexpr> code object) -> strict drop.
# ===========================================================================
@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires 3.10+")
class GenexprComprehensionTest(unittest.TestCase):
    def test_C_mangled_genexpr_reaches_zero(self):
        gt_src = "r = sum(x*x for x in xs)\n"
        der_src = "r = sum(y for y in xs)\n"
        with tempfile.TemporaryDirectory() as d:
            gt_pyc, der_pyc, fails = _failing(gt_src, der_src, d)
            base = _combined_distance(gt_pyc, der_pyc)
            self.assertGreater(base, 0)
            tr = next(t for t in fails if t.names().endswith("<genexpr>"))
            self.assertEqual(tr.message, "Different bytecode")
            out = comprehension_op_candidate(tr.bc_a, tr.bc_b, der_src, _ctx(tr.message))
            self.assertIsNotNone(out)
            self.assertEqual(out["kind"], "comprehension")
            # rebuilt from GT body (element x*x, target x) reusing derived iterable xs.
            self.assertIn("for x in xs", out["text"])
            compile(out["text"], "<m>", "exec")

            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            fixed = _combined_distance(gt_pyc, fixed_pyc)
            self.assertLess(fixed, base)   # strict drop
            self.assertEqual(fixed, 0)     # in fact exact


# ===========================================================================
# D: adversarial DEFER / SAFETY — must return None, never raise.
# ===========================================================================
@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 12), "requires 3.12 host")
class DeferTest(unittest.TestCase):
    def _first_diff(self, gt_src, der_src, d):
        _, _, fails = _failing(gt_src, der_src, d)
        return next((t for t in fails if t.message == "Different bytecode"), None)

    def test_nested_comprehension_defers(self):
        with tempfile.TemporaryDirectory() as d:
            tr = self._first_diff(
                "r = [[y for y in row] for row in grid]\n",
                "r = [[y for y in row] for row in wrong]\n", d)
            self.assertIsNotNone(tr)
            self.assertIsNone(comprehension_op_candidate(
                tr.bc_a, tr.bc_b, "r = [[y for y in row] for row in wrong]\n", _ctx()))

    def test_multi_for_defers(self):
        with tempfile.TemporaryDirectory() as d:
            tr = self._first_diff(
                "r = [a for row in grid for a in row]\n",
                "r = [a for row in grid for a in bad]\n", d)
            self.assertIsNotNone(tr)
            self.assertIsNone(comprehension_op_candidate(
                tr.bc_a, tr.bc_b, "r = [a for row in grid for a in bad]\n", _ctx()))

    def test_call_chain_element_defers(self):
        with tempfile.TemporaryDirectory() as d:
            tr = self._first_diff(
                "r = [a.b().c() for a in xs]\n",
                "r = [a for a in xs]\n", d)
            self.assertIsNotNone(tr)
            self.assertIsNone(comprehension_op_candidate(
                tr.bc_a, tr.bc_b, "r = [a for a in xs]\n", _ctx()))

    def test_wrong_regime_returns_none(self):
        # a plain non-comprehension "Different bytecode" divergence.
        with tempfile.TemporaryDirectory() as d:
            tr = self._first_diff("r = a + b\n", "r = a - b\n", d)
            self.assertIsNotNone(tr)
            self.assertIsNone(comprehension_op_candidate(
                tr.bc_a, tr.bc_b, "r = a - b\n", _ctx()))

    def test_wrong_failure_message_defers(self):
        gt = _BC([_Inst("LOAD_NAME", 0, "items", "items"), _Inst("GET_ITER"),
                  _Inst("LOAD_FAST_AND_CLEAR", 0, "x", "x")])
        self.assertIsNone(comprehension_op_candidate(gt, _BC([]), "r = [x for x in items]\n",
                                                     _ctx("Different control flow")))
        self.assertIsNone(comprehension_op_candidate(gt, _BC([]), "r = [x for x in items]\n", None))

    def test_garbage_never_raises(self):
        gt = _BC([_Inst("WEIRD", 3, 3, ""), _Inst("LOAD_FAST_AND_CLEAR", 9, "z", "z"),
                  _Inst("MAP_ADD", 2, 2, "")])
        self.assertIsNone(comprehension_op_candidate(gt, _BC([]), "def (:\n", _ctx()))
        self.assertIsNone(comprehension_op_candidate(gt, _BC([]), "not python at all {{{", _ctx()))

    def test_no_anchor_in_derived_defers(self):
        # GT has a real inlined listcomp, but the derived source has NO list display to
        # splice into (fully dropped) -> defer (cannot anchor).
        with tempfile.TemporaryDirectory() as d:
            tr = self._first_diff("r = [x*2 for x in items]\n", "r = 0\n", d)
            if tr is not None and tr.names() == "<module>":
                self.assertIsNone(comprehension_op_candidate(
                    tr.bc_a, tr.bc_b, "r = 0\n", _ctx()))


if __name__ == "__main__":
    unittest.main()
