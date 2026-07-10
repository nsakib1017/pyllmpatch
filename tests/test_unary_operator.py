"""Unit + end-to-end tests for the unary-form operator
(utils/semantic_operators/unary_op.py).

The operator fixes a SINGLE leaf whose UNARY form the decompiler got wrong: a dropped
``-``/``not``/``~`` (or 3.13 ``TO_BOOL``), a spurious added one, or the wrong one. Mock
``_Inst``/``_BC`` supply instruction streams for the helper + adversarial (defer/never-raise)
tests; the end-to-end tests drive the real pylingual ``compare_pyc`` on host CPython 3.12 and
assert the recompiled output reaches combined_distance 0 against GT. Mirrors
tests/test_literal_form_operator.py."""
import ast
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# The importable pylingual package lives at <repo>/pylingual/pylingual, so <repo>/pylingual
# must be on sys.path (alongside the repo root) for `import pylingual.equivalence_check` and
# for `utils` to resolve regardless of the launch cwd. Mirrors the repo's tools/tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.unary_op import (
    _find_unary_divergence,
    unary_op_candidate,
)


class _Inst:
    def __init__(self, opname, argval=None, argrepr="", offset=0):
        self.opname, self.argval, self.argrepr, self.offset = opname, argval, argrepr, offset


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


def _ctx(message="Different bytecode", failed_offset=4):
    return {"pylingual_failed_result": {"message": message}, "failed_offset": failed_offset}


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import (
        compare_code_object_distances,
        summarize_results,
    )
    return summarize_results(compare_code_object_distances(gt_pyc, derived_pyc)).get(
        "combined_distance"
    )


# ---------------------------------------------------------------------------
# Helper-level unit tests: stream alignment -> divergence descriptor
# ---------------------------------------------------------------------------
class FindDivergenceTest(unittest.TestCase):
    def test_dropped_negative(self):
        gt = [_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")]
        self.assertEqual(_find_unary_divergence(gt, der),
                         {"mode": "add", "unary": "UNARY_NEGATIVE", "ident": "x"})

    def test_dropped_not(self):
        gt = [_Inst("LOAD_FAST", "x"), _Inst("UNARY_NOT"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")]
        self.assertEqual(_find_unary_divergence(gt, der),
                         {"mode": "add", "unary": "UNARY_NOT", "ident": "x"})

    def test_dropped_invert(self):
        gt = [_Inst("LOAD_FAST", "n"), _Inst("UNARY_INVERT"), _Inst("STORE_FAST", "y")]
        der = [_Inst("LOAD_FAST", "n"), _Inst("STORE_FAST", "y")]
        self.assertEqual(_find_unary_divergence(gt, der),
                         {"mode": "add", "unary": "UNARY_INVERT", "ident": "n"})

    def test_dropped_to_bool(self):
        gt = [_Inst("LOAD_FAST", "x"), _Inst("TO_BOOL"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")]
        self.assertEqual(_find_unary_divergence(gt, der),
                         {"mode": "to_bool_add", "unary": "TO_BOOL", "ident": "x"})

    def test_added_unary(self):
        gt = [_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")]
        self.assertEqual(_find_unary_divergence(gt, der),
                         {"mode": "remove", "der_unary": "UNARY_NEGATIVE"})

    def test_swapped_unary(self):
        gt = [_Inst("LOAD_FAST", "x"), _Inst("UNARY_INVERT"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")]
        self.assertEqual(_find_unary_divergence(gt, der),
                         {"mode": "swap", "unary": "UNARY_INVERT", "der_unary": "UNARY_NEGATIVE"})

    def test_complex_operand_defers(self):
        # `-(a + b)`: the UNARY trails a BINARY_OP, not a leaf load -> defer.
        gt = [_Inst("LOAD_FAST", "a"), _Inst("LOAD_FAST", "b"), _Inst("BINARY_OP", "+"),
              _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_FAST", "a"), _Inst("LOAD_FAST", "b"), _Inst("BINARY_OP", "+"),
               _Inst("RETURN_VALUE")]
        self.assertIsNone(_find_unary_divergence(gt, der))

    def test_two_divergences_defer(self):
        # Two separate dropped unaries -> two non-equal chunks -> compound bug -> defer.
        # (A same-opcode LOAD_CONST value swap is invisible in the opname stream and is
        # leaf's job, so it can never be the "second chunk" here.)
        gt = [_Inst("LOAD_FAST", "a"), _Inst("UNARY_NEGATIVE"), _Inst("STORE_FAST", "t"),
              _Inst("LOAD_FAST", "b"), _Inst("UNARY_INVERT"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_FAST", "a"), _Inst("STORE_FAST", "t"),
               _Inst("LOAD_FAST", "b"), _Inst("RETURN_VALUE")]
        self.assertIsNone(_find_unary_divergence(gt, der))

    def test_identical_streams(self):
        gt = [_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")]
        self.assertIsNone(_find_unary_divergence(gt, list(gt)))

    def test_non_leaf_operand_load_defers(self):
        # unary over an attribute read (`-a.b`): preceding op is LOAD_ATTR, not a name -> defer.
        gt = [_Inst("LOAD_FAST", "a"), _Inst("LOAD_ATTR", "b"), _Inst("UNARY_NEGATIVE"),
              _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_FAST", "a"), _Inst("LOAD_ATTR", "b"), _Inst("RETURN_VALUE")]
        self.assertIsNone(_find_unary_divergence(gt, der))


# ---------------------------------------------------------------------------
# Candidate-level positive tests (mock bytecode)
# ---------------------------------------------------------------------------
class PositiveMockTest(unittest.TestCase):
    def test_add_negative(self):
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")])
        out = unary_op_candidate(gt, der, "def f(x):\n    return x\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n    return (-x)\n")
        self.assertEqual(out["kind"], "unary")
        self.assertEqual(out["opname"], "UNARY_NEGATIVE")
        self.assertEqual(out["confidence"], "unique")
        compile(out["text"], "<m>", "exec")

    def test_add_not(self):
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("UNARY_NOT"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")])
        out = unary_op_candidate(gt, der, "def f(x):\n    return x\n", _ctx())
        self.assertEqual(out["text"], "def f(x):\n    return (not x)\n")
        compile(out["text"], "<m>", "exec")

    def test_add_invert_preserves_indent(self):
        gt = _BC([_Inst("LOAD_FAST", "n"), _Inst("UNARY_INVERT"), _Inst("STORE_FAST", "y")])
        der = _BC([_Inst("LOAD_FAST", "n"), _Inst("STORE_FAST", "y")])
        frag = "class C:\n    def m(self, n):\n        y = n\n        return y\n"
        out = unary_op_candidate(gt, der, frag, _ctx())
        self.assertEqual(
            out["text"],
            "class C:\n    def m(self, n):\n        y = (~n)\n        return y\n")

    def test_remove_spurious_unary(self):
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")])
        out = unary_op_candidate(gt, der, "def f(x):\n    return -x\n", _ctx())
        self.assertEqual(out["text"], "def f(x):\n    return x\n")
        self.assertEqual(out["opname"], "UNARY_NEGATIVE")

    def test_swap_unary(self):
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("UNARY_INVERT"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")])
        out = unary_op_candidate(gt, der, "def f(x):\n    return -x\n", _ctx())
        self.assertEqual(out["text"], "def f(x):\n    return (~x)\n")
        self.assertIn("->", out["operator"])

    def test_to_bool_add(self):
        # Host is 3.12 (no TO_BOOL), so this exercises only the source transform: a dropped
        # explicit-truthiness op becomes an explicit bool() wrap. Mock stream, gated in prod.
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("TO_BOOL"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")])
        out = unary_op_candidate(gt, der, "def f(x):\n    return x\n", _ctx())
        self.assertEqual(out["text"], "def f(x):\n    return bool(x)\n")
        self.assertEqual(out["opname"], "TO_BOOL")


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests (never raise; unique-or-defer)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_ambiguous_two_operands_defer(self):
        # (D) exactly one dropped-unary in bytecode, but two matching Load names in source
        # -> cannot localize the operand uniquely -> defer.
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("LOAD_FAST", "x"),
                  _Inst("BINARY_OP", "+"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("LOAD_FAST", "x"),
                   _Inst("BINARY_OP", "+"), _Inst("RETURN_VALUE")])
        self.assertIsNone(unary_op_candidate(gt, der, "def f(x):\n    return x + x\n", _ctx()))

    def test_wrong_regime_defers(self):
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")])
        self.assertIsNone(unary_op_candidate(gt, der, "def f(x):\n    return x\n",
                                             _ctx("Different control flow")))
        self.assertIsNone(unary_op_candidate(gt, der, "def f(x):\n    return x\n", None))

    def test_no_unary_divergence_defers(self):
        # identical streams -> nothing to do -> None (not our regime).
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")])
        self.assertIsNone(unary_op_candidate(gt, der, "def f(x):\n    return x\n", _ctx()))

    def test_unparseable_fragment_defers(self):
        gt = _BC([_Inst("LOAD_FAST", "x"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "x"), _Inst("RETURN_VALUE")])
        self.assertIsNone(unary_op_candidate(gt, der, "def (:\n", _ctx()))

    def test_operand_missing_from_source_defers(self):
        # bytecode says the operand is `q`, but the source has no such Load name -> defer.
        gt = _BC([_Inst("LOAD_FAST", "q"), _Inst("UNARY_NEGATIVE"), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_FAST", "q"), _Inst("RETURN_VALUE")])
        self.assertIsNone(unary_op_candidate(gt, der, "def f(x):\n    return x\n", _ctx()))

    def test_never_raises_on_garbage(self):
        # malformed / underflowing streams must return None, never raise.
        self.assertIsNone(unary_op_candidate(_BC([None]), _BC([None]), "x = 1\n", _ctx()))
        self.assertIsNone(unary_op_candidate(
            _BC([_Inst("WEIRD_OP"), _Inst("MAP_ADD")]), _BC([None]), "x = 1\n", _ctx()))
        self.assertIsNone(unary_op_candidate(_BC([]), _BC([]), "x = 1\n", _ctx()))


# ---------------------------------------------------------------------------
# End-to-end tests through the real pylingual compare_pyc + oracle distance
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


def _e2e_candidate(der_src, gt_pyc, der_pyc):
    """Feed every failing per-code-object trace to the operator; return the first candidate
    (module-level traces whose opnames match produce no unary divergence -> None)."""
    from pylingual.equivalence_check import compare_pyc

    for tr in compare_pyc(gt_pyc, der_pyc):
        if tr.success or tr.bc_a is None or tr.bc_b is None:
            continue
        cand = unary_op_candidate(
            tr.bc_a, tr.bc_b, der_src,
            {"pylingual_failed_result": {"message": tr.message},
             "failed_offset": getattr(tr, "failed_offset", None)},
        )
        if cand is not None:
            return cand
    return None


def _run_end_to_end(tc, gt_src, der_src, expect_substr):
    with tempfile.TemporaryDirectory() as d:
        gt_py, der_py = os.path.join(d, "gt.py"), os.path.join(d, "der.py")
        Path(gt_py).write_text(gt_src, encoding="utf-8")
        Path(der_py).write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
        der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))

        # Sanity: the derived source really does diverge from GT before repair.
        tc.assertNotEqual(_combined_distance(gt_pyc, der_pyc), 0)

        out = _e2e_candidate(der_src, gt_pyc, der_pyc)
        tc.assertIsNotNone(out, "operator produced no candidate")
        tc.assertEqual(out["kind"], "unary")
        tc.assertIn(expect_substr, out["operator"])

        fixed_py = os.path.join(d, "fixed.py")
        Path(fixed_py).write_text(out["text"], encoding="utf-8")
        fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"),
                                            doraise=True))
        tc.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)


def _run_expect_defer(tc, gt_src, der_src):
    with tempfile.TemporaryDirectory() as d:
        gt_py, der_py = os.path.join(d, "gt.py"), os.path.join(d, "der.py")
        Path(gt_py).write_text(gt_src, encoding="utf-8")
        Path(der_py).write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
        der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
        tc.assertIsNone(_e2e_candidate(der_src, gt_pyc, der_pyc))


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class EndToEndTest(unittest.TestCase):
    """(A)/(B)/(C): the decompiler dropped the unary; the operator adds it back and the
    recompiled output reaches combined_distance 0 against GT. Plus swap/remove variants."""

    def test_A_negate(self):
        # (A) GT `return -x` vs derived `return x` -> add `-` -> distance 0.
        _run_end_to_end(self, "def f(x):\n    return -x\n",
                        "def f(x):\n    return x\n", "add unary -")

    def test_B_not(self):
        # (B) GT `return not x` vs derived `return x` -> add `not` -> distance 0.
        _run_end_to_end(self, "def f(x):\n    return not x\n",
                        "def f(x):\n    return x\n", "add unary not")

    def test_C_invert(self):
        # (C) GT `y = ~n` vs derived `y = n` -> add `~` -> distance 0.
        _run_end_to_end(self, "def f(n):\n    y = ~n\n    return y\n",
                        "def f(n):\n    y = n\n    return y\n", "add unary ~")

    def test_swap_invert_for_negate(self):
        # GT `return ~x` vs derived `return -x` -> swap the unary -> distance 0.
        _run_end_to_end(self, "def f(x):\n    return ~x\n",
                        "def f(x):\n    return -x\n", "->")

    def test_remove_spurious_negate(self):
        # GT `return x` vs derived `return -x` -> drop the spurious unary -> distance 0.
        _run_end_to_end(self, "def f(x):\n    return x\n",
                        "def f(x):\n    return -x\n", "drop unary -")

    def test_precedence_grouped_negate(self):
        # GT `(-x) ** 2` vs derived `x ** 2`: the operand is the leaf `x` (UNARY trails
        # LOAD x), and parenthesizing on splice keeps `(-x) ** 2` (not `-x**2`) -> distance 0.
        _run_end_to_end(self, "def f(x):\n    return (-x) ** 2\n",
                        "def f(x):\n    return x ** 2\n", "add unary -")


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class EndToEndDeferTest(unittest.TestCase):
    """(D) adversarial through the real pipeline: ambiguity and wrong-regime must defer."""

    def test_ambiguous_two_operands_defers(self):
        # GT `-x + x` vs derived `x + x`: one dropped UNARY, but two `x` Load names in source
        # -> operand not unique -> defer (no candidate from any trace).
        _run_expect_defer(self, "def f(x):\n    return -x + x\n",
                          "def f(x):\n    return x + x\n")

    def test_complex_operand_defers(self):
        # GT `-(a + b)` vs derived `a + b`: UNARY over a BINARY_OP result, not a leaf -> defer.
        _run_expect_defer(self, "def f(a, b):\n    return -(a + b)\n",
                          "def f(a, b):\n    return a + b\n")

    def test_precedence_trap_defers(self):
        # GT `-x ** 2` == `-(x ** 2)`: the UNARY trails the BINARY_OP `**`, so the operand is
        # the power result (not a leaf) -> defer, never mis-wrap the base `x`.
        _run_expect_defer(self, "def f(x):\n    return -x ** 2\n",
                          "def f(x):\n    return x ** 2\n")

    def test_non_bytecode_regime_defers(self):
        # Identical sources -> no bytecode divergence -> no candidate.
        _run_expect_defer(self, "def f(x):\n    return -x\n",
                          "def f(x):\n    return -x\n")


if __name__ == "__main__":
    unittest.main()
