"""Unit + end-to-end tests for the call-shape operator
(utils/semantic_operators/call_shape.py).

The operator fixes a call the decompiler rendered POSITIONALLY where GT passed trailing
arguments as KEYWORDS. 3.11+ bytecode makes the keyword names explicit (``KW_NAMES`` const
tuple before ``CALL`` on 3.11/3.12; ``CALL_KW`` on 3.13), so the correct call shape is
deterministically recoverable from GT bytecode.

Mock ``_Inst``/``_BC`` supply instruction streams for the helper + adversarial/positive unit
tests; the two end-to-end tests drive the real pylingual ``compare_pyc`` and assert the
recompiled module reaches combined_distance 0 (strictly lower than the mismatched derived)
against GT. Mirrors tests/test_literal_form_operator.py."""
import ast
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# <repo>/pylingual (the importable pylingual package) and <repo> both onto sys.path, so
# `import pylingual.equivalence_check` and `utils` resolve regardless of launch cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.call_shape import (  # noqa: E402
    _dotted_name,
    _find_call_shape_divergences,
    call_shape_candidate,
)


class _Inst:
    def __init__(self, opname, arg=None, argval=None, argrepr="", offset=0):
        self.opname, self.arg, self.argval, self.argrepr, self.offset = (
            opname, arg, argval, argrepr, offset)


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


def _ctx(message="Different bytecode"):
    # failed_offset is intentionally omitted/None: the real pylingual trace reports None for
    # this divergence, so the operator must align streams rather than lean on an offset.
    return {"pylingual_failed_result": {"message": message}, "failed_offset": None}


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import (
        compare_code_object_distances,
        summarize_results,
    )
    return summarize_results(compare_code_object_distances(gt_pyc, derived_pyc)).get(
        "combined_distance"
    )


# Canonical GT (keyword) vs derived (positional) streams for `g(1, key=2)` (3.11/3.12 shape:
# a KW_NAMES const tuple immediately before CALL). The value loads are byte-identical between
# the streams; only the KW_NAMES differs -> a lone SequenceMatcher `delete`.
def _kw_streams():
    gt = _BC([
        _Inst("LOAD_GLOBAL", argval="g", argrepr="NULL + g"),
        _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
        _Inst("KW_NAMES", arg=3, argval=("key",)), _Inst("CALL", arg=2, argval=2),
        _Inst("RETURN_VALUE"),
    ])
    der = _BC([
        _Inst("LOAD_GLOBAL", argval="g", argrepr="NULL + g"),
        _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
        _Inst("CALL", arg=2, argval=2), _Inst("RETURN_VALUE"),
    ])
    return gt, der


def _multi_kw_streams():
    """`g(1, x=2, y=3)` (two trailing keywords) vs the all-positional `g(1, 2, 3)`."""
    gt = _BC([
        _Inst("LOAD_GLOBAL", argval="g", argrepr="NULL + g"),
        _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2), _Inst("LOAD_CONST", argval=3),
        _Inst("KW_NAMES", arg=4, argval=("x", "y")), _Inst("CALL", arg=3, argval=3),
        _Inst("RETURN_VALUE"),
    ])
    der = _BC([
        _Inst("LOAD_GLOBAL", argval="g", argrepr="NULL + g"),
        _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2), _Inst("LOAD_CONST", argval=3),
        _Inst("CALL", arg=3, argval=3), _Inst("RETURN_VALUE"),
    ])
    return gt, der


# ---------------------------------------------------------------------------
# Helper-level unit tests
# ---------------------------------------------------------------------------
class HelperTest(unittest.TestCase):
    def test_finds_single_kw_divergence(self):
        gt, der = _kw_streams()
        self.assertEqual(_find_call_shape_divergences(list(gt), list(der)),
                         [{"names": ("key",), "total": 2, "n_positional": 1}])

    def test_finds_multi_kw_divergence(self):
        gt, der = _multi_kw_streams()
        self.assertEqual(_find_call_shape_divergences(list(gt), list(der)),
                         [{"names": ("x", "y"), "total": 3, "n_positional": 1}])

    def test_same_shape_call_has_no_divergence(self):
        # both sides keyword the call the same way -> KW_NAMES aligns `equal` -> no divergence.
        gt, _ = _kw_streams()
        self.assertEqual(_find_call_shape_divergences(list(gt), list(gt)), [])

    def test_no_keyword_call_has_no_divergence(self):
        _, der = _kw_streams()
        self.assertEqual(_find_call_shape_divergences(list(der), list(der)), [])

    def test_dotted_name(self):
        self.assertEqual(_dotted_name(ast.parse("g(1)", mode="eval").body.func), "g")
        self.assertEqual(_dotted_name(ast.parse("a.b.c(1)", mode="eval").body.func), "a.b.c")
        self.assertIsNone(_dotted_name(ast.parse("d[0](1)", mode="eval").body.func))


# ---------------------------------------------------------------------------
# Positive mock tests
# ---------------------------------------------------------------------------
class PositiveMockTest(unittest.TestCase):
    def test_single_kwarg_fixed(self):
        gt, der = _kw_streams()
        out = call_shape_candidate(gt, der, "def f():\n    return g(1, 2)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f():\n    return g(1, key=2)\n")
        self.assertEqual(out["kind"], "call_shape")
        self.assertEqual(out["opname"], "KW_NAMES")
        self.assertEqual(out["confidence"], "unique")
        self.assertIn("key", out["operator"])
        compile(out["text"], "<m>", "exec")

    def test_multi_kwarg_fixed(self):
        gt, der = _multi_kw_streams()
        out = call_shape_candidate(gt, der, "def f():\n    return g(1, 2, 3)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f():\n    return g(1, x=2, y=3)\n")
        compile(out["text"], "<m>", "exec")

    def test_preserves_indentation(self):
        gt, der = _kw_streams()
        frag = "def f():\n        return g(1, 2)\n"
        out = call_shape_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f():\n        return g(1, key=2)\n")

    def test_keeps_value_expressions_verbatim(self):
        # only the pos/kw FORM changes; a non-constant value expression is preserved as-is.
        gt, der = _kw_streams()
        out = call_shape_candidate(gt, der, "def f(a, b):\n    return g(a, b + 1)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b):\n    return g(a, key=b + 1)\n")


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests (mock bytecode) -- (C)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_more_than_one_matching_call_defers(self):
        # (C) >1 positional call with the divergence's arg count -> cannot localize -> defer.
        gt, der = _kw_streams()
        self.assertIsNone(
            call_shape_candidate(gt, der, "def f():\n    return g(1, 2) + g(3, 4)\n", _ctx()))

    def test_same_shape_call_returns_none(self):
        # (C) derived already matches GT's keyword shape -> no divergence -> None.
        gt, _ = _kw_streams()
        self.assertIsNone(
            call_shape_candidate(gt, gt, "def f():\n    return g(1, key=2)\n", _ctx()))

    def test_no_keyword_regime_returns_none(self):
        # neither side keywords the call -> nothing for this operator -> None.
        _, der = _kw_streams()
        self.assertIsNone(
            call_shape_candidate(der, der, "def f():\n    return g(1, 2)\n", _ctx()))

    def test_arg_count_mismatch_defers(self):
        # divergence wants a 2-arg positional call; the fragment has none -> defer.
        gt, der = _kw_streams()
        self.assertIsNone(
            call_shape_candidate(gt, der, "def f():\n    return g()\n", _ctx()))

    def test_already_keyworded_call_defers(self):
        # the sole call is already (differently) keyworded -> not fully positional -> defer.
        gt, der = _kw_streams()
        self.assertIsNone(
            call_shape_candidate(gt, der, "def f():\n    return g(1, other=2)\n", _ctx()))

    def test_wrong_regime_and_missing_context_return_none(self):
        gt, der = _kw_streams()
        self.assertIsNone(
            call_shape_candidate(gt, der, "def f():\n    return g(1, 2)\n",
                                 _ctx("Different control flow")))
        self.assertIsNone(call_shape_candidate(gt, der, "def f():\n    return g(1, 2)\n", None))

    def test_unparseable_fragment_defers(self):
        gt, der = _kw_streams()
        self.assertIsNone(call_shape_candidate(gt, der, "def (:\n", _ctx()))

    def test_never_raises_on_garbage(self):
        gt = _BC([_Inst("WEIRD_OP", 3), _Inst("KW_NAMES", argval="not-a-tuple"), _Inst("MAP_ADD", 5)])
        der = _BC([None])  # attribute access on None inside -> swallowed by try/except
        self.assertIsNone(call_shape_candidate(gt, der, "def f():\n    return g(1, 2)\n", _ctx()))


# ---------------------------------------------------------------------------
# End-to-end tests through the real pylingual compare_pyc + oracle distance
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


# A `def g` fixture in the SAME module makes the calls real; only `f` diverges (GT keywords
# the trailing args, derived passes them positionally).
_FIXTURE = "def g(a, key=None, y=None):\n    return (a, key, y)\n"


def _run_end_to_end(testcase, gt_f, der_f, expect_operator_substr):
    from pylingual.equivalence_check import compare_pyc

    gt_src = _FIXTURE + "\n" + gt_f
    der_src = _FIXTURE + "\n" + der_f
    with tempfile.TemporaryDirectory() as d:
        Path(d + "/gt.py").write_text(gt_src, encoding="utf-8")
        Path(d + "/der.py").write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(d + "/gt.py", cfile=d + "/gt.pyc", doraise=True))
        der_pyc = Path(py_compile.compile(d + "/der.py", cfile=d + "/der.pyc", doraise=True))

        # baseline: derived really is mismatched.
        base = _combined_distance(gt_pyc, der_pyc)
        testcase.assertGreater(base, 0, "expected the positional derived to diverge from GT")

        # the failing trace for `f` (its GT stream carries the KW_NAMES / CALL_KW keyword names).
        failures = [tr for tr in compare_pyc(gt_pyc, der_pyc)
                    if not tr.success and tr.bc_a is not None and tr.bc_b is not None]
        testcase.assertTrue(failures, "expected a bytecode divergence")
        tr = next((t for t in failures
                   if any(getattr(i, "opname", None) in ("KW_NAMES", "CALL_KW") for i in t.bc_a)),
                  failures[0])
        testcase.assertEqual(tr.message, "Different bytecode")

        out = call_shape_candidate(
            tr.bc_a, tr.bc_b, der_f,
            {"pylingual_failed_result": {"message": tr.message}, "failed_offset": tr.failed_offset},
        )
        testcase.assertIsNotNone(out)
        testcase.assertIn(expect_operator_substr, out["operator"])

        # apply the fix to the whole module (only `f` changes) and re-score.
        fixed_src = _FIXTURE + "\n" + out["text"]
        Path(d + "/fixed.py").write_text(fixed_src, encoding="utf-8")
        fixed_pyc = Path(py_compile.compile(d + "/fixed.py", cfile=d + "/fixed.pyc", doraise=True))
        fixed_dist = _combined_distance(gt_pyc, fixed_pyc)
        testcase.assertLess(fixed_dist, base, "fix must strictly lower the combined distance")
        testcase.assertEqual(fixed_dist, 0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 11), "KW_NAMES/CALL_KW is CPython 3.11+")
class SyntheticSingleKwargTest(unittest.TestCase):
    """(A) GT `g(1, key=2)`; derived rendered it positionally as `g(1, 2)`. The operator
    restores the keyword form and the recompiled module reaches combined_distance 0."""

    def test_recompiles_to_distance_zero(self):
        _run_end_to_end(self, "def f():\n    return g(1, key=2)\n",
                        "def f():\n    return g(1, 2)\n", "positional->keyword g(key)")


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 11), "KW_NAMES/CALL_KW is CPython 3.11+")
class SyntheticMultiKwargTest(unittest.TestCase):
    """(B) GT `g(1, key=2, y=3)`; derived rendered all three positionally. Both trailing
    keyword args are restored to distance 0."""

    def test_recompiles_to_distance_zero(self):
        _run_end_to_end(self, "def f():\n    return g(1, key=2, y=3)\n",
                        "def f():\n    return g(1, 2, 3)\n", "positional->keyword g(key, y)")


if __name__ == "__main__":
    unittest.main()
