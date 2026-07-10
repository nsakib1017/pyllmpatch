"""Unit + end-to-end tests for the unpacking/splat operator
(utils/semantic_operators/unpack_op.py).

The operator restores a STAR-unpacking form the decompiler flattened away, reading the
shape from ground-truth bytecode: a call splat ``f(*args)`` / ``f(**kw)``
(``CALL_FUNCTION_EX``), a starred assignment ``a, *b = it`` (``UNPACK_EX``), or a display
unpack ``{**d, ...}`` / ``[*a, ...]`` / ``{*a, ...}`` (``DICT_UPDATE`` / ``LIST_EXTEND`` /
``SET_UPDATE``).

Mock ``_Inst``/``_BC`` supply instruction streams for the helper + adversarial/positive unit
tests; the end-to-end tests drive the real pylingual ``compare_pyc`` on host CPython 3.12 and
assert the recompiled module reaches (A/B) combined_distance 0 or (C) a strict distance drop
against GT. Mirrors tests/test_container_operator.py and tests/test_call_shape_operator.py."""
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

from utils.semantic_operators.unpack_op import (  # noqa: E402
    _callex_star,
    _reconstruct_spread_displays,
    unpack_op_candidate,
)


class _Inst:
    def __init__(self, opname, arg=None, argval=None, argrepr="", offset=0,
                 is_jump=False):
        self.opname, self.arg, self.argval, self.argrepr = opname, arg, argval, argrepr
        self.offset, self.is_jump = offset, is_jump


class _BC:
    def __init__(self, insts):
        self._insts = insts

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


# --- canonical mock streams (real CPython 3.12 shapes, CACHE-stripped) -----------------
def _call_star_streams():
    """GT ``f(*args)`` vs derived ``f(args)`` (plain CALL)."""
    gt = _BC([_Inst("LOAD_GLOBAL", argval="f", argrepr="NULL + f"),
              _Inst("LOAD_GLOBAL", argval="args"),
              _Inst("CALL_FUNCTION_EX", arg=0, argval=0), _Inst("RETURN_VALUE")])
    der = _BC([_Inst("LOAD_GLOBAL", argval="f", argrepr="NULL + f"),
               _Inst("LOAD_GLOBAL", argval="args"),
               _Inst("CALL", arg=1, argval=1), _Inst("RETURN_VALUE")])
    return gt, der


def _call_dstar_streams():
    """GT ``f(**kw)`` vs derived ``f(kw)`` (plain CALL)."""
    gt = _BC([_Inst("LOAD_GLOBAL", argval="f", argrepr="NULL + f"),
              _Inst("LOAD_CONST", argval=()), _Inst("BUILD_MAP", arg=0, argval=0),
              _Inst("LOAD_GLOBAL", argval="kw"), _Inst("DICT_MERGE", arg=1, argval=1),
              _Inst("CALL_FUNCTION_EX", arg=1, argval=1), _Inst("RETURN_VALUE")])
    der = _BC([_Inst("LOAD_GLOBAL", argval="f", argrepr="NULL + f"),
               _Inst("LOAD_GLOBAL", argval="kw"),
               _Inst("CALL", arg=1, argval=1), _Inst("RETURN_VALUE")])
    return gt, der


def _unpack_ex_streams(count):
    """GT ``UNPACK_EX count`` vs derived ``UNPACK_SEQUENCE 2`` for a 2-target assign."""
    gt = _BC([_Inst("LOAD_GLOBAL", argval="it"),
              _Inst("UNPACK_EX", arg=count, argval=count),
              _Inst("STORE_FAST", argval="a"), _Inst("STORE_FAST", argval="b"),
              _Inst("RETURN_CONST", argval=None)])
    der = _BC([_Inst("LOAD_GLOBAL", argval="it"),
               _Inst("UNPACK_SEQUENCE", arg=2, argval=2),
               _Inst("STORE_FAST", argval="a"), _Inst("STORE_FAST", argval="b"),
               _Inst("RETURN_CONST", argval=None)])
    return gt, der


def _dict_display_stream():
    """GT ``{**base, 'k': 1}``."""
    return _BC([_Inst("BUILD_MAP", arg=0, argval=0),
                _Inst("LOAD_GLOBAL", argval="base"), _Inst("DICT_UPDATE", arg=1, argval=1),
                _Inst("LOAD_CONST", argval="k"), _Inst("LOAD_CONST", argval=1),
                _Inst("BUILD_MAP", arg=1, argval=1), _Inst("DICT_UPDATE", arg=1, argval=1),
                _Inst("STORE_FAST", argval="d"), _Inst("RETURN_CONST", argval=None)])


def _list_display_stream():
    """GT ``[*a, b]``."""
    return _BC([_Inst("BUILD_LIST", arg=0, argval=0),
                _Inst("LOAD_GLOBAL", argval="a"), _Inst("LIST_EXTEND", arg=1, argval=1),
                _Inst("LOAD_GLOBAL", argval="b"), _Inst("LIST_APPEND", arg=1, argval=1),
                _Inst("RETURN_VALUE")])


# ---------------------------------------------------------------------------
# Helper-level unit tests
# ---------------------------------------------------------------------------
class HelperTest(unittest.TestCase):
    def test_callex_star_single_positional(self):
        gt, _ = _call_star_streams()
        self.assertEqual(_callex_star(list(gt)), "*")

    def test_callex_star_single_keyword(self):
        gt, _ = _call_dstar_streams()
        self.assertEqual(_callex_star(list(gt)), "**")

    def test_callex_star_defers_on_mixed_positional(self):
        # f(a, *b): the positional is built with BUILD_LIST/LIST_EXTEND + LIST_TO_TUPLE ->
        # the instruction before CALL_FUNCTION_EX is CALL_INTRINSIC_1, not a lone load.
        gt = _BC([_Inst("LOAD_GLOBAL", argval="f", argrepr="NULL + f"),
                  _Inst("LOAD_GLOBAL", argval="a"), _Inst("BUILD_LIST", arg=1, argval=1),
                  _Inst("LOAD_GLOBAL", argval="b"), _Inst("LIST_EXTEND", arg=1, argval=1),
                  _Inst("CALL_INTRINSIC_1", arg=6, argval=6),
                  _Inst("CALL_FUNCTION_EX", arg=0, argval=0), _Inst("RETURN_VALUE")])
        self.assertIsNone(_callex_star(list(gt)))

    def test_reconstruct_dict_display(self):
        self.assertEqual(_reconstruct_spread_displays(list(_dict_display_stream())),
                         [("dict", "{**base, 'k': 1}")])

    def test_reconstruct_list_display(self):
        self.assertEqual(_reconstruct_spread_displays(list(_list_display_stream())),
                         [("list", "[*a, b]")])

    def test_reconstruct_ignores_spread_free_display(self):
        # a plain all-constant list has no spread -> not ours (container.py owns it).
        insts = [_Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
                 _Inst("BUILD_LIST", arg=2, argval=2), _Inst("RETURN_VALUE")]
        self.assertEqual(_reconstruct_spread_displays(insts), [])

    def test_reconstruct_defers_nonconst_literal(self):
        # {**base, k: v} with a non-constant literal value (MAP_ADD of a name) -> not
        # reconstructable verbatim from GT -> dropped.
        insts = [_Inst("BUILD_MAP", arg=0, argval=0),
                 _Inst("LOAD_GLOBAL", argval="base"), _Inst("DICT_UPDATE", arg=1, argval=1),
                 _Inst("LOAD_GLOBAL", argval="k"), _Inst("LOAD_GLOBAL", argval="v"),
                 _Inst("MAP_ADD", arg=1, argval=1),
                 _Inst("STORE_FAST", argval="d"), _Inst("RETURN_CONST", argval=None)]
        self.assertEqual(_reconstruct_spread_displays(insts), [])


# ---------------------------------------------------------------------------
# Positive mock tests
# ---------------------------------------------------------------------------
class PositiveMockTest(unittest.TestCase):
    def test_call_star_fixed(self):
        gt, der = _call_star_streams()
        out = unpack_op_candidate(gt, der, "def h():\n    return f(args)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def h():\n    return f(*args)\n")
        self.assertEqual(out["kind"], "unpack")
        self.assertEqual(out["opname"], "CALL_FUNCTION_EX")
        self.assertEqual(out["confidence"], "unique")
        compile(out["text"], "<m>", "exec")

    def test_call_dstar_fixed(self):
        gt, der = _call_dstar_streams()
        out = unpack_op_candidate(gt, der, "def h():\n    return f(kw)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def h():\n    return f(**kw)\n")
        compile(out["text"], "<m>", "exec")

    def test_call_star_keeps_expression_verbatim(self):
        # only the splat FORM is added; a compound argument expression is preserved as-is.
        gt, der = _call_star_streams()
        out = unpack_op_candidate(gt, der, "def h():\n    return f(build() + extra)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def h():\n    return f(*build() + extra)\n")

    def test_unpack_star_middle(self):
        gt, der = _unpack_ex_streams(1)  # a, *b  (before=1, after=0)
        out = unpack_op_candidate(gt, der, "def h():\n    a, b = it\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def h():\n    a, *b = it\n")
        self.assertEqual(out["opname"], "UNPACK_EX")
        compile(out["text"], "<m>", "exec")

    def test_unpack_star_head(self):
        gt, der = _unpack_ex_streams(256)  # *a, b  (before=0, after=1)
        out = unpack_op_candidate(gt, der, "def h():\n    a, b = it\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def h():\n    *a, b = it\n")
        compile(out["text"], "<m>", "exec")

    def test_dict_display_reconstructed(self):
        gt = _dict_display_stream()
        out = unpack_op_candidate(gt, _BC([]), "def h():\n    d = {'k': 1}\n    return d\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def h():\n    d = {**base, 'k': 1}\n    return d\n")
        self.assertEqual(out["opname"], "DICT_UPDATE")
        compile(out["text"], "<m>", "exec")

    def test_list_display_reconstructed(self):
        gt = _list_display_stream()
        out = unpack_op_candidate(gt, _BC([]), "def h():\n    return [nope]\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def h():\n    return [*a, b]\n")
        compile(out["text"], "<m>", "exec")

    def test_preserves_indentation(self):
        gt, der = _call_star_streams()
        out = unpack_op_candidate(gt, der, "def h():\n        return f(args)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def h():\n        return f(*args)\n")


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests -- (D)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_derived_already_splatted_defers(self):
        # derived already has CALL_FUNCTION_EX -> nothing to attribute -> defer.
        gt, _ = _call_star_streams()
        self.assertIsNone(unpack_op_candidate(gt, gt, "def h():\n    return f(*args)\n", _ctx()))

    def test_derived_already_starred_assignment_defers(self):
        gt, _ = _unpack_ex_streams(1)
        self.assertIsNone(unpack_op_candidate(gt, gt, "def h():\n    a, *b = it\n", _ctx()))

    def test_two_matching_calls_defers(self):
        # >1 single-positional call -> cannot localize the splat -> defer.
        gt, der = _call_star_streams()
        self.assertIsNone(
            unpack_op_candidate(gt, der, "def h():\n    return f(args) + g(other)\n", _ctx()))

    def test_two_matching_targets_defers(self):
        gt, der = _unpack_ex_streams(1)
        self.assertIsNone(
            unpack_op_candidate(gt, der, "def h():\n    a, b = it\n    c, d = other\n", _ctx()))

    def test_wrong_arity_target_defers(self):
        # GT wants a 2-element target; the fragment's tuple has 3 elements -> no match.
        gt, der = _unpack_ex_streams(1)
        self.assertIsNone(
            unpack_op_candidate(gt, der, "def h():\n    a, b, c = it\n", _ctx()))

    def test_wrong_regime_and_missing_context_return_none(self):
        gt, der = _call_star_streams()
        self.assertIsNone(unpack_op_candidate(gt, der, "def h():\n    return f(args)\n",
                                              _ctx("Different control flow")))
        self.assertIsNone(unpack_op_candidate(gt, der, "def h():\n    return f(args)\n", None))

    def test_unparseable_fragment_defers(self):
        gt, der = _call_star_streams()
        self.assertIsNone(unpack_op_candidate(gt, der, "def (:\n", _ctx()))

    def test_no_marker_opcode_defers(self):
        # neither side has any unpack marker -> nothing for this operator.
        plain = _BC([_Inst("LOAD_GLOBAL", argval="f"), _Inst("LOAD_GLOBAL", argval="a"),
                     _Inst("CALL", arg=1, argval=1), _Inst("RETURN_VALUE")])
        self.assertIsNone(unpack_op_candidate(plain, plain, "def h():\n    return f(a)\n", _ctx()))

    def test_never_raises_on_garbage(self):
        gt = _BC([_Inst("WEIRD_OP", 3), _Inst("UNPACK_EX", argval="not-an-int"),
                  _Inst("DICT_UPDATE", arg=9, argval=9), _Inst("CALL_FUNCTION_EX", argval=None)])
        der = _BC([None])  # attribute access on None inside -> swallowed by try/except
        self.assertIsNone(unpack_op_candidate(gt, der, "def h():\n    a, b = it\n", _ctx()))

    def test_garbage_fragment_never_raises(self):
        gt = _dict_display_stream()
        self.assertIsNone(unpack_op_candidate(gt, _BC([]), "\x00\x01 not python {[(", _ctx()))


# ---------------------------------------------------------------------------
# End-to-end tests through the real pylingual compare_pyc + oracle distance
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


def _find_diverging_trace(gt_pyc, der_pyc, marker_opnames):
    """The failing (Different bytecode) trace whose GT stream carries a marker opcode."""
    from pylingual.equivalence_check import compare_pyc
    failures = [tr for tr in compare_pyc(gt_pyc, der_pyc)
                if not tr.success and tr.bc_a is not None and tr.bc_b is not None]
    return next((t for t in failures
                 if any(getattr(i, "opname", None) in marker_opnames for i in t.bc_a)), None)


def _run_end_to_end(testcase, gt_src, der_src, markers, expect_distance=None):
    with tempfile.TemporaryDirectory() as d:
        Path(d + "/gt.py").write_text(gt_src, encoding="utf-8")
        Path(d + "/der.py").write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(d + "/gt.py", cfile=d + "/gt.pyc", doraise=True))
        der_pyc = Path(py_compile.compile(d + "/der.py", cfile=d + "/der.pyc", doraise=True))

        base = _combined_distance(gt_pyc, der_pyc)
        testcase.assertGreater(base, 0, "expected the flattened derived to diverge from GT")

        tr = _find_diverging_trace(gt_pyc, der_pyc, markers)
        testcase.assertIsNotNone(tr, "expected a bytecode divergence carrying the marker opcode")
        testcase.assertEqual(tr.message, "Different bytecode")

        out = unpack_op_candidate(
            tr.bc_a, tr.bc_b, der_src, {"pylingual_failed_result": {"message": tr.message}})
        testcase.assertIsNotNone(out, "operator should produce a candidate")
        testcase.assertEqual(out["kind"], "unpack")

        Path(d + "/fixed.py").write_text(out["text"], encoding="utf-8")
        fixed_pyc = Path(py_compile.compile(d + "/fixed.py", cfile=d + "/fixed.pyc", doraise=True))
        fixed_dist = _combined_distance(gt_pyc, fixed_pyc)
        testcase.assertLess(fixed_dist, base, "fix must strictly lower the combined distance")
        if expect_distance is not None:
            testcase.assertEqual(fixed_dist, expect_distance)
        return fixed_dist


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 11), "CALL_FUNCTION_EX splat signal is 3.11+")
class EndToEndCallStarTest(unittest.TestCase):
    """(A) GT ``f(*args)`` rendered positionally as ``f(args)`` -> distance 0."""

    def test_star_recompiles_to_distance_zero(self):
        _run_end_to_end(self,
                        "def h():\n    return f(*args)\n",
                        "def h():\n    return f(args)\n",
                        {"CALL_FUNCTION_EX"}, expect_distance=0)

    def test_dstar_recompiles_to_distance_zero(self):
        _run_end_to_end(self,
                        "def h():\n    return f(**kw)\n",
                        "def h():\n    return f(kw)\n",
                        {"CALL_FUNCTION_EX"}, expect_distance=0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 11), "UNPACK_EX/CACHE shape is 3.11+")
class EndToEndUnpackExTest(unittest.TestCase):
    """(B) GT ``a, *b = items`` rendered as ``a, b = items`` -> distance 0 (UNPACK_EX)."""

    def test_star_middle_recompiles_to_distance_zero(self):
        _run_end_to_end(self,
                        "def h():\n    a, *b = items\n    return (a, b)\n",
                        "def h():\n    a, b = items\n    return (a, b)\n",
                        {"UNPACK_EX"}, expect_distance=0)

    def test_star_head_recompiles_to_distance_zero(self):
        _run_end_to_end(self,
                        "def h():\n    *a, b = items\n    return (a, b)\n",
                        "def h():\n    a, b = items\n    return (a, b)\n",
                        {"UNPACK_EX"}, expect_distance=0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 11), "DICT_UPDATE display shape is 3.11+")
class EndToEndDictDisplayTest(unittest.TestCase):
    """(C) GT ``d = {**base, 'k': 1}`` rendered without the merge (``d = {'k': 1}``) ->
    the reconstructed spread display strictly lowers the combined distance."""

    def test_dict_merge_reconstruction_strictly_drops_distance(self):
        _run_end_to_end(self,
                        "def h():\n    d = {**base, 'k': 1}\n    return d\n",
                        "def h():\n    d = {'k': 1}\n    return d\n",
                        {"DICT_UPDATE"})

    def test_list_spread_reconstruction_strictly_drops_distance(self):
        _run_end_to_end(self,
                        "def h():\n    return [*a, b]\n",
                        "def h():\n    return [a, b]\n",
                        {"LIST_EXTEND"})


if __name__ == "__main__":
    unittest.main()
