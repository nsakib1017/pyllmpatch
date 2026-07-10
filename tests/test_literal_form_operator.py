"""Unit tests for the literal-form (quoting) operator
(utils/semantic_operators/literal_form.py).

The operator fixes a SINGLE leaf whose OPCODE differs between GT and derived bytecode: a
string constant the decompiler rendered as a bare name (quote) or vice-versa (unquote).
Mock ``_Inst``/``_BC`` supply instruction streams for the adversarial/positive unit tests;
the two end-to-end tests drive the real pylingual ``compare_pyc`` and assert the recompiled
output reaches combined_distance 0 against GT. Mirrors tests/test_container_operator.py."""
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

from utils.semantic_operators.literal_form import (
    _find_quoting_divergences,
    _name_of,
    _real_instructions,
    literal_form_candidate,
)


class _Inst:
    def __init__(self, opname, arg=None, argval=None, argrepr="", offset=0,
                 is_jump_target=False, is_jump=False):
        self.opname, self.arg, self.argval, self.argrepr, self.offset = (
            opname, arg, argval, argrepr, offset)
        self.is_jump_target, self.is_jump = is_jump_target, is_jump


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


def _ctx(message="Different bytecode", failed_offset=2):
    return {"pylingual_failed_result": {"message": message}, "failed_offset": failed_offset}


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import (
        compare_code_object_distances,
        summarize_results,
    )
    return summarize_results(compare_code_object_distances(gt_pyc, derived_pyc)).get(
        "combined_distance"
    )


# Canonical GT/derived streams for a single ``'TIMESTAMP'`` leaf inside a runtime-built
# tuple (a non-constant element keeps the string a separate LOAD_CONST, so the divergence
# is a clean 1-for-1 opcode swap rather than a folded whole-tuple constant).
def _quote_streams():
    """GT const-str vs derived bare-name (the decompiler dropped the quotes)."""
    gt = _BC([
        _Inst("LOAD_FAST", argval="x"), _Inst("LOAD_CONST", argval="TIMESTAMP"),
        _Inst("BUILD_TUPLE", 2), _Inst("RETURN_VALUE"),
    ])
    der = _BC([
        _Inst("LOAD_FAST", argval="x"), _Inst("LOAD_GLOBAL", argval="TIMESTAMP"),
        _Inst("BUILD_TUPLE", 2), _Inst("RETURN_VALUE"),
    ])
    return gt, der


def _unquote_streams():
    """GT bare-name vs derived const-str (the decompiler added the quotes)."""
    gt = _BC([
        _Inst("LOAD_FAST", argval="x"), _Inst("LOAD_GLOBAL", argval="TIMESTAMP"),
        _Inst("BUILD_TUPLE", 2), _Inst("RETURN_VALUE"),
    ])
    der = _BC([
        _Inst("LOAD_FAST", argval="x"), _Inst("LOAD_CONST", argval="TIMESTAMP"),
        _Inst("BUILD_TUPLE", 2), _Inst("RETURN_VALUE"),
    ])
    return gt, der


# ---------------------------------------------------------------------------
# Helper-level unit tests
# ---------------------------------------------------------------------------
class HelperTest(unittest.TestCase):
    def test_real_instructions_strips_cache(self):
        bc = _BC([_Inst("LOAD_GLOBAL", argval="x"), _Inst("CACHE"),
                  _Inst("CACHE"), _Inst("BUILD_TUPLE", 1)])
        self.assertEqual([i.opname for i in _real_instructions(bc)],
                         ["LOAD_GLOBAL", "BUILD_TUPLE"])

    def test_name_of_from_argval(self):
        self.assertEqual(_name_of(_Inst("LOAD_NAME", argval="TIMESTAMP")), "TIMESTAMP")

    def test_name_of_strips_null_prefix(self):
        # 3.11 LOAD_GLOBAL argrepr fallback: "NULL + name".
        self.assertEqual(_name_of(_Inst("LOAD_GLOBAL", argval=None, argrepr="NULL + TIMESTAMP")),
                         "TIMESTAMP")

    def test_name_of_rejects_non_identifier(self):
        self.assertIsNone(_name_of(_Inst("LOAD_CONST", argval=("a", "b"))))

    def test_finds_single_quote_divergence(self):
        gt, der = _quote_streams()
        divs = _find_quoting_divergences(list(gt), list(der))
        self.assertEqual(divs, [{"direction": "quote", "identifier": "TIMESTAMP",
                                 "const_value": "TIMESTAMP"}])

    def test_finds_single_unquote_divergence(self):
        gt, der = _unquote_streams()
        divs = _find_quoting_divergences(list(gt), list(der))
        self.assertEqual(divs, [{"direction": "unquote", "identifier": "TIMESTAMP",
                                 "const_value": "TIMESTAMP"}])

    def test_const_str_not_equal_to_name_is_not_quoting(self):
        # GT LOAD_CONST 'HELLO' vs derived LOAD_NAME WORLD: opcodes differ but the const
        # spelling != the identifier, so it is NOT a quoting bug (a real name/const swap).
        gt = [_Inst("LOAD_CONST", argval="HELLO"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_NAME", argval="WORLD"), _Inst("RETURN_VALUE")]
        self.assertEqual(_find_quoting_divergences(gt, der), [])

    def test_same_opcode_value_swap_is_not_quoting(self):
        # both LOAD_CONST -> a SequenceMatcher "equal" opname run, never a "replace"; leaf's
        # swap_literal owns this, so literal_form must not see a divergence.
        gt = [_Inst("LOAD_CONST", argval="A"), _Inst("RETURN_VALUE")]
        der = [_Inst("LOAD_CONST", argval="B"), _Inst("RETURN_VALUE")]
        self.assertEqual(_find_quoting_divergences(gt, der), [])


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests (mock bytecode)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_same_opcode_const_value_diff_defers(self):
        # (C) both sides LOAD_CONST with different values -> leaf's job, we defer.
        gt = _BC([_Inst("LOAD_CONST", argval="A"), _Inst("STORE_NAME", argval="X"),
                  _Inst("RETURN_CONST", argval=None)])
        der = _BC([_Inst("LOAD_CONST", argval="B"), _Inst("STORE_NAME", argval="X"),
                   _Inst("RETURN_CONST", argval=None)])
        self.assertIsNone(literal_form_candidate(gt, der, "X = 'B'\n", _ctx()))

    def test_ambiguous_two_name_nodes_defers(self):
        # (D) exactly one quoting divergence in bytecode, but two matching Name nodes in the
        # source -> cannot localize uniquely -> defer.
        gt = _BC([_Inst("LOAD_CONST", argval="TIMESTAMP"), _Inst("STORE_NAME", argval="X"),
                  _Inst("RETURN_CONST", argval=None)])
        der = _BC([_Inst("LOAD_NAME", argval="TIMESTAMP"), _Inst("STORE_NAME", argval="X"),
                   _Inst("RETURN_CONST", argval=None)])
        self.assertIsNone(literal_form_candidate(gt, der, "X = TIMESTAMP + TIMESTAMP\n", _ctx()))

    def test_ambiguous_two_string_constants_defers(self):
        # unquote mirror of (D): two 'TIMESTAMP' string constants -> not unique -> defer.
        gt = _BC([_Inst("LOAD_NAME", argval="TIMESTAMP"), _Inst("STORE_NAME", argval="X"),
                  _Inst("RETURN_CONST", argval=None)])
        der = _BC([_Inst("LOAD_CONST", argval="TIMESTAMP"), _Inst("STORE_NAME", argval="X"),
                   _Inst("RETURN_CONST", argval=None)])
        self.assertIsNone(literal_form_candidate(gt, der, "X = ('TIMESTAMP', 'TIMESTAMP')\n", _ctx()))

    def test_two_quoting_divergences_defer(self):
        # >1 quoting divergence -> ambiguous which to fix -> defer.
        gt = _BC([_Inst("LOAD_CONST", argval="A"), _Inst("LOAD_CONST", argval="B"),
                  _Inst("BUILD_TUPLE", 2), _Inst("RETURN_VALUE")])
        der = _BC([_Inst("LOAD_NAME", argval="A"), _Inst("LOAD_NAME", argval="B"),
                   _Inst("BUILD_TUPLE", 2), _Inst("RETURN_VALUE")])
        self.assertIsNone(literal_form_candidate(gt, der, "X = (A, B)\n", _ctx()))

    def test_wrong_regime_defers(self):
        # (E) not the "Different bytecode" regime, and a missing context.
        gt, der = _quote_streams()
        self.assertIsNone(literal_form_candidate(gt, der, "X = TIMESTAMP\n",
                                                 _ctx("Different control flow")))
        self.assertIsNone(literal_form_candidate(gt, der, "X = TIMESTAMP\n", None))

    def test_unparseable_fragment_defers(self):
        # (E) syntactically broken fragment must return None, not raise.
        gt, der = _quote_streams()
        self.assertIsNone(literal_form_candidate(gt, der, "def (:\n", _ctx()))

    def test_never_raises_on_garbage(self):
        # (E) malformed / underflowing instruction stream must not raise.
        gt = _BC([_Inst("WEIRD_UNKNOWN_OP", 3), _Inst("MAP_ADD", 5)])
        der = _BC([None])  # attribute access on None inside -> swallowed by try/except
        self.assertIsNone(literal_form_candidate(gt, der, "X = 1\n", _ctx()))


class PositiveMockTest(unittest.TestCase):
    def test_quote_fires_on_bare_name(self):
        gt, der = _quote_streams()
        out = literal_form_candidate(gt, der, "def f(x):\n    return (x, TIMESTAMP)\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n    return (x, 'TIMESTAMP')\n")
        self.assertEqual(out["kind"], "literal_form")
        self.assertEqual(out["opname"], "LOAD_CONST")
        self.assertEqual(out["confidence"], "unique")
        compile(out["text"], "<m>", "exec")

    def test_unquote_fires_on_overquoted_string(self):
        gt, der = _unquote_streams()
        out = literal_form_candidate(gt, der, "def f(x):\n    return (x, 'TIMESTAMP')\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n    return (x, TIMESTAMP)\n")
        self.assertEqual(out["kind"], "literal_form")
        self.assertEqual(out["opname"], "LOAD_NAME")
        compile(out["text"], "<m>", "exec")

    def test_preserves_indentation(self):
        gt, der = _quote_streams()
        frag = "def f(x):\n        return (x, TIMESTAMP)\n"
        out = literal_form_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n        return (x, 'TIMESTAMP')\n")


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
        testcase.assertEqual(tr.message, "Different bytecode")

        out = literal_form_candidate(
            tr.bc_a, tr.bc_b, der_src,
            {"pylingual_failed_result": {"message": tr.message}, "failed_offset": tr.failed_offset},
        )
        testcase.assertIsNotNone(out)
        testcase.assertIn(expect_operator_substr, out["operator"])

        fixed_py = os.path.join(d, "fixed.py")
        Path(fixed_py).write_text(out["text"], encoding="utf-8")
        fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
        testcase.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticQuoteTest(unittest.TestCase):
    """(A) GT loads the string constant ``'TIMESTAMP'``; the derived source dropped the
    quotes into a bare name. The operator quotes it back and the recompiled output reaches
    combined_distance 0 against GT."""

    GT_SRC = "def f(x):\n    return (x, 'TIMESTAMP')\n"
    DER_SRC = "def f(x):\n    return (x, TIMESTAMP)\n"

    def test_recompiles_to_distance_zero(self):
        _run_end_to_end(self, self.GT_SRC, self.DER_SRC, "quote name TIMESTAMP")


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticUnquoteTest(unittest.TestCase):
    """(B) The reverse: GT loads the name ``TIMESTAMP`` while the derived source over-quoted
    it into the string ``'TIMESTAMP'``. The operator unquotes it back to distance 0."""

    GT_SRC = "def f(x):\n    return (x, TIMESTAMP)\n"
    DER_SRC = "def f(x):\n    return (x, 'TIMESTAMP')\n"

    def test_recompiles_to_distance_zero(self):
        _run_end_to_end(self, self.GT_SRC, self.DER_SRC, "unquote")


if __name__ == "__main__":
    unittest.main()
