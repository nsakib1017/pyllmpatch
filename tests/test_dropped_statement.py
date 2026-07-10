"""Unit + end-to-end tests for the dropped-simple-statement operator
(utils/semantic_operators/dropped_statement.dropped_statement_candidate).

The operator deterministically reconstructs a single simple statement the decompiler
DROPPED ENTIRELY, reading its full source verbatim from GT bytecode: ``del``, ``import`` /
``from ... import``, ``assert`` (simple test) and ``global``. It requires EXACTLY ONE
dropped statement of ONE kind and defers on anything ambiguous; it NEVER raises.

Adversarial / helper cases supply GT/derived bytecode as explicit instruction lists. The
end-to-end cases (A del, B import x2, C assert, D global) drive the real pylingual
``compare_pyc`` + the oracle ``combined_distance`` and assert the recompiled output reaches
(or moves toward) distance 0. No LLM anywhere.
"""
import ast
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# <repo>/pylingual (for `import pylingual.*`) and <repo> (for `utils`) on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.dropped_statement import (
    _all_imports,
    _extract_asserts,
    _gt_dels,
    _global_flips,
    _reconstruct_import,
    dropped_statement_candidate,
)


class _Inst:
    def __init__(self, opname, arg=None, argval=None, argrepr="",
                 is_jump_target=False):
        self.opname, self.arg, self.argval = opname, arg, argval
        self.argrepr = argrepr
        self.is_jump_target = is_jump_target


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


def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


# ===========================================================================
# Helper-level unit tests (no pylingual)
# ===========================================================================
class ReconstructHelpersTest(unittest.TestCase):
    def test_del_simple_name(self):
        insts = [_Inst("DELETE_NAME", 0, "x", "x")]
        self.assertEqual(_gt_dels(insts), [("del x", 0)])

    def test_del_attr_chain(self):
        insts = [
            _Inst("LOAD_NAME", 0, "a", "a"),
            _Inst("LOAD_ATTR", 2, "b", "b"),
            _Inst("DELETE_ATTR", 2, "c", "c"),
        ]
        self.assertEqual(_gt_dels(insts), [("del a.b.c", 2)])

    def test_del_subscr_name_and_const_keys(self):
        name_key = [
            _Inst("LOAD_NAME", 0, "d", "d"),
            _Inst("LOAD_NAME", 1, "k", "k"),
            _Inst("DELETE_SUBSCR"),
        ]
        self.assertEqual(_gt_dels(name_key), [("del d[k]", 2)])
        const_key = [
            _Inst("LOAD_NAME", 0, "d", "d"),
            _Inst("LOAD_CONST", 0, "k", "'k'"),
            _Inst("DELETE_SUBSCR"),
        ]
        self.assertEqual(_gt_dels(const_key), [("del d['k']", 2)])

    def test_import_plain_and_dotted(self):
        # import os
        i1 = [_Inst("LOAD_CONST", 0, 0, "0"), _Inst("LOAD_CONST", 1, None, "None"),
              _Inst("IMPORT_NAME", 0, "os", "os"), _Inst("STORE_NAME", 0, "os", "os")]
        self.assertEqual(_all_imports(i1), ["import os"])
        # import a.b.c  -> binds top package `a`
        i2 = [_Inst("LOAD_CONST", 0, 0, "0"), _Inst("LOAD_CONST", 1, None, "None"),
              _Inst("IMPORT_NAME", 0, "a.b.c", "a.b.c"), _Inst("STORE_NAME", 1, "a", "a")]
        self.assertEqual(_all_imports(i2), ["import a.b.c"])
        # import os as o
        i3 = [_Inst("LOAD_CONST", 0, 0, "0"), _Inst("LOAD_CONST", 1, None, "None"),
              _Inst("IMPORT_NAME", 0, "os", "os"), _Inst("STORE_NAME", 1, "o", "o")]
        self.assertEqual(_all_imports(i3), ["import os as o"])
        # import a.b as x  (dotted-as: IMPORT_FROM then STORE)
        i4 = [_Inst("LOAD_CONST", 0, 0, "0"), _Inst("LOAD_CONST", 1, None, "None"),
              _Inst("IMPORT_NAME", 0, "a.b", "a.b"), _Inst("IMPORT_FROM", 1, "b", "b"),
              _Inst("STORE_NAME", 2, "x", "x"), _Inst("POP_TOP")]
        self.assertEqual(_all_imports(i4), ["import a.b as x"])

    def test_from_import_variants(self):
        # from m import n, p
        f1 = [_Inst("LOAD_CONST", 0, 0, "0"), _Inst("LOAD_CONST", 1, ("n", "p"), "('n', 'p')"),
              _Inst("IMPORT_NAME", 0, "m", "m"),
              _Inst("IMPORT_FROM", 1, "n", "n"), _Inst("STORE_NAME", 1, "n", "n"),
              _Inst("IMPORT_FROM", 2, "p", "p"), _Inst("STORE_NAME", 2, "p", "p"),
              _Inst("POP_TOP")]
        self.assertEqual(_all_imports(f1), ["from m import n, p"])
        # from a import b as c
        f2 = [_Inst("LOAD_CONST", 0, 0, "0"), _Inst("LOAD_CONST", 1, ("b",), "('b',)"),
              _Inst("IMPORT_NAME", 0, "a", "a"),
              _Inst("IMPORT_FROM", 1, "b", "b"), _Inst("STORE_NAME", 2, "c", "c"),
              _Inst("POP_TOP")]
        self.assertEqual(_all_imports(f2), ["from a import b as c"])
        # from ..pkg import y  (relative, level 2)
        f3 = [_Inst("LOAD_CONST", 0, 2, "2"), _Inst("LOAD_CONST", 1, ("y",), "('y',)"),
              _Inst("IMPORT_NAME", 0, "pkg", "pkg"),
              _Inst("IMPORT_FROM", 1, "y", "y"), _Inst("STORE_NAME", 1, "y", "y"),
              _Inst("POP_TOP")]
        self.assertEqual(_all_imports(f3), ["from ..pkg import y"])
        # from . import x  (level 1, empty module)
        f4 = [_Inst("LOAD_CONST", 0, 1, "1"), _Inst("LOAD_CONST", 1, ("x",), "('x',)"),
              _Inst("IMPORT_NAME", 0, "", ""),
              _Inst("IMPORT_FROM", 1, "x", "x"), _Inst("STORE_NAME", 1, "x", "x"),
              _Inst("POP_TOP")]
        self.assertEqual(_all_imports(f4), ["from . import x"])
        # from m import *
        f5 = [_Inst("LOAD_CONST", 0, 0, "0"), _Inst("LOAD_CONST", 1, ("*",), "('*',)"),
              _Inst("IMPORT_NAME", 0, "m", "m"), _Inst("CALL_INTRINSIC_1", 2), _Inst("POP_TOP")]
        self.assertEqual(_all_imports(f5), ["from m import *"])

    def test_assert_simple_tests(self):
        # assert x
        a1 = [_Inst("LOAD_NAME", 0, "x", "x"), _Inst("POP_JUMP_IF_TRUE", 2),
              _Inst("LOAD_ASSERTION_ERROR"), _Inst("RAISE_VARARGS", 1)]
        self.assertEqual(_extract_asserts(a1), [("assert x", 2)])
        # assert x == 1
        a2 = [_Inst("LOAD_NAME", 0, "x", "x"), _Inst("LOAD_CONST", 0, 1, "1"),
              _Inst("COMPARE_OP", 40, "==", "=="), _Inst("POP_JUMP_IF_TRUE", 2),
              _Inst("LOAD_ASSERTION_ERROR"), _Inst("RAISE_VARARGS", 1)]
        self.assertEqual(_extract_asserts(a2), [("assert x == 1", 4)])
        # assert x is None  (POP_JUMP_IF_NONE guard)
        a3 = [_Inst("LOAD_NAME", 0, "x", "x"), _Inst("POP_JUMP_IF_NONE", 2),
              _Inst("LOAD_ASSERTION_ERROR"), _Inst("RAISE_VARARGS", 1)]
        self.assertEqual(_extract_asserts(a3), [("assert x is None", 2)])
        # assert x, 'msg'  (constant message reconstructed)
        a4 = [_Inst("LOAD_NAME", 0, "x", "x"), _Inst("POP_JUMP_IF_TRUE", 7),
              _Inst("LOAD_ASSERTION_ERROR"), _Inst("LOAD_CONST", 0, "msg", "'msg'"),
              _Inst("CALL", 0), _Inst("RAISE_VARARGS", 1)]
        self.assertEqual(_extract_asserts(a4), [("assert x, 'msg'", 2)])

    def test_assert_complex_test_is_not_extracted(self):
        # assert f(x) -> the test is a CALL -> not a recoverable simple expr -> no assert.
        a = [_Inst("PUSH_NULL"), _Inst("LOAD_NAME", 0, "f", "f"),
             _Inst("LOAD_NAME", 1, "x", "x"), _Inst("CALL", 1),
             _Inst("POP_JUMP_IF_TRUE", 2), _Inst("LOAD_ASSERTION_ERROR"),
             _Inst("RAISE_VARARGS", 1)]
        self.assertEqual(_extract_asserts(a), [])

    def test_assert_complex_message_defers(self):
        # assert x, err()  -> message is not a lone constant -> defer (must not drop it).
        a = [_Inst("LOAD_NAME", 0, "x", "x"), _Inst("POP_JUMP_IF_TRUE", 9),
             _Inst("LOAD_ASSERTION_ERROR"), _Inst("PUSH_NULL"),
             _Inst("LOAD_NAME", 1, "err", "err"), _Inst("CALL", 0),
             _Inst("CALL", 0), _Inst("RAISE_VARARGS", 1)]
        self.assertEqual(_extract_asserts(a), [])

    def test_global_flip_detected(self):
        gt = [_Inst("LOAD_CONST", 1, 1, "1"), _Inst("STORE_GLOBAL", 0, "g", "g")]
        der = [_Inst("LOAD_CONST", 1, 1, "1"), _Inst("STORE_FAST", 0, "g", "g")]
        self.assertEqual(_global_flips(gt, der), {"g"})
        # no flip if derived already stores it as global
        self.assertEqual(_global_flips(gt, gt), set())

    def test_reconstruct_import_defers_on_missing_prologue(self):
        # IMPORT_NAME without the LOAD_CONST/LOAD_CONST prologue -> defer.
        self.assertIsNone(_reconstruct_import([_Inst("IMPORT_NAME", 0, "os", "os")], 0))


# ===========================================================================
# (E) Adversarial DEFER / never-raise tests (mock bytecode)
# ===========================================================================
class AdversarialTest(unittest.TestCase):
    def test_more_than_one_drop_defers(self):
        # two dropped dels -> ambiguous count -> defer.
        gt = _BC([
            _Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "a", "a"),
            _Inst("DELETE_NAME", 1, "x", "x"), _Inst("DELETE_NAME", 2, "y", "y"),
            _Inst("RETURN_CONST", 3, None, "None"),
        ])
        der = _BC([_Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "a", "a"),
                   _Inst("RETURN_CONST", 1, None, "None")])
        self.assertIsNone(dropped_statement_candidate(gt, der, "a = 5\n", _ctx()))

    def test_mixed_kinds_two_drops_defers(self):
        # one dropped del AND one dropped import -> two kinds -> defer.
        gt = _BC([
            _Inst("LOAD_CONST", 0, 0, "0"), _Inst("LOAD_CONST", 1, None, "None"),
            _Inst("IMPORT_NAME", 0, "os", "os"), _Inst("STORE_NAME", 0, "os", "os"),
            _Inst("LOAD_CONST", 2, 5, "5"), _Inst("STORE_NAME", 1, "a", "a"),
            _Inst("DELETE_NAME", 1, "a", "a"),
            _Inst("RETURN_CONST", 3, None, "None"),
        ])
        der = _BC([_Inst("LOAD_CONST", 2, 5, "5"), _Inst("STORE_NAME", 1, "a", "a"),
                   _Inst("RETURN_CONST", 3, None, "None")])
        self.assertIsNone(dropped_statement_candidate(gt, der, "a = 5\n", _ctx()))

    def test_wrong_regime_defers(self):
        gt = _BC([_Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "a", "a"),
                  _Inst("DELETE_NAME", 1, "x", "x"),
                  _Inst("RETURN_CONST", 2, None, "None")])
        der = _BC([_Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "a", "a"),
                   _Inst("RETURN_CONST", 1, None, "None")])
        self.assertIsNone(dropped_statement_candidate(gt, der, "a = 5\n", _ctx("Equal")))
        self.assertIsNone(dropped_statement_candidate(gt, der, "a = 5\n", None))
        self.assertIsNone(dropped_statement_candidate(gt, der, "a = 5\n", {}))

    def test_control_flow_regime_only_considers_assert(self):
        # A dropped del under the control-flow regime is NOT eligible (only assert is).
        gt = _BC([_Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "a", "a"),
                  _Inst("DELETE_NAME", 1, "x", "x"),
                  _Inst("RETURN_CONST", 2, None, "None")])
        der = _BC([_Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "a", "a"),
                   _Inst("RETURN_CONST", 1, None, "None")])
        self.assertIsNone(
            dropped_statement_candidate(gt, der, "a = 5\n", _ctx("Different control flow")))

    def test_complex_assert_call_defers(self):
        # assert f(x): the only divergence, but the test is a call -> defer.
        gt = _BC([_Inst("PUSH_NULL"), _Inst("LOAD_NAME", 0, "f", "f"),
                  _Inst("LOAD_NAME", 1, "x", "x"), _Inst("CALL", 1),
                  _Inst("POP_JUMP_IF_TRUE", 2), _Inst("LOAD_ASSERTION_ERROR"),
                  _Inst("RAISE_VARARGS", 1), _Inst("RETURN_CONST", 0, None, "None")])
        der = _BC([_Inst("RETURN_CONST", 0, None, "None")])
        self.assertIsNone(
            dropped_statement_candidate(gt, der, "pass\n", _ctx("Different control flow")))

    def test_garbage_never_raises(self):
        garbage = _BC([_Inst("WEIRD_OP", 3), _Inst("DELETE_SUBSCR"),
                       _Inst("MAP_ADD", 9), _Inst("IMPORT_NAME", 0, None, "")])
        for frag in ("def (:\n", "", "x = \n", "class C\n"):
            self.assertIsNone(dropped_statement_candidate(garbage, _BC([]), frag, _ctx()))
        # nothing dropped at all -> None
        self.assertIsNone(dropped_statement_candidate(_BC([]), _BC([]), "x = 1\n", _ctx()))

    def test_present_del_is_not_a_drop(self):
        # derived still deletes x -> nothing dropped -> defer.
        gt = _BC([_Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "x", "x"),
                  _Inst("DELETE_NAME", 0, "x", "x"),
                  _Inst("RETURN_CONST", 1, None, "None")])
        der = _BC([_Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "x", "x"),
                   _Inst("DELETE_NAME", 0, "x", "x"),
                   _Inst("RETURN_CONST", 1, None, "None")])
        self.assertIsNone(dropped_statement_candidate(gt, der, "x = 5\ndel x\n", _ctx()))


# ===========================================================================
# Positive mock test: del splices beside its unique store neighbour.
# ===========================================================================
class DelSpliceMockTest(unittest.TestCase):
    def test_del_spliced_after_preceding_store(self):
        gt = _BC([
            _Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "x", "x"),
            _Inst("DELETE_NAME", 0, "x", "x"),
            _Inst("LOAD_CONST", 1, 6, "6"), _Inst("STORE_NAME", 1, "y", "y"),
            _Inst("RETURN_CONST", 2, None, "None"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", 0, 5, "5"), _Inst("STORE_NAME", 0, "x", "x"),
            _Inst("LOAD_CONST", 1, 6, "6"), _Inst("STORE_NAME", 1, "y", "y"),
            _Inst("RETURN_CONST", 2, None, "None"),
        ])
        out = dropped_statement_candidate(gt, der, "x = 5\ny = 6\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "x = 5\ndel x\ny = 6\n")
        self.assertEqual(out["kind"], "dropped_statement")
        self.assertEqual(out["opname"], "DELETE_NAME")
        self.assertEqual(out["confidence"], "unique")
        compile(out["text"], "<m>", "exec")


# ===========================================================================
# End-to-end tests: real pylingual compare_pyc + oracle combined_distance.
# ===========================================================================
def _run_case(testcase, gt_src, der_src, want_name, message, fragment=None,
              expect_zero=True):
    """Compile GT/DER, find the diverging code object, run the operator, recompile its
    output and return (distance_before, distance_after, out)."""
    from pylingual.equivalence_check import compare_pyc

    fragment = der_src if fragment is None else fragment
    with tempfile.TemporaryDirectory() as d:
        gt_py = os.path.join(d, "gt.py")
        der_py = os.path.join(d, "der.py")
        Path(gt_py).write_text(gt_src, encoding="utf-8")
        Path(der_py).write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
        der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
        before = _combined_distance(gt_pyc, der_pyc)

        tr = next((t for t in compare_pyc(gt_pyc, der_pyc)
                   if not t.success and t.names() == want_name), None)
        testcase.assertIsNotNone(tr, f"expected a divergence at {want_name}")
        testcase.assertEqual(tr.message, message)

        out = dropped_statement_candidate(
            tr.bc_a, tr.bc_b, fragment, {"pylingual_failed_result": {"message": tr.message}})
        testcase.assertIsNotNone(out, "operator declined a recoverable drop")
        compile(out["text"], "<fixed>", "exec")

        fixed_py = os.path.join(d, "fixed.py")
        Path(fixed_py).write_text(out["text"], encoding="utf-8")
        fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
        after = _combined_distance(gt_pyc, fixed_pyc)
        return before, after, out


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class EndToEndTest(unittest.TestCase):
    def test_A_dropped_del(self):
        before, after, out = _run_case(
            self,
            gt_src="x = 5\ndel x\ny = 6\nprint(y)\n",
            der_src="x = 5\ny = 6\nprint(y)\n",
            want_name="<module>", message="Different bytecode")
        self.assertGreater(before, 0)
        self.assertEqual(after, 0)
        self.assertEqual(out["text"], "x = 5\ndel x\ny = 6\nprint(y)\n")

    def test_B1_dropped_import_os(self):
        before, after, out = _run_case(
            self,
            gt_src="import os\nx = 5\nprint(x)\n",
            der_src="x = 5\nprint(x)\n",
            want_name="<module>", message="Different bytecode")
        self.assertGreater(before, 0)
        self.assertEqual(after, 0)
        self.assertEqual(out["text"], "import os\nx = 5\nprint(x)\n")

    def test_B2_dropped_from_import_as(self):
        before, after, out = _run_case(
            self,
            gt_src="from a import b as c\nx = 5\nprint(x)\n",
            der_src="x = 5\nprint(x)\n",
            want_name="<module>", message="Different bytecode")
        self.assertGreater(before, 0)
        self.assertEqual(after, 0)
        self.assertEqual(out["text"], "from a import b as c\nx = 5\nprint(x)\n")

    def test_C_dropped_assert(self):
        before, after, out = _run_case(
            self,
            gt_src="x = 1\nassert x == 1\ny = 2\n",
            der_src="x = 1\ny = 2\n",
            want_name="<module>", message="Different control flow")
        self.assertGreater(before, 0)
        self.assertLess(after, before)          # distance drops
        self.assertEqual(after, 0)              # in fact reaches 0
        self.assertEqual(out["text"], "x = 1\nassert x == 1\ny = 2\n")

    def test_D_dropped_global(self):
        # The nested function `f` surfaces as `<module>.f`; the fragment is the def source.
        before, after, out = _run_case(
            self,
            gt_src="def f():\n    global g\n    g = 1\n    return g\n",
            der_src="def f():\n    g = 1\n    return g\n",
            want_name="<module>.f", message="Different bytecode",
            fragment="def f():\n    g = 1\n    return g\n")
        self.assertGreater(before, 0)
        self.assertLess(after, before)          # distance drops
        self.assertEqual(after, 0)
        self.assertEqual(out["text"], "def f():\n    global g\n    g = 1\n    return g\n")


if __name__ == "__main__":
    unittest.main()
