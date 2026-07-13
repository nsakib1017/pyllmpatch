"""Unit tests for the decorator / default-argument operator
(utils/semantic_operators/decorator_default.py).

The operator restores a dropped positional default value or ``@decorator`` from ground-truth
``MAKE_FUNCTION`` bytecode. Because a ``def`` and its wrapper are two separate code objects and
the *body* is byte-identical whether or not a default/decorator was present, the diverging code
object is always the **enclosing** (module / class) one — which is what ``compare_pyc`` reports
and hands the operator, so no parent-climbing is needed.

Bytecode is supplied either as explicit instruction lists (matching pylingual's editable
bytecode shapes) for the helper / defer tests, or via the real pylingual ``compare_pyc`` for the
two end-to-end tests, which assert the recompiled output reaches combined_distance 0."""
import ast
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# <repo>/pylingual (the importable package root) and <repo> must both be on sys.path so
# `import pylingual.equivalence_check` and `utils` resolve regardless of launch cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.decorator_default import (
    _decorator_expr_before,
    _def_sites,
    decorator_default_candidate,
)


class _Inst:
    def __init__(self, opname, arg=None, argval=None, argrepr=None):
        self.opname = opname
        self.arg = arg
        self.argval = argval
        self.argrepr = argrepr if argrepr is not None else ("" if argval is None else str(argval))


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


def _ctx(message="Different bytecode"):
    return {"pylingual_failed_result": {"message": message}}


def _code(src):
    """Compile ``src`` (a single top-level ``def``) and return that def's code object, for use
    as a ``LOAD_CONST`` argval (it carries co_argcount / co_varnames like the real thing)."""
    for c in compile(src, "<m>", "exec").co_consts:
        if hasattr(c, "co_argcount"):
            return c
    raise AssertionError("no code object in source")


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import (
        compare_code_object_distances,
        summarize_results,
    )
    return summarize_results(compare_code_object_distances(gt_pyc, derived_pyc)).get(
        "combined_distance"
    )


# ---------------------------------------------------------------------------
# Helper-level unit tests
# ---------------------------------------------------------------------------
class DefSitesTest(unittest.TestCase):
    def test_recovers_positional_defaults(self):
        code_f = _code("def f(a, b): return a + b")
        gt = _BC([
            _Inst("LOAD_CONST", argval=(5,)),
            _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1, argrepr="defaults"),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
            _Inst("RETURN_CONST", argval=None),
        ])
        sites = _def_sites(list(gt))
        self.assertIn("f", sites)
        self.assertTrue(sites["f"].defaults_ok)
        self.assertEqual(sites["f"].defaults, (5,))
        self.assertEqual(sites["f"].flags, 1)

    def test_recovers_single_decorator(self):
        code_f = _code("def f(): return 1")
        gt = _BC([
            _Inst("LOAD_NAME", argval="deco", argrepr="deco"),
            _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=0),
            _Inst("CALL", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
            _Inst("RETURN_CONST", argval=None),
        ])
        sites = _def_sites(list(gt))
        self.assertEqual(sites["f"].decorator, "deco")
        self.assertEqual(sites["f"].decorator_count, 1)

    def test_duplicate_name_is_marked_ambiguous(self):
        code_f = _code("def f(): return 1")
        gt = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        self.assertFalse(_def_sites(list(gt))["f"].clean)

    def test_dotted_decorator_expr(self):
        seq = [
            _Inst("LOAD_NAME", argval="m", argrepr="m"),
            _Inst("LOAD_ATTR", argval="deco", argrepr="deco"),
            _Inst("LOAD_CONST", argval=_code("def f(): return 1")),
        ]
        self.assertEqual(_decorator_expr_before(seq, 2), "m.deco")

    def test_call_form_decorator_is_not_a_bare_name(self):
        # @deco(1): PUSH_NULL / LOAD deco / LOAD_CONST 1 / CALL 1 / LOAD_CONST code -> the
        # instruction just before code is a CALL, not a name load -> unrecoverable here.
        seq = [
            _Inst("PUSH_NULL"), _Inst("LOAD_NAME", argval="deco", argrepr="deco"),
            _Inst("LOAD_CONST", argval=1), _Inst("CALL", arg=1),
            _Inst("LOAD_CONST", argval=_code("def f(): return 1")),
        ]
        self.assertIsNone(_decorator_expr_before(seq, 4))


# ---------------------------------------------------------------------------
# Positive mock tests (bytecode shapes without a full compile)
# ---------------------------------------------------------------------------
class PositiveMockTest(unittest.TestCase):
    def test_default_added_to_matching_param(self):
        code_f = _code("def f(a, b): return a + b")
        gt = _BC([
            _Inst("LOAD_CONST", argval=(5,)), _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="f", argrepr="f"),
            _Inst("RETURN_CONST", argval=None),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"), _Inst("RETURN_CONST", argval=None),
        ])
        out = decorator_default_candidate(gt, der, "def f(a, b):\n    return a + b\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b=5):\n    return a + b\n")
        self.assertEqual(out["kind"], "decorator_default")
        self.assertEqual(out["opname"], "MAKE_FUNCTION")
        self.assertEqual(out["confidence"], "unique")
        compile(out["text"], "<m>", "exec")

    def test_two_trailing_defaults(self):
        code_f = _code("def f(a, b, c): return a")
        gt = _BC([
            _Inst("LOAD_CONST", argval=(1, 2)), _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        out = decorator_default_candidate(gt, der, "def f(a, b, c):\n    return a\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b=1, c=2):\n    return a\n")

    def test_decorator_prepended(self):
        code_f = _code("def f(): return 1")
        gt = _BC([
            _Inst("LOAD_NAME", argval="deco", argrepr="deco"),
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("CALL", arg=0), _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        out = decorator_default_candidate(gt, der, "def f():\n    return 1\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "@deco\ndef f():\n    return 1\n")
        compile(out["text"], "<m>", "exec")

    def test_default_on_indented_method(self):
        code_m = _code("def m(self, x): return x")
        gt = _BC([
            _Inst("LOAD_CONST", argval=(7,)), _Inst("LOAD_CONST", argval=code_m),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="m", argrepr="m"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_m), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="m", argrepr="m"),
        ])
        frag = "class C:\n    def m(self, x):\n        return x\n"
        out = decorator_default_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "class C:\n    def m(self, x=7):\n        return x\n")


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_non_const_default_defers(self):
        # def f(a, b=[]) -> BUILD_LIST 0 / BUILD_TUPLE 1 (no LOAD_CONST tuple) -> defer.
        code_f = _code("def f(a, b): return a")
        gt = _BC([
            _Inst("BUILD_LIST", arg=0), _Inst("BUILD_TUPLE", arg=1),
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=1),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        self.assertIsNone(decorator_default_candidate(gt, der, "def f(a, b):\n    return a\n", _ctx()))

    def test_duplicate_def_name_defers(self):
        code_f = _code("def f(a, b): return a")
        gt = _BC([
            _Inst("LOAD_CONST", argval=(5,)), _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="f", argrepr="f"),
            _Inst("LOAD_CONST", argval=(5,)), _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        frag = "def f(a, b):\n    return a\ndef f(a, b):\n    return a\n"
        self.assertIsNone(decorator_default_candidate(gt, der, frag, _ctx()))

    def test_param_name_mismatch_defers(self):
        # GT default (5,) attaches to a 2-arg (a, b) def, but the fragment's def has different
        # param names -> alignment fails -> defer.
        code_f = _code("def f(a, b): return a")
        gt = _BC([
            _Inst("LOAD_CONST", argval=(5,)), _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        self.assertIsNone(decorator_default_candidate(gt, der, "def f(x, y):\n    return x\n", _ctx()))

    def test_call_form_decorator_defers(self):
        code_f = _code("def f(): return 1")
        gt = _BC([
            _Inst("PUSH_NULL"), _Inst("LOAD_NAME", argval="deco", argrepr="deco"),
            _Inst("LOAD_CONST", argval=1), _Inst("CALL", arg=1),
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("CALL", arg=0), _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        self.assertIsNone(decorator_default_candidate(gt, der, "def f():\n    return 1\n", _ctx()))

    def test_wrong_regime_defers(self):
        code_f = _code("def f(a, b): return a")
        gt = _BC([
            _Inst("LOAD_CONST", argval=(5,)), _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        frag = "def f(a, b):\n    return a\n"
        self.assertIsNone(decorator_default_candidate(gt, der, frag, _ctx("Different control flow")))
        self.assertIsNone(decorator_default_candidate(gt, der, frag, None))

    def test_no_divergence_defers(self):
        # GT and derived both already have the default -> nothing to add.
        code_f = _code("def f(a, b): return a")
        insts = [
            _Inst("LOAD_CONST", argval=(5,)), _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="f", argrepr="f"),
        ]
        self.assertIsNone(decorator_default_candidate(_BC(insts), _BC(list(insts)),
                                                      "def f(a, b=5):\n    return a\n", _ctx()))

    def test_garbage_never_raises(self):
        gt = _BC([
            _Inst("WEIRD_OP", arg=3), _Inst("MAKE_FUNCTION", arg=9),
            _Inst("LOAD_CONST", argval=object()), _Inst("MAKE_FUNCTION", arg=1),
            _Inst("STORE_NAME", argval="?", argrepr="?"),
        ])
        # Un-parseable fragment + junk bytecode must return None, never raise.
        self.assertIsNone(decorator_default_candidate(gt, _BC([]), "def (:\n", _ctx()))
        self.assertIsNone(decorator_default_candidate(_BC([]), _BC([]), "x = 1\n", _ctx()))


# ---------------------------------------------------------------------------
# End-to-end tests through real pylingual compare_pyc + oracle distance
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


def _module_divergence(gt_pyc: Path, der_pyc: Path):
    from pylingual.equivalence_check import compare_pyc
    for tr in compare_pyc(gt_pyc, der_pyc):
        if not tr.success and tr.names() == "<module>":
            return tr
    return None


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticDefaultTest(unittest.TestCase):
    """(A) GT ``def f(a, b=5)`` vs derived ``def f(a, b)``: the operator restores ``=5`` and the
    recompiled module reaches combined_distance 0 against GT."""

    GT_SRC = "def f(a, b=5):\n    return a + b\n\nprint(f(1))\n"
    DER_SRC = "def f(a, b):\n    return a + b\n\nprint(f(1))\n"

    def test_recompiles_to_distance_zero(self):
        with tempfile.TemporaryDirectory() as d:
            gt_py, der_py = os.path.join(d, "gt.py"), os.path.join(d, "der.py")
            Path(gt_py).write_text(self.GT_SRC, encoding="utf-8")
            Path(der_py).write_text(self.DER_SRC, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))

            self.assertNotEqual(_combined_distance(gt_pyc, der_pyc), 0)
            tr = _module_divergence(gt_pyc, der_pyc)
            self.assertIsNotNone(tr)
            self.assertEqual(tr.message, "Different bytecode")

            out = decorator_default_candidate(
                tr.bc_a, tr.bc_b, self.DER_SRC, {"pylingual_failed_result": {"message": tr.message}}
            )
            self.assertIsNotNone(out)
            self.assertEqual(out["text"], self.GT_SRC)

            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            self.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticDecoratorTest(unittest.TestCase):
    """(B) GT wraps ``f`` in a module-level ``@deco`` (deco is itself a def) that the derived
    source dropped: the operator restores ``@deco`` and the recompiled module strictly drops the
    distance (to 0)."""

    GT_SRC = "def deco(g):\n    return g\n\n@deco\ndef f():\n    return 1\n"
    DER_SRC = "def deco(g):\n    return g\n\ndef f():\n    return 1\n"

    def test_recompiles_with_strict_drop(self):
        with tempfile.TemporaryDirectory() as d:
            gt_py, der_py = os.path.join(d, "gt.py"), os.path.join(d, "der.py")
            Path(gt_py).write_text(self.GT_SRC, encoding="utf-8")
            Path(der_py).write_text(self.DER_SRC, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))

            before = _combined_distance(gt_pyc, der_pyc)
            self.assertGreater(before, 0)
            tr = _module_divergence(gt_pyc, der_pyc)
            self.assertIsNotNone(tr)

            out = decorator_default_candidate(
                tr.bc_a, tr.bc_b, self.DER_SRC, {"pylingual_failed_result": {"message": tr.message}}
            )
            self.assertIsNotNone(out)
            self.assertEqual(out["text"], self.GT_SRC)

            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            after = _combined_distance(gt_pyc, fixed_pyc)
            self.assertLess(after, before)   # strict drop
            self.assertEqual(after, 0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticNonConstDefaultDefersTest(unittest.TestCase):
    """(C) GT ``def f(a, b=[])`` — a non-constant (mutable) default — must DEFER through the
    real compare_pyc path, never emitting a guessed value."""

    GT_SRC = "def f(a, b=[]):\n    return a\n\nprint(f(1))\n"
    DER_SRC = "def f(a, b):\n    return a\n\nprint(f(1))\n"

    def test_defers(self):
        with tempfile.TemporaryDirectory() as d:
            gt_py, der_py = os.path.join(d, "gt.py"), os.path.join(d, "der.py")
            Path(gt_py).write_text(self.GT_SRC, encoding="utf-8")
            Path(der_py).write_text(self.DER_SRC, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
            tr = _module_divergence(gt_pyc, der_pyc)
            self.assertIsNotNone(tr)
            out = decorator_default_candidate(
                tr.bc_a, tr.bc_b, self.DER_SRC, {"pylingual_failed_result": {"message": tr.message}}
            )
            self.assertIsNone(out)


# ---------------------------------------------------------------------------
# EXTENSION (task #35): annotation-entangled defaults, SET_FUNCTION_ATTRIBUTE form,
# and multiple diverging functions. See docs/superpowers/specs/
# 2026-07-12-deterministic-module-operator-coverage-design.md.
# ---------------------------------------------------------------------------
class AnnotationEntangledDefaultMockTest(unittest.TestCase):
    """3.11/3.12 flag form: a function with BOTH defaults and annotations
    (``MAKE_FUNCTION 5``) whose derived dropped only the defaults (``MAKE_FUNCTION 4``).
    The defaults tuple must be recovered even though it is not immediately before the
    code load (the annotations const sits between)."""

    def _sites_flags5(self):
        code_f = _code("def f(a, b): return a")
        # stack order (bit order): defaults(0x01), annotations(0x04), code.
        gt = _BC([
            _Inst("LOAD_CONST", argval=(5,)),                 # defaults
            _Inst("LOAD_CONST", argval=("return", "int")),    # annotations tuple
            _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=5, argrepr="defaults, annotations"),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=("return", "int")),    # annotations only
            _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=4, argrepr="annotations"),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        return gt, der

    def test_def_sites_recovers_defaults_under_annotations(self):
        gt, _der = self._sites_flags5()
        sites = _def_sites(list(gt))
        self.assertIn("f", sites)
        self.assertTrue(sites["f"].defaults_ok, "defaults must be recovered when flags & 0x01 with other bits set")
        self.assertEqual(sites["f"].defaults, (5,))

    def test_candidate_restores_default_under_annotation(self):
        gt, der = self._sites_flags5()
        out = decorator_default_candidate(gt, der, "def f(a, b) -> int:\n    return a\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b=5) -> int:\n    return a\n")


class SetFunctionAttributeDefaultMockTest(unittest.TestCase):
    """3.13-3.15 form: ``MAKE_FUNCTION`` takes no flags; defaults are applied by a
    trailing ``SET_FUNCTION_ATTRIBUTE 1``. Derived dropped the whole thing."""

    def _sites_sfa(self, with_annotation=False):
        code_f = _code("def f(a, b): return a")
        gt_ins = [_Inst("LOAD_CONST", argval=(5,))]           # defaults value
        if with_annotation:
            gt_ins.append(_Inst("LOAD_CONST", argval=("return", "int")))
        gt_ins += [
            _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=None),
        ]
        if with_annotation:
            gt_ins.append(_Inst("SET_FUNCTION_ATTRIBUTE", arg=4, argrepr="annotations"))
        gt_ins += [
            _Inst("SET_FUNCTION_ATTRIBUTE", arg=1, argrepr="defaults"),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ]
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=None),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
        ])
        return _BC(gt_ins), der

    def test_def_sites_recovers_defaults_from_sfa(self):
        gt, _der = self._sites_sfa()
        sites = _def_sites(list(gt))
        self.assertIn("f", sites)
        self.assertTrue(sites["f"].defaults_ok, "SET_FUNCTION_ATTRIBUTE 1 defaults must be recovered (3.13+)")
        self.assertEqual(sites["f"].defaults, (5,))

    def test_candidate_restores_default_sfa(self):
        gt, der = self._sites_sfa()
        out = decorator_default_candidate(gt, der, "def f(a, b):\n    return a\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(a, b=5):\n    return a\n")


class MultiFunctionDefaultTest(unittest.TestCase):
    """Multiple functions dropped their defaults: the operator fixes ONE (deterministically
    the first by source order) and defers the rest to the next prepass iteration, rather than
    deferring everything because more than one candidate applies."""

    def test_fixes_one_of_many(self):
        code_f = _code("def f(a, b): return a")
        code_g = _code("def g(c, d): return c")
        gt = _BC([
            _Inst("LOAD_CONST", argval=(5,)), _Inst("LOAD_CONST", argval=code_f),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="f", argrepr="f"),
            _Inst("LOAD_CONST", argval=(9,)), _Inst("LOAD_CONST", argval=code_g),
            _Inst("MAKE_FUNCTION", arg=1), _Inst("STORE_NAME", argval="g", argrepr="g"),
        ])
        der = _BC([
            _Inst("LOAD_CONST", argval=code_f), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="f", argrepr="f"),
            _Inst("LOAD_CONST", argval=code_g), _Inst("MAKE_FUNCTION", arg=0),
            _Inst("STORE_NAME", argval="g", argrepr="g"),
        ])
        frag = "def f(a, b):\n    return a\n\ndef g(c, d):\n    return c\n"
        out = decorator_default_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out, "must fix one function rather than defer on multiple")
        # deterministically the first (f); exactly one edit applied, still valid source.
        self.assertEqual(out["text"], "def f(a, b=5):\n    return a\n\ndef g(c, d):\n    return c\n")
        compile(out["text"], "<m>", "exec")


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticAnnotationEntangledDefaultTest(unittest.TestCase):
    """End-to-end on the host interpreter, mirroring the real target 8eedb5648e (pandas):
    ``from __future__ import annotations`` makes annotations const strings, so a function with
    BOTH defaults and a return annotation compiles to two const tuples before the code load
    (``MAKE_FUNCTION 5`` on 3.11/3.12, ``SET_FUNCTION_ATTRIBUTE`` on 3.13+). The derived source
    dropped only the defaults. The operator restores ``=5`` and recompiles to distance 0."""

    GT_SRC = "from __future__ import annotations\n\ndef f(a, b=5) -> int:\n    return a + b\n\nprint(f(1))\n"
    DER_SRC = "from __future__ import annotations\n\ndef f(a, b) -> int:\n    return a + b\n\nprint(f(1))\n"

    def test_recompiles_to_distance_zero(self):
        with tempfile.TemporaryDirectory() as d:
            gt_py, der_py = os.path.join(d, "gt.py"), os.path.join(d, "der.py")
            Path(gt_py).write_text(self.GT_SRC, encoding="utf-8")
            Path(der_py).write_text(self.DER_SRC, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
            self.assertNotEqual(_combined_distance(gt_pyc, der_pyc), 0)
            tr = _module_divergence(gt_pyc, der_pyc)
            self.assertIsNotNone(tr)
            out = decorator_default_candidate(
                tr.bc_a, tr.bc_b, self.DER_SRC, {"pylingual_failed_result": {"message": tr.message}})
            self.assertIsNotNone(out)
            self.assertEqual(out["text"], self.GT_SRC)
            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            self.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticComputedAnnotationDefersTest(unittest.TestCase):
    """Safety boundary: a *computed* return annotation (``-> int`` with NO
    ``from __future__ import annotations``) builds the annotation via ``LOAD_NAME int;
    BUILD_TUPLE`` — a multi-instruction attribute slot the defaults localizer cannot positively
    separate — so the operator DEFERS rather than guess. The oracle gate / LLM handle it."""

    GT_SRC = "def f(a, b=5) -> int:\n    return a + b\n\nprint(f(1))\n"
    DER_SRC = "def f(a, b) -> int:\n    return a + b\n\nprint(f(1))\n"

    def test_defers(self):
        with tempfile.TemporaryDirectory() as d:
            gt_py, der_py = os.path.join(d, "gt.py"), os.path.join(d, "der.py")
            Path(gt_py).write_text(self.GT_SRC, encoding="utf-8")
            Path(der_py).write_text(self.DER_SRC, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
            tr = _module_divergence(gt_pyc, der_pyc)
            self.assertIsNotNone(tr)
            out = decorator_default_candidate(
                tr.bc_a, tr.bc_b, self.DER_SRC, {"pylingual_failed_result": {"message": tr.message}})
            self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
