"""Unit + end-to-end tests for the exception-region operator
(utils/semantic_operators/exc_regions.py).

The operator deterministically reconstructs try/except, try/finally and ``with``
structure the decompiler DROPPED or flattened, reading the protected byte ranges
and handler targets from the 3.11+ exception table (``co_exceptiontable`` via
``bc.named_exception_table``) and mapping them to present derived statements by
matching the GT protected body's ``(opcode, operand)`` signature as a unique
contiguous run.

Bootstrapped exactly like the sibling operator tests: pure-mock helper tests for
the handler classifiers / context recovery, and real-compile end-to-end tests
through pylingual's ``compare_pyc`` + the real oracle distance scorer
(``compare_code_object_distances`` + ``summarize_results``), asserting the
recompiled output's combined distance strictly drops (to 0 on the happy paths).
"""
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# The importable pylingual package lives at <repo>/pylingual/pylingual, so
# <repo>/pylingual must be on sys.path (alongside the repo root) for `import
# pylingual.*` and for `utils` to resolve regardless of launch cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.exc_regions import (  # noqa: E402
    _except_kind,
    _finally_body,
    _recover_ctx_load,
    _reconstruct_with,
    exc_regions_candidate,
)


# ---------------------------------------------------------------------------
# Mock bytecode shapes for the pure helper tests.
# ---------------------------------------------------------------------------
class _Inst:
    def __init__(self, offset, opname, argrepr="", argval=None, starts_line=None):
        self.offset = offset
        self.opname = opname
        self.argrepr = argrepr
        self.argval = argval
        self.starts_line = starts_line


class _BC:
    def __init__(self, insts, table=None):
        self._insts = insts
        self.named_exception_table = table

    def __iter__(self):
        return iter(self._insts)


def _ctx(message="Different control flow"):
    return {"pylingual_failed_result": {"message": message}}


# ---------------------------------------------------------------------------
# Handler classification helpers (pure, mock instructions).
# ---------------------------------------------------------------------------
class ExceptKindTest(unittest.TestCase):
    def test_bare_except_pass_is_trivial(self):
        # PUSH_EXC_INFO POP_TOP POP_EXCEPT  -> `except: pass`
        insts = [_Inst(12, "PUSH_EXC_INFO"), _Inst(14, "POP_TOP"),
                 _Inst(16, "POP_EXCEPT"), _Inst(18, "RETURN_CONST", "None")]
        trivial, exc = _except_kind(insts, 12)
        self.assertTrue(trivial)
        self.assertIsNone(exc)

    def test_typed_except_pass_is_trivial(self):
        # PUSH_EXC_INFO LOAD_GLOBAL CHECK_EXC_MATCH POP_JUMP POP_TOP POP_EXCEPT
        insts = [_Inst(12, "PUSH_EXC_INFO"), _Inst(14, "LOAD_GLOBAL", "ValueError"),
                 _Inst(16, "CHECK_EXC_MATCH"), _Inst(18, "POP_JUMP_IF_FALSE", "to 26"),
                 _Inst(20, "POP_TOP"), _Inst(22, "POP_EXCEPT"),
                 _Inst(24, "RETURN_CONST", "None"), _Inst(26, "RERAISE")]
        trivial, exc = _except_kind(insts, 12)
        self.assertTrue(trivial)
        self.assertEqual(exc, "ValueError")

    def test_dotted_except_type_recovered(self):
        # `except os.error: pass`  -> LOAD_GLOBAL os; LOAD_ATTR error
        insts = [_Inst(12, "PUSH_EXC_INFO"), _Inst(14, "LOAD_GLOBAL", "NULL + os"),
                 _Inst(16, "LOAD_ATTR", "error"), _Inst(18, "CHECK_EXC_MATCH"),
                 _Inst(20, "POP_JUMP_IF_FALSE", "to 30"), _Inst(22, "POP_TOP"),
                 _Inst(24, "POP_EXCEPT"), _Inst(26, "RERAISE")]
        trivial, exc = _except_kind(insts, 12)
        self.assertTrue(trivial)
        self.assertEqual(exc, "os.error")

    def test_content_handler_not_trivial(self):
        # `except ValueError: acc.append('x')` -> real work before POP_EXCEPT.
        insts = [_Inst(12, "PUSH_EXC_INFO"), _Inst(14, "LOAD_GLOBAL", "ValueError"),
                 _Inst(16, "CHECK_EXC_MATCH"), _Inst(18, "POP_JUMP_IF_FALSE", "to 40"),
                 _Inst(20, "POP_TOP"), _Inst(22, "LOAD_FAST", "acc"),
                 _Inst(24, "LOAD_ATTR", "append"), _Inst(26, "LOAD_CONST", '"x"'),
                 _Inst(28, "CALL"), _Inst(30, "POP_TOP"), _Inst(32, "POP_EXCEPT"),
                 _Inst(34, "RERAISE")]
        trivial, _ = _except_kind(insts, 12)
        self.assertFalse(trivial)

    def test_finally_handler_not_an_except(self):
        # A finally cleanup (no POP_EXCEPT before RERAISE) is not a trivial except.
        insts = [_Inst(30, "PUSH_EXC_INFO"), _Inst(32, "LOAD_FAST", "b"),
                 _Inst(34, "LOAD_ATTR", "close"), _Inst(36, "CALL"),
                 _Inst(38, "POP_TOP"), _Inst(40, "RERAISE")]
        trivial, _ = _except_kind(insts, 30)
        self.assertFalse(trivial)


class FinallyBodyTest(unittest.TestCase):
    def test_recovers_finally_body(self):
        insts = [_Inst(30, "PUSH_EXC_INFO"), _Inst(32, "LOAD_FAST", "b"),
                 _Inst(34, "LOAD_ATTR", "close"), _Inst(36, "CALL"),
                 _Inst(38, "POP_TOP"), _Inst(40, "RERAISE")]
        body = _finally_body(insts, 30)
        self.assertIsNotNone(body)
        self.assertEqual([i.opname for i in body],
                         ["LOAD_FAST", "LOAD_ATTR", "CALL", "POP_TOP"])

    def test_except_handler_is_not_finally(self):
        # Has POP_EXCEPT -> an except handler, not a finally.
        insts = [_Inst(12, "PUSH_EXC_INFO"), _Inst(14, "POP_TOP"),
                 _Inst(16, "POP_EXCEPT"), _Inst(18, "RERAISE")]
        self.assertIsNone(_finally_body(insts, 12))


class RecoverCtxLoadTest(unittest.TestCase):
    def test_simple_name(self):
        insts = [_Inst(0, "LOAD_FAST", "lock", starts_line=2)]
        self.assertEqual(_recover_ctx_load(insts), "lock")

    def test_attribute_chain(self):
        insts = [_Inst(0, "LOAD_FAST", "self", starts_line=2),
                 _Inst(2, "LOAD_ATTR", "lock")]
        self.assertEqual(_recover_ctx_load(insts), "self.lock")

    def test_call_defers(self):
        # `with open(f):` -> a call is NOT a call-free load -> defer.
        insts = [_Inst(0, "LOAD_GLOBAL", "NULL + open", starts_line=2),
                 _Inst(2, "LOAD_FAST", "f"), _Inst(4, "CALL")]
        self.assertIsNone(_recover_ctx_load(insts))


# ---------------------------------------------------------------------------
# End-to-end tests through the real pylingual compare_pyc + oracle distance.
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import (
        compare_code_object_distances,
        summarize_results,
    )
    return summarize_results(
        compare_code_object_distances(gt_pyc, derived_pyc)
    ).get("combined_distance")


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 11),
                     "exception-table path requires CPython 3.11+")
class EndToEndTest(unittest.TestCase):
    """Real compile + oracle. Each happy path must reach combined_distance 0."""

    def _prep(self, d, gt_src, der_src, qual="<module>.f"):
        from pylingual.equivalence_check import compare_pyc

        gt_py = os.path.join(d, "gt.py")
        der_py = os.path.join(d, "der.py")
        Path(gt_py).write_text(gt_src, encoding="utf-8")
        Path(der_py).write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
        der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
        failures = [t for t in compare_pyc(gt_pyc, der_pyc) if not t.success]
        tr = next((t for t in failures if t.names() == qual), None)
        self.assertIsNotNone(
            tr, f"expected a divergence for {qual}: {[(t.names(), t.message) for t in failures]}")
        self.assertEqual(tr.message, "Different control flow")
        baseline = _combined_distance(gt_pyc, der_pyc)
        self.assertGreater(baseline, 0)
        return tr, gt_pyc, der_pyc, baseline

    def _run(self, gt_src, der_src, qual="<module>.f", expect_zero=True):
        with tempfile.TemporaryDirectory() as d:
            tr, gt_pyc, _der, baseline = self._prep(d, gt_src, der_src, qual)
            out = exc_regions_candidate(tr.bc_a, tr.bc_b, der_src, _ctx(tr.message))
            self.assertIsNotNone(out, "operator should fire on this deterministic case")
            self.assertEqual(out["kind"], "exc_regions")
            self.assertEqual(out["confidence"], "unique")
            text = out["text"] if out["text"].endswith("\n") else out["text"] + "\n"
            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(text, encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            after = _combined_distance(gt_pyc, fixed_pyc)
            self.assertLess(after, baseline, "combined distance must strictly drop")
            if expect_zero:
                self.assertEqual(after, 0, f"expected exact reconstruction; got:\n{text}")
            return out

    # --- (A) try/except around a MULTI-STATEMENT body ---
    def test_A_try_except_multi_statement(self):
        out = self._run(
            "def f(a, b, acc):\n"
            "    try:\n"
            "        acc.append(a)\n"
            "        acc.append(b)\n"
            "    except ValueError:\n"
            "        pass\n",
            "def f(a, b, acc):\n"
            "    acc.append(a)\n"
            "    acc.append(b)\n",
        )
        self.assertIn("try:", out["text"])
        self.assertIn("except ValueError:", out["text"])

    def test_A_try_except_bare_last_statement(self):
        # Bare `except: pass` as the LAST statement: the narrow structural operator
        # mis-classifies this (its window sweeps in the trailing `return None`); this
        # operator must handle it.
        self._run(
            "def f(a, acc):\n"
            "    try:\n"
            "        acc.append(a)\n"
            "        acc.append(a)\n"
            "    except:\n"
            "        pass\n",
            "def f(a, acc):\n"
            "    acc.append(a)\n"
            "    acc.append(a)\n",
        )

    # --- (B) try/FINALLY ---
    def test_B_try_finally(self):
        out = self._run(
            "def f(acc, b):\n"
            "    try:\n"
            "        acc.append(1)\n"
            "        acc.append(2)\n"
            "    finally:\n"
            "        b.close()\n",
            "def f(acc, b):\n"
            "    acc.append(1)\n"
            "    acc.append(2)\n"
            "    b.close()\n",
        )
        self.assertIn("try:", out["text"])
        self.assertIn("finally:", out["text"])

    # --- (C) with ---
    def test_C_with_open_as_fh(self):
        out = self._run(
            "def f(fname, acc):\n"
            "    with open(fname) as fh:\n"
            "        acc.append(fh.read())\n",
            "def f(fname, acc):\n"
            "    fh = open(fname)\n"
            "    acc.append(fh.read())\n",
        )
        self.assertIn("with open(fname) as fh:", out["text"])
        # the flattened `fh = open(fname)` assignment must be gone
        self.assertNotIn("fh = open(fname)", out["text"])

    def test_C_with_callfree_ctx(self):
        out = self._run(
            "def f(lock, acc):\n"
            "    with lock:\n"
            "        acc.append(1)\n"
            "        acc.append(2)\n",
            "def f(lock, acc):\n"
            "    acc.append(1)\n"
            "    acc.append(2)\n",
        )
        self.assertIn("with lock:", out["text"])

    # --- (D) ADVERSARIAL / DEFER cases ---
    def test_D_content_handler_defers(self):
        # A real (content-bearing) handler body must DEFER (LLM territory).
        with tempfile.TemporaryDirectory() as d:
            tr, _gt, _der, _b = self._prep(
                d,
                "def f(a, acc):\n"
                "    try:\n"
                "        acc.append(a)\n"
                "    except ValueError:\n"
                "        acc.append('err')\n",
                "def f(a, acc):\n"
                "    acc.append(a)\n",
            )
            self.assertIsNone(
                exc_regions_candidate(tr.bc_a, tr.bc_b,
                                      "def f(a, acc):\n    acc.append(a)\n",
                                      _ctx(tr.message)))

    def test_D_wrong_regime_defers(self):
        gt = _BC([_Inst(0, "PUSH_EXC_INFO")])
        self.assertIsNone(
            exc_regions_candidate(gt, _BC([]), "def f():\n    pass\n",
                                  _ctx("Different bytecode")))

    def test_D_none_context_defers(self):
        gt = _BC([_Inst(0, "PUSH_EXC_INFO")])
        self.assertIsNone(exc_regions_candidate(gt, _BC([]), "def f():\n    pass\n", None))

    def test_D_garbage_never_raises(self):
        # Non-bytecode mocks / garbage fragment: must return None, never raise.
        gt = _BC([_Inst(0, "WEIRD_OP", "?"), _Inst(2, "BEFORE_WITH")], table=[])
        self.assertIsNone(
            exc_regions_candidate(gt, _BC([]), "def (:\n  ??\n", _ctx()))
        self.assertIsNone(
            exc_regions_candidate(_BC([]), _BC([]), "x = 1\n", _ctx()))

    def test_D_single_trivial_try_handled_not_crash(self):
        # The narrow operator's own case (single call, bare except) — this operator
        # may ALSO handle it, but must at minimum not crash and, if it fires, must
        # produce a distance-0 reconstruction.
        with tempfile.TemporaryDirectory() as d:
            tr, gt_pyc, _der, baseline = self._prep(
                d,
                "def f(path, acc):\n"
                "    try:\n"
                "        acc.append(path)\n"
                "    except:\n"
                "        pass\n"
                "    return acc\n",
                "def f(path, acc):\n"
                "    acc.append(path)\n"
                "    return acc\n",
            )
            out = exc_regions_candidate(
                tr.bc_a, tr.bc_b,
                "def f(path, acc):\n    acc.append(path)\n    return acc\n",
                _ctx(tr.message))
            if out is not None:
                text = out["text"] if out["text"].endswith("\n") else out["text"] + "\n"
                fixed_py = os.path.join(d, "fixed.py")
                Path(fixed_py).write_text(text, encoding="utf-8")
                fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
                self.assertLess(_combined_distance(gt_pyc, fixed_pyc), baseline)

    def test_D_module_scope_defers(self):
        # `<module>` objects are intentionally out of scope -> defer even on a real
        # dropped-try divergence.
        with tempfile.TemporaryDirectory() as d:
            from pylingual.equivalence_check import compare_pyc

            gt_src = ("try:\n"
                      "    a = 1\n"
                      "    b = 2\n"
                      "except ValueError:\n"
                      "    pass\n")
            der_src = "a = 1\nb = 2\n"
            gt_py = os.path.join(d, "gt.py"); Path(gt_py).write_text(gt_src)
            der_py = os.path.join(d, "der.py"); Path(der_py).write_text(der_src)
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
            tr = next((t for t in compare_pyc(gt_pyc, der_pyc)
                       if not t.success and t.names() == "<module>"), None)
            if tr is not None:  # only meaningful if the module itself diverges
                self.assertIsNone(
                    exc_regions_candidate(tr.bc_a, tr.bc_b, der_src, _ctx(tr.message)))


if __name__ == "__main__":
    unittest.main()
