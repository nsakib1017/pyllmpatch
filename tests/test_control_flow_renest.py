"""Unit + end-to-end tests for the control-flow RE-NESTING operator
(utils/semantic_operators/control_flow_renest.py).

The operator re-parents a contiguous run of derived statements into the single ``if`` /
``for`` construct the ground truth nests them in but the decompiler emitted one level
SHALLOWER (flattened). Correctness is proven from GT bytecode (the header + body content)
and PEP 657 co_positions COLUMN offsets (the re-parenting direction), and every emitted
fragment is checked to recompile; the happy-path cases assert the recompiled output reaches
a STRICTLY LOWER oracle combined_distance against GT (ideally 0) through the real scorer.

Honest scope, restated as executable assertions:
  * (A) dropped ``if`` re-nest and (B) dropped ``for`` re-nest CONVERT deterministically —
    columns give a clean, keyword-invariant proof and recompilation collapses the distance.
  * (C) the real 3.10 ``find_module_override`` case DEFERS: 3.10 predates PEP 657, so no
    columns exist, and the operator declines rather than guess. This documents, with a real
    end-to-end assertion, that the column-based lever is simply unavailable pre-3.11 (and,
    separately, that this case is arbitrary re-parenting into a non-flat derived body — the
    harder regime that needs the LLM even where columns exist).
  * (D) adversarial inputs (missing body content, wrong regime, no columns, garbage) all
    return None and NEVER raise.
"""
import ast
import os
import py_compile
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.control_flow_renest import (
    _cols_by_line,
    _find_single_construct,
    _proves_reparenting,
    _recover_expr,
    _recover_for_header,
    _renest_signature,
    control_flow_renest_candidate,
)


class _Inst:
    def __init__(self, opname, argrepr="", arg=None, argval=None, starts_line=None, offset=0):
        self.opname = opname
        self.argrepr = argrepr
        self.arg = arg
        self.argval = argval
        self.starts_line = starts_line
        self.offset = offset


class _BC:
    """Minimal EditableBytecode stand-in: iterable of instructions, optionally carrying a
    raw code object for the co_positions path."""

    def __init__(self, insts, codeobj=None):
        self._insts = insts
        self.codeobj = codeobj

    def __iter__(self):
        return iter(self._insts)


class _FakeCode:
    """A raw-code-object stand-in exposing only ``co_positions`` (columns can be forced to
    None to emulate CPython 3.10)."""

    def __init__(self, positions):
        self._positions = positions

    def co_positions(self):
        return iter(self._positions)


def _ctx(message="Different control flow"):
    return {"pylingual_failed_result": {"message": message}}


# ---------------------------------------------------------------------------
# Pure helper tests (mock instructions / code objects)
# ---------------------------------------------------------------------------
class ExprRecoveryTest(unittest.TestCase):
    def test_name(self):
        self.assertEqual(_recover_expr([_Inst("LOAD_FAST", "items")]), "items")

    def test_attribute_chain(self):
        self.assertEqual(
            _recover_expr([_Inst("LOAD_FAST", "self"), _Inst("LOAD_ATTR", "items")]),
            "self.items",
        )

    def test_range_call(self):
        insts = [
            _Inst("PUSH_NULL"),
            _Inst("LOAD_GLOBAL", "NULL + range"),
            _Inst("LOAD_FAST", "n"),
            _Inst("CALL", "", arg=1),
        ]
        self.assertEqual(_recover_expr(insts), "range(n)")

    def test_subscript(self):
        insts = [_Inst("LOAD_FAST", "d"), _Inst("LOAD_CONST", argval="k"), _Inst("BINARY_SUBSCR")]
        self.assertEqual(_recover_expr(insts), "d['k']")

    def test_defers_on_unknown_op(self):
        self.assertIsNone(_recover_expr([_Inst("BUILD_SET", "", arg=2)]))

    def test_defers_on_empty(self):
        self.assertIsNone(_recover_expr([]))


class ForHeaderTest(unittest.TestCase):
    def test_simple_name_iterable(self):
        insts = [
            _Inst("LOAD_FAST", "items", starts_line=2),
            _Inst("GET_ITER"),
            _Inst("FOR_ITER", "to 34"),
            _Inst("STORE_FAST", "it"),
        ]
        self.assertEqual(_recover_for_header(insts), ("it", "items"))

    def test_range_iterable(self):
        insts = [
            _Inst("PUSH_NULL", starts_line=2),
            _Inst("LOAD_GLOBAL", "NULL + range"),
            _Inst("LOAD_FAST", "n"),
            _Inst("CALL", "", arg=1),
            _Inst("GET_ITER"),
            _Inst("FOR_ITER", "to 40"),
            _Inst("STORE_FAST", "i"),
        ]
        self.assertEqual(_recover_for_header(insts), ("i", "range(n)"))

    def test_tuple_target_defers(self):
        insts = [
            _Inst("LOAD_FAST", "pairs", starts_line=2),
            _Inst("GET_ITER"),
            _Inst("FOR_ITER", "to 34"),
            _Inst("UNPACK_SEQUENCE", "", arg=2),
            _Inst("STORE_FAST", "a"),
            _Inst("STORE_FAST", "b"),
        ]
        self.assertIsNone(_recover_for_header(insts))

    def test_no_for_iter_defers(self):
        self.assertIsNone(_recover_for_header([_Inst("LOAD_FAST", "x")]))


class SignatureTest(unittest.TestCase):
    def test_load_family_is_binding_insensitive(self):
        # The whole point: LOAD_FAST it (GT local) and LOAD_GLOBAL it (flattened derived)
        # must produce the SAME signature so the run can be located.
        self.assertEqual(
            _renest_signature([_Inst("LOAD_FAST", "it")]),
            _renest_signature([_Inst("LOAD_GLOBAL", "NULL + it")]),
        )

    def test_attr_and_const_keyed(self):
        sig = _renest_signature([_Inst("LOAD_ATTR", "NULL|self + append"), _Inst("LOAD_CONST", argval=1)])
        self.assertEqual(sig[0], ("LOAD_ATTR", "append"))
        self.assertEqual(sig[1], ("LOAD_CONST", "1"))


class ColsAndProofTest(unittest.TestCase):
    def test_cols_by_line_takes_min(self):
        co = _FakeCode([(2, 2, 7, 11), (2, 2, 15, 19), (3, 3, 8, 9), (3, 3, 12, 21)])
        self.assertEqual(_cols_by_line(_BC([], codeobj=co)), {2: 7, 3: 8})

    def test_cols_none_on_310_style(self):
        co = _FakeCode([(2, 2, None, None), (3, 3, None, None)])
        self.assertIsNone(_cols_by_line(_BC([], codeobj=co)))

    def test_cols_none_without_codeobj(self):
        self.assertIsNone(_cols_by_line(_BC([])))

    def test_proof_constant_positive_delta(self):
        gt_cols = {3: 8, 4: 8, 5: 8}
        der_cols = {2: 4, 3: 4, 4: 4}
        gt_body = [_Inst("A", starts_line=3), _Inst("B", starts_line=4), _Inst("C", starts_line=5)]
        der_matched = [_Inst("A", starts_line=2), _Inst("B", starts_line=3), _Inst("C", starts_line=4)]
        self.assertTrue(_proves_reparenting(gt_cols, der_cols, gt_body, der_matched))

    def test_proof_rejects_zero_delta(self):
        gt_cols = {3: 4}
        der_cols = {2: 4}
        self.assertFalse(_proves_reparenting(
            gt_cols, der_cols, [_Inst("A", starts_line=3)], [_Inst("A", starts_line=2)]))

    def test_proof_rejects_inconsistent_delta(self):
        gt_cols = {3: 8, 4: 12}
        der_cols = {2: 4, 3: 4}
        gt_body = [_Inst("A", starts_line=3), _Inst("B", starts_line=4)]
        der_matched = [_Inst("A", starts_line=2), _Inst("B", starts_line=3)]
        self.assertFalse(_proves_reparenting(gt_cols, der_cols, gt_body, der_matched))


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests (never raise) — (D)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_wrong_regime_defers(self):
        gt = _BC([_Inst("LOAD_FAST", "x")])
        self.assertIsNone(control_flow_renest_candidate(gt, _BC([]), "def f():\n    pass\n",
                                                        _ctx("Different bytecode")))

    def test_none_context_defers(self):
        gt = _BC([_Inst("LOAD_FAST", "x")])
        self.assertIsNone(control_flow_renest_candidate(gt, _BC([]), "def f():\n    pass\n", None))

    def test_no_columns_defers(self):
        # 3.10-style: co_positions present but every column is None -> defer.
        co = _FakeCode([(1, 1, None, None), (2, 2, None, None)])
        gt = _BC([_Inst("LOAD_FAST", "x", starts_line=2), _Inst("POP_JUMP_IF_FALSE")], codeobj=co)
        self.assertIsNone(control_flow_renest_candidate(gt, _BC([]), "x = 1\n", _ctx()))

    def test_garbage_bytecode_never_raises(self):
        gt = _BC([_Inst("WEIRD_OP", "?"), _Inst("POP_JUMP_IF_FALSE", "to 4")])
        self.assertIsNone(control_flow_renest_candidate(gt, _BC([]), "def f():\n    pass\n", _ctx()))

    def test_empty_bytecode_defers(self):
        self.assertIsNone(control_flow_renest_candidate(_BC([]), _BC([]), "x = 1\n", _ctx()))


# ---------------------------------------------------------------------------
# End-to-end tests through the real pylingual compare_pyc + oracle distance
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import compare_code_object_distances, summarize_results
    return summarize_results(compare_code_object_distances(gt_pyc, derived_pyc)).get("combined_distance")


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 11), "requires PEP 657 columns (CPython 3.11+)")
class SyntheticReNestTest(unittest.TestCase):
    """(A)/(B): GT nests a contiguous run inside one construct; the derived module
    flattened it out. The operator must re-nest exactly that run with the bytecode-recovered
    header, and the recompiled output must reach a STRICTLY LOWER combined_distance."""

    def _run(self, gt_src, der_src, qual="<module>.f", expect_fire=True, expect_zero=True):
        from pylingual.equivalence_check import compare_pyc

        with tempfile.TemporaryDirectory() as d:
            gt_py = os.path.join(d, "gt.py")
            der_py = os.path.join(d, "der.py")
            Path(gt_py).write_text(gt_src, encoding="utf-8")
            Path(der_py).write_text(der_src, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))

            failures = [tr for tr in compare_pyc(gt_pyc, der_pyc) if not tr.success]
            tr = next((t for t in failures if t.names() == qual), None)
            self.assertIsNotNone(
                tr, f"expected a divergence for {qual}: {[(t.names(), t.message) for t in failures]}")
            self.assertEqual(tr.message, "Different control flow")
            baseline = _combined_distance(gt_pyc, der_pyc)
            self.assertGreater(baseline, 0)

            out = control_flow_renest_candidate(tr.bc_a, tr.bc_b, der_src, _ctx(tr.message))
            if not expect_fire:
                self.assertIsNone(out)
                return None
            self.assertIsNotNone(out, "operator was expected to fire")
            self.assertEqual(out["kind"], "control_flow_renest")
            self.assertEqual(out["confidence"], "unique")

            text = out["text"] if out["text"].endswith("\n") else out["text"] + "\n"
            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(text, encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            after = _combined_distance(gt_pyc, fixed_pyc)
            self.assertLess(after, baseline, "combined distance must strictly drop")
            if expect_zero:
                self.assertEqual(after, 0, f"expected exact recovery; fixed source was:\n{text}")
            return out

    # --- (A) dropped-if re-nest ---
    def test_a_dropped_if_renest(self):
        out = self._run(
            "def f(m, d):\n    if m:\n        a = d.read()\n        b = a + 1\n        return b\n    return 0\n",
            "def f(m, d):\n    a = d.read()\n    b = a + 1\n    return b\n    return 0\n",
        )
        self.assertIn("if m", out["text"])

    def test_a_dropped_if_comparison(self):
        self._run(
            "def f(mode, data):\n    if mode == 'rb':\n        x = data.read()\n        return x\n    return None\n",
            "def f(mode, data):\n    x = data.read()\n    return x\n    return None\n",
        )

    # --- (B) dropped-for re-nest ---
    def test_b_dropped_for_renest(self):
        out = self._run(
            "def f(items, acc):\n    for it in items:\n        acc.append(it)\n        acc.append(it + 1)\n    return acc\n",
            "def f(items, acc):\n    acc.append(it)\n    acc.append(it + 1)\n    return acc\n",
        )
        self.assertIn("for it in items", out["text"])

    def test_b_dropped_for_range(self):
        out = self._run(
            "def f(n, acc):\n    for i in range(n):\n        acc.append(i)\n        acc.append(i * 2)\n    return acc\n",
            "def f(n, acc):\n    acc.append(i)\n    acc.append(i * 2)\n    return acc\n",
        )
        self.assertIn("for i in range(n)", out["text"])

    # --- end-to-end DEFER cases ---
    def test_missing_body_content_defers(self):
        # Derived is flat but the if-body statements are ABSENT (replaced by unrelated code)
        # -> the run cannot be located -> defer.
        self._run(
            "def f(m, d):\n    if m:\n        a = d.read()\n        b = a + 1\n        return b\n    return 0\n",
            "def f(m, d):\n    x = 99\n    return 0\n",
            expect_fire=False,
        )

    def test_unrelated_trivial_branch_still_reduces(self):
        # Generalization over control_flow_candidate: the derived body carries an unrelated
        # (empty) branch, yet the operator still locates + re-nests the target run and
        # STRICTLY reduces the distance. The residual (the unrelated `if c: pass`, which GT
        # lacks) is a separate divergence for another step -> not exact-zero here.
        out = self._run(
            "def f(m, d, c):\n    if m:\n        a = d.read()\n        b = a + 1\n        return b\n    return 0\n",
            "def f(m, d, c):\n    if c:\n        pass\n    a = d.read()\n    b = a + 1\n    return b\n    return 0\n",
            expect_zero=False,
        )
        self.assertIn("if m", out["text"])

    def test_real_nesting_before_run_defers(self):
        # Genuine re-parenting: derived nests REAL statements before the target run, so the
        # bytecode-ordinal -> AST-ordinal mapping no longer holds. The operator must DEFER
        # (arbitrary re-parenting is the LLM's job) rather than mis-place the wrapper.
        self._run(
            "def f(m, d, c):\n    if m:\n        a = d.read()\n        b = a + 1\n        return b\n    return 0\n",
            "def f(m, d, c):\n    if c:\n        z = 1\n        w = 2\n    a = d.read()\n    b = a + 1\n    return b\n    return 0\n",
            expect_fire=False,
        )

    def test_unrecoverable_if_header_defers(self):
        # Condition is a call -> header not recoverable -> defer.
        self._run(
            "def f(data, acc):\n    if data.ready():\n        acc.append(1)\n        acc.append(2)\n",
            "def f(data, acc):\n    acc.append(1)\n    acc.append(2)\n",
            expect_fire=False,
        )


# ---------------------------------------------------------------------------
# (C) Real artifact — 3.10 find_module_override: no PEP 657 columns -> DEFER.
# ---------------------------------------------------------------------------
def _real_paths():
    try:
        from utils.file_helpers import fetch_pyllmpatch_repair_paths
        g, d, ds = fetch_pyllmpatch_repair_paths(
            "642dfddb80c1dc618acb68885b304c68487de8e8011ddafa7d4e30e1b06fd0cf", "PyLingual.IO"
        )
        if g and d and ds and Path(g).exists() and Path(d).exists() and Path(ds).exists():
            return Path(g), Path(d), Path(ds)
    except Exception:  # noqa: BLE001
        pass
    return None, None, None


_R_GT, _R_DER, _R_SRC = _real_paths()


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(_R_GT is not None, "642dfddb real artifacts not present")
class RealFindModuleOverrideTest(unittest.TestCase):
    """(C) The real ``find_module_override`` case is a genuine "Different control flow"
    divergence, but it is 3.10 bytecode — which predates PEP 657 — so co_positions carries
    no columns and the operator DEFERS (returns None). It is ALSO arbitrary re-parenting into
    a non-flat derived body, so it would defer even on 3.11+. This is the mandated
    unique-or-defer behavior on a real case, asserted end-to-end."""

    def test_real_case_defers(self):
        from pylingual.equivalence_check import compare_pyc

        baseline = _combined_distance(_R_GT, _R_DER)
        self.assertGreater(baseline, 0)
        tr = next(
            (t for t in compare_pyc(_R_GT, _R_DER)
             if not t.success and t.names() == "<module>.find_module_override"),
            None,
        )
        self.assertIsNotNone(tr)
        self.assertEqual(tr.message, "Different control flow")

        # No PEP 657 columns on 3.10 -> the operator's column gate must fail closed.
        self.assertIsNone(_cols_by_line(tr.bc_a), "3.10 GT must expose no columns")

        src = _R_SRC.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        func = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "find_module_override")
        fragment = ast.get_source_segment(src, func)

        out = control_flow_renest_candidate(tr.bc_a, tr.bc_b, fragment, _ctx(tr.message))
        self.assertIsNone(out, "operator must defer on the 3.10 / non-flat real case")


if __name__ == "__main__":
    unittest.main()
