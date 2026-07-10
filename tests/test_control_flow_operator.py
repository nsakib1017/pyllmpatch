"""Unit + end-to-end tests for the control-flow operator
(utils/semantic_operators/control_flow.py).

The operator deterministically re-wraps a DROPPED simple-``if`` guard around a
contiguous run of statements that the decompiler flattened out of the guard but left
present in the derived body. GT/derived bytecode is supplied either as mock instruction
lists (for the pure condition-recovery / matching helpers) or, for the happy-path and
real-artifact tests, via the real pylingual ``compare_pyc``; every emitted fragment is
checked to recompile, and each happy-path case asserts the recompiled output reaches
combined_distance 0 against GT through the real oracle scorer.

Honest scope note (see RealFindModuleOverrideTest): the two real "Different control
flow" rows suggested for this operator do NOT match its target pattern — their statements
are mis-nested (stranded after a ``raise`` in a dead ``else``) or dropped entirely, not
merely un-wrapped — so the operator correctly DEFERS on them. A scan of the whole
hybrid100 dataset (175 CF-failing objects) fired the operator zero times, which is the
intended unique-or-defer behavior: it never emits fragile output on a case it cannot
cleanly and deterministically prove from GT bytecode.
"""
import ast
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# The importable pylingual package lives at <repo>/pylingual/pylingual, so <repo>/pylingual
# must be on sys.path (alongside the repo root) for `import pylingual.*` and for `utils`
# to resolve regardless of launch cwd. Mirrors the repo's other operator tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.control_flow import (
    _const_repr,
    _recover_if_condition,
    _signature,
    _unique_contiguous_match,
    control_flow_candidate,
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


# ---------------------------------------------------------------------------
# Condition-recovery helper tests (pure, mock instructions)
# ---------------------------------------------------------------------------
class ConditionRecoveryTest(unittest.TestCase):
    def test_comparison_eq_const(self):
        # `if mode == 'rb':`  (3.12 form)
        insts = [
            _Inst("LOAD_FAST", "mode", starts_line=2),
            _Inst("LOAD_CONST", argval="rb"),
            _Inst("COMPARE_OP", "=="),
            _Inst("POP_JUMP_IF_FALSE", "to 20"),
        ]
        self.assertEqual(_recover_if_condition(insts), "mode == 'rb'")

    def test_truthiness(self):
        insts = [_Inst("LOAD_FAST", "flag", starts_line=2), _Inst("POP_JUMP_IF_FALSE", "to 10")]
        self.assertEqual(_recover_if_condition(insts), "flag")

    def test_is_none_specialized_jump_3_12(self):
        # `if v is None:` compiles to LOAD v; POP_JUMP_IF_NOT_NONE on 3.11+
        insts = [_Inst("LOAD_FAST", "v", starts_line=2), _Inst("POP_JUMP_IF_NOT_NONE", "to 16")]
        self.assertEqual(_recover_if_condition(insts), "v is None")

    def test_is_not_none_specialized_jump(self):
        insts = [_Inst("LOAD_FAST", "v", starts_line=2), _Inst("POP_JUMP_IF_NONE", "to 16")]
        self.assertEqual(_recover_if_condition(insts), "v is not None")

    def test_is_op_form_3_10(self):
        # `if path is None:` on 3.10 uses IS_OP + POP_JUMP_IF_FALSE
        insts = [
            _Inst("LOAD_FAST", "path", starts_line=2),
            _Inst("LOAD_CONST", argval=None),
            _Inst("IS_OP", "is", arg=0),
            _Inst("POP_JUMP_IF_FALSE", "to 26"),
        ]
        self.assertEqual(_recover_if_condition(insts), "path is None")

    def test_membership(self):
        insts = [
            _Inst("LOAD_FAST", "k", starts_line=2), _Inst("LOAD_FAST", "d"),
            _Inst("CONTAINS_OP", "in", arg=0), _Inst("POP_JUMP_IF_FALSE", "to 20"),
        ]
        self.assertEqual(_recover_if_condition(insts), "k in d")

    def test_attribute_chain(self):
        insts = [
            _Inst("LOAD_FAST", "self", starts_line=2), _Inst("LOAD_ATTR", "mode"),
            _Inst("LOAD_CONST", argval="rb"), _Inst("COMPARE_OP", "=="),
            _Inst("POP_JUMP_IF_FALSE", "to 20"),
        ]
        self.assertEqual(_recover_if_condition(insts), "self.mode == 'rb'")

    def test_uses_only_last_statement_group(self):
        # A leading `y = 1` shares the header basic block; only the final group (the
        # condition) must be recovered, and the leading assignment ignored.
        insts = [
            _Inst("LOAD_CONST", argval=1, starts_line=2), _Inst("STORE_FAST", "y"),
            _Inst("LOAD_FAST", "mode", starts_line=3), _Inst("LOAD_CONST", argval="rb"),
            _Inst("COMPARE_OP", "=="), _Inst("POP_JUMP_IF_FALSE", "to 20"),
        ]
        self.assertEqual(_recover_if_condition(insts), "mode == 'rb'")

    def test_defers_on_call_condition(self):
        # `if foo(mode):` — a CALL is not cleanly recoverable.
        insts = [
            _Inst("LOAD_GLOBAL", "foo", starts_line=2), _Inst("LOAD_FAST", "mode"),
            _Inst("CALL_FUNCTION", "1 positional argument", arg=1),
            _Inst("POP_JUMP_IF_FALSE", "to 20"),
        ]
        self.assertIsNone(_recover_if_condition(insts))

    def test_defers_on_true_jump(self):
        # POP_JUMP_IF_TRUE (negated) — inversion is error-prone, defer.
        insts = [_Inst("LOAD_FAST", "flag", starts_line=2), _Inst("POP_JUMP_IF_TRUE", "to 10")]
        self.assertIsNone(_recover_if_condition(insts))

    def test_defers_when_no_header_jump(self):
        insts = [_Inst("LOAD_FAST", "flag", starts_line=2), _Inst("RETURN_VALUE")]
        self.assertIsNone(_recover_if_condition(insts))

    def test_defers_on_empty(self):
        self.assertIsNone(_recover_if_condition([]))


class ConstReprTest(unittest.TestCase):
    def test_simple_scalars(self):
        self.assertEqual(_const_repr("rb"), "'rb'")
        self.assertEqual(_const_repr(3), "3")
        self.assertEqual(_const_repr(None), "None")
        self.assertEqual(_const_repr(True), "True")

    def test_tuple_of_scalars(self):
        self.assertEqual(_const_repr((1, 2)), "(1, 2)")

    def test_rejects_code_object(self):
        self.assertIsNone(_const_repr(compile("x", "<m>", "eval")))


class MatchTest(unittest.TestCase):
    def test_unique_match(self):
        hay = [("A", None), ("B", "x"), ("C", None), ("D", None)]
        self.assertEqual(_unique_contiguous_match(hay, [("B", "x"), ("C", None)]), 1)

    def test_no_match(self):
        self.assertIsNone(_unique_contiguous_match([("A", None)], [("Z", None)]))

    def test_ambiguous_match_defers(self):
        hay = [("A", None), ("A", None), ("A", None)]
        self.assertIsNone(_unique_contiguous_match(hay, [("A", None)]))

    def test_empty_needle(self):
        self.assertIsNone(_unique_contiguous_match([("A", None)], []))


class SignatureTest(unittest.TestCase):
    def test_skips_noise_and_keys_names(self):
        insts = [
            _Inst("RESUME"), _Inst("CACHE"),
            _Inst("LOAD_FAST", "data"), _Inst("LOAD_ATTR", "NULL|self + read"),
            _Inst("CALL", "", arg=0), _Inst("STORE_FAST", "x"),
        ]
        sig = _signature(insts)
        self.assertEqual(sig[0], ("LOAD_FAST", "data"))
        self.assertEqual(sig[1], ("LOAD_ATTR", "read"))  # cleaned of NULL|self prefix
        self.assertIn(("STORE_FAST", "x"), sig)
        self.assertNotIn(("RESUME", None), sig)


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests (never raise)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_wrong_regime_defers(self):
        gt = _BC([_Inst("LOAD_FAST", "x")])
        self.assertIsNone(control_flow_candidate(gt, _BC([]), "def f():\n    pass\n",
                                                 _ctx("Different bytecode")))

    def test_none_context_defers(self):
        gt = _BC([_Inst("LOAD_FAST", "x")])
        self.assertIsNone(control_flow_candidate(gt, _BC([]), "def f():\n    pass\n", None))

    def test_garbage_bytecode_never_raises(self):
        # Mock objects that are not real EditableBytecode: bc_to_cft will fail internally;
        # the operator must swallow it and return None rather than raise.
        gt = _BC([_Inst("WEIRD_OP", "?"), _Inst("POP_JUMP_IF_FALSE", "to 4")])
        self.assertIsNone(control_flow_candidate(gt, _BC([]), "def f():\n    pass\n", _ctx()))

    def test_empty_bytecode_defers(self):
        self.assertIsNone(control_flow_candidate(_BC([]), _BC([]), "x = 1\n", _ctx()))


# ---------------------------------------------------------------------------
# End-to-end tests through the real pylingual compare_pyc + oracle distance
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


def _combined_distance(gt_pyc: Path, derived_pyc: Path):
    from utils.pyc_code_object_distance import compare_code_object_distances, summarize_results
    return summarize_results(compare_code_object_distances(gt_pyc, derived_pyc)).get("combined_distance")


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticHappyPathTest(unittest.TestCase):
    """(A) A ground-truth module wraps a contiguous run of statements in a simple ``if``
    guard; the derived module flattened that guard away. The operator must re-wrap the
    right statements with the bytecode-recovered condition, and the recompiled output
    must reach combined_distance 0 against GT through the real oracle scorer."""

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
            self.assertIsNotNone(tr, f"expected a divergence for {qual}: {[ (t.names(), t.message) for t in failures]}")
            self.assertEqual(tr.message, "Different control flow")
            baseline = _combined_distance(gt_pyc, der_pyc)
            self.assertGreater(baseline, 0)

            out = control_flow_candidate(tr.bc_a, tr.bc_b, der_src, _ctx(tr.message))
            if not expect_fire:
                self.assertIsNone(out)
                return
            self.assertIsNotNone(out)
            self.assertEqual(out["kind"], "control_flow")
            self.assertEqual(out["confidence"], "unique")

            text = out["text"] if out["text"].endswith("\n") else out["text"] + "\n"
            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(text, encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            after = _combined_distance(gt_pyc, fixed_pyc)
            self.assertLess(after, baseline, "combined distance must strictly drop")
            if expect_zero:
                self.assertEqual(after, 0)

    def test_dropped_eq_guard(self):
        self._run(
            "def f(mode, data):\n    if mode == 'rb':\n        x = data.read()\n        return x\n    return None\n",
            "def f(mode, data):\n    x = data.read()\n    return x\n    return None\n",
        )

    def test_dropped_truthiness_guard(self):
        self._run(
            "def f(flag, data):\n    if flag:\n        x = data.read()\n        return x\n    return None\n",
            "def f(flag, data):\n    x = data.read()\n    return x\n    return None\n",
        )

    def test_dropped_is_none_guard(self):
        self._run(
            "def f(v, data):\n    if v is None:\n        x = data.read()\n        return x\n    return None\n",
            "def f(v, data):\n    x = data.read()\n    return x\n    return None\n",
        )

    def test_dropped_membership_guard(self):
        self._run(
            "def f(k, d):\n    if k in d:\n        x = d.pop()\n        return x\n    return None\n",
            "def f(k, d):\n    x = d.pop()\n    return x\n    return None\n",
        )

    def test_dropped_attribute_guard(self):
        self._run(
            "def f(self, data):\n    if self.mode == 'rb':\n        x = data.read()\n        return x\n    return None\n",
            "def f(self, data):\n    x = data.read()\n    return x\n    return None\n",
        )

    def test_leading_statement_preserved(self):
        self._run(
            "def f(mode, data):\n    y = 1\n    if mode == 'rb':\n        x = data.read()\n        return x\n    return None\n",
            "def f(mode, data):\n    y = 1\n    x = data.read()\n    return x\n    return None\n",
        )

    def test_pure_ifthen_no_trailing(self):
        self._run(
            "def f(mode, acc):\n    if mode == 'rb':\n        acc.append(1)\n        acc.append(2)\n",
            "def f(mode, acc):\n    acc.append(1)\n    acc.append(2)\n",
        )

    # --- end-to-end DEFER cases (adversarial control-flow diffs) ---

    def test_two_constructs_defer(self):
        # GT has TWO ifs -> not a single-missing-wrapper -> defer.
        self._run(
            "def f(a, b, acc):\n    if a:\n        acc.append(1)\n    if b:\n        acc.append(2)\n",
            "def f(a, b, acc):\n    acc.append(1)\n    acc.append(2)\n",
            expect_fire=False,
        )

    def test_unrecoverable_header_defers(self):
        # Condition is a call -> header not bytecode-recoverable -> defer.
        self._run(
            "def f(data, acc):\n    if data.ready():\n        acc.append(1)\n        acc.append(2)\n",
            "def f(data, acc):\n    acc.append(1)\n    acc.append(2)\n",
            expect_fire=False,
        )

    def test_true_else_defers(self):
        # A real if/else with a non-terminating body: leaving the "else" statements
        # un-wrapped would be wrong, so the operator must decline.
        self._run(
            "def f(c):\n    if c:\n        a = 1\n    else:\n        a = 2\n    return a\n",
            "def f(c):\n    a = 1\n    a = 2\n    return a\n",
            expect_fire=False,
        )

    def test_unparseable_fragment_defers(self):
        # A genuine CF divergence, but a garbage fragment -> parse fails -> None, no raise.
        from pylingual.equivalence_check import compare_pyc

        with tempfile.TemporaryDirectory() as d:
            gt_src = "def f(mode, data):\n    if mode == 'rb':\n        x = data.read()\n        return x\n    return None\n"
            der_src = "def f(mode, data):\n    x = data.read()\n    return x\n    return None\n"
            gt_py = os.path.join(d, "gt.py"); Path(gt_py).write_text(gt_src)
            der_py = os.path.join(d, "der.py"); Path(der_py).write_text(der_src)
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
            tr = next(t for t in compare_pyc(gt_pyc, der_pyc) if not t.success and t.names() == "<module>.f")
            self.assertIsNone(control_flow_candidate(tr.bc_a, tr.bc_b, "def (:\n  ??\n", _ctx(tr.message)))


# ---------------------------------------------------------------------------
# (B) Real artifact: the suggested real case does NOT fit the operator's pattern.
# ---------------------------------------------------------------------------
def _real_find_module_paths():
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


_R_GT, _R_DER, _R_SRC = _real_find_module_paths()


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(_R_GT is not None, "642dfddb real artifacts not present")
class RealFindModuleOverrideTest(unittest.TestCase):
    """(B) The suggested real case ``<module>.find_module_override`` is a genuine
    "Different control flow" divergence, but NOT the operator's target pattern: the GT's
    ``file = io.BytesIO() if mode == 'rb' else io.StringIO()`` and ``pathname = filename``
    were decompiled but stranded after a ``raise`` in a dead ``else`` block (a re-nesting
    problem), not merely un-wrapped from a dropped ``if``. The operator must therefore
    DEFER (return None) rather than emit fragile output — exactly the mandated
    unique-or-defer behavior. This documents, with a real end-to-end assertion, that a
    clean deterministic re-wrap is not achievable for this case.
    """

    def test_real_divergence_exists_but_operator_defers(self):
        import ast as _ast

        from pylingual.equivalence_check import compare_pyc

        self.assertGreater(_combined_distance(_R_GT, _R_DER), 0)
        tr = next(
            (t for t in compare_pyc(_R_GT, _R_DER)
             if not t.success and t.names() == "<module>.find_module_override"),
            None,
        )
        self.assertIsNotNone(tr)
        self.assertEqual(tr.message, "Different control flow")

        # Build the same fragment the prepass would (the decompiled function's source).
        src = _R_SRC.read_text(encoding="utf-8", errors="replace")
        tree = _ast.parse(src)
        func = next(n for n in tree.body
                    if isinstance(n, _ast.FunctionDef) and n.name == "find_module_override")
        fragment = _ast.get_source_segment(src, func)

        out = control_flow_candidate(tr.bc_a, tr.bc_b, fragment, _ctx(tr.message))
        self.assertIsNone(out, "operator must defer on the mis-nested real case, not emit fragile output")


if __name__ == "__main__":
    unittest.main()
