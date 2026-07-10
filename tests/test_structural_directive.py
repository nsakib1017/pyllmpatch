"""Tests for the deterministic STRUCTURAL-DIRECTIVE generator
(utils/semantic_operators/structural_directive.py).

The generator turns the ground-truth control-flow SKELETON DIFF (constructs GT has that
the decompiled source is missing/mis-nesting) into a short, imperative natural-language
directive for injection into the LLM repair prompt on "Different control flow" objects.

These tests mirror tests/test_control_flow_operator.py's real-compile approach: GT and
derived source are compiled with py_compile on the host (3.12), the diverging code object
is pulled from pylingual's real ``compare_pyc``, and the directive is generated from the
real GT/derived bytecode (``tr.bc_a`` / ``tr.bc_b``). Pure-safety cases use mock
instruction lists to prove the public function never raises.
"""
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

# The importable pylingual package lives at <repo>/pylingual/pylingual, so <repo>/pylingual
# must be on sys.path (alongside the repo root). Mirrors the repo's other operator tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "pylingual"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_operators.structural_directive import (
    _describe_statement,
    structural_directive_or_empty,
    structural_repair_directive,
)


class _Inst:
    """Mock instruction matching pylingual's editable-bytecode instruction shape."""

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


def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


def _diverging_bc(gt_src, der_src, qual="<module>.f"):
    """Compile both sources on the host and return (tr.bc_a, tr.bc_b, message) for the
    named divergence, or (None, None, None) if there is none."""
    from pylingual.equivalence_check import compare_pyc

    with tempfile.TemporaryDirectory() as d:
        gt_py = os.path.join(d, "gt.py")
        der_py = os.path.join(d, "der.py")
        Path(gt_py).write_text(gt_src, encoding="utf-8")
        Path(der_py).write_text(der_src, encoding="utf-8")
        gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
        der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))
        tr = next((t for t in compare_pyc(gt_pyc, der_pyc)
                   if not t.success and t.names() == qual), None)
        if tr is None:
            return None, None, None
        return tr.bc_a, tr.bc_b, tr.message


# ---------------------------------------------------------------------------
# Pure statement-describer unit tests (mock instructions).
# ---------------------------------------------------------------------------
class DescribeStatementTest(unittest.TestCase):
    def test_call_assignment(self):
        group = [
            _Inst("LOAD_FAST", "data", starts_line=3), _Inst("LOAD_ATTR", "NULL|self + read"),
            _Inst("CALL", "", arg=0), _Inst("STORE_FAST", "x"),
        ]
        self.assertEqual(_describe_statement(group), "x = data.read(...)")

    def test_augmented_assignment(self):
        group = [
            _Inst("LOAD_FAST", "total", starts_line=4), _Inst("LOAD_FAST", "n"),
            _Inst("BINARY_OP", "+="), _Inst("STORE_FAST", "total"),
        ]
        self.assertEqual(_describe_statement(group), "total += n")

    def test_return_name(self):
        group = [_Inst("LOAD_FAST", "result", starts_line=3), _Inst("RETURN_VALUE")]
        self.assertEqual(_describe_statement(group), "return result")

    def test_bare_call(self):
        group = [
            _Inst("LOAD_GLOBAL", "io", starts_line=5), _Inst("LOAD_METHOD", "BytesIO"),
            _Inst("CALL", "", arg=0), _Inst("JUMP_FORWARD", "to 40"),
        ]
        self.assertEqual(_describe_statement(group), "io.BytesIO(...)")


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests (never raise).
# ---------------------------------------------------------------------------
class SafetyTest(unittest.TestCase):
    def test_garbage_bytecode_never_raises(self):
        gt = _BC([_Inst("WEIRD_OP", "?"), _Inst("POP_JUMP_IF_FALSE", "to 4")])
        der = _BC([_Inst("NONSENSE")])
        self.assertIsNone(structural_repair_directive(gt, der))

    def test_empty_bytecode_returns_none(self):
        self.assertIsNone(structural_repair_directive(_BC([]), _BC([])))

    def test_or_empty_helper_returns_string(self):
        self.assertEqual(structural_directive_or_empty(_BC([]), _BC([])), "")

    def test_none_inputs_never_raise(self):
        self.assertIsNone(structural_repair_directive(None, None))


# ---------------------------------------------------------------------------
# End-to-end tests through real pylingual compare_pyc.
# ---------------------------------------------------------------------------
@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticSkeletonDiffTest(unittest.TestCase):
    # (A) DROPPED IF -------------------------------------------------------
    def test_dropped_if_guard(self):
        gt = ("def f(mode, data):\n    if mode == 'rb':\n        x = data.read()\n"
              "        return x\n    return None\n")
        der = "def f(mode, data):\n    x = data.read()\n    return x\n    return None\n"
        bc_a, bc_b, msg = _diverging_bc(gt, der)
        self.assertEqual(msg, "Different control flow")
        directive = structural_repair_directive(bc_a, bc_b)
        self.assertIsInstance(directive, str)
        self.assertTrue(directive)
        self.assertIn("if", directive)
        # condition token recovered verbatim from GT bytecode
        self.assertTrue(any(tok in directive for tok in ("rb", "mode", "m")))
        # references the guarded body
        self.assertTrue(any(tok in directive for tok in ("read", "x")))

    # (B) DROPPED FOR ------------------------------------------------------
    def test_dropped_for_loop(self):
        gt = ("def f(items):\n    total = 0\n    for n in items:\n        total += n\n"
              "    return total\n")
        der = "def f(items):\n    total = 0\n    total += items\n    return total\n"
        bc_a, bc_b, msg = _diverging_bc(gt, der)
        self.assertEqual(msg, "Different control flow")
        directive = structural_repair_directive(bc_a, bc_b)
        self.assertIsInstance(directive, str)
        self.assertTrue(directive)
        self.assertIn("for", directive)
        # iterator/target recovered from GT bytecode
        self.assertIn("n", directive)
        self.assertIn("items", directive)

    # (C) DROPPED TRY ------------------------------------------------------
    def test_dropped_try_except(self):
        gt = ("def f(data):\n    try:\n        result = data.read()\n        return result\n"
              "    except OSError:\n        return None\n")
        der = "def f(data):\n    result = data.read()\n    return result\n"
        bc_a, bc_b, msg = _diverging_bc(gt, der)
        self.assertEqual(msg, "Different control flow")
        directive = structural_repair_directive(bc_a, bc_b)
        self.assertIsInstance(directive, str)
        self.assertTrue(directive)
        self.assertIn("try", directive)
        self.assertIn("except", directive)
        self.assertIn("OSError", directive)

    # (D) NEGATIVE: identical control structure, only a leaf differs -------
    def test_identical_structure_leaf_diff_returns_none(self):
        # Same `if path is None:` guard on both sides; only a returned literal differs
        # (a leaf, NOT a control-flow change) -> the recovered-condition sets cancel.
        gt = ("def f(path, x):\n    if path is None:\n        x = x + 1\n        return x\n"
              "    return 7\n")
        der = ("def f(path, x):\n    if path is None:\n        x = x + 1\n        return x\n"
               "    return 9\n")
        bc_a, bc_b, msg = _diverging_bc(gt, der)
        self.assertIsNotNone(bc_a, "expected a divergence to obtain bytecode")
        self.assertIsNone(structural_repair_directive(bc_a, bc_b))

    def test_flat_leaf_diff_returns_none(self):
        # No control flow at all on either side, only an operand differs -> None.
        gt = "def f(a, b):\n    c = a + b\n    return c\n"
        der = "def f(a, b):\n    c = a - b\n    return c\n"
        bc_a, bc_b, _ = _diverging_bc(gt, der)
        self.assertIsNotNone(bc_a)
        self.assertIsNone(structural_repair_directive(bc_a, bc_b))


# ---------------------------------------------------------------------------
# (E) REAL artifact: 642dfddb .../find_module_override
# ---------------------------------------------------------------------------
def _real_find_module_paths():
    try:
        from utils.file_helpers import fetch_pyllmpatch_repair_paths
        g, d, ds = fetch_pyllmpatch_repair_paths(
            "642dfddb80c1dc618acb68885b304c68487de8e8011ddafa7d4e30e1b06fd0cf", "PyLingual.IO"
        )
        if g and d and Path(g).exists() and Path(d).exists():
            return Path(g), Path(d)
    except Exception:  # noqa: BLE001
        pass
    return None, None


_R_GT, _R_DER = _real_find_module_paths()


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(_R_GT is not None, "642dfddb real artifacts not present")
class RealFindModuleOverrideTest(unittest.TestCase):
    """The real ``find_module_override`` is a genuine "Different control flow" divergence
    that ``bc_to_cft`` CANNOT fully structure (both sides fall back to the irreducible
    ``CDG``). The bytecode-window fallback still recovers the precise delta: the ground
    truth has a conditional on ``mode == 'rb'`` (the ternary ``io.BytesIO() if mode == 'rb'
    else io.StringIO()``) that the decompiled source dropped by stranding it as dead code
    after a ``raise``. The directive must therefore name that conditional construct."""

    def test_real_divergence_produces_directive(self):
        from pylingual.equivalence_check import compare_pyc

        tr = next(
            (t for t in compare_pyc(_R_GT, _R_DER)
             if not t.success and t.names() == "<module>.find_module_override"),
            None,
        )
        self.assertIsNotNone(tr)
        self.assertEqual(tr.message, "Different control flow")

        directive = structural_repair_directive(tr.bc_a, tr.bc_b)
        self.assertIsInstance(directive, str)
        self.assertTrue(directive, "expected a non-empty directive on the real case")
        # names the relevant construct + the bytecode-recovered condition
        self.assertIn("if", directive)
        self.assertIn("mode == 'rb'", directive)


if __name__ == "__main__":
    unittest.main()
