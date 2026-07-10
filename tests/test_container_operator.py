"""Unit tests for the container-construction operator
(utils/semantic_operators/container.py).

The operator regenerates an all-constant dict/set/list literal from GT bytecode when the
decompiler mangled the display. GT/derived bytecode is supplied either as explicit
instruction lists (matching the shapes pylingual's editable bytecode yields) or, for the
happy-path / real-artifact tests, via the real pylingual ``compare_pyc``; every emitted
fragment is checked to recompile, and the two end-to-end tests assert the recompiled
output reaches combined_distance 0 against GT."""
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

from utils.semantic_operators.container import (
    _recover_constant_containers,
    _round_trips,
    _top_level_displays,
    container_construction_candidate,
)

C4D3_GT_PYC = (
    "/home/mxs220189/pylingual_collaboration/pylingual_download/decompiler_workspace/"
    "c4d351c4a1705b4035531dc2a9e8bb2ea4c067e29c97a6f406a39cac88cea4a2/constants.pyc"
)
C4D3_DER_PY = (
    "/home/mxs220189/pylingual_collaboration/pylingual_download/decompiler_workspace/"
    "c4d351c4a1705b4035531dc2a9e8bb2ea4c067e29c97a6f406a39cac88cea4a2/"
    "decompiler_output/indented_0.py"
)


class _Inst:
    def __init__(self, opname, arg=None, argval=None, offset=0,
                 is_jump_target=False, is_jump=False):
        self.opname, self.arg, self.argval, self.offset = opname, arg, argval, offset
        self.is_jump_target, self.is_jump = is_jump_target, is_jump


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


# ---------------------------------------------------------------------------
# Helper-level unit tests
# ---------------------------------------------------------------------------
class RecoverContainersTest(unittest.TestCase):
    def test_build_map_then_const_key_map_merge_is_one_dict(self):
        # Mirrors the c4d3 shape: BUILD_MAP + MAP_ADD for the head, a BUILD_CONST_KEY_MAP
        # for the tail, then DICT_UPDATE merges the tail in -> ONE flat dict, in order.
        insts = [
            _Inst("BUILD_MAP", 0),
            _Inst("LOAD_CONST", argval=10), _Inst("LOAD_CONST", argval="A"), _Inst("MAP_ADD", 1),
            _Inst("LOAD_CONST", argval=4), _Inst("LOAD_CONST", argval="B"), _Inst("MAP_ADD", 1),
            _Inst("LOAD_CONST", argval="C"), _Inst("LOAD_CONST", argval="D"),
            _Inst("LOAD_CONST", argval=(8, 1)),  # keys tuple
            _Inst("BUILD_CONST_KEY_MAP", 2),
            _Inst("DICT_UPDATE", 1),
            _Inst("STORE_NAME", argval="T"),
            _Inst("RETURN_CONST", argval=None),
        ]
        got = _recover_constant_containers(insts)
        self.assertEqual(got, [("dict", {10: "A", 4: "B", 8: "C", 1: "D"})])

    def test_build_list_recovered(self):
        insts = [
            _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
            _Inst("BUILD_LIST", 2), _Inst("STORE_NAME", argval="L"),
        ]
        self.assertEqual(_recover_constant_containers(insts), [("list", [1, 2])])

    def test_nonconstant_element_taints_container(self):
        # [1, x] -> the LOAD_NAME pushes NONCONST -> the list is not fully constant.
        insts = [
            _Inst("LOAD_CONST", argval=1), _Inst("LOAD_NAME", argval="x"),
            _Inst("BUILD_LIST", 2), _Inst("STORE_NAME", argval="L"),
        ]
        self.assertEqual(_recover_constant_containers(insts), [])

    def test_empty_container_is_insignificant(self):
        insts = [_Inst("BUILD_MAP", 0), _Inst("STORE_NAME", argval="D"),
                 _Inst("RETURN_CONST", argval=None)]
        self.assertEqual(_recover_constant_containers(insts), [])

    def test_set_from_const_elements(self):
        insts = [
            _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
            _Inst("BUILD_SET", 2), _Inst("STORE_NAME", argval="S"),
        ]
        self.assertEqual(_recover_constant_containers(insts), [("set", {1, 2})])

    def test_never_raises_on_garbage(self):
        # underflowing / unknown opcodes must not raise.
        insts = [_Inst("MAP_ADD", 5), _Inst("DICT_UPDATE", 9),
                 _Inst("WEIRD_UNKNOWN_OP", 3), _Inst("BUILD_MAP", 4)]
        self.assertEqual(_recover_constant_containers(insts), [])


class TopLevelDisplaysTest(unittest.TestCase):
    def test_nested_same_kind_counts_as_one(self):
        tree = ast.parse("T = {1: 'a', X: {2: 'b'}}\n")
        self.assertEqual(len(_top_level_displays(tree, ast.Dict)), 1)

    def test_two_disjoint_dicts_count_as_two(self):
        tree = ast.parse("A = {1: 2}\nB = {3: 4}\n")
        self.assertEqual(len(_top_level_displays(tree, ast.Dict)), 2)


class RoundTripTest(unittest.TestCase):
    def test_dict_round_trips(self):
        self.assertTrue(_round_trips({1: "a", 2: "b"}))

    def test_code_object_does_not_round_trip(self):
        self.assertFalse(_round_trips(compile("x + 1", "<m>", "eval")))


# ---------------------------------------------------------------------------
# Adversarial DEFER / SAFETY tests (mock bytecode)
# ---------------------------------------------------------------------------
class DeferTest(unittest.TestCase):
    def test_nonconstant_value_defers(self):
        # T = {1: some_func()}  -> value is a CALL result, dict not fully constant.
        gt = _BC([
            _Inst("BUILD_MAP", 0), _Inst("LOAD_CONST", argval=1),
            _Inst("PUSH_NULL"), _Inst("LOAD_NAME", argval="some_func"),
            _Inst("CALL", 0), _Inst("MAP_ADD", 1), _Inst("STORE_NAME", argval="T"),
            _Inst("RETURN_CONST", argval=None),
        ])
        self.assertIsNone(container_construction_candidate(gt, _BC([]), "T = {1: some_func()}\n", _ctx()))

    def test_ambiguous_two_displays_defers(self):
        # GT builds exactly one dict, but the fragment has two dict displays -> defer.
        gt = _BC([
            _Inst("BUILD_MAP", 0), _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
            _Inst("MAP_ADD", 1), _Inst("STORE_NAME", argval="T"), _Inst("RETURN_CONST", argval=None),
        ])
        self.assertIsNone(container_construction_candidate(gt, _BC([]), "A = {1: 2}\nB = {3: 4}\n", _ctx()))

    def test_no_container_in_gt_defers(self):
        gt = _BC([_Inst("LOAD_CONST", argval=5), _Inst("STORE_NAME", argval="X"),
                  _Inst("RETURN_CONST", argval=None)])
        self.assertIsNone(container_construction_candidate(gt, _BC([]), "X = 5\n", _ctx()))

    def test_wrong_failure_regime_defers(self):
        gt = _BC([
            _Inst("BUILD_MAP", 0), _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
            _Inst("MAP_ADD", 1), _Inst("STORE_NAME", argval="T"), _Inst("RETURN_CONST", argval=None),
        ])
        self.assertIsNone(container_construction_candidate(gt, _BC([]), "T = {1: 2}\n", _ctx("Different control flow")))
        self.assertIsNone(container_construction_candidate(gt, _BC([]), "T = {1: 2}\n", None))

    def test_never_raises_on_bad_fragment(self):
        gt = _BC([
            _Inst("BUILD_MAP", 0), _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
            _Inst("MAP_ADD", 1), _Inst("STORE_NAME", argval="T"), _Inst("RETURN_CONST", argval=None),
        ])
        # Un-parseable fragment must return None rather than raise.
        self.assertIsNone(container_construction_candidate(gt, _BC([]), "def (:\n", _ctx()))


class PositiveMockTest(unittest.TestCase):
    def test_fires_on_mangled_nested_dict(self):
        # GT: flat dict {1:'a', 2:'b'}. Derived display mangled with a bare-name key and a
        # nested dict (parses, does not evaluate). Operator replaces the outermost dict.
        gt = _BC([
            _Inst("BUILD_MAP", 0),
            _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval="a"), _Inst("MAP_ADD", 1),
            _Inst("LOAD_CONST", argval=2), _Inst("LOAD_CONST", argval="b"), _Inst("MAP_ADD", 1),
            _Inst("STORE_NAME", argval="T"), _Inst("RETURN_CONST", argval=None),
        ])
        out = container_construction_candidate(gt, _BC([]), "T = {1: 'a', BOGUS: {2: 'b'}}\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "T = {1: 'a', 2: 'b'}\n")
        self.assertEqual(out["kind"], "container_construction")
        self.assertEqual(out["confidence"], "unique")
        compile(out["text"], "<m>", "exec")

    def test_preserves_indentation(self):
        gt = _BC([
            _Inst("LOAD_CONST", argval=1), _Inst("LOAD_CONST", argval=2),
            _Inst("BUILD_LIST", 2), _Inst("STORE_FAST", argval="L"),
            _Inst("LOAD_CONST", argval=None), _Inst("RETURN_VALUE"),
        ])
        frag = "def f():\n        L = [9, X, 9]\n        return L\n"
        out = container_construction_candidate(gt, _BC([]), frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f():\n        L = [1, 2]\n        return L\n")
        compile(out["text"], "<m>", "exec")


# ---------------------------------------------------------------------------
# End-to-end tests through the real pylingual compare_pyc + oracle distance
# ---------------------------------------------------------------------------
def _pylingual_available():
    try:
        import pylingual.equivalence_check  # noqa: F401
        return True
    except Exception:  # pragma: no cover - pylingual missing
        return False


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(sys.version_info[:2] >= (3, 10), "requires CPython 3.10+")
class SyntheticHappyPathTest(unittest.TestCase):
    """(A) Build a GT module with a flat all-const dict and a derived module that mangles
    it into a nested dict with a bare-name key; the operator's output must recompile to
    combined_distance 0 against GT."""

    GT_SRC = "T = {10: 'A', 4: 'B', 8: 'C', 1: 'D', 13: 'E', 18: 'F'}\nX = 5\nprint(X)\n"
    DER_SRC = "T = {10: 'A', 4: 'B', F: {8: 'C', 1: 'D', 13: 'E'}}\nX = 5\nprint(X)\n"

    def test_recompiles_to_distance_zero(self):
        from pylingual.equivalence_check import compare_pyc

        with tempfile.TemporaryDirectory() as d:
            gt_py = os.path.join(d, "gt.py")
            der_py = os.path.join(d, "der.py")
            Path(gt_py).write_text(self.GT_SRC, encoding="utf-8")
            Path(der_py).write_text(self.DER_SRC, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc"), doraise=True))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc"), doraise=True))

            failures = [tr for tr in compare_pyc(gt_pyc, der_pyc) if not tr.success]
            self.assertTrue(failures, "expected a module bytecode divergence")
            tr = next(tr for tr in failures if tr.names() == "<module>")
            self.assertEqual(tr.message, "Different bytecode")

            out = container_construction_candidate(
                tr.bc_a, tr.bc_b, self.DER_SRC, {"pylingual_failed_result": {"message": tr.message}}
            )
            self.assertIsNotNone(out)

            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            self.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)


@unittest.skipUnless(_pylingual_available(), "pylingual unavailable")
@unittest.skipUnless(os.path.exists(C4D3_GT_PYC) and os.path.exists(C4D3_DER_PY),
                     "c4d3 real artifacts not present")
@unittest.skipUnless(sys.version_info[:2] == (3, 12), "c4d3 pyc is 3.12 bytecode")
class RealC4d3Test(unittest.TestCase):
    """(B) The proven real case: a module-level TYPE_NAMES dict PyLingual rendered as a
    nested dict with a bare-name key. The operator recovers GT's flat 21-item dict and the
    recompiled module reaches combined_distance 0."""

    def test_recompiles_to_distance_zero(self):
        from pylingual.equivalence_check import compare_pyc

        gt_pyc = Path(C4D3_GT_PYC)
        fragment = Path(C4D3_DER_PY).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            der_pyc = Path(py_compile.compile(C4D3_DER_PY, cfile=os.path.join(d, "der.pyc"), doraise=True))
            self.assertEqual(_combined_distance(gt_pyc, der_pyc), 5)

            tr = next(tr for tr in compare_pyc(gt_pyc, der_pyc)
                      if not tr.success and tr.names() == "<module>")
            out = container_construction_candidate(
                tr.bc_a, tr.bc_b, fragment, {"pylingual_failed_result": {"message": tr.message}}
            )
            self.assertIsNotNone(out)
            self.assertIn("21 items", out["operator"])

            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc"), doraise=True))
            self.assertEqual(_combined_distance(gt_pyc, fixed_pyc), 0)


if __name__ == "__main__":
    unittest.main()
