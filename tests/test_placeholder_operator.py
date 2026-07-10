"""Unit tests for the line-too-long placeholder operator
(utils/semantic_operators/placeholder.py). No LLM; GT/derived bytecode is supplied
as explicit instruction lists (matching the shapes pylingual's editable bytecode
yields) and every emitted fragment is checked to recompile."""
import os
import tempfile
import unittest
from pathlib import Path

from utils.semantic_operators.placeholder import (
    LINE_TOO_LONG_MARKER,
    line_too_long_candidate,
)


class _Inst:
    def __init__(self, offset, opname, argrepr="", argval=None, arg=None):
        self.offset, self.opname, self.argrepr = offset, opname, argrepr
        self.argval, self.arg = argval, arg


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


def _ctx(offset=None):
    return {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": offset}


# A derived module whose deferred statement became the marker docstring
# (LOAD_CONST marker; STORE_NAME __doc__).
DER = _BC([
    _Inst(0, "LOAD_CONST", repr(LINE_TOO_LONG_MARKER), LINE_TOO_LONG_MARKER),
    _Inst(2, "STORE_NAME", "__doc__", "__doc__"),
    _Inst(4, "RETURN_CONST", "None", None),
])

MARKER_FRAG = '"""' + LINE_TOO_LONG_MARKER + '"""\n'


def _gt_scalar(name, value):
    return _BC([
        _Inst(0, "LOAD_CONST", repr(value), value, 0),
        _Inst(2, "STORE_NAME", name, name, 0),
        _Inst(4, "RETURN_CONST", "None", None, 1),
    ])


def _gt_list_extend(name, value):
    return _BC([
        _Inst(0, "BUILD_LIST", "", 0, 0),
        _Inst(2, "LOAD_CONST", repr(value), value, 0),
        _Inst(4, "LIST_EXTEND", "", 1, 1),
        _Inst(6, "STORE_NAME", name, name, 0),
        _Inst(8, "RETURN_CONST", "None", None, 1),
    ])


class PlaceholderPositiveTest(unittest.TestCase):
    def test_bare_tuple(self):
        out = line_too_long_candidate(_gt_scalar("X", (1, 2, 3)), DER, MARKER_FRAG, _ctx(0))
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "X = (1, 2, 3)\n")
        compile(out["text"], "<m>", "exec")  # recompiles

    def test_list_extend_idiom(self):
        out = line_too_long_candidate(_gt_list_extend("PAYLOAD", (1, 2, 3)), DER, MARKER_FRAG, _ctx(None))
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "PAYLOAD = [1, 2, 3]\n")
        compile(out["text"], "<m>", "exec")

    def test_string_literal_value(self):
        out = line_too_long_candidate(_gt_scalar("SECRET", "abc" * 40), DER, MARKER_FRAG, _ctx(0))
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "SECRET = %r\n" % ("abc" * 40))
        compile(out["text"], "<m>", "exec")

    def test_preserves_indentation_and_surrounding_lines(self):
        frag = (
            "class C:\n"
            "    a = 1\n"
            '    """' + LINE_TOO_LONG_MARKER + '"""\n'
            "    b = 2\n"
        )
        # STORE_FAST-free class body; only the marker line is rewritten.
        out = line_too_long_candidate(_gt_scalar("jobs", (7, 8)), DER, frag, _ctx(0))
        self.assertIsNotNone(out)
        self.assertEqual(
            out["text"],
            "class C:\n    a = 1\n    jobs = (7, 8)\n    b = 2\n",
        )
        compile(out["text"], "<m>", "exec")


class PlaceholderDeclineTest(unittest.TestCase):
    def test_multi_instruction_non_literal_call(self):
        # GT: X = foo()  ->  STORE preceded by CALL, not a literal store.
        gt = _BC([
            _Inst(0, "PUSH_NULL", ""),
            _Inst(2, "LOAD_NAME", "foo", "foo"),
            _Inst(4, "CALL", "", 0, 0),
            _Inst(6, "STORE_NAME", "X", "X"),
            _Inst(8, "RETURN_CONST", "None", None),
        ])
        self.assertIsNone(line_too_long_candidate(gt, DER, MARKER_FRAG, _ctx(0)))

    def test_non_round_trippable_inf(self):
        gt = _gt_scalar("X", float("inf"))
        self.assertIsNone(line_too_long_candidate(gt, DER, MARKER_FRAG, _ctx(0)))

    def test_non_round_trippable_code_object(self):
        code_val = compile("x + 1", "<m>", "eval")
        gt = _gt_scalar("X", code_val)
        self.assertIsNone(line_too_long_candidate(gt, DER, MARKER_FRAG, _ctx(0)))

    def test_marker_must_appear_exactly_once(self):
        gt = _gt_scalar("X", (1, 2, 3))
        two = MARKER_FRAG + '"""' + LINE_TOO_LONG_MARKER + '"""\n'
        self.assertIsNone(line_too_long_candidate(gt, DER, two, _ctx(0)))
        self.assertIsNone(line_too_long_candidate(gt, DER, "y = 1\n", _ctx(0)))

    def test_marker_not_a_bare_statement(self):
        # marker as a call argument, not a standalone Expr statement -> defer.
        frag = "print(%r)\n" % LINE_TOO_LONG_MARKER
        gt = _gt_scalar("X", (1, 2, 3))
        self.assertIsNone(line_too_long_candidate(gt, DER, frag, _ctx(0)))

    def test_ambiguous_multiple_literal_stores(self):
        # Two literal stores in GT, neither in derived -> cannot align to one marker.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "(1, 2)", (1, 2), 0),
            _Inst(2, "STORE_NAME", "A", "A", 0),
            _Inst(4, "LOAD_CONST", "(3, 4)", (3, 4), 1),
            _Inst(6, "STORE_NAME", "B", "B", 1),
            _Inst(8, "RETURN_CONST", "None", None, 2),
        ])
        self.assertIsNone(line_too_long_candidate(gt, DER, MARKER_FRAG, _ctx(0)))

    def test_target_also_stored_in_derived_defers(self):
        # If the derived object already stores X, X is not the deferred name -> defer.
        der = _BC([
            _Inst(0, "LOAD_CONST", "5", 5, 0),
            _Inst(2, "STORE_NAME", "X", "X", 0),
            _Inst(4, "RETURN_CONST", "None", None, 1),
        ])
        gt = _gt_scalar("X", (1, 2, 3))
        self.assertIsNone(line_too_long_candidate(gt, der, MARKER_FRAG, _ctx(0)))

    def test_annotated_assignment_defers(self):
        # GT: X: tuple = (1, 2, 3)  emits SETUP_ANNOTATIONS + __annotations__['X'] store.
        gt = _BC([
            _Inst(0, "SETUP_ANNOTATIONS", ""),
            _Inst(2, "LOAD_CONST", "(1, 2, 3)", (1, 2, 3), 0),
            _Inst(4, "STORE_NAME", "X", "X", 0),
            _Inst(6, "LOAD_NAME", "tuple", "tuple", 1),
            _Inst(8, "LOAD_NAME", "__annotations__", "__annotations__", 2),
            _Inst(10, "LOAD_CONST", "X", "X", 1),
            _Inst(12, "STORE_SUBSCR", ""),
            _Inst(14, "RETURN_CONST", "None", None, 2),
        ])
        self.assertIsNone(line_too_long_candidate(gt, DER, MARKER_FRAG, _ctx(0)))

    def test_wrong_failure_regime_defers(self):
        gt = _gt_scalar("X", (1, 2, 3))
        ctx = {"pylingual_failed_result": {"message": "Different control flow"}, "failed_offset": None}
        self.assertIsNone(line_too_long_candidate(gt, DER, MARKER_FRAG, ctx))
        self.assertIsNone(line_too_long_candidate(gt, DER, MARKER_FRAG, None))


class PlaceholderPrepassIntegrationTest(unittest.TestCase):
    """End-to-end: build a GT pyc and a marker-only derived pyc, run the real
    pylingual compare_pyc, apply the operator to the derived fragment, recompile,
    and confirm the <module> object is no longer reported as a failure."""

    def test_recompiled_candidate_matches_gt(self):
        try:
            from pylingual.equivalence_check import compare_pyc  # noqa: F401
        except Exception as exc:  # pragma: no cover - pylingual missing
            self.skipTest(f"pylingual unavailable: {exc}")
        import py_compile

        from pylingual.equivalence_check import compare_pyc

        with tempfile.TemporaryDirectory() as d:
            gt_py = os.path.join(d, "gt.py")
            der_py = os.path.join(d, "der.py")
            Path(gt_py).write_text("X = (1, 2, 3)\n", encoding="utf-8")
            Path(der_py).write_text(MARKER_FRAG, encoding="utf-8")
            gt_pyc = Path(py_compile.compile(gt_py, cfile=os.path.join(d, "gt.pyc")))
            der_pyc = Path(py_compile.compile(der_py, cfile=os.path.join(d, "der.pyc")))

            failures = [tr for tr in compare_pyc(gt_pyc, der_pyc) if not tr.success]
            self.assertTrue(failures, "expected a module-level bytecode divergence")
            tr = failures[0]
            self.assertEqual(tr.message, "Different bytecode")

            ctx = {
                "pylingual_failed_result": {"message": tr.message},
                "failed_offset": tr.failed_offset,
            }
            out = line_too_long_candidate(tr.bc_a, tr.bc_b, MARKER_FRAG, ctx)
            self.assertIsNotNone(out)
            self.assertEqual(out["text"], "X = (1, 2, 3)\n")

            fixed_py = os.path.join(d, "fixed.py")
            Path(fixed_py).write_text(out["text"], encoding="utf-8")
            fixed_pyc = Path(py_compile.compile(fixed_py, cfile=os.path.join(d, "fixed.pyc")))
            self.assertTrue(all(tr2.success for tr2 in compare_pyc(gt_pyc, fixed_pyc)))


if __name__ == "__main__":
    unittest.main()
