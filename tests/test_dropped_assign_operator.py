"""Unit tests for the dropped-scalar-literal-assignment operator
(utils/semantic_operators/placeholder.dropped_assignment_candidate).

No LLM. GT/derived bytecode is supplied as explicit instruction lists (matching the
shapes pylingual's editable bytecode yields). The operator NEVER emits a value it did
not read verbatim from GT: it enumerates GT ``LOAD_CONST <literal>; STORE <name>``
targets, keeps only those absent from the derived object's stores, requires EXACTLY ONE
survivor, then splices ``name = repr(value)`` next to the uniquely-mappable GT store
neighbour in source. Any ambiguity -> None."""
import unittest

from utils.semantic_operators.placeholder import (
    LINE_TOO_LONG_MARKER,
    dropped_assignment_candidate,
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


def _ctx(message="Different bytecode", offset=None):
    return {"pylingual_failed_result": {"message": message}, "failed_offset": offset}


def _store(name):
    return _Inst(0, "STORE_NAME", name, name, 0)


def _der_stores(*names):
    return _BC([_Inst(0, "STORE_NAME", n, n, 0) for n in names]
               + [_Inst(0, "RETURN_CONST", "None", None, 0)])


class DroppedAssignPositiveTest(unittest.TestCase):
    def test_happy_path_insert_before_successor(self):
        # GT stores X (=6.0) then Y (=1); derived only stores Y. X is the single dropped
        # literal assignment. X has no preceding store, so it anchors on its successor Y
        # (present + unique in source) and is spliced BEFORE `Y = 1`.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "6.0", 6.0, 0),
            _Inst(2, "STORE_NAME", "X", "X", 0),
            _Inst(4, "LOAD_CONST", "1", 1, 1),
            _Inst(6, "STORE_NAME", "Y", "Y", 1),
            _Inst(8, "RETURN_CONST", "None", None, 2),
        ])
        der = _der_stores("Y")
        out = dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "X = 6.0\nY = 1\n")
        self.assertEqual(out["opname"], "STORE_NAME")
        self.assertEqual(out["kind"], "dropped_assign")
        self.assertEqual(out["confidence"], "unique")
        self.assertTrue(out["operator"].startswith("insert X = "))
        compile(out["text"], "<m>", "exec")

    def test_string_value_verbatim(self):
        author = "Putiatin Philipp: the great decompiler tester"
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'name'", "name", 0),
            _Inst(2, "STORE_NAME", "__name__slot", "__name__slot", 0),  # not a dunder
            _Inst(4, "LOAD_CONST", repr(author), author, 1),
            _Inst(6, "STORE_NAME", "__author__", "__author__", 1),
            _Inst(8, "RETURN_CONST", "None", None, 2),
        ])
        # __author__ is a real source assignment, NOT a synthesized compiler dunder.
        der = _der_stores("__name__slot")
        frag = "__name__slot = 'name'\n"
        out = dropped_assignment_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "__name__slot = 'name'\n__author__ = %r\n" % author)
        compile(out["text"], "<m>", "exec")


class DroppedAssignSpliceTest(unittest.TestCase):
    def test_insert_after_preceding_neighbor(self):
        # GT: A=0 ; X=6.0 ; Y=1.  X missing.  X's preceding store is A (present + unique in
        # source) -> X spliced immediately AFTER the `A = 0` statement.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "0", 0, 0),
            _Inst(2, "STORE_NAME", "A", "A", 0),
            _Inst(4, "LOAD_CONST", "6.0", 6.0, 1),
            _Inst(6, "STORE_NAME", "X", "X", 1),
            _Inst(8, "LOAD_CONST", "1", 1, 2),
            _Inst(10, "STORE_NAME", "Y", "Y", 2),
            _Inst(12, "RETURN_CONST", "None", None, 3),
        ])
        der = _der_stores("A", "Y")
        frag = "import os\nA = 0\nY = 1\n"
        out = dropped_assignment_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "import os\nA = 0\nX = 6.0\nY = 1\n")
        compile(out["text"], "<m>", "exec")

    def test_none_when_neighbor_missing_in_source(self):
        # Preceding neighbor A is not present as an assignment in the source fragment.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "0", 0, 0),
            _Inst(2, "STORE_NAME", "A", "A", 0),
            _Inst(4, "LOAD_CONST", "6.0", 6.0, 1),
            _Inst(6, "STORE_NAME", "X", "X", 1),
            _Inst(8, "LOAD_CONST", "1", 1, 2),
            _Inst(10, "STORE_NAME", "Y", "Y", 2),
            _Inst(12, "RETURN_CONST", "None", None, 3),
        ])
        der = _der_stores("A", "Y")
        self.assertIsNone(dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx()))

    def test_none_when_neighbor_non_unique_in_source(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "0", 0, 0),
            _Inst(2, "STORE_NAME", "A", "A", 0),
            _Inst(4, "LOAD_CONST", "6.0", 6.0, 1),
            _Inst(6, "STORE_NAME", "X", "X", 1),
            _Inst(8, "LOAD_CONST", "1", 1, 2),
            _Inst(10, "STORE_NAME", "Y", "Y", 2),
            _Inst(12, "RETURN_CONST", "None", None, 3),
        ])
        der = _der_stores("A", "Y")
        # A assigned twice -> not a unique anchor.
        self.assertIsNone(
            dropped_assignment_candidate(gt, der, "A = 0\nA = 9\nY = 1\n", _ctx()))


class DroppedAssignDeclineTest(unittest.TestCase):
    def test_ambiguous_two_absent_targets(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "6.0", 6.0, 0),
            _Inst(2, "STORE_NAME", "X", "X", 0),
            _Inst(4, "LOAD_CONST", "2.0", 2.0, 1),
            _Inst(6, "STORE_NAME", "Z", "Z", 1),
            _Inst(8, "LOAD_CONST", "1", 1, 2),
            _Inst(10, "STORE_NAME", "Y", "Y", 2),
            _Inst(12, "RETURN_CONST", "None", None, 3),
        ])
        der = _der_stores("Y")
        self.assertIsNone(dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx()))

    def test_compiler_dunder_is_skipped(self):
        for dunder in ("__doc__", "__module__", "__qualname__", "__annotations__",
                       "__static_attributes__", "__firstlineno__"):
            gt = _BC([
                _Inst(0, "LOAD_CONST", "'d'", "d", 0),
                _Inst(2, "STORE_NAME", dunder, dunder, 0),
                _Inst(4, "LOAD_CONST", "1", 1, 1),
                _Inst(6, "STORE_NAME", "Y", "Y", 1),
                _Inst(8, "RETURN_CONST", "None", None, 2),
            ])
            der = _der_stores("Y")
            self.assertIsNone(
                dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx()),
                f"{dunder} must be skipped")

    def test_annotated_assignment_defers(self):
        # GT: X: tuple = 6.0 style -> __annotations__['X'] store present -> defer.
        gt = _BC([
            _Inst(0, "SETUP_ANNOTATIONS", ""),
            _Inst(2, "LOAD_CONST", "6.0", 6.0, 0),
            _Inst(4, "STORE_NAME", "X", "X", 0),
            _Inst(6, "LOAD_NAME", "float", "float", 1),
            _Inst(8, "LOAD_NAME", "__annotations__", "__annotations__", 2),
            _Inst(10, "LOAD_CONST", "X", "X", 1),
            _Inst(12, "STORE_SUBSCR", ""),
            _Inst(14, "LOAD_CONST", "1", 1, 2),
            _Inst(16, "STORE_NAME", "Y", "Y", 2),
            _Inst(18, "RETURN_CONST", "None", None, 3),
        ])
        der = _der_stores("Y")
        self.assertIsNone(dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx()))

    def test_non_round_trippable_inf(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "inf", float("inf"), 0),
            _Inst(2, "STORE_NAME", "X", "X", 0),
            _Inst(4, "LOAD_CONST", "1", 1, 1),
            _Inst(6, "STORE_NAME", "Y", "Y", 1),
            _Inst(8, "RETURN_CONST", "None", None, 2),
        ])
        der = _der_stores("Y")
        self.assertIsNone(dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx()))

    def test_non_round_trippable_code_object(self):
        code_val = compile("x + 1", "<m>", "eval")
        gt = _BC([
            _Inst(0, "LOAD_CONST", "<code>", code_val, 0),
            _Inst(2, "STORE_NAME", "X", "X", 0),
            _Inst(4, "LOAD_CONST", "1", 1, 1),
            _Inst(6, "STORE_NAME", "Y", "Y", 1),
            _Inst(8, "RETURN_CONST", "None", None, 2),
        ])
        der = _der_stores("Y")
        self.assertIsNone(dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx()))

    def test_list_extend_idiom_is_not_a_scalar_drop(self):
        # BUILD_LIST 0 / LOAD_CONST tuple / LIST_EXTEND 1 / STORE is NOT a scalar literal
        # store; this operator declines it (owned by line_too_long_candidate's list path).
        gt = _BC([
            _Inst(0, "BUILD_LIST", "", 0, 0),
            _Inst(2, "LOAD_CONST", "(1, 2)", (1, 2), 0),
            _Inst(4, "LIST_EXTEND", "", 1, 1),
            _Inst(6, "STORE_NAME", "P", "P", 0),
            _Inst(8, "LOAD_CONST", "1", 1, 1),
            _Inst(10, "STORE_NAME", "Y", "Y", 1),
            _Inst(12, "RETURN_CONST", "None", None, 2),
        ])
        der = _der_stores("Y")
        self.assertIsNone(dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx()))

    def test_wrong_failure_regime_defers(self):
        gt = _BC([
            _Inst(0, "LOAD_CONST", "6.0", 6.0, 0),
            _Inst(2, "STORE_NAME", "X", "X", 0),
            _Inst(4, "LOAD_CONST", "1", 1, 1),
            _Inst(6, "STORE_NAME", "Y", "Y", 1),
            _Inst(8, "RETURN_CONST", "None", None, 2),
        ])
        der = _der_stores("Y")
        self.assertIsNone(
            dropped_assignment_candidate(gt, der, "Y = 1\n", _ctx("Different control flow")))
        self.assertIsNone(dropped_assignment_candidate(gt, der, "Y = 1\n", None))

    def test_hand_off_line_too_long_marker(self):
        # A fragment that still carries the line-too-long marker stays owned by
        # line_too_long_candidate; this operator declines.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "6.0", 6.0, 0),
            _Inst(2, "STORE_NAME", "X", "X", 0),
            _Inst(4, "LOAD_CONST", "1", 1, 1),
            _Inst(6, "STORE_NAME", "Y", "Y", 1),
            _Inst(8, "RETURN_CONST", "None", None, 2),
        ])
        der = _der_stores("Y")
        frag = '"""' + LINE_TOO_LONG_MARKER + '"""\nY = 1\n'
        self.assertIsNone(dropped_assignment_candidate(gt, der, frag, _ctx()))

    def test_target_present_in_derived_defers(self):
        # If the derived object already stores X, X was not dropped -> nothing absent.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "6.0", 6.0, 0),
            _Inst(2, "STORE_NAME", "X", "X", 0),
            _Inst(4, "LOAD_CONST", "1", 1, 1),
            _Inst(6, "STORE_NAME", "Y", "Y", 1),
            _Inst(8, "RETURN_CONST", "None", None, 2),
        ])
        der = _der_stores("X", "Y")
        self.assertIsNone(dropped_assignment_candidate(gt, der, "X = 6.0\nY = 1\n", _ctx()))


class DroppedAssignClassBodyTest(unittest.TestCase):
    def test_class_body_none_drop_with_indentation(self):
        # Real-world shape: a class body where `HAS_NOT_MET = None` was dropped. GT class
        # body stores __module__, __qualname__, HAS_MET, HAS_NOT_MET.
        gt = _BC([
            _Inst(0, "LOAD_CONST", "'m'", "m", 0),
            _Inst(2, "STORE_NAME", "__module__", "__module__", 0),
            _Inst(4, "LOAD_CONST", "'C'", "C", 1),
            _Inst(6, "STORE_NAME", "__qualname__", "__qualname__", 1),
            _Inst(8, "LOAD_CONST", "None", None, 2),
            _Inst(10, "STORE_NAME", "HAS_MET", "HAS_MET", 2),
            _Inst(12, "LOAD_CONST", "None", None, 3),
            _Inst(14, "STORE_NAME", "HAS_NOT_MET", "HAS_NOT_MET", 3),
            _Inst(16, "RETURN_CONST", "None", None, 4),
        ])
        der = _BC([
            _Inst(0, "STORE_NAME", "__module__", "__module__", 0),
            _Inst(2, "STORE_NAME", "__qualname__", "__qualname__", 1),
            _Inst(4, "STORE_NAME", "HAS_MET", "HAS_MET", 2),
            _Inst(6, "RETURN_CONST", "None", None, 3),
        ])
        frag = "class C:\n    HAS_MET = None\n"
        out = dropped_assignment_candidate(gt, der, frag, _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "class C:\n    HAS_MET = None\n    HAS_NOT_MET = None\n")
        compile(out["text"], "<m>", "exec")


if __name__ == "__main__":
    unittest.main()


class DroppedAssignBoundaryFixTest(unittest.TestCase):
    def test_declines_conditional_rhs_ending_in_const(self):
        """Review fix: a dropped `K = a if c else 99` ends in LOAD_CONST 99 but is NOT a
        scalar assignment — the const-load is reached via a branch, not a statement start,
        so _gt_literal_store_targets must not classify it as a scalar target."""
        from utils.semantic_operators.placeholder import _gt_literal_store_targets
        import dis
        code = compile("K = a if c else 99", "<t>", "exec")
        scalars = [t for t in _gt_literal_store_targets(list(dis.get_instructions(code))) if t[2] == "scalar"]
        self.assertEqual(scalars, [], f"conditional RHS must not be a scalar target; got {scalars}")

    def test_still_accepts_plain_scalar(self):
        from utils.semantic_operators.placeholder import _gt_literal_store_targets
        import dis
        code = compile("K = 99", "<t>", "exec")
        scalars = [t for t in _gt_literal_store_targets(list(dis.get_instructions(code))) if t[2] == "scalar"]
        self.assertEqual([(n, v) for n, v, _ in scalars], [("K", 99)])
