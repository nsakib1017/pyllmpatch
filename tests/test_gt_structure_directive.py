"""gt_structure_directive renders a GT-STRUCTURE-ONLY control-flow directive for the syntactic
repair prompt.

Syntactic repair operates on source that does NOT compile, so there is no derived code object
to diff against (unlike the semantic structural_repair_directive). This renders the ground
truth's own control-flow constructs (try/if/for/while/with, recovered from bc_to_cft) as
readable reconstruction clauses -- the info the offset-based brief ("loop @1..@318") lacks and
that the ~47% control-flow-damaged malware decompilations need. Here _constructs is patched so
the aggregation/format logic is tested hermetically; the bytecode extraction it wraps is
covered by the semantic structural-directive tests.
"""
import unittest
from unittest import mock

from utils.semantic_operators import structural_directive as SD


class GtStructureDirectiveTest(unittest.TestCase):
    def test_empty_when_no_constructs(self):
        with mock.patch.object(SD, "_constructs", return_value=[]):
            self.assertEqual(SD.gt_structure_directive(object()), "")

    def test_single_construct_is_rendered(self):
        recs = [{"kind": "if", "sig": "if::x>0", "cond": "x > 0", "body_desc": "y = f()", "offset": 4}]
        with mock.patch.object(SD, "_constructs", return_value=recs):
            out = SD.gt_structure_directive(object())
        self.assertIn("if x > 0", out)
        self.assertTrue(out.strip().endswith("."))

    def test_multiple_constructs_numbered_and_capped(self):
        recs = [
            {"kind": "for", "sig": "for::i::items", "target": "i", "iterable": "items", "body_desc": None, "offset": 2},
            {"kind": "while", "sig": "while::run", "cond": "run", "body_desc": None, "offset": 8},
            {"kind": "if", "sig": "if::a", "cond": "a", "body_desc": None, "offset": 12},
            {"kind": "if", "sig": "if::b", "cond": "b", "body_desc": None, "offset": 20},
            {"kind": "if", "sig": "if::c", "cond": "c", "body_desc": None, "offset": 30},
        ]
        with mock.patch.object(SD, "_constructs", return_value=recs):
            out = SD.gt_structure_directive(object(), max_clauses=3)
        self.assertIn("(1)", out)
        self.assertIn("(3)", out)
        self.assertNotIn("(4)", out)  # capped at 3

    def test_never_raises_returns_empty(self):
        with mock.patch.object(SD, "_constructs", side_effect=RuntimeError("boom")):
            self.assertEqual(SD.gt_structure_directive(object()), "")


if __name__ == "__main__":
    unittest.main()
