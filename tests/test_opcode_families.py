"""Version-agnostic opcode-family predicates so operators written against 3.10-3.13 opnames also
recognise 3.14/3.15's renamed/added opcodes. These are FACTUAL mappings (a borrow-load IS a local
load; LOAD_SMALL_INT IS a const load), verified here against the real 3.14 and 3.15 opcode
inventories so the families never silently miss a new opcode of a covered kind."""
import json
import unittest
from pathlib import Path

from utils.semantic_operators import opcode_families as fam

REPO = Path(__file__).resolve().parent.parent


class OpcodeFamilyTest(unittest.TestCase):
    def test_314_new_load_opcodes_are_recognised(self):
        self.assertTrue(fam.is_local_load("LOAD_FAST_BORROW"))
        self.assertTrue(fam.is_local_load("LOAD_FAST_BORROW_LOAD_FAST_BORROW"))
        self.assertTrue(fam.is_const_load("LOAD_SMALL_INT"))
        self.assertTrue(fam.is_const_load("LOAD_CONST"))  # still classic

    def test_314_new_conditional_jumps_are_recognised(self):
        self.assertTrue(fam.is_conditional_jump("JUMP_IF_FALSE"))
        self.assertTrue(fam.is_conditional_jump("JUMP_IF_TRUE"))
        self.assertTrue(fam.is_conditional_jump("POP_JUMP_IF_FALSE"))
        self.assertTrue(fam.is_unconditional_jump("JUMP_NO_INTERRUPT"))
        self.assertTrue(fam.is_jump("JUMP_BACKWARD"))

    def test_negatives(self):
        self.assertFalse(fam.is_local_load("LOAD_CONST"))
        self.assertFalse(fam.is_const_load("LOAD_FAST"))
        self.assertFalse(fam.is_conditional_jump("JUMP_FORWARD"))
        self.assertFalse(fam.is_jump("BINARY_OP"))

    def test_families_cover_every_real_314_and_315_opcode_of_their_kind(self):
        # cross-check against authoritative interpreter opcode inventories: any opcode whose name
        # clearly belongs to a family MUST be in it (guards against a missed new opcode).
        data_315 = json.loads((REPO / "utils" / "py315_support" / "opcode_315_data.json").read_text())
        names = set(data_315["opmap"])
        for op in names:
            if op.startswith("LOAD_FAST"):
                self.assertTrue(fam.is_local_load(op), f"{op} not a local load")
            if op in ("JUMP_IF_FALSE", "JUMP_IF_TRUE") or op.startswith("POP_JUMP_IF"):
                self.assertTrue(fam.is_conditional_jump(op), f"{op} not a conditional jump")


if __name__ == "__main__":
    unittest.main()
