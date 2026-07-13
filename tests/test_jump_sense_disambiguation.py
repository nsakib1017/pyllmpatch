"""REGRESSION: the deterministic prepass already drives two clustered dead-pass-else nodes
(`if C: pass else: BODY` -> `if C: BODY`, the pure jump-flip idiom) to byte-perfect, oracle-gated,
with no LLM. (Investigation for #74 found this case is NOT a coverage gap: across 215 real
diverging objects only 1 is a clean pure-flip and it is already handled; the residual jump-opcode
differences are embedded in larger STRUCTURAL restructures — decompiler/LLM territory, not
deterministically fixable. So there is no jump operator to add; this pins that the clean case
stays covered.)"""
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual"))

import utils.reattach_source_code_object as R
from utils.generate_bytecode import compile_version

HOST = f"{sys.version_info[0]}.{sys.version_info[1]}"


def _scenario(gt_src: str, der_src: str) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "gt.py").write_text(gt_src)
    (d / "der.py").write_text(der_src)
    compile_version(str(d / "gt.py"), str(d / "gt.pyc"), HOST)
    compile_version(str(d / "der.py"), str(d / "der.pyc"), HOST)
    return d


class JumpSenseDisambiguationTest(unittest.TestCase):
    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE  # repair_* mutates the global; restore for isolation

    def tearDown(self):
        R.ACCEPTANCE_MODE = self._prev_mode

    def test_two_clustered_dead_pass_else_are_disambiguated(self):
        # GT: two plain `if C: BODY` (no else). Derived: the decompiler rendered BOTH as the
        # inverted dead-pass-else idiom -> two pure jump-flips in one fragment.
        gt = "def f(x, y):\n    if x:\n        z = 1\n    if y:\n        w = 3\n    return 0\n"
        der = (
            "def f(x, y):\n"
            "    if x:\n        pass\n    else:\n        z = 1\n"
            "    if y:\n        pass\n    else:\n        w = 3\n"
            "    return 0\n"
        )
        d = _scenario(gt, der)
        res = R.repair_mismatching_code_objects(
            d / "gt.pyc", d / "der.pyc", d / "der.py", gt_source=d / "gt.py",
            output_dir=d / "out", fragment_fixer=None, acceptance_mode="oracle",
        )
        # the deterministic prepass must collapse BOTH (disambiguating each by its failed offset)
        self.assertTrue(res["pylingual_verification"]["all_equal"],
                        "clustered dead-pass-else nodes were not disambiguated to byte-perfect")


if __name__ == "__main__":
    unittest.main()
