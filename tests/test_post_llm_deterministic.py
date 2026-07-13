"""Post-LLM deterministic reconciliation: after each LLM iteration, if not perfect, re-run the
oracle-gated deterministic operators on the residual. Flag SEMANTIC_POST_LLM_DETERMINISTIC
(default off) + hard-require ACCEPTANCE_MODE=='oracle'. These tests drive the integration with
synthetic pycs and no live LLM (the pre-LLM prepass / oracle fixer are the testable seams)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class _ModeIsolatedTest(unittest.TestCase):
    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE  # repair_* mutates the global; restore for isolation

    def tearDown(self):
        R.ACCEPTANCE_MODE = self._prev_mode


class PostLLMFlagTest(_ModeIsolatedTest):
    def test_flag_off_is_pure_noop(self):
        d = _scenario("def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n")
        with mock.patch.object(R, "SEMANTIC_POST_LLM_DETERMINISTIC", False):
            res = R.repair_mismatching_code_objects(
                d / "gt.pyc", d / "der.pyc", d / "der.py",
                output_dir=d / "out", fragment_fixer=None, acceptance_mode="oracle",
            )
        # the run-level attribution keys must exist and report no post-LLM activity
        self.assertIn("post_llm_deterministic_edits", res)
        self.assertIn("post_llm_deterministic_enabled", res)
        self.assertEqual(res["post_llm_deterministic_edits"], 0)
        self.assertFalse(res["post_llm_deterministic_enabled"])
        self.assertFalse(any(s.get("phase") == "post_llm" for s in res.get("steps", [])))

    def test_requires_oracle_mode(self):
        # flag ON but distance mode -> feature must NOT enable (closes distance-mode thrash)
        d = _scenario("def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n")
        with mock.patch.object(R, "SEMANTIC_POST_LLM_DETERMINISTIC", True):
            res = R.repair_mismatching_code_objects(
                d / "gt.pyc", d / "der.pyc", d / "der.py",
                output_dir=d / "out", fragment_fixer=None, acceptance_mode="distance",
            )
        self.assertFalse(res["post_llm_deterministic_enabled"])
        self.assertEqual(res["post_llm_deterministic_edits"], 0)


def _partial_fixer(qualname, gt_bytecode, derived_bytecode, extracted_before, repair_context):
    """Engine-shaped fixer callable: fix only `a` (return the correct GT fragment); no-op every
    other object by echoing its derived fragment, so `b` stays diverged for the post-LLM det pass."""
    if qualname.split(".")[-1] == "a":
        return ["def a(x):\n    return x + 1\n"]
    return [extracted_before]


class PostLLMFiringTest(_ModeIsolatedTest):
    def test_post_det_reconciles_residual_after_llm_accept(self):
        # two diverged leaf objects; the fixer fixes only `a`, leaving `b` for the post-LLM det pass.
        gt = "def a(x):\n    return x + 1\n\ndef b(x):\n    return x + 2\n"
        der = "def a(x):\n    return x - 1\n\ndef b(x):\n    return x - 2\n"

        def run(flag_on: bool):
            d = _scenario(gt, der)
            # disable the PRE-LLM prepass so both leaves survive to the loop (else it fixes both
            # before the LLM runs and the hook never gets a residual to prove itself on)
            with mock.patch.object(R, "SEMANTIC_DETERMINISTIC_PREPASS", False), \
                 mock.patch.object(R, "SEMANTIC_POST_LLM_DETERMINISTIC", flag_on):
                return R.repair_mismatching_code_objects(
                    d / "gt.pyc", d / "der.pyc", d / "der.py", gt_source=d / "gt.py",
                    output_dir=d / "out", fragment_fixer=_partial_fixer, acceptance_mode="oracle",
                    max_iterations=1,
                )

        off = run(False)
        on = run(True)
        # flag OFF: the LLM fixed `a` but `b`'s leaf residual survives -> not perfect
        self.assertFalse(off["pylingual_verification"]["all_equal"])
        self.assertEqual(off["post_llm_deterministic_edits"], 0)
        # flag ON: the post-LLM deterministic pass mops up `b` -> file-perfect
        self.assertTrue(on["pylingual_verification"]["all_equal"])
        self.assertGreater(on["post_llm_deterministic_edits"], 0)
        self.assertTrue(any(s.get("phase") == "post_llm" for s in on["steps"]))


if __name__ == "__main__":
    unittest.main()
