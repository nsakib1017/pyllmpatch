"""Module repair must consume ALL fixer candidates, not just the first.

`fragment_fixer` returns a LIST of candidates. The fragment path honours that
(`reattach_source_code_object.py:4573` -> `raw_candidates`), but the module path collapses it to a
single value (`:4412` -> `replacement_text`) and hands one candidate to
`_apply_module_statement_candidate` (`:4462`). Module repair is therefore SINGLE-SHOT.

Measured consequence over the live run (results/experiment_outputs/20260713T212214Z):
`candidate_results` is EMPTY for all 3,035 module steps, module acceptance is 81/3035 = 2.7%, and on
the 418-file module-only-residual cohort 27% of rejections are purely mechanical -- 256 "source
parsing failed" + 279 "compilation failed" + 30 "could not localize". Those are fatal only because
there is no second candidate: one unparseable generation loses the whole file.

That cohort is worth ~22 points of file-perfect ceiling, so these tests pin the fix: every candidate
gets a turn, and a candidate that cannot parse or compile costs one candidate rather than the file.
Acceptance itself is unchanged -- the oracle gate still decides.
"""
from __future__ import annotations

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

# A module-level constant divergence: the residual lives in <module>, which routes to the
# repair_module_statement branch rather than repair_source_fragment.
GT_SRC = "X = 1\n"
DER_SRC = "X = 2\n"

# Every deterministic prepass operator must defer, otherwise the prepass fixes this before the
# module branch is ever reached and the test would pass vacuously.
_PREPASS_OPERATORS = (
    "leaf_value_candidate", "collapse_dead_pass_else_candidate", "leaf_value_no_offset_candidate",
    "literal_form_candidate", "container_construction_candidate", "call_shape_candidate",
    "unary_op_candidate", "unpack_op_candidate", "decorator_default_candidate",
    "dropped_assignment_candidate", "line_too_long_candidate", "fstring_segment_candidate",
    "comprehension_op_candidate",
)


def _scenario(gt_src: str = GT_SRC, der_src: str = DER_SRC) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "gt.py").write_text(gt_src)
    (d / "der.py").write_text(der_src)
    compile_version(str(d / "gt.py"), str(d / "gt.pyc"), HOST)
    compile_version(str(d / "der.py"), str(d / "der.pyc"), HOST)
    return d


class _DeferPrepass(unittest.TestCase):
    """Force the LLM module branch by making every deterministic operator defer."""

    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE
        self._patches = []
        for name in _PREPASS_OPERATORS:
            if hasattr(R, name):
                p = mock.patch.object(R, name, lambda *a, **k: None)
                p.start()
                self._patches.append(p)

    def tearDown(self):
        for p in self._patches:
            p.stop()
        R.ACCEPTANCE_MODE = self._prev_mode

    def _repair(self, d: Path, fixer) -> dict:
        return R.repair_mismatching_code_objects(
            d / "gt.pyc", d / "der.pyc", d / "der.py",
            output_dir=d / "out", fragment_fixer=fixer, acceptance_mode="oracle",
        )


class ModuleBestOfNTest(_DeferPrepass):
    def test_second_candidate_is_tried_when_first_cannot_parse(self):
        # The decisive case: candidate 1 is unparseable garbage, candidate 2 is GT-correct.
        # Single-shot loses the file; best-of-N must reach candidate 2 and land it.
        d = _scenario()
        fixer = lambda *a, **k: ["X = (((\n", "X = 1\n"]
        res = self._repair(d, fixer)
        self.assertTrue(
            (res.get("pylingual_verification") or {}).get("all_equal"),
            "a GT-correct 2nd candidate must be tried after the 1st fails to parse",
        )

    def test_second_candidate_is_tried_when_first_does_not_compile(self):
        d = _scenario()
        fixer = lambda *a, **k: ["X = 1 +\n", "X = 1\n"]
        res = self._repair(d, fixer)
        self.assertTrue(
            (res.get("pylingual_verification") or {}).get("all_equal"),
            "a GT-correct 2nd candidate must be tried after the 1st fails to compile",
        )

    def test_second_candidate_is_tried_when_first_is_merely_wrong(self):
        # Candidate 1 parses and compiles but is semantically wrong -> the oracle rejects it.
        # The engine must move on rather than burn the whole attempt.
        d = _scenario()
        fixer = lambda *a, **k: ["X = 99\n", "X = 1\n"]
        res = self._repair(d, fixer)
        self.assertTrue(
            (res.get("pylingual_verification") or {}).get("all_equal"),
            "a GT-correct 2nd candidate must be tried after the 1st is oracle-rejected",
        )

    def test_module_step_records_candidate_results(self):
        # Module steps currently record NO candidate_results (0 across 3,035 live steps), which is
        # why module repair is invisible to every best-of-N diagnostic we have.
        d = _scenario()
        fixer = lambda *a, **k: ["X = 99\n", "X = 1\n"]
        res = self._repair(d, fixer)
        mod_steps = [s for s in res.get("steps", [])
                     if s.get("repair_operation") == "repair_module_statement"]
        self.assertTrue(mod_steps, "the module branch must have run")
        self.assertTrue(
            any(s.get("candidate_results") for s in mod_steps),
            "module steps must record per-candidate results like the fragment path does",
        )


class ModuleSingleCandidateUnchangedTest(_DeferPrepass):
    def test_single_correct_candidate_still_accepted(self):
        # Regression guard: the common one-candidate case must behave exactly as before.
        d = _scenario()
        res = self._repair(d, lambda *a, **k: ["X = 1\n"])
        self.assertTrue((res.get("pylingual_verification") or {}).get("all_equal"))

    def test_all_candidates_bad_is_rejected_not_crashed(self):
        # No candidate works -> the file stays unrepaired, recorded, and the loop does NOT explode.
        d = _scenario()
        res = self._repair(d, lambda *a, **k: ["X = (((\n", "X = 77\n"])
        self.assertFalse((res.get("pylingual_verification") or {}).get("all_equal"))
        self.assertTrue(res.get("module_rejected_attempts"),
                        "exhausted module candidates must be recorded as rejected attempts")


if __name__ == "__main__":
    unittest.main()
