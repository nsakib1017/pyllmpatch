"""`<module>` must go through the best-of-N candidate machinery like every other target.

`CodeObjectRepairLoop._fix_fragment` (pipeline/code_object_repair_loop.py:2342) gates the
candidate path on `qualname != "<module>"`, so module targets fall through to
`generate_candidate` (SINGULAR, :2378) and get exactly one greedy shot. Every other target gets up
to 5 strategy-conditioned candidates.

Measured over the live run (results/experiment_outputs/20260713T212214Z): `candidate_results` is
EMPTY for all 3,035 module steps; module acceptance is 81/3035 = 2.7%. On the 418-file
module-only-residual cohort -- worth ~22 points of file-perfect ceiling -- 27% of rejections are
purely mechanical (256 parse failures + 279 compile failures + 30 localization), fatal only because
there is no second candidate.

This is the OTHER half of the fix: `utils/reattach_source_code_object.py` now consumes a candidate
list for module targets, but it can only do so if this layer produces one. Either half alone is a
no-op -- the same trap as the candidate_count clamp, which was capped in two independent places.

Acceptance is unchanged throughout: the oracle still decides which candidate lands.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pipeline.code_object_repair_loop as L


class _FixerWithCandidates:
    """Engine-shaped fixer exposing BOTH APIs, so we can tell which one the loop chose."""

    def __init__(self):
        self.generate_candidates_calls = []
        self.generate_candidate_calls = []

    def generate_candidates(self, *, qualname, gt_code_object, derived_code_object,
                            derived_source_fragment, repair_context=None):
        self.generate_candidates_calls.append(qualname)
        return [
            {"candidate_index": 0, "strategy": "retrieval_guided", "text": "X = 1\n"},
            {"candidate_index": 1, "strategy": "control_flow_residual", "text": "X = 2\n"},
        ]

    def generate_candidate(self, *, qualname, gt_code_object, derived_code_object,
                           derived_source_fragment, repair_context=None):
        self.generate_candidate_calls.append(qualname)
        return "X = 9\n"


def _loop(fixer):
    loop = L.CodeObjectRepairLoop.__new__(L.CodeObjectRepairLoop)
    loop.fixer = fixer
    return loop


class ModuleUsesCandidateMachineryTest(unittest.TestCase):
    def setUp(self):
        # Isolate from the deterministic-leaf short circuit: this test is about the routing gate,
        # not about operators. The operator path is covered by test_deterministic_dispatch_coverage.
        self._p = mock.patch.object(L, "SEMANTIC_DETERMINISTIC_OPERATORS", False)
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_module_target_uses_generate_candidates(self):
        fixer = _FixerWithCandidates()
        out = _loop(fixer)._fix_fragment("<module>", object(), object(), "X = 0\n", {})
        self.assertEqual(
            fixer.generate_candidates_calls, ["<module>"],
            "<module> must route to the best-of-N candidate generator, not the single-shot path",
        )
        self.assertEqual(fixer.generate_candidate_calls, [],
                         "<module> must NOT fall through to generate_candidate (singular)")
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2, "every generated candidate must reach the engine")

    def test_non_module_target_is_unchanged(self):
        # Regression guard: the fragment path must behave exactly as before.
        fixer = _FixerWithCandidates()
        out = _loop(fixer)._fix_fragment("<module>.f", object(), object(), "def f(): ...\n", {})
        self.assertEqual(fixer.generate_candidates_calls, ["<module>.f"])
        self.assertIsInstance(out, list)

    def test_falls_back_to_single_candidate_without_generate_candidates(self):
        # A fixer that only implements the singular API (e.g. OracleFragmentFixer) must still work.
        class _OnlySingle:
            def generate_candidate(self, *, qualname, gt_code_object, derived_code_object,
                                   derived_source_fragment, repair_context=None):
                return "X = 1\n"

        out = _loop(_OnlySingle())._fix_fragment("<module>", object(), object(), "X = 0\n", {})
        self.assertEqual(out, "X = 1\n")

    def test_module_with_no_derived_code_object_falls_back(self):
        # derived_code_object is None -> no bytecode to condition on -> single-shot is correct.
        fixer = _FixerWithCandidates()
        out = _loop(fixer)._fix_fragment("<module>", object(), None, "X = 0\n", {})
        self.assertEqual(fixer.generate_candidates_calls, [])
        self.assertEqual(out, "X = 9\n")


if __name__ == "__main__":
    unittest.main()
