"""The firing audit must distinguish "deferred" from "fired but rejected".

`tools/deterministic_firing_audit.py:100` counts an operator as having FIRED only when
`s.get("accepted")` is true, then line 118 sweeps every other object into `deferred` -- a bucket
its own docstring declares "SAFE: the operator gates declined an ambiguous case ... NOT a bug".

That inference is unsound. Before #92 the engine discarded rejected prepass candidates silently
(reattach:4131 compile / :4138 gate both `continue` after `step_index -= 1`), so `deferred` was
really {operator returned None} UNION {operator fired, produced a candidate, and lost}. Those two
have opposite meanings: the first says the gate was appropriately conservative, the second says the
operator IS firing and its output is wrong or unhelpful. Reading the union as "safe" is how a
firing-but-broken operator hides, and it is the ambiguity that refuted two prior repair designs.

`repair_mismatching_code_objects` now returns `prepass_rejected_attempts` (see
tests/test_prepass_reject_instrumentation.py). These tests pin the audit consuming it: an object
whose operator fired and was rejected must be reported as `fired_rejected` with its cause, never as
`deferred`. Classification only -- no engine behaviour changes.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.deterministic_firing_audit import classify_objects


class ClassifyObjectsTest(unittest.TestCase):
    def test_accepted_operator_counts_as_fired(self):
        counts, _ = classify_objects(
            init={"<module>.f": "leaf"},
            steps=[{"qualname": "<module>.f", "accepted": True,
                    "repair_operation": "deterministic_prepass_leaf",
                    "selected_candidate_strategy": "deterministic_leaf"}],
            rejected=[],
            final_fail=set(),
        )
        self.assertEqual(counts["leaf"]["fired"], 1)
        self.assertEqual(counts["leaf"]["deferred"], 0)
        self.assertEqual(counts["leaf"]["fired_rejected"], 0)

    def test_no_operator_output_counts_as_deferred(self):
        # The genuinely safe case: nothing fired, nothing was rejected.
        counts, _ = classify_objects(
            init={"<module>.f": "leaf"}, steps=[], rejected=[], final_fail={"<module>.f"},
        )
        self.assertEqual(counts["leaf"]["deferred"], 1)
        self.assertEqual(counts["leaf"]["fired_rejected"], 0)

    def test_rejected_operator_is_fired_rejected_not_deferred(self):
        # The decisive case: the operator DID fire; the gate threw its candidate away.
        # Calling this "deferred" asserts the operator declined, which is false.
        counts, by_cause = classify_objects(
            init={"<module>.f": "leaf"},
            steps=[],
            rejected=[{"qualname": "<module>.f", "operator": "leaf_value_candidate",
                       "strategy": "deterministic_leaf", "reason": "regressed"}],
            final_fail={"<module>.f"},
        )
        self.assertEqual(counts["leaf"]["fired_rejected"], 1)
        self.assertEqual(counts["leaf"]["deferred"], 0,
                         "a rejected candidate must NOT be reported as a deferral")
        self.assertEqual(by_cause["regressed"], 1)

    def test_rejection_causes_are_broken_out(self):
        # compile_error means the operator emits invalid Python -- a real defect, unlike
        # no_improvement, which is the gate working as designed. They must not be pooled.
        counts, by_cause = classify_objects(
            init={"a": "leaf", "b": "leaf", "c": "control_flow"},
            steps=[],
            rejected=[
                {"qualname": "a", "operator": "leaf_value_candidate", "reason": "compile_error"},
                {"qualname": "b", "operator": "leaf_value_candidate", "reason": "no_improvement"},
                {"qualname": "c", "operator": "try_except_candidate", "reason": "compile_error"},
            ],
            final_fail={"a", "b", "c"},
        )
        self.assertEqual(by_cause["compile_error"], 2)
        self.assertEqual(by_cause["no_improvement"], 1)
        self.assertEqual(counts["leaf"]["fired_rejected"], 2)
        self.assertEqual(counts["control_flow"]["fired_rejected"], 1)

    def test_accepted_wins_over_a_rejected_sibling_attempt(self):
        # An object can have several attempts: an early candidate rejected, a later one accepted.
        # The object fired and SUCCEEDED -- it must count once, as fired.
        counts, _ = classify_objects(
            init={"<module>.f": "leaf"},
            steps=[{"qualname": "<module>.f", "accepted": True,
                    "repair_operation": "deterministic_prepass_leaf",
                    "selected_candidate_strategy": "deterministic_leaf"}],
            rejected=[{"qualname": "<module>.f", "operator": "leaf_value_candidate",
                       "reason": "no_improvement"}],
            final_fail=set(),
        )
        self.assertEqual(counts["leaf"]["fired"], 1)
        self.assertEqual(counts["leaf"]["fired_rejected"], 0)
        self.assertEqual(counts["leaf"]["deferred"], 0)

    def test_every_object_lands_in_exactly_one_bucket(self):
        # The three buckets must partition the objects, or the totals silently lie.
        counts, _ = classify_objects(
            init={"a": "leaf", "b": "leaf", "c": "leaf"},
            steps=[{"qualname": "a", "accepted": True,
                    "repair_operation": "deterministic_prepass_leaf"}],
            rejected=[{"qualname": "b", "operator": "leaf_value_candidate", "reason": "regressed"}],
            final_fail={"b", "c"},
        )
        r = counts["leaf"]
        self.assertEqual(r["objects"], 3)
        self.assertEqual(r["fired"] + r["fired_rejected"] + r["deferred"], r["objects"])


if __name__ == "__main__":
    unittest.main()
