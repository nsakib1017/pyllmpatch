"""classify_repair_outcome turns a syntactic run-log record into the reported outcome bucket.

Two measured defects in the ad-hoc classification this replaces:

1. ZERO-DELETION MISCOUNT. A file can engage the delete-only fallback and delete NOTHING (the
   fallback's best-of candidate compiled as-is). 20 such files in the malware run were reported as
   non-genuine even though no code was removed -- they are genuine repairs by definition.

2. FIDELITY. `balance_delimiters` closes decompiler-TRUNCATED literals, which compiles and deletes
   no lines, so it scores "genuine" while discarding the literal's tail. 26% of genuine malware
   repairs carry that damage. The bucket therefore splits genuine into FAITHFUL and LOSSY so the
   headline rate cannot silently reward data loss.

The function is pure so the same rule applies to already-finished runs (no re-run needed) and to
future ones.
"""
import unittest

from pipeline.logging_utils import classify_repair_outcome


class ClassifyRepairOutcomeTest(unittest.TestCase):
    def test_plain_repair_with_no_fallback_is_genuine(self):
        rec = {"compiled_success": True, "delete_only_fallback_used": False}
        self.assertEqual(classify_repair_outcome(rec), "genuine")

    def test_fallback_that_deleted_nothing_is_genuine(self):
        # The 20-file miscount: fallback engaged, zero lines removed.
        rec = {"compiled_success": True, "delete_only_fallback_used": True, "delete_only_deletions": 0}
        self.assertEqual(classify_repair_outcome(rec), "genuine")

    def test_light_and_heavy_deletion_are_separated(self):
        light = {"compiled_success": True, "delete_only_fallback_used": True, "delete_only_deletions": 50}
        heavy = {"compiled_success": True, "delete_only_fallback_used": True, "delete_only_deletions": 51}
        self.assertEqual(classify_repair_outcome(light), "light_deletion")
        self.assertEqual(classify_repair_outcome(heavy), "heavy_deletion")

    def test_genuine_but_lossy_is_reported_separately(self):
        rec = {
            "compiled_success": True,
            "delete_only_fallback_used": False,
            "lossy_literal_closure": True,
        }
        self.assertEqual(classify_repair_outcome(rec), "genuine_lossy")

    def test_failure_and_out_of_scope_are_distinct(self):
        self.assertEqual(classify_repair_outcome({"compiled_success": False}), "failed")
        self.assertEqual(classify_repair_outcome({"compiled_success": None}), "out_of_scope")
        self.assertEqual(
            classify_repair_outcome({"compiled_success": None, "unwindowable_error_line": True}),
            "out_of_scope",
        )

    def test_missing_fields_never_raise(self):
        self.assertEqual(classify_repair_outcome({}), "out_of_scope")
        self.assertEqual(classify_repair_outcome(None), "out_of_scope")

    def test_deletion_count_none_with_fallback_counts_as_genuine(self):
        # No count recorded means nothing was reported deleted; do not penalise the file.
        rec = {"compiled_success": True, "delete_only_fallback_used": True, "delete_only_deletions": None}
        self.assertEqual(classify_repair_outcome(rec), "genuine")


if __name__ == "__main__":
    unittest.main()


class ForcedCompileNeutralisationTest(unittest.TestCase):
    """The forced-compile (neutralise-not-delete) fallback must never count as genuine.

    A forced-compiled file compiles WITHOUT deletion, but unparseable regions were turned into inert
    string literals -- it is analysable, not repaired, so it gets its own bucket disqualified from
    the genuine numerator exactly like deletion.
    """

    def test_forced_compile_is_its_own_bucket_not_genuine(self):
        rec = {"compiled_success": True, "forced_compile_used": True,
               "forced_compile_neutralised_lines": 12}
        self.assertEqual(classify_repair_outcome(rec), "neutralised_compile")

    def test_forced_compile_disqualifies_even_with_zero_deletion(self):
        rec = {"compiled_success": True, "forced_compile_used": True,
               "forced_compile_neutralised_lines": 1, "delete_only_fallback_used": False}
        self.assertNotIn(classify_repair_outcome(rec), ("genuine", "genuine_lossy"))

    def test_deletion_still_takes_precedence_if_both_somehow_set(self):
        rec = {"compiled_success": True, "forced_compile_used": True,
               "delete_only_fallback_used": True, "delete_only_deletions": 5}
        self.assertIn(classify_repair_outcome(rec), ("light_deletion", "heavy_deletion"))

    def test_absent_forced_compile_still_genuine(self):
        rec = {"compiled_success": True}
        self.assertEqual(classify_repair_outcome(rec), "genuine")
