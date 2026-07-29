"""Only MECHANICAL deletion disqualifies a repair; LLM-side deletion is repair work.

The decompiler emits genuine garbage -- `# postinserted` scaffolding, duplicated statements,
orphaned fragments -- and a correct repair removes it on purpose. So lines the LLM deletes must NOT
count against the repair. What is not a repair is the delete-only fallback stripping lines until the
file happens to parse: that is compilation by attrition, and it is what the "genuine" metric excludes.

`count_deleted_lines` therefore measures the SIZE of a mechanical deletion (it sees every removed
line, where the fallback's own tally can undercount), but outside the fallback it is diagnostic
only -- useful for spotting an LLM that removed 251 lines, never a reason to re-bucket the outcome.
"""
import unittest

from pipeline.logging_utils import classify_repair_outcome, count_deleted_lines


class CountDeletedLinesTest(unittest.TestCase):
    def test_no_change_deletes_nothing(self):
        src = "a = 1\nb = 2\n"
        self.assertEqual(count_deleted_lines(src, src), 0)

    def test_removed_line_is_counted(self):
        self.assertEqual(count_deleted_lines("a = 1\nb = 2\nc = 3\n", "a = 1\nc = 3\n"), 1)

    def test_multiple_removals_counted(self):
        self.assertEqual(count_deleted_lines("a\nb\nc\nd\n", "a\nd\n"), 2)

    def test_added_lines_are_not_deletions(self):
        self.assertEqual(count_deleted_lines("a = 1\n", "a = 1\nexcept_added = 2\n"), 0)

    def test_reordering_is_not_deletion(self):
        self.assertEqual(count_deleted_lines("a\nb\n", "b\na\n"), 0)

    def test_whitespace_only_change_is_not_deletion(self):
        self.assertEqual(count_deleted_lines("if x:\n    a\n", "if x:\n        a\n"), 0)

    def test_missing_inputs_are_safe(self):
        self.assertEqual(count_deleted_lines(None, "a\n"), 0)
        self.assertEqual(count_deleted_lines("a\n", None), 0)
        self.assertEqual(count_deleted_lines("", ""), 0)


class MechanicalDeletionOnlyTest(unittest.TestCase):
    def test_llm_deletions_remain_genuine(self):
        # The LLM stripped 7 lines of decompiler scaffolding -- that is the repair, not damage.
        rec = {"compiled_success": True, "delete_only_fallback_used": False,
               "lines_deleted_vs_original": 7}
        self.assertEqual(classify_repair_outcome(rec), "genuine")

    def test_large_llm_deletion_is_still_genuine(self):
        rec = {"compiled_success": True, "delete_only_fallback_used": False,
               "lines_deleted_vs_original": 251}
        self.assertEqual(classify_repair_outcome(rec), "genuine")

    def test_mechanical_deletion_is_bucketed_by_measured_size(self):
        light = {"compiled_success": True, "delete_only_fallback_used": True,
                 "lines_deleted_vs_original": 50}
        heavy = {"compiled_success": True, "delete_only_fallback_used": True,
                 "lines_deleted_vs_original": 51}
        self.assertEqual(classify_repair_outcome(light), "light_deletion")
        self.assertEqual(classify_repair_outcome(heavy), "heavy_deletion")

    def test_fallback_that_removed_nothing_is_genuine(self):
        rec = {"compiled_success": True, "delete_only_fallback_used": True,
               "lines_deleted_vs_original": 0}
        self.assertEqual(classify_repair_outcome(rec), "genuine")

    def test_measured_size_overrides_the_fallback_tally(self):
        # The fallback under-counted its own deletions; the diff is authoritative for SIZE.
        rec = {"compiled_success": True, "delete_only_fallback_used": True,
               "delete_only_deletions": 3, "lines_deleted_vs_original": 90}
        self.assertEqual(classify_repair_outcome(rec), "heavy_deletion")

    def test_legacy_logs_without_the_diff_are_unchanged(self):
        rec = {"compiled_success": True, "delete_only_fallback_used": True,
               "delete_only_deletions": 5}
        self.assertEqual(classify_repair_outcome(rec), "light_deletion")
        rec2 = {"compiled_success": True, "delete_only_fallback_used": False}
        self.assertEqual(classify_repair_outcome(rec2), "genuine")


if __name__ == "__main__":
    unittest.main()
