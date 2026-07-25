"""PEP-649 `__annotate__` synthetic code objects have NO source representation
(the decompiled source lost the annotations), so they surface as *missing* repair
targets that can never be reattached -> `ReattachError` aborts the whole file.

They must be skipped from target selection exactly like expression-level children
(`.<lambda>`, `.<listcomp>`, ...) already are, so the rest of the file still repairs.
"""
import unittest

from utils.reattach_source_code_object import (
    select_missing_repair_targets,
    select_repair_targets,
    _is_synthetic_annotation_qualname,
)


class SyntheticAnnotationSkipTest(unittest.TestCase):
    def test_missing_annotate_target_is_skipped(self):
        rows = [
            {"status": "missing", "gt_name": "<module>.Config.__annotate__"},
            {"status": "missing", "gt_name": "<module>.real_function"},
        ]
        targets = select_missing_repair_targets(rows)
        self.assertIn("<module>.real_function", targets)
        self.assertNotIn("<module>.Config.__annotate__", targets)

    def test_module_level_annotate_target_is_skipped(self):
        rows = [{"status": "missing", "gt_name": "<module>.__annotate__"}]
        self.assertEqual(select_missing_repair_targets(rows), [])

    def test_matched_annotate_target_is_skipped(self):
        # If an __annotate__ ever shows up matched-but-divergent it is still
        # unreattachable (no source node), so it must not be selected either.
        rows = [
            {
                "status": "matched",
                "gt_name": "<module>.Config.__annotate__",
                "derived_name": "<module>.Config.__annotate__",
                "combined_distance": 5,
            },
            {
                "status": "matched",
                "gt_name": "<module>.real_function",
                "derived_name": "<module>.real_function",
                "combined_distance": 5,
            },
        ]
        targets = select_repair_targets(rows)
        self.assertIn("<module>.real_function", targets)
        self.assertNotIn("<module>.Config.__annotate__", targets)

    def test_helper_predicate(self):
        self.assertTrue(_is_synthetic_annotation_qualname("<module>.X.__annotate__"))
        self.assertTrue(_is_synthetic_annotation_qualname("<module>.__annotate__"))
        self.assertFalse(_is_synthetic_annotation_qualname("<module>.__annotate__x"))
        self.assertFalse(_is_synthetic_annotation_qualname("<module>.annotate"))
        self.assertFalse(_is_synthetic_annotation_qualname("<module>.real_function"))
        self.assertFalse(_is_synthetic_annotation_qualname(None))


if __name__ == "__main__":
    unittest.main()
