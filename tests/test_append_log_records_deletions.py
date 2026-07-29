"""append_log records the measured deletion count so no repair route can escape the audit.

The runner writes `path_out` from four different places (prepass-solo fix, LLM success, retry
success, delete-only fallback). Computing the diff at each site invites exactly the omission that
created this problem in the first place -- the old `delete_only_fallback_used` flag covered only one
route and was blind to the LLM's own deletions, which is why 84% of repairs labelled "genuine" had
in fact removed lines. Deriving it centrally at log time, from the paths already present in the
record, makes the measurement structurally unmissable.

`path_in` is the pristine decompiled source (verified byte-identical to the dataset's
`decompiled_source` on 200/200 files -- the pipeline copies rather than mutating in place).
"""
import json
import os
import tempfile
import unittest

from pipeline.logging_utils import append_log


class AppendLogDeletionAuditTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.log = os.path.join(self.dir, "run.jsonl")

    def _write(self, name, text):
        p = os.path.join(self.dir, name)
        with open(p, "w") as f:
            f.write(text)
        return p

    def _logged(self):
        with open(self.log) as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_deletion_is_measured_and_recorded(self):
        a = self._write("in.py", "a = 1\nb = 2\nc = 3\n")
        b = self._write("out.py", "a = 1\nc = 3\n")
        append_log(self.log, {"path_in": a, "path_out": b})
        self.assertEqual(self._logged()[0]["lines_deleted_vs_original"], 1)

    def test_faithful_repair_records_zero(self):
        a = self._write("in.py", "a = 1\nb = 2\n")
        b = self._write("out.py", "a = 1\nb = 2\nd = 4\n")  # added only
        append_log(self.log, {"path_in": a, "path_out": b})
        self.assertEqual(self._logged()[0]["lines_deleted_vs_original"], 0)

    def test_existing_value_is_not_overwritten(self):
        a = self._write("in.py", "a = 1\nb = 2\n")
        b = self._write("out.py", "a = 1\n")
        append_log(self.log, {"path_in": a, "path_out": b, "lines_deleted_vs_original": 99})
        self.assertEqual(self._logged()[0]["lines_deleted_vs_original"], 99)

    def test_records_without_both_paths_are_untouched(self):
        append_log(self.log, {"path_in": None, "path_out": None, "file_hash": "h"})
        rec = self._logged()[0]
        self.assertNotIn("lines_deleted_vs_original", rec)
        self.assertEqual(rec["file_hash"], "h")

    def test_missing_files_never_raise(self):
        append_log(self.log, {"path_in": "/nope/a.py", "path_out": "/nope/b.py"})
        self.assertEqual(len(self._logged()), 1)


if __name__ == "__main__":
    unittest.main()
