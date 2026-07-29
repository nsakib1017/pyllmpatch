"""copy_file must produce a WRITABLE working copy, even from a read-only source.

Root cause of ~37% of malware syntactic files erroring with PermissionError: the decompiler
artifacts are mode 0o555 (read-only), and copy_file used shutil.copy2 which preserves mode, so
the copy was read-only too. The repair loop then overwrites that copy with each candidate
(pipeline.runner._apply_candidate_and_compile -> open(copy_dir, "w")), which raised
`PermissionError: [Errno 13] Permission denied`. A copy made specifically to be edited must be
writable regardless of the source's mode.
"""
import os
import stat
import tempfile
import unittest
from pathlib import Path

from utils.file_helpers import copy_file


class CopyFileWritableTest(unittest.TestCase):
    def test_copy_of_readonly_source_is_writable(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.py"
            src.write_text("x = 1\n", encoding="utf-8")
            os.chmod(src, 0o555)  # read-only, like the 0o555 malware artifacts
            dst = Path(td) / "out" / "copy.py"

            copy_file(src, dst)

            self.assertTrue(dst.exists())
            self.assertTrue(os.access(dst, os.W_OK), "copy must be writable")
            # the operation the repair loop actually performs must not raise
            with open(dst, "w", encoding="utf-8") as f:
                f.write("x = 2\n")
            self.assertEqual(dst.read_text(encoding="utf-8"), "x = 2\n")

    def test_content_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.py"
            src.write_text("hello = 'world'\n", encoding="utf-8")
            dst = Path(td) / "copy.py"
            copy_file(src, dst)
            self.assertEqual(dst.read_text(encoding="utf-8"), "hello = 'world'\n")


if __name__ == "__main__":
    unittest.main()
