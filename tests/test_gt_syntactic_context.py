"""TDD spec for utils.gt_syntactic_context.build_gt_object_context.

The syntactic-repair LLM sometimes faces a broken decompiled code object it cannot
reconstruct blind (leaked PEP-695 generics like `def <generic parameters of cc_zip>(.defaults):`,
`class .type_params(...)`, `@overload` stacks, `# ***<qualname>: Failure` markers). This module
extracts the TRUE structure (signature/placement) of the enclosing object from the ground-truth
`.pyc`, so it can be handed to the LLM as extra context.

This suite uses a REAL row from dataset/syntactic_corpus_gt.csv (file_hash starting
`42539768a344`): its broken source has `def <generic parameters of cc_zip>(.defaults):` at
line 23, the enclosing real object (found by scanning upward for the nearest def/class header)
is `cc_zip`, and the paired GT `.pyc` (`cc_tools.pyc`) contains a code object named
`<module>.cc_zip`.

`build_gt_object_context` must NEVER raise -- it returns "" on any failure (no gt path, missing
file, load error, module-scope error with no enclosing object, or no name match in the GT).
"""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

from utils.gt_syntactic_context import build_gt_object_context

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_CSV = REPO_ROOT / "dataset" / "syntactic_corpus_gt.csv"
TARGET_HASH_PREFIX = "42539768a344"


def _find_row(hash_prefix: str) -> dict:
    with open(CORPUS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["file_hash"].startswith(hash_prefix):
                return row
    raise AssertionError(f"no row in {CORPUS_CSV} with file_hash prefix {hash_prefix}")


def _target_row_available() -> tuple[dict, str] | None:
    """Returns (row, reason_unavailable). reason is None if usable."""
    try:
        row = _find_row(TARGET_HASH_PREFIX)
    except AssertionError as exc:
        return None, str(exc)
    src_path = Path(row["file_path"])
    gt_path = Path(row["gt_pyc"])
    if not src_path.is_file():
        return None, f"broken source missing on disk: {src_path}"
    if not gt_path.is_file():
        return None, f"gt pyc missing on disk: {gt_path}"
    return (row, None)


_ROW, _UNAVAILABLE_REASON = _target_row_available()


class BuildGtObjectContextRealCorpusTest(unittest.TestCase):
    @unittest.skipIf(_ROW is None, f"target corpus row unavailable: {_UNAVAILABLE_REASON}")
    def test_returns_nonempty_brief_mentioning_true_object_name_for_cc_zip(self):
        row = _ROW
        broken_source = Path(row["file_path"]).read_text(encoding="utf-8")
        gt_pyc_path = row["gt_pyc"]

        result = build_gt_object_context(broken_source, 23, gt_pyc_path)

        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        self.assertIn("cc_zip", result)


class BuildGtObjectContextFailureModesTest(unittest.TestCase):
    def setUp(self):
        row, reason = _target_row_available()
        if row is None:
            self.skipTest(f"target corpus row unavailable: {reason}")
        self.row = row
        self.broken_source = Path(row["file_path"]).read_text(encoding="utf-8")

    def test_empty_gt_pyc_path_returns_empty_string(self):
        result = build_gt_object_context(self.broken_source, 23, "")
        self.assertEqual(result, "")

    def test_nonexistent_gt_pyc_path_returns_empty_string_without_raising(self):
        result = build_gt_object_context(
            self.broken_source, 23, "/nonexistent/path/does/not/exist.pyc"
        )
        self.assertEqual(result, "")

    def test_module_scope_error_returns_empty_string(self):
        # No enclosing def/class header above line 1 -- the scanner finds no qualname.
        result = build_gt_object_context("x = f(1, 2\n", 1, self.row["gt_pyc"])
        self.assertEqual(result, "")


class BuildGtObjectContextNeverRaisesTest(unittest.TestCase):
    def test_never_raises_on_garbage_inputs(self):
        try:
            result = build_gt_object_context("", 0, "")
            self.assertEqual(result, "")
            result = build_gt_object_context("not python (((", 1, "/tmp/does-not-exist.pyc")
            self.assertEqual(result, "")
        except Exception as exc:  # pragma: no cover - the whole point of the test
            self.fail(f"build_gt_object_context raised {exc!r} instead of returning \"\"")


if __name__ == "__main__":
    unittest.main()
