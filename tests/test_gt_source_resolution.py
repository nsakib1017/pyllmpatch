"""TDD spec for `utils.file_helpers.fetch_pyllmpatch_gt_source_path`.

BUG: `fetch_pyllmpatch_source_path(file_hash, source)` returns the DECOMPILED source
(`decompiler_output/indented_*.py`) for the pylingual (non-PyPi) tree. That value was wired
in as `gt_source` in `pipeline.code_object_repair_loop._resolve_dataset_repair_paths` and fed
to `OracleFragmentFixer(gt_pyc, gt_source)`, so the "oracle" extracted DERIVED fragments
(no-ops), not ground truth -- silently breaking every oracle run on the adapter tree.

The real GT source lives at `<ROOT_FOR_FILES>/pyllm_final_dataset/pyllm_py_dataset/<base_hash>/*.py`
where `base_hash` is `file_hash` with its trailing `_<digits>` version suffix stripped.

`fetch_pyllmpatch_gt_source_path` must:
- passthrough unchanged for PyPi rows (already the real source),
- resolve the real GT tree for pylingual rows and return it when found,
- fall back to the previous (decompiled) behavior when the GT tree/file is absent -- never a
  regression, never an exception.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from utils import file_helpers


class GtSourceResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self.base_hash = "abc123deadbeef"
        self.row_hash = f"{self.base_hash}_315"

        # Adapter (pylingual) tree: BASE_DIR_PYTHON_FILES_PYLINGUAL/<row_hash>/decompiler_output/indented_*.py
        self.pylingual_root = self.root / "decompiler_workspace"
        self.adapter_out_dir = self.pylingual_root / self.row_hash / "decompiler_output"
        self.adapter_out_dir.mkdir(parents=True)
        self.decompiled_py = self.adapter_out_dir / "indented_1.py"
        self.decompiled_py.write_text("x = 1\n", encoding="utf-8")

        # Real GT tree: <ROOT_FOR_FILES>/pyllm_final_dataset/pyllm_py_dataset/<base_hash>/*.py
        self.gt_root = self.root / "pyllm_final_dataset" / "pyllm_py_dataset"
        self.gt_dir = self.gt_root / self.base_hash
        self.gt_dir.mkdir(parents=True)
        self.gt_py = self.gt_dir / "mod.py"
        self.gt_py.write_text("x: int = 1\n", encoding="utf-8")
        (self.gt_dir / "meta.json").write_text("{}", encoding="utf-8")

        # PyPi tree, for the passthrough case.
        self.pypi_root = self.root / "pypi_downloaded"
        self.pypi_hash_dir = self.pypi_root / self.row_hash
        self.pypi_hash_dir.mkdir(parents=True)
        self.pypi_source_py = self.pypi_hash_dir / "real_source.py"
        self.pypi_source_py.write_text("y: int = 2\n", encoding="utf-8")

        self._orig_root_for_files = file_helpers.ROOT_FOR_FILES
        self._orig_base_dir_pylingual = file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL
        self._orig_base_dir_pypi = file_helpers.BASE_DIR_PYTHON_FILES_PYPI
        self._orig_env = os.environ.get("PYLINGUAL_GT_SOURCE_DIR")

        file_helpers.ROOT_FOR_FILES = self.root
        file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL = self.pylingual_root
        file_helpers.BASE_DIR_PYTHON_FILES_PYPI = self.pypi_root
        os.environ.pop("PYLINGUAL_GT_SOURCE_DIR", None)

    def tearDown(self) -> None:
        file_helpers.ROOT_FOR_FILES = self._orig_root_for_files
        file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL = self._orig_base_dir_pylingual
        file_helpers.BASE_DIR_PYTHON_FILES_PYPI = self._orig_base_dir_pypi
        if self._orig_env is None:
            os.environ.pop("PYLINGUAL_GT_SOURCE_DIR", None)
        else:
            os.environ["PYLINGUAL_GT_SOURCE_DIR"] = self._orig_env

    def test_resolves_real_gt_source_for_pylingual_tree(self) -> None:
        result = file_helpers.fetch_pyllmpatch_gt_source_path(self.row_hash, "pylingual")
        self.assertIsNotNone(result)
        self.assertIn("x: int", Path(result).read_text(encoding="utf-8"))
        self.assertEqual(Path(result), self.gt_py)

    def test_falls_back_to_decompiled_when_gt_tree_absent(self) -> None:
        # Remove the real GT tree entirely -- only the decompiled adapter output remains.
        self.gt_py.unlink()
        (self.gt_dir / "meta.json").unlink()
        self.gt_dir.rmdir()

        result = file_helpers.fetch_pyllmpatch_gt_source_path(self.row_hash, "pylingual")
        self.assertIsNotNone(result)
        self.assertEqual(Path(result), self.decompiled_py)
        self.assertNotIn("x: int", Path(result).read_text(encoding="utf-8"))

    def test_pypi_rows_are_unchanged_passthrough(self) -> None:
        expected = file_helpers.fetch_pyllmpatch_source_path(self.row_hash, "PyPi")
        result = file_helpers.fetch_pyllmpatch_gt_source_path(self.row_hash, "PyPi")
        self.assertEqual(result, expected)
        self.assertEqual(Path(result), self.pypi_source_py)


if __name__ == "__main__":
    unittest.main()
