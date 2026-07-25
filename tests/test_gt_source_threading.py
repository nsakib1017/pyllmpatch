"""GT source must reach `repair_mismatching_code_objects` so the annotation-reconciliation
Stage-1 operator (utils/reattach_source_code_object.py) can fire in dataset/oracle runs.

BUG (pre-fix): `CodeObjectRepairLoop.run()` (pipeline/code_object_repair_loop.py) never passed
`gt_source` through to `repair_mismatching_code_objects`, so it always defaulted to None and the
annotation-reconciliation branch was dead in every dataset run -- even though `OracleFragmentFixer`
already resolves and holds a usable `self.gt_source` (set in its __init__).

Fix under test: `run()` gains a keyword-only `gt_source: Path | None = None` parameter. If not
passed explicitly, it falls back to `getattr(self.fixer, "gt_source", None)` -- so oracle-backed
dataset runs get the GT source "for free," while fixers with no such attribute (e.g. an LLM fixer)
leave the argument at None, preserving prior behavior exactly.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pipeline.code_object_repair_loop as L


class _StubFixerWithGtSource:
    """Fixer-shaped stub carrying a resolved `.gt_source`, like OracleFragmentFixer."""

    def __init__(self, gt_source: Path):
        self.gt_source = gt_source


class _StubFixerNoGtSource:
    """Fixer-shaped stub with no `.gt_source` attribute at all, like LLMFragmentFixer."""


def _loop(fixer):
    loop = L.CodeObjectRepairLoop.__new__(L.CodeObjectRepairLoop)
    loop.fixer = fixer
    return loop


class GtSourceThreadingTest(unittest.TestCase):
    def test_oracle_fixer_gt_source_reaches_repair_call(self):
        gt_source = Path("/fake/gt_source.py")
        fixer = _StubFixerWithGtSource(gt_source)
        loop = _loop(fixer)

        fake_result = {"final_summary": {}}
        with mock.patch.object(L, "repair_mismatching_code_objects", return_value=fake_result) as mocked:
            loop.run(
                gt_pyc=Path("/fake/gt.pyc"),
                derived_pyc=Path("/fake/derived.pyc"),
                derived_source=Path("/fake/derived.py"),
                output_dir=Path("/fake/out"),
            )

        mocked.assert_called_once()
        _, kwargs = mocked.call_args
        self.assertEqual(
            kwargs.get("gt_source"), gt_source,
            "run() must thread the fixer's gt_source into repair_mismatching_code_objects "
            "so the annotation-reconciliation operator can fire",
        )

    def test_explicit_gt_source_kwarg_overrides_fixer(self):
        fixer_gt_source = Path("/fake/fixer_gt_source.py")
        explicit_gt_source = Path("/fake/explicit_gt_source.py")
        fixer = _StubFixerWithGtSource(fixer_gt_source)
        loop = _loop(fixer)

        fake_result = {"final_summary": {}}
        with mock.patch.object(L, "repair_mismatching_code_objects", return_value=fake_result) as mocked:
            loop.run(
                gt_pyc=Path("/fake/gt.pyc"),
                derived_pyc=Path("/fake/derived.pyc"),
                derived_source=Path("/fake/derived.py"),
                output_dir=Path("/fake/out"),
                gt_source=explicit_gt_source,
            )

        _, kwargs = mocked.call_args
        self.assertEqual(kwargs.get("gt_source"), explicit_gt_source)

    def test_fixer_without_gt_source_stays_none(self):
        fixer = _StubFixerNoGtSource()
        loop = _loop(fixer)

        fake_result = {"final_summary": {}}
        with mock.patch.object(L, "repair_mismatching_code_objects", return_value=fake_result) as mocked:
            loop.run(
                gt_pyc=Path("/fake/gt.pyc"),
                derived_pyc=Path("/fake/derived.pyc"),
                derived_source=Path("/fake/derived.py"),
                output_dir=Path("/fake/out"),
            )

        _, kwargs = mocked.call_args
        self.assertIsNone(kwargs.get("gt_source"))


if __name__ == "__main__":
    unittest.main()
