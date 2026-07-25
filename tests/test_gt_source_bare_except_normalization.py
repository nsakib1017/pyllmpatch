"""Wiring spec: GT source text must be normalized with
``utils.reattach_source_code_object.parenthesize_bare_except`` at every site that loads it
for splicing/parsing, so a decompiler-artifact ``except A, B:`` clause anywhere in a
ground-truth source file no longer breaks the oracle fixer or the annotation-reconciliation
operator.

See ``tests/test_parenthesize_bare_except.py`` for the normalizer's own unit tests; this
file only checks that the two GT-source-load call sites named in the diagnosis actually
apply it:

1. ``pipeline.code_object_repair_loop.OracleFragmentFixer.__init__`` (sets
   ``self.gt_source_text`` from the GT source file).
2. ``utils.reattach_source_code_object._gt_def_source_by_qualname``'s caller (the
   annotation-reconciliation operator), via the same normalize-then-parse composition.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.code_object_repair_loop import OracleFragmentFixer
from utils.reattach_source_code_object import (
    _gt_def_source_by_qualname,
    parenthesize_bare_except,
)


class OracleFragmentFixerBareExceptNormalizationTest(unittest.TestCase):
    def test_gt_source_text_is_normalized_on_load(self):
        with tempfile.TemporaryDirectory() as td:
            gt_source = Path(td) / "gt.py"
            gt_source.write_text(
                "def f():\n    try:\n        pass\n    except X, Y:\n        pass\n",
                encoding="utf-8",
            )
            fixer = OracleFragmentFixer(Path(td) / "gt.pyc", gt_source)

        self.assertIn("except (X, Y):", fixer.gt_source_text)
        self.assertNotIn("except X, Y:", fixer.gt_source_text)
        compile(fixer.gt_source_text, "<test>", "exec")  # must not raise


class AnnotationOperatorBareExceptCompositionTest(unittest.TestCase):
    """`headers.py`-shaped repro: a bare-except clause ANYWHERE in the GT file makes
    `ast.parse` fail for the WHOLE file, so `_gt_def_source_by_qualname` returns None even
    for an unrelated qualname (e.g. `Config`, which has no except clause of its own) --
    unless the text is normalized first, matching the annotation operator's call site."""

    SRC = (
        "class Config:\n"
        "    x: int = 1\n"
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except json.JSONDecodeError, ValueError:\n"
        "        pass\n"
    )

    def test_raw_source_is_unparseable(self):
        self.assertIsNone(_gt_def_source_by_qualname(self.SRC, "<module>.Config"))

    def test_normalized_source_parses(self):
        normalized = parenthesize_bare_except(self.SRC)
        seg = _gt_def_source_by_qualname(normalized, "<module>.Config")
        self.assertIsNotNone(seg)
        self.assertTrue(seg.startswith("class Config:"))


if __name__ == "__main__":
    unittest.main()
