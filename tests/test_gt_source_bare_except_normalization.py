"""Wiring spec: GT source text must be normalized with
``utils.reattach_source_code_object.parenthesize_bare_except`` when the oracle fixer loads it,
so a decompiler-artifact ``except A, B:`` clause anywhere in a ground-truth source file no
longer breaks the oracle fixer.

See ``tests/test_parenthesize_bare_except.py`` for the normalizer's own unit tests; this file
only checks that ``pipeline.code_object_repair_loop.OracleFragmentFixer.__init__`` (which sets
``self.gt_source_text`` from the GT source file) actually applies it.

(The gt-source annotation-reconciliation operator, which was the other consumer, has been
removed so no ground-truth SOURCE is read in the deployable LLM path — see
``tests/test_annotation_reconciliation.py``.)
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.code_object_repair_loop import OracleFragmentFixer


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


if __name__ == "__main__":
    unittest.main()
