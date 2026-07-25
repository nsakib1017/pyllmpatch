"""TDD spec: the source-mapping parse path must tolerate the decompiler artifact
``except A, B:`` (unparenthesized multiple exception types -- invalid Python 3 syntax).

``utils.map_source_code_objects.collect_source_code_objects`` calls ``ast.parse`` directly
on the raw GT source file. Even though
``utils.reattach_source_code_object.parenthesize_bare_except`` normalizes GT source text at
the fragment-load sites (see ``tests/test_gt_source_bare_except_normalization.py``), the
oracle fixer's ``_find_target_row`` -> ``map_source_to_pyc`` -> ``collect_source_code_objects``
path reaches ``ast.parse`` on the UNNORMALIZED file, so a bare-except clause anywhere in a
ground-truth source still raises ``SyntaxError: multiple exception types must be
parenthesized`` there -- this is the "mapping path" the fragment-load fix did not cover.

This test is end-to-end on the mapping path: a real temp ``.py`` file (not just a string
passed straight to the normalizer) containing the real artifact shape
(``except json.JSONDecodeError, ValueError:``) is fed to ``collect_source_code_objects``.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.map_source_code_objects import collect_source_code_objects


SRC = (
    "class Config:\n"
    "    x: int = 1\n"
    "\n"
    "    def load(self, payload):\n"
    "        try:\n"
    "            return payload.decode()\n"
    "        except json.JSONDecodeError, ValueError:\n"
    "            return None\n"
)


class CollectSourceCodeObjectsBareExceptTest(unittest.TestCase):
    def test_does_not_raise_on_bare_except_source(self):
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "gt.py"
            source_path.write_text(SRC, encoding="utf-8")

            # Must not raise SyntaxError: multiple exception types must be parenthesized
            records = collect_source_code_objects(source_path)

        qualnames = {record.qualname for record in records}
        self.assertIn("<module>", qualnames)
        self.assertIn("<module>.Config", qualnames)
        self.assertIn("<module>.Config.load", qualnames)


if __name__ == "__main__":
    unittest.main()
