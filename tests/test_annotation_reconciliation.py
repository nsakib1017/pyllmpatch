import unittest
from utils.reattach_source_code_object import _annotate_enclosing_qualname, _gt_def_source_by_qualname

class AnnotateEnclosingQualnameTest(unittest.TestCase):
    def test_maps_annotate_child_to_enclosing_def(self):
        self.assertEqual(_annotate_enclosing_qualname("<module>.Config.__annotate__"), "<module>.Config")
    def test_maps_nested(self):
        self.assertEqual(_annotate_enclosing_qualname("<module>.A.B.__annotate__"), "<module>.A.B")
    def test_module_level_annotate_has_no_enclosing_def(self):
        self.assertIsNone(_annotate_enclosing_qualname("<module>.__annotate__"))
    def test_non_annotate_returns_none(self):
        self.assertIsNone(_annotate_enclosing_qualname("<module>.Config"))
        self.assertIsNone(_annotate_enclosing_qualname(None))


SRC = (
    "class Outer:\n"
    "    x: int = 1\n"
    "    class Inner:\n"
    "        y: str = 'a'\n"
    "def top(a: int) -> str:\n"
    "    return str(a)\n"
)


class GtDefSourceTest(unittest.TestCase):
    def test_top_level_class(self):
        seg = _gt_def_source_by_qualname(SRC, "<module>.Outer")
        self.assertTrue(seg.startswith("class Outer:")); self.assertIn("class Inner:", seg)
    def test_nested_class(self):
        seg = _gt_def_source_by_qualname(SRC, "<module>.Outer.Inner")
        self.assertTrue(seg.startswith("class Inner:")); self.assertIn("y: str", seg)
    def test_function(self):
        seg = _gt_def_source_by_qualname(SRC, "<module>.top")
        self.assertTrue(seg.startswith("def top(")); self.assertIn("-> str", seg)
    def test_missing_returns_none(self):
        self.assertIsNone(_gt_def_source_by_qualname(SRC, "<module>.Nope"))
