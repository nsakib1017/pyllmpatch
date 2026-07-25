import unittest
from utils.reattach_source_code_object import _annotate_enclosing_qualname

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
