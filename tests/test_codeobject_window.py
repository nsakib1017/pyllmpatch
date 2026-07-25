import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.syntactic_prepass import codeobject_span, codeobject_window, reattach_window, SyntaxErrorInfo

NESTED = "class A:\n    def m(self):\n        x = f(1, 2\n        y = 3\n    def n(self):\n        pass\n"

def test_span_is_innermost_enclosing_method():
    # error on line 3 (inside A.m) -> span covers 'def m' (line 2) .. line 4, NOT def n
    s = codeobject_span(NESTED.splitlines(), 3)
    assert s == (2, 4)

def test_module_scope_error_returns_none():
    # a top-level (indent 0) garbage statement is enclosed by nothing
    src = "def cc_zip():\n    return\n@overload\ndef bad(.defaults):\n    pass\n".splitlines()
    assert codeobject_span(src, 4) is None  # line 4 at indent 0 is a sibling of cc_zip, not inside it

def test_window_covers_object_and_roundtrips():
    err = SyntaxErrorInfo(lineno=3, offset=None, msg="'(' was never closed")
    w = codeobject_window(NESTED, err, 0)
    assert "def m(self):" in w.text and "def n" not in w.text
    assert reattach_window(NESTED, w, w.text) == NESTED

def test_module_scope_degenerates_to_minimal_window():
    src = "x = f(1, 2\n"
    err = SyntaxErrorInfo(lineno=1, offset=None, msg="'(' was never closed")
    w = codeobject_window(src, err, 0)
    assert w.text.strip() == "x = f(1, 2"  # same as minimal/cause-aware

def test_giant_object_degenerates_to_minimal():
    body = "\n".join("        a = 1" for _ in range(500))
    src = "class Big:\n    def m(self):\n" + body + "\n        y = g(1, 2\n"
    err = SyntaxErrorInfo(lineno=len(src.splitlines()), offset=None, msg="'(' was never closed")
    w = codeobject_window(src, err, 0)
    assert (w.end_line - w.start_line + 1) < 400  # capped -> minimal window, not the 500-line object

def test_expansion_widens_to_outer_object():
    err = SyntaxErrorInfo(lineno=3, offset=None, msg="x")
    w0 = codeobject_window(NESTED, err, 0)
    w1 = codeobject_window(NESTED, err, 1)
    assert w1.start_line <= w0.start_line and (w1.end_line - w1.start_line) >= (w0.end_line - w0.start_line)
