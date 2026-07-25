import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.syntactic_prepass import minimal_repair_window, reattach_window, RepairWindow, SyntaxErrorInfo

SRC = "def outer():\n    x = 1\n    y = f(1, 2\n    z = 3\n"

def test_window_is_single_logical_line_at_expansion_0():
    w = minimal_repair_window(SRC, SyntaxErrorInfo(3, 0, "'(' was never closed"), expansion=0)
    assert w.start_line == 3 and w.end_line == 3
    assert w.text.strip() == "y = f(1, 2"          # dedented
    assert w.indent == "    "

def test_window_pulls_continuation_lines_when_brackets_open():
    src = "a = [\n    1,\n    2\n"                    # open bracket spans 3 physical lines
    w = minimal_repair_window(src, SyntaxErrorInfo(1, 0, "'[' was never closed"), expansion=0)
    assert w.start_line == 1 and w.end_line == 3

def test_expansion_widens_to_enclosing_block():
    w0 = minimal_repair_window(SRC, SyntaxErrorInfo(3, 0, "x"), expansion=0)
    w1 = minimal_repair_window(SRC, SyntaxErrorInfo(3, 0, "x"), expansion=1)
    assert (w1.end_line - w1.start_line) > (w0.end_line - w0.start_line)

def test_reattach_is_exact_inverse_for_unchanged_fix():
    w = minimal_repair_window(SRC, SyntaxErrorInfo(3, 0, "x"), expansion=0)
    assert reattach_window(SRC, w, w.text) == SRC     # round-trip identity

def test_reattach_reindents_and_splices():
    w = minimal_repair_window(SRC, SyntaxErrorInfo(3, 0, "x"), expansion=0)
    out = reattach_window(SRC, w, "y = f(1, 2)")       # fixed snippet (dedented)
    assert "    y = f(1, 2)\n" in out and "z = 3" in out


# ---------------------------------------------------------------------------
# Finding 1: bracket-depth (not indentation) must decide window extent, so
# Black-style dedented closers are included and never-closed brackets never
# swallow an unrelated following statement.
# ---------------------------------------------------------------------------

BLACK_TOP_LEVEL = "foo(\n    a,\n    b,\n)\nbar = 1\n"
BLACK_NESTED = "def outer():\n    foo(\n        a,\n        b,\n    )\n    bar = 1\n"


def test_black_style_dedented_closer_included_at_expansion_0_top_level():
    w = minimal_repair_window(BLACK_TOP_LEVEL, SyntaxErrorInfo(1, 0, "'(' was never closed"), expansion=0)
    assert w.start_line == 1 and w.end_line == 4       # must include line 4's ')'
    assert w.text.strip().endswith(")")
    assert "bar" not in w.text


def test_black_style_dedented_closer_included_at_expansion_0_nested():
    w = minimal_repair_window(BLACK_NESTED, SyntaxErrorInfo(2, 0, "'(' was never closed"), expansion=0)
    assert w.start_line == 2 and w.end_line == 5       # must include the dedented ')'
    assert w.text.strip().endswith(")")
    assert "bar" not in w.text


def test_black_style_window_round_trips_exactly():
    w = minimal_repair_window(BLACK_TOP_LEVEL, SyntaxErrorInfo(1, 0, "'(' was never closed"), expansion=0)
    assert reattach_window(BLACK_TOP_LEVEL, w, w.text) == BLACK_TOP_LEVEL


def test_black_style_expansion_1_does_not_balloon_to_sibling_statement_top_level():
    w0 = minimal_repair_window(BLACK_TOP_LEVEL, SyntaxErrorInfo(1, 0, "'(' was never closed"), expansion=0)
    w1 = minimal_repair_window(BLACK_TOP_LEVEL, SyntaxErrorInfo(1, 0, "'(' was never closed"), expansion=1)
    # no enclosing block exists at the top level -> expansion is a no-op here
    assert w1.start_line == w0.start_line and w1.end_line == w0.end_line == 4
    assert "bar" not in w1.text


def test_truly_never_closed_bracket_does_not_swallow_next_statement_top_level():
    src = "y = f(1, 2\nz = 3\n"
    w = minimal_repair_window(src, SyntaxErrorInfo(1, 0, "'(' was never closed"), expansion=0)
    assert w.start_line == 1 and w.end_line == 1
    assert "z" not in w.text


def test_truly_never_closed_bracket_does_not_swallow_next_statement_nested():
    # SRC's error line ("y = f(1, 2") is inside outer() at indent 4; the next
    # line "z = 3" sits at the very same indent, so it must not be pulled in.
    w = minimal_repair_window(SRC, SyntaxErrorInfo(3, 0, "'(' was never closed"), expansion=0)
    assert w.start_line == 3 and w.end_line == 3
    assert "z" not in w.text


def test_empty_source_returns_valid_window():
    w = minimal_repair_window("", SyntaxErrorInfo(1, 0, "x"), expansion=0)
    assert w.start_line == 1 and w.end_line == 1 and w.text == ""


def test_whitespace_only_source_returns_valid_window():
    w = minimal_repair_window("   \n\n", SyntaxErrorInfo(1, 0, "x"), expansion=0)
    assert w.start_line == 1 and w.end_line == 1
