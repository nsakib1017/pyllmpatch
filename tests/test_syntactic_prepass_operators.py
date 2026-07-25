import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.syntactic_prepass import balance_delimiters, probe_syntax, host_compile

def _fix(src):
    e = probe_syntax(src, host_compile); assert e is not None
    return balance_delimiters(src, e)

def test_closes_trailing_unclosed_paren():
    src = "x = f(1, 2\n"
    out = _fix(src)
    assert out is not None and probe_syntax(out, host_compile) is None

def test_closes_nested_missing_double_paren():
    src = "y = all((a for a in items\n"
    out = _fix(src)
    assert out is not None and probe_syntax(out, host_compile) is None

def test_closes_unterminated_string():
    src = "s = 'hello\n"
    out = _fix(src)
    assert out is not None and probe_syntax(out, host_compile) is None

def test_closes_unterminated_triple_quoted_string_single_line():
    src = "s = '''hello\n"
    out = _fix(src)
    assert out is not None and probe_syntax(out, host_compile) is None
    assert out == "s = '''hello'''\n"


def test_closes_unterminated_triple_quoted_string_multi_line():
    src = "s = '''hello\nworld\n"
    out = _fix(src)
    assert out is not None and probe_syntax(out, host_compile) is None
    assert out == "s = '''hello\nworld'''\n"
    assert "world" in out                # continuation content is preserved, not dropped


def test_closes_unterminated_triple_double_quoted_string():
    src = 's = """hello\nworld\n'
    out = _fix(src)
    assert out is not None and probe_syntax(out, host_compile) is None


def test_returns_none_when_balanced():
    # a non-delimiter error (bad indent) -> operator defers
    from utils.syntactic_prepass import SyntaxErrorInfo
    assert balance_delimiters("def f():\nx=1\n", SyntaxErrorInfo(2,0,"expected an indented block")) is None


def test_dedents_stray_indented_line():
    from utils.syntactic_prepass import dedent_stray_block, probe_syntax, host_compile
    src = "a = 1\n    b = 2\n"   # b over-indented, a is not an opener
    e = probe_syntax(src, host_compile); assert e and "indent" in e.msg
    out = dedent_stray_block(src, e)
    assert out is not None and probe_syntax(out, host_compile) is None

def test_defers_when_previous_is_opener():
    from utils.syntactic_prepass import dedent_stray_block, SyntaxErrorInfo
    # 'if x:' IS a valid opener -> the indent is legitimate, operator must defer
    assert dedent_stray_block("if x:\n        y = 1\n", SyntaxErrorInfo(2,0,"unexpected indent")) is None

def test_defers_on_non_indent_error():
    from utils.syntactic_prepass import dedent_stray_block, SyntaxErrorInfo
    assert dedent_stray_block("x = (\n", SyntaxErrorInfo(1,0,"'(' was never closed")) is None


def test_line_continuation_strips_trailing():
    from utils.syntactic_prepass import fix_line_continuation, probe_syntax, host_compile
    src = "x = 1 + \\ tail\n2\n"
    e = probe_syntax(src, host_compile); assert e
    out = fix_line_continuation(src, e)
    # everything after the backslash on the error line is dropped verbatim
    assert out == "x = 1 +\n2\n"


def test_line_continuation_strips_trailing_and_fixes_when_line_is_otherwise_complete():
    from utils.syntactic_prepass import fix_line_continuation, probe_syntax, host_compile
    src = "x = 1 \\ y\n"
    e = probe_syntax(src, host_compile); assert e
    out = fix_line_continuation(src, e)
    assert out == "x = 1\n"
    assert probe_syntax(out, host_compile) is None            # this one actually compiles

def test_numeric_defers_when_ambiguous():
    from utils.syntactic_prepass import fix_numeric_literal, SyntaxErrorInfo
    assert fix_numeric_literal("x = 0xabcg\n", SyntaxErrorInfo(1,0,"invalid hexadecimal literal")) is None
