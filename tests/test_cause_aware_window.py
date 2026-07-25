"""Cause-aware repair window: the LLM window must contain the ACTUAL cause
of a syntax error (often several lines above the reported symptom line),
not just the symptom line itself, plus the minimal context needed to fix it.

Task 1: locate_cause(source, error) -> CauseAnchor -- error-type-aware
backward anchoring from the reported symptom to the true cause line.

Task 2: cause_aware_window(source, error, expansion) -> RepairWindow --
builds the window from the cause anchor + minimal necessary context, while
preserving minimal_repair_window's minimality/round-trip guarantees.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.syntactic_prepass import (
    CauseAnchor,
    SyntaxErrorInfo,
    cause_aware_window,
    locate_cause,
    minimal_repair_window,
    reattach_window,
)


# ---------------------------------------------------------------------------
# Task 1: locate_cause -- one test per rule, realistic census-style snippets.
# ---------------------------------------------------------------------------

def test_unexpected_indent_cause_is_preceding_line():
    src = "a = 1\n    b = 2\n"
    err = SyntaxErrorInfo(2, 4, "unexpected indent")
    anchor = locate_cause(src, err)
    assert anchor.line == 1
    assert anchor.reason == "unexpected_indent_prev_line"
    assert anchor.include_preceding is True


def test_unexpected_indent_cause_is_unclosed_paren_line_above():
    # Realistic decompiler shape: an unclosed generator-expr paren on the
    # line above triggers a downstream "unexpected indent" on the next
    # physical line -- the TRUE cause is the unclosed-paren line, not the
    # indented line itself.
    src = "x = all((y for y in items\n    b = 2\n"
    err = SyntaxErrorInfo(2, 4, "unexpected indent")
    anchor = locate_cause(src, err)
    assert anchor.line == 1
    assert anchor.reason == "unexpected_indent_prev_line"


def test_empty_block_header_cause_is_the_colon_header():
    src = "if x:\ny = 1\n"
    err = SyntaxErrorInfo(2, 1, "expected an indented block after 'if' statement on line 1")
    anchor = locate_cause(src, err)
    assert anchor.line == 1
    assert anchor.reason == "empty_block_header"
    assert anchor.include_preceding is True


def test_unclosed_delimiter_anchor_is_the_opener_itself():
    src = "foo(\n    a,\n    b,\n)\nbar = 1\n"
    err = SyntaxErrorInfo(1, 4, "'(' was never closed")
    anchor = locate_cause(src, err)
    assert anchor.line == 1
    assert anchor.reason == "unclosed_delimiter"
    assert anchor.span_statement is True


def test_unterminated_string_anchor_is_the_opener_itself():
    src = "x = 'abc\ny = 1\n"
    err = SyntaxErrorInfo(1, 5, "unterminated string literal (detected at line 1)")
    anchor = locate_cause(src, err)
    assert anchor.line == 1
    assert anchor.reason == "unclosed_delimiter"


def test_unindent_mismatch_cause_is_enclosing_block_start():
    src = "def f():\n    if True:\n        x = 1\n      y = 2\n"
    err = SyntaxErrorInfo(4, 7, "unindent does not match any outer indentation level")
    anchor = locate_cause(src, err)
    assert anchor.line == 2  # "    if True:" -- the enclosing block we dedented out of
    assert anchor.reason == "unindent_mismatch"
    assert anchor.include_preceding is True


def test_mid_expression_invalid_syntax_spans_three_lines():
    src = "result = (\n    1 +\n    * 2\n)\n"
    err = SyntaxErrorInfo(3, 5, "invalid syntax")
    anchor = locate_cause(src, err)
    assert anchor.line == 1  # walk up through the open '(' to the statement start
    assert anchor.reason == "mid_statement"
    assert anchor.span_statement is True


def test_forgot_a_comma_is_treated_as_mid_statement():
    src = "d = {\n    'a': 1\n    'b': 2\n}\n"
    err = SyntaxErrorInfo(3, 5, "invalid syntax. Perhaps you forgot a comma?")
    anchor = locate_cause(src, err)
    assert anchor.reason == "mid_statement"
    assert anchor.span_statement is True


def test_pep695_generic_leak_is_detected_on_error_line():
    src = "def <generic parameters of foo>(.defaults):\n    return 1\n"
    err = SyntaxErrorInfo(1, 5, "invalid syntax")
    anchor = locate_cause(src, err)
    assert anchor.line == 1
    assert anchor.reason == "pep695_generic_leak"
    assert anchor.include_decorators is True
    # BUG 1 fix: pep695 leaks must NOT span into the function body -- the
    # cause is local to the header line itself.
    assert anchor.span_statement is False


def test_decorated_def_error_flags_decorators_and_span():
    src = "@dec\ndef f(:\n    pass\n"
    err = SyntaxErrorInfo(2, 7, "invalid syntax")
    anchor = locate_cause(src, err)
    assert anchor.line == 2
    assert anchor.reason == "def_header"
    assert anchor.include_decorators is True
    # BUG 1 fix: def-header errors must NOT span into the function body --
    # the cause is local to the header line itself.
    assert anchor.span_statement is False


def test_async_def_error_also_flags_decorators_and_span():
    src = "@dec\nasync def f(:\n    pass\n"
    err = SyntaxErrorInfo(2, 13, "invalid syntax")
    anchor = locate_cause(src, err)
    assert anchor.reason == "def_header"
    assert anchor.include_decorators is True


def test_class_header_error_flags_decorators_and_span():
    src = "@dec\nclass C(:\n    pass\n"
    err = SyntaxErrorInfo(2, 9, "invalid syntax")
    anchor = locate_cause(src, err)
    assert anchor.reason == "def_header"


def test_default_rule_anchors_on_reported_line():
    src = "def outer():\n    x = 1\n    y = f(1, 2\n    z = 3\n"
    err = SyntaxErrorInfo(3, 0, "x")
    anchor = locate_cause(src, err)
    assert anchor.line == 3
    assert anchor.reason == "reported_line"
    assert anchor.include_preceding is False
    assert anchor.include_decorators is False
    assert anchor.span_statement is False


def test_cause_anchor_is_a_dataclass_with_expected_fields():
    anchor = CauseAnchor(line=1, reason="reported_line", include_preceding=False,
                          include_decorators=False, span_statement=False)
    assert anchor.line == 1
    assert anchor.reason == "reported_line"


# ---------------------------------------------------------------------------
# Task 2: cause_aware_window -- build the window from the cause + minimal
# necessary context, while preserving minimal_repair_window's minimality
# and reattach_window's round-trip identity.
# ---------------------------------------------------------------------------

def test_unexpected_indent_window_includes_the_preceding_cause_line():
    src = "a = 1\n    b = 2\n"
    err = SyntaxErrorInfo(2, 4, "unexpected indent")
    w = cause_aware_window(src, err, expansion=0)
    assert "a = 1" in w.text
    assert w.start_line == 1


def test_unclosed_paren_window_spans_opener_to_balanced_close():
    src = "foo(\n    a,\n    b,\n)\nbar = 1\n"
    err = SyntaxErrorInfo(1, 4, "'(' was never closed")
    w = cause_aware_window(src, err, expansion=0)
    assert w.start_line == 1 and w.end_line == 4
    assert w.text.strip().endswith(")")
    assert "bar" not in w.text


def test_pep695_leak_window_is_tight_and_excludes_the_body():
    """BUG 1 fix (regression pin, superseding the old body-inclusion
    expectation): a PEP-695 generic-leak window must stay tight to the
    header line -- pulling in the (possibly broken/huge) function body
    confused the LLM in the real A/B regression this fix addresses."""
    src = "def <generic parameters of foo>(.defaults):\n    return 1\n"
    err = SyntaxErrorInfo(1, 5, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert "generic parameters of foo" in w.text
    assert "return 1" not in w.text
    assert w.start_line == 1 and w.end_line == 1


def test_decorated_def_window_includes_the_decorator_line():
    src = "@dec\ndef f(:\n    pass\n"
    err = SyntaxErrorInfo(2, 7, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert "@dec" in w.text
    assert w.start_line == 1


def test_cause_aware_window_round_trips_for_preceding_line_case():
    src = "a = 1\n    b = 2\n"
    err = SyntaxErrorInfo(2, 4, "unexpected indent")
    w = cause_aware_window(src, err, expansion=0)
    assert reattach_window(src, w, w.text) == src


def test_cause_aware_window_round_trips_for_pep695_case():
    src = "def <generic parameters of foo>(.defaults):\n    return 1\n"
    err = SyntaxErrorInfo(1, 5, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert reattach_window(src, w, w.text) == src


def test_cause_aware_window_round_trips_for_decorated_def_case():
    src = "@dec\ndef f(:\n    pass\n"
    err = SyntaxErrorInfo(2, 7, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert reattach_window(src, w, w.text) == src


def test_cause_aware_window_round_trips_for_unindent_mismatch_case():
    src = "def f():\n    if True:\n        x = 1\n      y = 2\n"
    err = SyntaxErrorInfo(4, 7, "unindent does not match any outer indentation level")
    w = cause_aware_window(src, err, expansion=0)
    assert reattach_window(src, w, w.text) == src


def test_cause_aware_window_round_trips_for_mid_statement_case():
    src = "result = (\n    1 +\n    * 2\n)\n"
    err = SyntaxErrorInfo(3, 5, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert reattach_window(src, w, w.text) == src


def test_minimality_matches_minimal_repair_window_when_no_cause_above():
    """MINIMALITY REGRESSION (pinned): for a plain single-line error with no
    cause above the reported line, cause_aware_window must add NOTHING --
    it must equal minimal_repair_window exactly."""
    src = "def outer():\n    x = 1\n    y = f(1, 2\n    z = 3\n"
    err = SyntaxErrorInfo(3, 0, "x")
    base = minimal_repair_window(src, err, expansion=0)
    aware = cause_aware_window(src, err, expansion=0)
    assert aware.start_line == base.start_line
    assert aware.end_line == base.end_line
    assert aware.text == base.text
    assert aware.indent == base.indent


def test_minimality_matches_minimal_repair_window_at_higher_expansion_too():
    src = "def outer():\n    x = 1\n    y = f(1, 2\n    z = 3\n"
    err = SyntaxErrorInfo(3, 0, "x")
    base = minimal_repair_window(src, err, expansion=1)
    aware = cause_aware_window(src, err, expansion=1)
    assert (aware.start_line, aware.end_line, aware.text, aware.indent) == \
           (base.start_line, base.end_line, base.text, base.indent)


def test_mid_statement_window_pulls_in_the_full_statement():
    src = "result = (\n    1 +\n    * 2\n)\n"
    err = SyntaxErrorInfo(3, 5, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert w.start_line == 1
    assert "result = (" in w.text


def test_cap_stops_at_enclosing_def_class_block():
    """CAP: cause-anchoring never escapes the enclosing def/class block --
    the mid-statement walk here must not reach up into the OUTER function's
    own header, even though nothing else shallower separates them."""
    src = (
        "def outer():\n"
        "    def inner():\n"
        "        result = (\n"
        "            1 +\n"
        "            * 2\n"
        "        )\n"
    )
    err = SyntaxErrorInfo(5, 13, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert "def outer" not in w.text
    assert "result = (" in w.text


# ---------------------------------------------------------------------------
# BUG 1 regression (A/B: 19/20 -> 17/20): cause_aware_window over-expanded
# into the broken function BODY for def-header / PEP-695-leak errors, e.g. a
# 65-line balloon (lines 81-146) around a single-line-local def header. The
# window must stay TIGHT: the header line plus any contiguous @decorator
# lines directly above it, and nothing from the body below.
# ---------------------------------------------------------------------------

def test_pep695_leak_window_stays_tight_and_excludes_the_body_realistic():
    src = (
        "@overload\n"
        "def <generic parameters of cc_zip>(.defaults, *, strict=False):\n"
        "    \"\"\"A\"\"\"\n"
        "    x = 1\n"
    )
    err = SyntaxErrorInfo(2, 5, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert "@overload" in w.text
    assert "generic parameters of cc_zip" in w.text
    assert "x = 1" not in w.text
    assert w.start_line == 1 and w.end_line == 2


# ---------------------------------------------------------------------------
# BUG 2 regression: when the reported line (e.g. a def header) sits INSIDE an
# unclosed bracket/string opened on a preceding line, the true cause is that
# preceding opener line, not the reported line. Concrete real-world shape:
# an unterminated multiline call/string on one line, followed by a `def` on
# the next line that CPython reports as "invalid syntax".
# ---------------------------------------------------------------------------

def test_unclosed_delimiter_above_is_detected_before_def_header_rule():
    src = "y = f((1, 2\ndef g():\n    pass\n"
    err = SyntaxErrorInfo(2, 1, "invalid syntax")
    anchor = locate_cause(src, err)
    assert anchor.line == 1
    assert anchor.reason == "unclosed_delimiter_above"
    assert anchor.span_statement is True


def test_unclosed_delimiter_above_window_includes_the_opener_line():
    src = "y = f((1, 2\ndef g():\n    pass\n"
    err = SyntaxErrorInfo(2, 1, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert w.start_line == 1
    assert "y = f((1, 2" in w.text
    assert "def g():" in w.text


def test_unclosed_delimiter_above_round_trips():
    src = "y = f((1, 2\ndef g():\n    pass\n"
    err = SyntaxErrorInfo(2, 1, "invalid syntax")
    w = cause_aware_window(src, err, expansion=0)
    assert reattach_window(src, w, w.text) == src
