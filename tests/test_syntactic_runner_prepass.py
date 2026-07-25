import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.runner import maybe_prepass, minimal_window_syntax_context


# ---------------------------------------------------------------------------
# maybe_prepass (Task 6, bullet 1) — pure, testable, env-flag gated.
# ---------------------------------------------------------------------------

def test_prepass_fixes_without_llm(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_DETERMINISTIC_PREPASS", "1")
    new, ok, ops = maybe_prepass(
        "x = f(1, 2\n", "3.12", compile_version_fn=lambda s, v: compile(s, "<t>", "exec")
    )
    assert ok and "balance_delimiters" in ops


def test_prepass_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_DETERMINISTIC_PREPASS", "0")
    new, ok, ops = maybe_prepass(
        "x = f(1, 2\n", "3.12", compile_version_fn=lambda s, v: compile(s, "<t>", "exec")
    )
    assert ok is False and ops == [] and new == "x = f(1, 2\n"


def test_prepass_default_is_enabled_when_env_unset(monkeypatch):
    monkeypatch.delenv("SYNTACTIC_DETERMINISTIC_PREPASS", raising=False)
    new, ok, ops = maybe_prepass(
        "x = f(1, 2\n", "3.12", compile_version_fn=lambda s, v: compile(s, "<t>", "exec")
    )
    assert ok and "balance_delimiters" in ops


def test_prepass_leaves_unfixable_source_unchanged_but_deferred(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_DETERMINISTIC_PREPASS", "1")
    src = "def f():\n    return 1\n    except X:\n        pass\n"
    new, ok, ops = maybe_prepass(src, "3.12", compile_version_fn=lambda s, v: compile(s, "<t>", "exec"))
    assert ok is False
    assert new  # never crashes, returns best-effort source


def test_prepass_disabled_never_calls_compile_fn(monkeypatch):
    """FIX 3: the flag-off short-circuit must be a genuine no-op -- exact
    passthrough tuple AND zero calls into compile_version_fn (no side
    effects at all when the pre-pass is disabled)."""
    monkeypatch.setenv("SYNTACTIC_DETERMINISTIC_PREPASS", "0")
    calls = []

    def spy_compile_version_fn(s, v):
        calls.append((s, v))
        raise AssertionError("compile_version_fn must not be called when the prepass is disabled")

    source = "x = f(1, 2\n"
    new, ok, ops = maybe_prepass(source, "3.12", compile_version_fn=spy_compile_version_fn)

    assert (new, ok, ops) == (source, False, [])
    assert calls == []


# ---------------------------------------------------------------------------
# FIX 2: all 7 pre-existing Task-6 tests above probe with
# `lambda s, v: compile(s, "<t>", "exec")`, which behaves like host_compile
# (raises a real, structured SyntaxError) -- NOT like production's
# utils.generate_bytecode.compile_version, which always raises a plain
# `CompileError` (a text-only Exception with no structured .lineno). That
# left the wired production path untested. This models the real shape.
# ---------------------------------------------------------------------------

class _FakeCompileError(Exception):
    """Mirrors utils.generate_bytecode.CompileError: a plain Exception (NOT a
    SyntaxError), whose message is text-only -- no structured .lineno."""


def _production_shaped_compile_version_fn(source: str, version: str) -> None:
    try:
        compile(source, "<t>", "exec")
    except SyntaxError as e:
        # Structured lineno stripped -- only text remains, matching
        # utils.generate_bytecode.CompileError(str(py_compile.PyCompileError(...))).
        raise _FakeCompileError(f"CompileError: <t>, line {e.lineno}: {e.msg}") from None


def test_maybe_prepass_fixes_source_against_production_shaped_compile_error(monkeypatch):
    monkeypatch.setenv("SYNTACTIC_DETERMINISTIC_PREPASS", "1")
    new, ok, ops = maybe_prepass(
        "x = f(1, 2\n", "3.12", compile_version_fn=_production_shaped_compile_version_fn
    )
    assert ok and "balance_delimiters" in ops


# ---------------------------------------------------------------------------
# minimal_window_syntax_context (Task 6, bullet 3) — minimal LLM residual
# window, segment_syntax_context-compatible, expansion widens per retry.
# ---------------------------------------------------------------------------

SRC = "def outer():\n    x = 1\n    y = f(1, 2\n    z = 3\n"


def test_first_llm_window_is_exactly_the_failing_line_not_enclosing_def(tmp_path):
    py_file = tmp_path / "sample.py"
    py_file.write_text(SRC)

    segment = minimal_window_syntax_context(
        py_file, error_line=3, error_description="'(' was never closed", expansion_level=0
    )

    assert segment is not None
    assert segment.start_line == 3 and segment.end_line == 3
    assert segment.text.strip() == "y = f(1, 2"
    assert "def outer" not in segment.text
    assert "z = 3" not in segment.text


def test_llm_window_widens_on_retry_via_expansion_level(tmp_path):
    py_file = tmp_path / "sample.py"
    py_file.write_text(SRC)

    w0 = minimal_window_syntax_context(py_file, error_line=3, error_description="x", expansion_level=0)
    w1 = minimal_window_syntax_context(py_file, error_line=3, error_description="x", expansion_level=1)

    assert w0 is not None and w1 is not None
    assert (w1.end_line - w1.start_line) > (w0.end_line - w0.start_line)


def test_llm_window_falls_back_when_error_line_missing(tmp_path):
    py_file = tmp_path / "sample.py"
    py_file.write_text(SRC)

    # error_line=None -> existing segment_syntax_context behavior (returns None);
    # must not guess a line number.
    segment = minimal_window_syntax_context(py_file, error_line=None, error_description="x", expansion_level=0)
    assert segment is None


# ---------------------------------------------------------------------------
# Cause-aware window wiring (Task 3): minimal_window_syntax_context must use
# cause_aware_window, not minimal_repair_window, so an "unexpected indent"
# whose true cause is the line above reaches the LLM WITH that cause line,
# not just the symptom line.
# ---------------------------------------------------------------------------

UNEXPECTED_INDENT_SRC = "a = 1\n    b = 2\n"


def test_llm_window_for_unexpected_indent_includes_the_cause_line_above(tmp_path):
    py_file = tmp_path / "sample.py"
    py_file.write_text(UNEXPECTED_INDENT_SRC)

    segment = minimal_window_syntax_context(
        py_file, error_line=2, error_description="unexpected indent", expansion_level=0
    )

    assert segment is not None
    assert segment.start_line == 1  # the cause line, not just the symptom line 2
    assert "a = 1" in segment.text
    assert "b = 2" in segment.text
