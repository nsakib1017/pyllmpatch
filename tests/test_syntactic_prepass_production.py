"""Regression coverage for the production defect: in production,
`utils.generate_bytecode.compile_version` ALWAYS raises `CompileError` (a
plain Exception, NOT a SyntaxError) on failure -- even for the host Python
version. `probe_syntax`'s `except SyntaxError:` branch is therefore dead in
production, and the `except Exception:` branch always ran, returning
`SyntaxErrorInfo(lineno=None, ...)`. That crippled every lineno-gated
downstream consumer (`advanced`, `dedent_stray_block`,
`fix_line_continuation`, `balance_delimiters`'s in-place closer placement).

These tests model a fake compile_fn that mimics the production shape: it
raises a plain Exception subclass whose str() is text-only (embeds
"line N" but carries NO structured .lineno attribute) -- exactly like
`utils.generate_bytecode.CompileError(str(py_compile.PyCompileError(...)))`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.syntactic_prepass import probe_syntax, run_syntactic_prepass


class FakeCompileError(Exception):
    """Mirrors utils.generate_bytecode.CompileError: a plain Exception (NOT a
    SyntaxError), whose message is text-only -- no structured .lineno."""


def test_probe_syntax_recovers_lineno_from_text_only_compile_error():
    def compile_fn(source):
        raise FakeCompileError("CompileError: some/path.py, line 3: invalid syntax")

    info = probe_syntax("irrelevant source\n", compile_fn)

    assert info is not None
    assert info.lineno == 3
    assert info.offset is None
    assert "FakeCompileError" in info.msg


def test_probe_syntax_lineno_none_when_message_has_no_line_number():
    def compile_fn(source):
        raise FakeCompileError("CompileError: something went wrong, no location info")

    info = probe_syntax("irrelevant source\n", compile_fn)

    assert info is not None
    assert info.lineno is None


def test_run_syntactic_prepass_fires_operators_against_text_only_compile_error():
    """The production-shape regression test: with a compile_fn that raises
    a text-only CompileError (no structured lineno) -- exactly the shape
    utils.generate_bytecode.compile_version uses in production -- the
    prepass driver must still recover the line number and let
    balance_delimiters fire and fully repair the source."""
    src = "def outer():\n    x = 1\n    y = f(1, 2\n    z = 3\n"

    def compile_fn(source):
        try:
            compile(source, "<t>", "exec")
        except SyntaxError as e:
            # Structured lineno is stripped -- only text remains, matching
            # utils.generate_bytecode.CompileError(str(...)) in production.
            raise FakeCompileError(f"CompileError: <t>, line {e.lineno}: {e.msg}") from None

    result = run_syntactic_prepass(src, compile_fn=compile_fn)

    assert result.compiled is True
    assert "balance_delimiters" in result.operations
