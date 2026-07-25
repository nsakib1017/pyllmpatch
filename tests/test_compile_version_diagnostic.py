"""Regression test for a diagnostic-corruption bug in compile_version().

_compile_uv() raises CompileError(stderr) for BOTH real uv-infra failures AND genuine
SyntaxErrors in the candidate (stderr contains the real "SyntaxError ... line N"
diagnostic). Previously, when _compile_uv raised, compile_version unconditionally fell
through to _compile_pyenv -- and on boxes with no matching pyenv interpreter installed,
_compile_pyenv's failure (e.g. "pyenv: version '3.13' is not installed") REPLACED the
real "line N" diagnostic from uv. Every caller that reads the compile error text (the
syntactic mechanical prepass via probe_syntax; LLM localization via extract_line_number)
lost the line number for all non-host versions.

The fix: when _compile_uv fails, still attempt _compile_pyenv (it may successfully
compile on boxes where it IS installed), but if pyenv ALSO fails, re-raise the ORIGINAL
uv CompileError (which carries the real diagnostic), not the pyenv error.
"""
import sys
import unittest
from unittest import mock

import utils.generate_bytecode as gb
from utils.generate_bytecode import CompileError, PyenvError, compile_version

HOST = (sys.version_info[0], sys.version_info[1])


def _cross_version():
    # any UV-managed version that is not the host, so the uv/pyenv branch is taken
    for cand in (3, 10), (3, 11), (3, 13):
        if cand != HOST:
            return cand
    return (3, 10)


class CompileVersionDiagnosticPreservationTest(unittest.TestCase):
    def test_genuine_syntax_error_survives_when_pyenv_cannot_help(self):
        """When uv reports a genuine SyntaxError and pyenv can't compile either (no
        matching interpreter installed), the raised error must still carry the real
        uv diagnostic (with the line number), not the pyenv infra-failure text."""
        v = _cross_version()
        uv_message = "  File \"file.py\", line 7\nSyntaxError: invalid syntax (file.py, line 7)"
        pyenv_message = "pyenv: version '3.13' is not installed"

        with mock.patch.object(gb, "_compile_uv", side_effect=CompileError(uv_message)), \
             mock.patch.object(gb, "_compile_pyenv", side_effect=PyenvError(pyenv_message)):
            with self.assertRaises(Exception) as ctx:
                compile_version("s.py", "o.pyc", v)

        raised_text = str(ctx.exception)
        self.assertIn("line 7", raised_text,
                       "the real uv SyntaxError diagnostic (with line number) was lost")
        self.assertNotIn("not installed", raised_text,
                          "the pyenv infra-failure text replaced the real diagnostic")

    def test_uv_success_never_invokes_pyenv(self):
        """When _compile_uv succeeds, pyenv must not be called at all."""
        v = _cross_version()

        def _fake_uv(py_file, out_file, version):
            return None

        with mock.patch.object(gb, "_compile_uv", side_effect=_fake_uv), \
             mock.patch.object(gb, "_compile_pyenv",
                                side_effect=AssertionError("pyenv should not be invoked")) as pyenv:
            compile_version("s.py", "o.pyc", v)  # must not raise

        pyenv.assert_not_called()

    def test_real_source_compiles_at_host_version(self):
        """Sanity check: a real, valid source file compiles via compile_version at the
        host interpreter's own version (native path, unaffected by this fix)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            py_file = Path(tmp) / "ok.py"
            out_file = Path(tmp) / "ok.pyc"
            py_file.write_text("x = 1 + 1\n")

            compile_version(str(py_file), str(out_file), HOST)

            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()
