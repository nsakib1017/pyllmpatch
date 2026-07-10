"""The uv/pyenv compile subprocess emits harmless broken-.pth startup noise on the 3.10
toolchain; _clean_compile_stderr must strip it (else every 3.10 candidate is wrongly
rejected -> 0 deterministic fixes on 3.10) while preserving real compile errors."""
import unittest
from utils.generate_bytecode import _clean_compile_stderr

_VENV_NOISE = (
    "Error processing line 1 of /tmp/uv-cache/archive-v0/QP/lib/python3.10/site-packages/_virtualenv.pth:\n\n"
    "  Traceback (most recent call last):\n"
    "    File \"/x/site.py\", line 195, in addpackage\n      exec(line)\n"
    "    File \"<string>\", line 1, in <module>\n"
    "  ModuleNotFoundError: No module named '_virtualenv'\n\nRemainder of file ignored\n"
)


class CleanCompileStderrTest(unittest.TestCase):
    def test_strips_virtualenv_pth_noise_to_empty(self):
        self.assertEqual(_clean_compile_stderr(_VENV_NOISE), "")

    def test_strips_repeated_pth_noise(self):
        self.assertEqual(_clean_compile_stderr(_VENV_NOISE + _VENV_NOISE), "")

    def test_strips_download_progress(self):
        self.assertEqual(_clean_compile_stderr("Downloading cpython-3.10 (20MiB)\n"), "")

    def test_preserves_real_syntax_error(self):
        real = "  File \"t.py\", line 2\n    return x +\n             ^\nSyntaxError: invalid syntax\n"
        self.assertIn("SyntaxError", _clean_compile_stderr(_VENV_NOISE + real))

    def test_preserves_version_assertion_error(self):
        real = "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nAssertionError\n"
        self.assertIn("AssertionError", _clean_compile_stderr(_VENV_NOISE + real))


if __name__ == "__main__":
    unittest.main()
