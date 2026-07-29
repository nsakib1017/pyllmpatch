"""Wiring elision into the repair loop: recover un-windowable files without losing their payloads.

327 malware files are discarded untried because their ERROR LINE ALONE exceeds the generation
budget -- a minified/packer one-liner carrying an embedded payload. Eliding the payload shrinks
that line enough to repair the file.

Two design constraints drive these tests:

  * **Blast radius.** Elision fires ONLY for a file that would otherwise be thrown away. Every file
    already in scope must take a byte-identical path, so the 1,218 in-scope files cannot regress
    while chasing the 327.
  * **Restoration or failure.** The payload goes back before anything is written, verified. A file
    whose placeholder the repair mangled is a FAILURE -- never a success. Writing it anyway would
    put `__PYLLM_PAYLOAD_0__` where hundreds of kilobytes used to be: output that compiles, reports
    zero deleted lines, and scores as a genuine repair while being silently corrupt.
"""
import os
import sys
import tempfile
import unittest

from pipeline.runner import elide_to_fit_window, finalize_and_compile
from utils.syntactic_prepass import elide_long_string_literals

PAYLOAD = "A" * 40000
# The error is ON the giant line: a packer one-liner that never closes its call.
UNWINDOWABLE = "x = f('%s', 1, 2\n" % PAYLOAD
ORDINARY_BROKEN = "def g():\n    return 1\nx = f(1, 2\n"


HOST_VERSION = "%d.%d" % sys.version_info[:2]


def _tokens(text):
    """Stand-in tokenizer: ~4 chars per token, the ratio the real one averages on base64."""
    return len(text) // 4


class ElideToFitWindowTest(unittest.TestCase):
    def test_unwindowable_line_becomes_windowable(self):
        source, mapping = elide_to_fit_window(UNWINDOWABLE, 1, 2048, _tokens)
        self.assertTrue(mapping)
        self.assertNotIn(PAYLOAD, source)
        self.assertLessEqual(_tokens(source.splitlines()[0]), 2048)

    def test_file_already_within_budget_is_untouched(self):
        source, mapping = elide_to_fit_window(ORDINARY_BROKEN, 3, 2048, _tokens)
        self.assertEqual(source, ORDINARY_BROKEN)
        self.assertEqual(mapping, {})

    def test_line_still_oversized_after_elision_is_not_adopted(self):
        # A genuinely huge line with no string payload to hide -- elision cannot help, so it must
        # report no change rather than churning the source for nothing.
        source = "x = f(" + ", ".join(str(i) for i in range(40000)) + "\n"
        out, mapping = elide_to_fit_window(source, 1, 2048, _tokens)
        self.assertEqual(out, source)
        self.assertEqual(mapping, {})

    def test_disabled_budget_is_a_no_op(self):
        self.assertEqual(elide_to_fit_window(UNWINDOWABLE, 1, 0, _tokens)[1], {})

    def test_kill_switch_is_honoured(self):
        prior = os.environ.get("SYNTACTIC_PAYLOAD_ELISION")
        os.environ["SYNTACTIC_PAYLOAD_ELISION"] = "0"
        try:
            self.assertEqual(elide_to_fit_window(UNWINDOWABLE, 1, 2048, _tokens)[1], {})
        finally:
            if prior is None:
                os.environ.pop("SYNTACTIC_PAYLOAD_ELISION", None)
            else:
                os.environ["SYNTACTIC_PAYLOAD_ELISION"] = prior

    def test_bad_inputs_never_raise(self):
        self.assertEqual(elide_to_fit_window(None, 1, 2048, _tokens)[1], {})
        self.assertEqual(elide_to_fit_window("", 1, 2048, _tokens)[1], {})
        self.assertEqual(elide_to_fit_window(UNWINDOWABLE, None, 2048, _tokens)[1], {})

    def test_error_line_out_of_range_is_safe(self):
        self.assertEqual(elide_to_fit_window(ORDINARY_BROKEN, 999, 2048, _tokens)[1], {})


class FinalizeAndCompileTest(unittest.TestCase):
    """The single write boundary. Every accepted repair passes through here, so restoration is
    enforced in one place rather than at each of the three sites that write the output."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.py = os.path.join(self.dir, "out.py")
        self.pyc = os.path.join(self.dir, "out.pyc")

    def test_payload_is_restored_into_the_written_file(self):
        source = "data = '%s'\nx = 1\n" % PAYLOAD
        elided, mapping = elide_long_string_literals(source)
        result = finalize_and_compile(elided, self.py, self.pyc, HOST_VERSION, mapping)
        self.assertTrue(result["is_compiled"])
        self.assertEqual(open(self.py).read(), source)

    def test_no_mapping_writes_the_source_unchanged(self):
        source = "x = 1\n"
        result = finalize_and_compile(source, self.py, self.pyc, HOST_VERSION, {})
        self.assertTrue(result["is_compiled"])
        self.assertEqual(open(self.py).read(), source)

    def test_failed_restoration_is_reported_as_a_compile_failure(self):
        source = "data = '%s'\nx = 1\n" % PAYLOAD
        elided, mapping = elide_long_string_literals(source)
        mangled = elided.replace("__PYLLM_PAYLOAD_0__", "__GONE__")
        result = finalize_and_compile(mangled, self.py, self.pyc, HOST_VERSION, mapping)
        self.assertFalse(result["is_compiled"])
        self.assertTrue(result.get("payload_restoration_failed"))

    def test_failed_restoration_never_writes_a_placeholder_to_disk(self):
        source = "data = '%s'\nx = 1\n" % PAYLOAD
        elided, mapping = elide_long_string_literals(source)
        mangled = elided.replace("__PYLLM_PAYLOAD_0__", "__GONE__")
        finalize_and_compile(mangled, self.py, self.pyc, HOST_VERSION, mapping)
        if os.path.exists(self.py):
            self.assertNotIn("__GONE__", open(self.py).read())
            self.assertNotIn("__PYLLM_PAYLOAD_", open(self.py).read())

    def test_a_genuine_syntax_error_still_reports_normally(self):
        result = finalize_and_compile("x = f(1, 2\n", self.py, self.pyc, HOST_VERSION, {})
        self.assertFalse(result["is_compiled"])
        self.assertFalse(result.get("payload_restoration_failed"))


if __name__ == "__main__":
    unittest.main()
