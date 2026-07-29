"""Elide oversized string-literal payloads so a file stops being un-windowable.

Measured: the lines that push malware files out of scope are overwhelmingly giant embedded
payloads -- `encrypted_data = '<744,000 chars of base64>'`. They are syntactically trivial, they are
never the cause of the syntax error, and the model gains nothing from reading them; they simply
exhaust the 2048-token window and get the whole file discarded.

Replacing each oversized literal with a short placeholder shrinks the window enough to repair the
file, and the mapping restores the exact original bytes afterwards -- so this costs no fidelity:
the payload is preserved verbatim, it is just hidden from the model.
"""
import unittest
from unittest import mock

from pipeline.runner import maybe_elide_payloads
from utils.syntactic_prepass import (
    elide_long_string_literals,
    restore_and_verify,
    restore_elided_literals,
)

PAYLOAD = "A" * 5000
BROKEN = "data = '%s'\nx = f(1, 2\n" % PAYLOAD


class PayloadElisionTest(unittest.TestCase):
    def test_oversized_literal_is_replaced(self):
        src = "data = '" + "A" * 5000 + "'\nx = 1\n"
        out, mapping = elide_long_string_literals(src, max_literal_chars=100)
        self.assertLess(len(out), 500)
        self.assertEqual(len(mapping), 1)
        self.assertIn("x = 1", out)

    def test_short_literals_are_left_alone(self):
        src = "a = 'hello'\nb = \"world\"\n"
        out, mapping = elide_long_string_literals(src, max_literal_chars=100)
        self.assertEqual(out, src)
        self.assertEqual(mapping, {})

    def test_round_trip_restores_the_payload_exactly(self):
        payload = "B" * 9000
        src = f"data = '{payload}'\nrun()\n"
        out, mapping = elide_long_string_literals(src, max_literal_chars=100)
        self.assertEqual(restore_elided_literals(out, mapping), src)

    def test_multiple_payloads_round_trip(self):
        src = "a = '" + "X" * 3000 + "'\nb = \"" + "Y" * 3000 + "\"\n"
        out, mapping = elide_long_string_literals(src, max_literal_chars=100)
        self.assertEqual(len(mapping), 2)
        self.assertEqual(restore_elided_literals(out, mapping), src)

    def test_elided_source_still_parses_when_original_did(self):
        import ast
        src = "data = '" + "Z" * 8000 + "'\ndef f():\n    return data\n"
        out, _ = elide_long_string_literals(src, max_literal_chars=100)
        ast.parse(out)

    def test_restore_with_empty_mapping_is_identity(self):
        self.assertEqual(restore_elided_literals("x = 1\n", {}), "x = 1\n")

    def test_never_raises(self):
        for src in ("", None, "data = 'unterminated" + "Q" * 5000):
            elide_long_string_literals(src, max_literal_chars=100)
        self.assertEqual(restore_elided_literals(None, None), None)


class RestoreAndVerifyTest(unittest.TestCase):
    """Restoration is byte-exact or the repair does not count.

    The model can rename, split, or delete a placeholder. When that happens the payload is
    unrecoverable, and shipping the file anyway would mean writing `__PYLLM_PAYLOAD_0__` where
    744KB of base64 used to be -- corrupted output that still compiles and would score as a genuine
    repair. So restoration is VERIFIED rather than assumed, and a failure is reported as one.
    """

    def test_clean_restoration_reports_success(self):
        elided, mapping = elide_long_string_literals(BROKEN)
        restored, ok = restore_and_verify(elided, mapping)
        self.assertTrue(ok)
        self.assertEqual(restored, BROKEN)

    def test_restoration_survives_edits_elsewhere_in_the_file(self):
        elided, mapping = elide_long_string_literals(BROKEN)
        repaired = elided.replace("x = f(1, 2", "x = f(1, 2)")
        restored, ok = restore_and_verify(repaired, mapping)
        self.assertTrue(ok)
        self.assertIn(PAYLOAD, restored)
        self.assertIn("x = f(1, 2)", restored)

    def test_a_dropped_placeholder_is_a_failed_restoration(self):
        elided, mapping = elide_long_string_literals(BROKEN)
        mangled = "\n".join(l for l in elided.splitlines() if "__PYLLM_PAYLOAD_" not in l)
        _, ok = restore_and_verify(mangled, mapping)
        self.assertFalse(ok)

    def test_a_renamed_placeholder_is_a_failed_restoration(self):
        elided, mapping = elide_long_string_literals(BROKEN)
        mangled = elided.replace("__PYLLM_PAYLOAD_0__", "__PYLLM_PAYLOAD_99__")
        _, ok = restore_and_verify(mangled, mapping)
        self.assertFalse(ok)

    def test_no_placeholder_may_survive_into_the_output(self):
        elided, mapping = elide_long_string_literals(BROKEN)
        restored, ok = restore_and_verify(elided, mapping)
        self.assertTrue(ok)
        for token in mapping:
            self.assertNotIn(token, restored)

    def test_empty_mapping_is_a_transparent_success(self):
        restored, ok = restore_and_verify("x = 1\n", {})
        self.assertTrue(ok)
        self.assertEqual(restored, "x = 1\n")

    def test_never_raises_on_junk(self):
        self.assertFalse(restore_and_verify(None, {"__PYLLM_PAYLOAD_0__": "'x'"})[1])
        self.assertEqual(restore_and_verify("", {})[0], "")


class MaybeElidePayloadsTest(unittest.TestCase):
    """The runner seam. Elision only pays for files carrying an oversized payload; for every other
    file it must be a byte-identical no-op so the existing repair path is untouched."""

    def test_oversized_payload_is_elided(self):
        elided, mapping = maybe_elide_payloads(BROKEN)
        self.assertTrue(mapping)
        self.assertNotIn(PAYLOAD, elided)

    def test_ordinary_source_is_a_no_op(self):
        src = "def f():\n    return 'hello'\n"
        elided, mapping = maybe_elide_payloads(src)
        self.assertEqual(elided, src)
        self.assertEqual(mapping, {})

    def test_kill_switch_disables_elision(self):
        with mock.patch.dict("os.environ", {"SYNTACTIC_PAYLOAD_ELISION": "0"}):
            elided, mapping = maybe_elide_payloads(BROKEN)
        self.assertEqual(elided, BROKEN)
        self.assertEqual(mapping, {})

    def test_enabled_by_default(self):
        import os

        os.environ.pop("SYNTACTIC_PAYLOAD_ELISION", None)
        _, mapping = maybe_elide_payloads(BROKEN)
        self.assertTrue(mapping)

    def test_empty_source_is_safe(self):
        self.assertEqual(maybe_elide_payloads("")[1], {})
        self.assertEqual(maybe_elide_payloads(None)[1], {})

    def test_elision_does_not_move_the_syntax_error(self):
        # The repair window is built from the error's line number. If elision shifted it, the
        # window would be constructed around the wrong part of the file.
        import ast

        elided, _ = maybe_elide_payloads(BROKEN)
        for source in (BROKEN, elided):
            with self.assertRaises(SyntaxError) as ctx:
                ast.parse(source)
            self.assertEqual(ctx.exception.lineno, 2)


if __name__ == "__main__":
    unittest.main()
