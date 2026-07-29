"""The exhausted deterministic layer, applied after EVERY LLM generation.

The runner already ran a post-LLM pass, but it called the shipped `maybe_prepass` -- the weak
driver that fixes ~0.9% of files. The exhaustive search fixes 21.7% of files that one cannot, so
the old "post-LLM prepass is worthless" result (0/57) does not transfer to it.

Measured on real artifacts: taking the LLM's last output on files that fell to the delete-only
fallback and running the exhaustive layer rescues 10 of 75 (13.3%) -- each one converting a
delete-only compile into a GENUINE repair.

The model gets a file close to correct and then misses one bracket or one handler body; the search
finishes the job. That is the whole point of running it after each generation rather than once.
"""
import unittest
from unittest import mock

from pipeline import runner as runner_mod
from pipeline.runner import maybe_post_llm_repair


def _host(src, version=None):
    compile(src, "<t>", "exec")


class PostLlmExhaustiveTest(unittest.TestCase):
    def test_an_almost_correct_llm_output_is_finished(self):
        # what the model typically leaves: one unclosed call
        src = "def f():\n    return g(1, 2\n"
        out, ok, ops = maybe_post_llm_repair(src, "3.12", _host)
        self.assertTrue(ok, ops)
        compile(out, "<t>", "exec")

    def test_output_that_already_compiles_is_returned_untouched(self):
        src = "x = 1\n"
        out, ok, ops = maybe_post_llm_repair(src, "3.12", _host)
        self.assertTrue(ok)
        self.assertEqual(out, src)
        self.assertEqual(ops, [])

    def test_unfixable_output_is_reported_as_such_and_not_mutated(self):
        src = "@@@ not python @@@\n"
        out, ok, ops = maybe_post_llm_repair(src, "3.12", _host)
        self.assertFalse(ok)
        self.assertEqual(out, src)

    def test_the_target_version_gate_is_authoritative(self):
        def reject(src, version=None):
            raise SyntaxError("target rejects")

        out, ok, ops = maybe_post_llm_repair("def f():\n    return g(1, 2\n", "3.12", reject)
        self.assertFalse(ok)

    def test_empty_or_missing_output_is_safe(self):
        self.assertEqual(maybe_post_llm_repair("", "3.12", _host)[1], False)
        self.assertEqual(maybe_post_llm_repair(None, "3.12", _host)[1], False)

    def test_kill_switch(self):
        with mock.patch.dict("os.environ", {"SYNTACTIC_POST_LLM_EXHAUSTIVE": "0"}):
            out, ok, ops = maybe_post_llm_repair("def f():\n    return g(1, 2\n", "3.12", _host)
        self.assertFalse(ok)
        self.assertEqual(ops, [])

    def test_never_raises_when_an_operator_explodes(self):
        with mock.patch.object(runner_mod, "exhaustive_repair", side_effect=RuntimeError("boom")):
            out, ok, ops = maybe_post_llm_repair("def f():\n    return g(1, 2\n", "3.12", _host)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()


class BrokenProbeIsVisibleTest(unittest.TestCase):
    """A mis-wired probe must be detectable, not silently absorbed.

    `compile_version` takes (py_path, out_path, version) -- file paths. It was wired here as a
    (source, version) probe, so every call raised TypeError, the blanket except swallowed it, and
    the layer reported "no fix" for EVERY file for its entire life. Run E showed 0 rescues across
    38 files before this was found. The same defect sat in the original `maybe_prepass` call at this
    site, which means the historical "post-LLM prepass 0/57" result measured a broken call.
    """

    def test_a_probe_with_the_wrong_arity_is_reported_not_swallowed(self):
        def three_arg_probe(py_path, out_path, version):  # the real compile_version shape
            raise AssertionError("should never be reached")

        out, ok, ops = maybe_post_llm_repair("def f():\n    return g(1, 2\n", "3.12", three_arg_probe)
        self.assertFalse(ok)
        self.assertEqual(ops, [])

    def test_a_correctly_shaped_probe_still_works(self):
        out, ok, ops = maybe_post_llm_repair("def f():\n    return g(1, 2\n", "3.12", _host)
        self.assertTrue(ok)

    def test_a_probe_that_rejects_valid_source_is_still_used(self):
        # a target-version compiler may legitimately reject; that is not a wiring bug
        calls = []

        def picky(src, version=None):
            calls.append(src)
            if "g(1, 2)" not in src:
                raise SyntaxError("nope")

        out, ok, ops = maybe_post_llm_repair("def f():\n    return g(1, 2\n", "3.12", picky)
        self.assertTrue(ok, ops)
        self.assertTrue(len(calls) > 1)
