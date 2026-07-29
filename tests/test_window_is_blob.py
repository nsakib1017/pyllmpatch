"""window_exceeds_token_budget gates the runtime window discard.

Policy: a syntactic-repair file is DISCARDED (not attempted, excluded from the analyzability
denominator) when its fix window exceeds the LLM generation budget (2048 tokens) -- the model
cannot regenerate a window larger than its output cap, so the fix is truncated and the attempt
is doomed. The token count comes from the model tokenizer; here it is injected so the decision
logic is tested purely.
"""
import unittest

from pipeline.logging_utils import window_exceeds_token_budget


def fake_count(mapping):
    return lambda text: mapping.get(text, len(text) // 4)


class WindowExceedsTokenBudgetTest(unittest.TestCase):
    def test_under_budget_is_kept(self):
        self.assertFalse(window_exceeds_token_budget("small window", 2048, count_tokens=lambda t: 500))

    def test_over_budget_is_discarded(self):
        self.assertTrue(window_exceeds_token_budget("big window", 2048, count_tokens=lambda t: 2049))

    def test_at_budget_is_kept(self):
        self.assertFalse(window_exceeds_token_budget("exactly", 2048, count_tokens=lambda t: 2048))

    def test_none_or_empty_never_exceeds(self):
        self.assertFalse(window_exceeds_token_budget(None, 2048, count_tokens=lambda t: 99999))
        self.assertFalse(window_exceeds_token_budget("", 2048, count_tokens=lambda t: 99999))

    def test_nonpositive_budget_disables_guard(self):
        self.assertFalse(window_exceeds_token_budget("x" * 100000, 0, count_tokens=lambda t: 99999))
        self.assertFalse(window_exceeds_token_budget("x" * 100000, -1, count_tokens=lambda t: 99999))

    def test_counter_is_actually_used(self):
        # a normal-looking multi-line window that tokenizes to >2048 must still be discarded
        window = "\n".join(["x = f(a, b, c)"] * 300)
        self.assertTrue(window_exceeds_token_budget(window, 2048, count_tokens=lambda t: 2500))


if __name__ == "__main__":
    unittest.main()
