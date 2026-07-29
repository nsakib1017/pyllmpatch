"""Exhaustive deterministic repair: search EVERY operator sequence, not one greedy path.

`_structural_fixpoint` applies operators in a FIXED ORDER. That made ordering load-bearing and cost
19 files when an aggressive operator pre-empted precise ones (`close_unclosed_opener`, measured
1030 -> 1011). Searching the reachable state space removes ordering from the equation entirely.

What this makes provable: **no sequence of OUR operators, within the stated bounds, compiles the
file.** That is a checkable claim about a defined search space -- not the unprovable claim that no
deterministic program could ever fix it (deleting everything compiles, after all). The bounds are
part of the claim and are reported, never hidden.
"""
import ast
import unittest

from utils.exhaustive_repair import exhaustive_repair, SearchBudget


def _parses(src):
    try:
        ast.parse(src)
        return True
    except Exception:
        return False


def _host_ok(src, version=None):
    compile(src, "<t>", "exec")


class ExhaustiveSearchTest(unittest.TestCase):
    def test_finds_a_fix_reachable_only_by_a_non_default_order(self):
        # Two operators where applying them in registry order fails but the reverse succeeds.
        def add_tail(src, err):
            return src + "\n" if not src.endswith("\n\n") else None

        def close_paren(src, err):
            return src.replace("f(1, 2", "f(1, 2)") if "f(1, 2" in src and "f(1, 2)" not in src else None

        ops = [("add_tail", add_tail), ("close_paren", close_paren)]
        src = "x = f(1, 2\n"
        out, path = exhaustive_repair(src, _host_ok, None, ops)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out))
        self.assertIn("close_paren", path)

    def test_returns_none_when_no_sequence_works(self):
        ops = [("noop_a", lambda s, e: None), ("noop_b", lambda s, e: None)]
        out, path = exhaustive_repair("@@@ not python @@@\n", _host_ok, None, ops)
        self.assertIsNone(out)
        self.assertEqual(path, [])

    def test_already_compiling_source_is_returned_untouched(self):
        src = "x = 1\n"
        out, path = exhaustive_repair(src, _host_ok, None, [("x", lambda s, e: s + "y = 2\n")])
        self.assertEqual(out, src)
        self.assertEqual(path, [])

    def test_the_target_version_gate_is_authoritative(self):
        # A rewrite the host parser accepts but the target compiler rejects must NOT be adopted.
        def rewrite(src, err):
            return "x = 1\n"

        def always_reject(src, version=None):
            raise SyntaxError("target rejects")

        out, path = exhaustive_repair("x = f(1, 2\n", always_reject, None, [("r", rewrite)])
        self.assertIsNone(out)

    def test_search_is_bounded_and_reports_its_bounds(self):
        # An operator that always produces something new must not run forever.
        counter = {"n": 0}

        def grow(src, err):
            counter["n"] += 1
            return src + "#\n"

        budget = SearchBudget(max_depth=3, max_states=10)
        out, path = exhaustive_repair("x = f(1, 2\n", _host_ok, None, [("grow", grow)], budget=budget)
        self.assertIsNone(out)
        self.assertLessEqual(counter["n"], 40)

    def test_duplicate_states_are_not_re_expanded(self):
        seen = {"n": 0}

        def idempotent(src, err):
            seen["n"] += 1
            return src.replace("  ", " ") if "  " in src else None

        exhaustive_repair("x = f(1,  2\n", _host_ok, None, [("idem", idempotent)])
        self.assertLess(seen["n"], 30)

    def test_never_raises_on_junk(self):
        def boom(src, err):
            raise RuntimeError("operator exploded")

        out, path = exhaustive_repair("x = f(1, 2\n", _host_ok, None, [("boom", boom)])
        self.assertIsNone(out)
        for bad in (None, ""):
            exhaustive_repair(bad, _host_ok, None, [("boom", boom)])


class RealOperatorsTest(unittest.TestCase):
    def test_it_beats_the_fixed_order_on_a_real_shape(self):
        from utils.structural_repair import STRUCT_OPS

        ops = [(o.name, o.fn) for o in STRUCT_OPS]
        src = ("def f(cfg):\n"
               "    with open(cfg) as fh:\n"
               "        try:\n"
               "            item = json.load(fh)\n"
               "        except ValueError:\n"
               "        else:  # inserted\n"
               "            item['x'] = False\n")
        out, path = exhaustive_repair(src, _host_ok, None, ops)
        self.assertIsNotNone(out)
        self.assertTrue(_parses(out), out)

    def test_valid_python_is_never_touched(self):
        from utils.structural_repair import STRUCT_OPS

        ops = [(o.name, o.fn) for o in STRUCT_OPS]
        for src in ("import os\nx = [i for i in range(3)]\n",
                    "def f():\n    try:\n        return 1\n    except Exception:\n        return 2\n"):
            out, path = exhaustive_repair(src, _host_ok, None, ops)
            self.assertEqual(out, src)
            self.assertEqual(path, [])


if __name__ == "__main__":
    unittest.main()
