"""Unit tests for the tail-return-value recovery operator
(utils/semantic_operators/leaf.py :: tail_return_candidate).

These use REAL compiled bytecode (dis) so the assertions are version-robust: the
tail signature (`LOAD_<name>; RETURN_VALUE` in GT vs an implicit `return None` in
the derived code) is read straight from the host interpreter's instruction stream.
"""
import ast
import dis
import textwrap
import unittest

from utils.semantic_operators.leaf import (
    const_inline_candidate,
    extract_operand_divergence,
    leaf_value_candidate,
    tail_return_candidate,
)


def _single_func_code(fragment: str):
    """Compile a (possibly indented) single-function fragment and return its code object."""
    module = compile(textwrap.dedent(fragment), "<frag>", "exec")
    funcs = [c for c in module.co_consts if hasattr(c, "co_code")]
    assert len(funcs) == 1, f"expected exactly one function, got {funcs!r}"
    return funcs[0]


def _instrs(fragment: str):
    return list(dis.get_instructions(_single_func_code(fragment)))


def _dis_sig(instrs):
    return [(i.opname, i.argrepr) for i in instrs]


def _first_divergence_offset(der, gt):
    """First derived offset where (opname, argval) diverges from GT — matches the
    `failed_offset` the pipeline hands the operator (its exact value is irrelevant to
    tail detection, but leaf_value_candidate requires it to be present)."""
    for k in range(max(len(der), len(gt))):
        a = der[k] if k < len(der) else None
        b = gt[k] if k < len(gt) else None
        if a is None or b is None or a.opname != b.opname or a.argval != b.argval:
            return a.offset if a is not None else (der[-1].offset if der else 0)
    return der[-1].offset if der else 0


def _ctx(der, gt):
    return {
        "pylingual_failed_result": {"message": "Different bytecode"},
        "failed_offset": _first_divergence_offset(der, gt),
    }


# --- shared fixtures --------------------------------------------------------

# module-level def, body indent 4: dead `return matches` over-indented after `break`.
DER_MODULE = (
    "def find_all_images_in_window(x):\n"
    "    matches = []\n"
    "    for pt in x:\n"
    "        matches.append(pt)\n"
    "        if len(matches) >= 2:\n"
    "            break\n"
    "            return matches\n"
)
GT_MODULE = (
    "def find_all_images_in_window(x):\n"
    "    matches = []\n"
    "    for pt in x:\n"
    "        matches.append(pt)\n"
    "        if len(matches) >= 2:\n"
    "            break\n"
    "    return matches\n"
)

# method fragment, raw body indent 8 (def at col 4, body at col 8).
DER_METHOD = (
    "    def getPromptCarFeatures(self, x):\n"
    "        responsesCarFeatures = []\n"
    "        for pt in x:\n"
    "            responsesCarFeatures.append(pt)\n"
    "            if pt:\n"
    "                break\n"
    "                return responsesCarFeatures\n"
)
GT_METHOD = (
    "    def getPromptCarFeatures(self, x):\n"
    "        responsesCarFeatures = []\n"
    "        for pt in x:\n"
    "            responsesCarFeatures.append(pt)\n"
    "            if pt:\n"
    "                break\n"
    "        return responsesCarFeatures\n"
)


class TailReturnGuardTest(unittest.TestCase):
    def test_fail_before_existing_handlers_decline(self):
        der, gt = _instrs(DER_MODULE), _instrs(GT_MODULE)
        ctx = _ctx(der, gt)
        # The pre-existing leaf handlers cannot classify this divergence...
        self.assertIsNone(extract_operand_divergence(gt, der, ctx["failed_offset"]))
        self.assertIsNone(const_inline_candidate(gt, der, DER_MODULE, ctx))
        # ...but the new tail-return operator produces a candidate.
        cand = tail_return_candidate(gt, der, DER_MODULE, ctx)
        self.assertIsNotNone(cand)
        self.assertEqual(cand["kind"], "tail_return")


class TailReturnPositiveTest(unittest.TestCase):
    def _assert_fixes(self, der_frag, gt_frag):
        der, gt = _instrs(der_frag), _instrs(gt_frag)
        cand = tail_return_candidate(gt, der, der_frag, _ctx(der, gt))
        self.assertIsNotNone(cand)
        fixed = _instrs(cand["text"])
        self.assertEqual(_dis_sig(fixed), _dis_sig(gt))

    def test_module_level_def_body_indent_4(self):
        self._assert_fixes(DER_MODULE, GT_MODULE)

    def test_method_body_indent_8(self):
        self._assert_fixes(DER_METHOD, GT_METHOD)

    def test_candidate_text_differs_and_parses(self):
        der, gt = _instrs(DER_MODULE), _instrs(GT_MODULE)
        cand = tail_return_candidate(gt, der, DER_MODULE, _ctx(der, gt))
        self.assertNotEqual(cand["text"], DER_MODULE)
        ast.parse(textwrap.dedent(cand["text"]))  # must not raise


class TailReturnMultiReturnTest(unittest.TestCase):
    def test_two_over_indented_returns_first_valid_fixes(self):
        der = (
            "def getLibDep(x):\n"
            "    libs = []\n"
            "    for a in x:\n"
            "        libs.append(a)\n"
            "        if a > 0:\n"
            "            break\n"
            "            return libs\n"   # 12sp, after break (dead)
            "        return libs\n"        # 8sp, over-indented inside for
        )
        gt = (
            "def getLibDep(x):\n"
            "    libs = []\n"
            "    for a in x:\n"
            "        libs.append(a)\n"
            "        if a > 0:\n"
            "            break\n"
            "    return libs\n"
        )
        di, gi = _instrs(der), _instrs(gt)
        cand = tail_return_candidate(gi, di, der, _ctx(di, gi))
        self.assertIsNotNone(cand)
        self.assertEqual(_dis_sig(_instrs(cand["text"])), _dis_sig(gi))


class TailReturnNegativeTest(unittest.TestCase):
    def test_gt_tail_also_none_declines(self):
        # 4a: GT itself falls through to an implicit `return None` -> nothing to recover.
        der = (
            "def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
        )
        gt = der
        di, gi = _instrs(der), _instrs(gt)
        self.assertIsNone(tail_return_candidate(gi, di, der, _ctx(di, gi)))

    def test_gt_tail_name_but_no_return_token_declines(self):
        # 4b: GT wants `return acc` but the derived fragment has NO `return acc` at all.
        der = (
            "def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
        )
        gt = (
            "def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "    return acc\n"
        )
        di, gi = _instrs(der), _instrs(gt)
        self.assertTrue(gi[-1].opname == "RETURN_VALUE")
        self.assertIsNone(tail_return_candidate(gi, di, der, _ctx(di, gi)))

    def test_return_already_at_body_indent_declines(self):
        # 4c: the `return acc` is already at the body indent -> not over-indented.
        der = (
            "def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "    return acc\n"
        )
        gt = der
        di, gi = _instrs(der), _instrs(gt)
        self.assertIsNone(tail_return_candidate(gi, di, der, _ctx(di, gi)))

    def test_wrong_failed_message_declines(self):
        der, gt = _instrs(DER_MODULE), _instrs(GT_MODULE)
        ctx = {"pylingual_failed_result": {"message": "Different control flow"},
               "failed_offset": 0}
        self.assertIsNone(tail_return_candidate(gt, der, DER_MODULE, ctx))

    def test_missing_repair_context_declines(self):
        der, gt = _instrs(DER_MODULE), _instrs(GT_MODULE)
        self.assertIsNone(tail_return_candidate(gt, der, DER_MODULE, None))

    def test_return_of_expression_not_bare_name_declines(self):
        # GT tail loads a bare name; a derived over-indented `return acc + [1]` is a
        # different value expression, not the name -> decline.
        der = (
            "def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "        if a:\n"
            "            break\n"
            "            return acc + [1]\n"
        )
        gt = _instrs(GT_MODULE.replace("find_all_images_in_window", "f")
                     .replace("matches", "acc"))
        di = _instrs(der)
        self.assertIsNone(tail_return_candidate(gt, di, der, _ctx(di, gt)))

    def test_over_indented_return_of_different_name_declines(self):
        der = (
            "def f(x):\n"
            "    acc = []\n"
            "    other = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "        if a:\n"
            "            break\n"
            "            return other\n"   # GT wants `acc`, not `other`
        )
        gt = _instrs(
            "def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "        if a:\n"
            "            break\n"
            "    return acc\n"
        )
        di = _instrs(der)
        self.assertIsNone(tail_return_candidate(gt, di, der, _ctx(di, gt)))

    def test_async_function_epilogue_declines(self):
        # Coroutines/generators append a STOPITERATION_ERROR;RERAISE epilogue after the
        # real return, so the stream tail is not a plain return -> safe decline.
        der = (
            "async def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "        if a:\n"
            "            break\n"
            "            return acc\n"
        )
        gt = _instrs(
            "async def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "        if a:\n"
            "            break\n"
            "    return acc\n"
        )
        di = _instrs(der)
        self.assertIsNone(tail_return_candidate(gt, di, der, _ctx(di, gt)))

    def test_return_inside_nested_def_not_recovered(self):
        # A `return name` that belongs to a nested function (a different code object)
        # must not be dragged up into the outer function's body.
        der = (
            "def outer(x):\n"
            "    def inner():\n"
            "        acc = 1\n"
            "        if x:\n"
            "            return acc\n"
            "    return inner\n"
        )
        # GT (hypothetically) wants outer to tail-return `acc` — but there is no
        # over-indented `return acc` in outer's OWN scope, only in inner's.
        gt = _instrs(
            "def outer(x):\n"
            "    acc = 2\n"
            "    return acc\n"
        )
        di = _instrs(der)
        self.assertIsNone(tail_return_candidate(gt, di, der, _ctx(di, gt)))


class TailReturnWiringTest(unittest.TestCase):
    def test_leaf_value_candidate_uses_tail_return_fallback(self):
        der, gt = _instrs(DER_MODULE), _instrs(GT_MODULE)
        cand = leaf_value_candidate(gt, der, DER_MODULE, _ctx(der, gt))
        self.assertIsNotNone(cand)
        self.assertEqual(cand["kind"], "tail_return")
        self.assertEqual(_dis_sig(_instrs(cand["text"])), _dis_sig(gt))


class TailReturnLastStatementTest(unittest.TestCase):
    def test_declines_when_recovered_return_would_not_be_last(self):
        # Review fix: dedenting the over-indented `return acc` lands it BEFORE the
        # trailing `acc.append(0)`, making that statement dead -> must decline.
        der = (
            "def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "        if a:\n"
            "            break\n"
            "            return acc\n"
            "    acc.append(0)\n"
        )
        gt = _instrs(
            "def f(x):\n"
            "    acc = []\n"
            "    for a in x:\n"
            "        acc.append(a)\n"
            "        if a:\n"
            "            break\n"
            "    acc.append(0)\n"
            "    return acc\n"
        )
        di = _instrs(der)
        self.assertIsNone(tail_return_candidate(gt, di, der, _ctx(di, gt)))


if __name__ == "__main__":
    unittest.main()
