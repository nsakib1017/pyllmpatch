"""Unit tests for the leaf-value operator (utils/semantic_operators/leaf.py, M3)."""
import unittest

from utils.semantic_operators.leaf import (
    const_inline_candidate,
    const_to_name_candidate,
    extract_operand_divergence,
    extract_operator_divergence,
    leaf_value_candidate,
    swap_attr,
    swap_compare_operator,
    swap_format_value,
    swap_is_contains,
    swap_literal,
    swap_local_rename,
    swap_name,
    swap_unique_binary_operator,
)


class _Inst:
    def __init__(self, offset, opname, argrepr, argval=None):
        self.offset, self.opname, self.argrepr = offset, opname, argrepr
        self.argval = argval


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


class SwapOperatorTest(unittest.TestCase):
    def test_simple_top_level(self):
        frag = "def f(x):\n    return x + 1\n"
        out = swap_unique_binary_operator(frag, "+", "*")
        self.assertEqual(out, "def f(x):\n    return x * 1\n")

    def test_indented_method_bitop(self):
        frag = "    def m(self):\n        return self.a | self.b\n"
        out = swap_unique_binary_operator(frag, "|", "/")
        self.assertEqual(out, "    def m(self):\n        return self.a / self.b\n")

    def test_non_unique_defers(self):
        frag = "def f(x):\n    return x + 1 + 2\n"  # two '+' -> ambiguous
        self.assertIsNone(swap_unique_binary_operator(frag, "+", "*"))

    def test_other_operators_present_but_target_unique(self):
        frag = "def f(a, b, c):\n    return a * b + c\n"  # one '+', one '*'
        self.assertEqual(
            swap_unique_binary_operator(frag, "+", "-"),
            "def f(a, b, c):\n    return a * b - c\n",
        )

    def test_unparseable_returns_none(self):
        self.assertIsNone(swap_unique_binary_operator("return x + (\n", "+", "*"))

    def test_non_ascii_line_defers(self):
        frag = "def f(x):\n    return x + 'ключ'\n"  # non-ascii on operator line
        self.assertIsNone(swap_unique_binary_operator(frag, "+", "*"))


class ExtractDivergenceTest(unittest.TestCase):
    def test_binary_op_mismatch(self):
        der = _BC([_Inst(0, "RESUME", ""), _Inst(18, "BINARY_OP", "+")])
        gt = _BC([_Inst(0, "RESUME", ""), _Inst(18, "BINARY_OP", "*")])
        div = extract_operator_divergence(gt, der, 18)
        self.assertEqual(div, {"opname": "BINARY_OP", "wrong": "+", "right": "*", "ordinal": 0})

    def test_ordinal_counts_preceding_same_operator(self):
        der = _BC([_Inst(4, "BINARY_OP", "+"), _Inst(10, "BINARY_OP", "+"), _Inst(16, "BINARY_OP", "+")])
        gt = _BC([_Inst(4, "BINARY_OP", "+"), _Inst(10, "BINARY_OP", "+"), _Inst(16, "BINARY_OP", "*")])
        div = extract_operator_divergence(gt, der, 16)
        self.assertEqual(div["ordinal"], 2)  # two '+' BINARY_OPs precede offset 16

    def test_different_opcode_defers(self):
        der = _BC([_Inst(14, "LOAD_NAME", "word_list")])
        gt = _BC([_Inst(14, "LOAD_CONST", "(...)")])
        self.assertIsNone(extract_operator_divergence(gt, der, 14))

    def test_inplace_or_unknown_symbol_defers(self):
        der = _BC([_Inst(2, "BINARY_OP", "+=")])
        gt = _BC([_Inst(2, "BINARY_OP", "-=")])
        self.assertIsNone(extract_operator_divergence(gt, der, 2))

    def test_offset_not_found(self):
        der = _BC([_Inst(0, "BINARY_OP", "+")])
        gt = _BC([_Inst(0, "BINARY_OP", "*")])
        self.assertIsNone(extract_operator_divergence(gt, der, 999))


class LeafCandidateTest(unittest.TestCase):
    def test_end_to_end_candidate(self):
        der = _BC([_Inst(0, "RESUME", ""), _Inst(18, "BINARY_OP", "+")])
        gt = _BC([_Inst(0, "RESUME", ""), _Inst(18, "BINARY_OP", "*")])
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 18}
        out = leaf_value_candidate(gt, der, "def f(x):\n    return x + 1\n", ctx)
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n    return x * 1\n")
        self.assertEqual(out["operator"], "+ -> *")

    def test_defers_without_offset(self):
        der = _BC([_Inst(18, "BINARY_OP", "+")])
        gt = _BC([_Inst(18, "BINARY_OP", "*")])
        ctx = {"pylingual_failed_result": {"message": "Different control flow"}, "failed_offset": None}
        self.assertIsNone(leaf_value_candidate(gt, der, "x + 1", ctx))

    def test_ordinal_disambiguation_on_non_unique(self):
        # Two '+' in source; the failing BINARY_OP is the first (ordinal 0).
        der = _BC([_Inst(18, "BINARY_OP", "+")])
        gt = _BC([_Inst(18, "BINARY_OP", "*")])
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 18}
        out = leaf_value_candidate(gt, der, "x + 1 + 2", ctx)
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "x * 1 + 2")
        self.assertEqual(out["confidence"], "ordinal")

    def test_swap_unique_wrapper_still_defers_on_non_unique(self):
        # The legacy unique-only wrapper must keep deferring when ambiguous.
        self.assertIsNone(swap_unique_binary_operator("x + 1 + 2", "+", "*"))


class Py310BinaryOpTest(unittest.TestCase):
    """3.10 encodes the binary operator in the OPCODE NAME (BINARY_ADD vs BINARY_SUBTRACT)
    with no argrepr, so a swap surfaces as *different opnames*. The operator must handle it."""

    def test_extract_operand_divergence_310(self):
        der = _BC([_Inst(0, "RESUME", ""), _Inst(18, "BINARY_ADD", "")])
        gt = _BC([_Inst(0, "RESUME", ""), _Inst(18, "BINARY_SUBTRACT", "")])
        div = extract_operand_divergence(gt, der, 18)
        self.assertEqual(div, {"opname": "BINARY_ADD", "kind": "binary_op",
                               "wrong": "+", "right": "-", "ordinal": 0})

    def test_extract_operator_divergence_310(self):
        der = _BC([_Inst(18, "BINARY_MULTIPLY", "")])
        gt = _BC([_Inst(18, "BINARY_TRUE_DIVIDE", "")])
        div = extract_operator_divergence(gt, der, 18)
        self.assertEqual(div, {"opname": "BINARY_MULTIPLY", "wrong": "*", "right": "/", "ordinal": 0})

    def test_leaf_candidate_310_end_to_end(self):
        der = _BC([_Inst(0, "RESUME", ""), _Inst(18, "BINARY_ADD", "")])
        gt = _BC([_Inst(0, "RESUME", ""), _Inst(18, "BINARY_SUBTRACT", "")])
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 18}
        out = leaf_value_candidate(gt, der, "def f(x):\n    return x + 1\n", ctx)
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f(x):\n    return x - 1\n")
        self.assertEqual(out["operator"], "+ -> -")

    def test_310_ordinal_on_non_unique(self):
        der = _BC([_Inst(18, "BINARY_ADD", "")])
        gt = _BC([_Inst(18, "BINARY_SUBTRACT", "")])
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 18}
        out = leaf_value_candidate(gt, der, "x + 1 + 2", ctx)
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "x - 1 + 2")

    def test_310_different_opname_non_binop_defers(self):
        # BINARY_ADD vs LOAD_CONST at the same offset is a STRUCTURAL diff, not an operator
        # swap — only one side is a mapped binary opcode, so we must not fire.
        der = _BC([_Inst(18, "BINARY_ADD", "")])
        gt = _BC([_Inst(18, "LOAD_CONST", "", 1)])
        self.assertIsNone(extract_operand_divergence(gt, der, 18))

    def test_310_inplace_augmented_defers(self):
        # INPLACE_* (x += y => ast.AugAssign) is intentionally unmapped; the BinOp swap
        # can't apply it, so the operator must defer rather than misfire.
        der = _BC([_Inst(18, "INPLACE_ADD", "")])
        gt = _BC([_Inst(18, "INPLACE_SUBTRACT", "")])
        self.assertIsNone(extract_operand_divergence(gt, der, 18))


class TupleConstSwapTest(unittest.TestCase):
    """Constant TUPLES fold to a single LOAD_CONST; decompilers mis-place elements
    (e.g. __slots__/__match_args__/column-def tuples). The literal swap must handle them."""

    def test_extract_tuple_const_divergence(self):
        der = _BC([_Inst(10, "LOAD_CONST", "", (None, "a", "b", None))])
        gt = _BC([_Inst(10, "LOAD_CONST", "", (None, "a", None, "b"))])
        div = extract_operand_divergence(gt, der, 10)
        self.assertIsNotNone(div)
        self.assertEqual(div["kind"], "literal")
        self.assertEqual(div["wrong"], (None, "a", "b", None))
        self.assertEqual(div["right"], (None, "a", None, "b"))

    def test_swap_literal_tuple(self):
        frag = "def f():\n    cols = (None, 'a', 'b', None)\n    return cols\n"
        out = swap_literal(frag, (None, "a", "b", None), (None, "a", None, "b"))
        self.assertIsNotNone(out)
        self.assertIn("cols = (None, 'a', None, 'b')", out[0])

    def test_leaf_candidate_tuple_const_end_to_end(self):
        der = _BC([_Inst(0, "RESUME", ""), _Inst(10, "LOAD_CONST", "", (None, "a", "b", None))])
        gt = _BC([_Inst(0, "RESUME", ""), _Inst(10, "LOAD_CONST", "", (None, "a", None, "b"))])
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 10}
        out = leaf_value_candidate(gt, der, "cols = (None, 'a', 'b', None)\n", ctx)
        self.assertIsNotNone(out)
        self.assertIn("(None, 'a', None, 'b')", out["text"])

    def test_nested_tuple_const_defers(self):
        # nested (non-flat) constant tuples are intentionally not handled -> defer safely.
        der = _BC([_Inst(10, "LOAD_CONST", "", (1, (2, 3)))])
        gt = _BC([_Inst(10, "LOAD_CONST", "", (1, (3, 2)))])
        self.assertIsNone(extract_operand_divergence(gt, der, 10))


class CompareOpTest(unittest.TestCase):
    def test_swap_comparison(self):
        self.assertEqual(swap_compare_operator("def f(a, b):\n    return a < b\n", "<", ">=", 0),
                         ("def f(a, b):\n    return a >= b\n", "unique"))

    def test_chained_comparison_ordinal(self):
        # a < b < c : two '<' ops; ordinal 1 targets the second.
        out = swap_compare_operator("def f(a, b, c):\n    return a < b < c\n", "<", "<=", 1)
        self.assertEqual(out, ("def f(a, b, c):\n    return a < b <= c\n", "ordinal"))

    def test_declines_unknown_symbol(self):
        self.assertIsNone(swap_compare_operator("a in b", "in", "not in", 0))  # not in cmp map


class LiteralSwapTest(unittest.TestCase):
    def test_number(self):
        self.assertEqual(swap_literal("def f():\n    return 0\n", 0, 1, 0),
                         ("def f():\n    return 1\n", "unique"))

    def test_string(self):
        self.assertEqual(swap_literal("def f():\n    return 'a'\n", "a", "b", 0),
                         ("def f():\n    return 'b'\n", "unique"))

    def test_bool_not_matched_as_int(self):
        # wrong_value True must not match a literal 1
        self.assertIsNone(swap_literal("def f():\n    return 1\n", True, 2, 0))

    def test_implicit_none_has_no_token(self):
        self.assertIsNone(swap_literal("def f():\n    pass\n", None, "en", 0))


class ConstInlineTest(unittest.TestCase):
    def _bc(self, insts):
        class _B:
            def __init__(s, i):
                s._i = i
            def __iter__(s):
                return iter(s._i)
        return _B(insts)

    def test_inline_self_reference_placeholder(self):
        der = self._bc([_Inst(14, "LOAD_NAME", "word_list")])
        gt = self._bc([_Inst(14, "LOAD_CONST", "(...)")])
        # give GT inst a real argval
        list(gt)[0].argval = ("a", "b", "c")
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 14}
        out = const_inline_candidate(gt, der, "class P:\n    word_list = word_list\n", ctx)
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "class P:\n    word_list = ('a', 'b', 'c')\n")
        self.assertEqual(out["kind"], "const_inline")

    def test_declines_non_roundtrip(self):
        der = self._bc([_Inst(14, "LOAD_NAME", "x")])
        gt = self._bc([_Inst(14, "LOAD_CONST", "")])
        list(gt)[0].argval = object()  # not a literal -> not round-trippable
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 14}
        self.assertIsNone(const_inline_candidate(gt, der, "x = x\n", ctx))


class NameAttrSwapTest(unittest.TestCase):
    def test_name_swap(self):
        self.assertEqual(swap_name("def f(a, b):\n    return a\n", "a", "b", 0),
                         ("def f(a, b):\n    return b\n", "unique"))

    def test_name_swap_only_load_context(self):
        # `a = x` : `a` is Store ctx, `x` is Load. Swapping load-name x->y.
        self.assertEqual(swap_name("a = x\nreturn a\n", "x", "y", 0),
                         ("a = y\nreturn a\n", "unique"))

    def test_attr_swap_trailing_only(self):
        # chained x.a.b : swapping .b must not touch .a
        self.assertEqual(swap_attr("def f(x):\n    return x.a.b\n", "b", "c", 0),
                         ("def f(x):\n    return x.a.c\n", "unique"))

    def test_attr_swap_declines_when_tail_mismatch(self):
        self.assertIsNone(swap_attr("def f(x):\n    return x.foo\n", "bar", "baz", 0))

    def test_extract_name_divergence(self):
        der = _BC([_Inst(6, "LOAD_FAST", "a")])
        gt = _BC([_Inst(6, "LOAD_FAST", "b")])
        div = extract_operand_divergence(gt, der, 6)
        self.assertEqual(div, {"opname": "LOAD_FAST", "kind": "name", "wrong": "a", "right": "b", "total": 1})

    def test_extract_name_total_counts_whole_object(self):
        # two loads of `a` in the object -> total == 2 (dispatch will then defer)
        der = _BC([_Inst(6, "LOAD_FAST", "a"), _Inst(12, "LOAD_FAST", "a")])
        gt = _BC([_Inst(6, "LOAD_FAST", "b"), _Inst(12, "LOAD_FAST", "a")])
        div = extract_operand_divergence(gt, der, 6)
        self.assertEqual(div["total"], 2)

    def test_name_swap_declines_multiple_occurrences(self):
        # `x` loaded twice -> not globally unique -> defer (bug: consistent-rename risk)
        self.assertIsNone(swap_name("def f():\n    return x + x\n", "x", "y"))

    def test_name_swap_declines_nested_scope(self):
        # `n` also used inside a nested lambda (separate code object) -> non-unique -> defer
        self.assertIsNone(swap_name("def f():\n    h = lambda: n\n    return n\n", "n", "m"))

    def test_attr_swap_ignores_store_context(self):
        # STORE_ATTR target `self.foo` must NOT be counted; only the Load `y.foo` is
        # unique here, so the swap lands on the read, not the write.
        self.assertEqual(
            swap_attr("def f(self, y):\n    self.foo = y\n    return y.foo\n", "foo", "bar"),
            ("def f(self, y):\n    self.foo = y\n    return y.bar\n", "unique"))

    def test_attr_swap_declines_store_only(self):
        # only a STORE_ATTR `foo` exists (no Load) -> no Load-attr node -> defer
        self.assertIsNone(swap_attr("def f(self, y):\n    self.foo = y\n", "foo", "bar"))

    def test_extract_attr_divergence(self):
        der = _BC([_Inst(6, "LOAD_ATTR", "foo")])
        gt = _BC([_Inst(6, "LOAD_ATTR", "bar")])
        div = extract_operand_divergence(gt, der, 6)
        self.assertEqual(div["kind"], "attr")
        self.assertEqual((div["wrong"], div["right"]), ("foo", "bar"))

    def test_extract_scope_change_declines(self):
        # LOAD_FAST vs LOAD_GLOBAL = different opcode -> not a name swap
        der = _BC([_Inst(6, "LOAD_FAST", "a")])
        gt = _BC([_Inst(6, "LOAD_GLOBAL", "a")])
        self.assertIsNone(extract_operand_divergence(gt, der, 6))


class IsContainsSwapTest(unittest.TestCase):
    def test_is_to_is_not(self):
        self.assertEqual(swap_is_contains("def f(a, b):\n    return a is b\n", "is", "is not"),
                         ("def f(a, b):\n    return a is not b\n", "unique"))

    def test_is_not_to_is(self):
        self.assertEqual(swap_is_contains("x = a is not b\n", "is not", "is"),
                         ("x = a is b\n", "unique"))

    def test_in_to_not_in(self):
        self.assertEqual(swap_is_contains("x = a in b\n", "in", "not in"),
                         ("x = a not in b\n", "unique"))

    def test_cross_family_declined(self):
        # never turn `is` into `in` (different opcode family)
        self.assertIsNone(swap_is_contains("x = a is b\n", "is", "in"))

    def test_chained_nested_ordinal_bytecode_order(self):
        # Review regression: `a is b is (c is d)` bytecode op order is
        # [outer0, inner, outer1]; ordinal 0 must edit the OUTER `a is b`, not the
        # nested `c is d` (post-order would have mislocalized).
        self.assertEqual(swap_is_contains("x = a is b is (c is d)\n", "is", "is not", 0),
                         ("x = a is not b is (c is d)\n", "ordinal"))
        # ordinal 1 is the nested `c is d`
        self.assertEqual(swap_is_contains("x = a is b is (c is d)\n", "is", "is not", 1),
                         ("x = a is b is (c is not d)\n", "ordinal"))

    def test_extract_is_op(self):
        der = _BC([_Inst(6, "IS_OP", "is")])
        gt = _BC([_Inst(6, "IS_OP", "is not")])
        div = extract_operand_divergence(gt, der, 6)
        self.assertEqual(div, {"opname": "IS_OP", "kind": "is_contains",
                               "wrong": "is", "right": "is not", "ordinal": 0})


class FormatValueSwapTest(unittest.TestCase):
    def test_add_repr_conversion(self):
        out = swap_format_value("x = f'{a}'\n", -1, 114)
        self.assertEqual(out, ("x = f'{a!r}'\n", "unique"))

    def test_change_conversion(self):
        out = swap_format_value("x = f'{a!s}'\n", 115, 114)
        self.assertEqual(out, ("x = f'{a!r}'\n", "unique"))

    def test_declines_when_ambiguous(self):
        # two fields with the same (no-)conversion -> not unique -> defer
        self.assertIsNone(swap_format_value("x = f'{a} {b}'\n", -1, 114))

    def test_extract_format_value(self):
        der = _BC([_Inst(6, "FORMAT_VALUE", "")])
        gt = _BC([_Inst(6, "FORMAT_VALUE", "!r")])
        div = extract_operand_divergence(gt, der, 6)
        self.assertEqual(div, {"opname": "FORMAT_VALUE", "kind": "format_value",
                               "wrong": -1, "right": 114, "total": 1})


class LocalRenameTest(unittest.TestCase):
    def test_consistent_rename_all_occurrences(self):
        # param + two uses all renamed together
        out = swap_local_rename("def f(x):\n    y = x + 1\n    return x * y\n", "x", "z")
        self.assertEqual(out, ("def f(z):\n    y = z + 1\n    return z * y\n", "rename"))

    def test_rename_store_and_load(self):
        out = swap_local_rename("def f():\n    a = 1\n    return a\n", "a", "b")
        self.assertEqual(out, ("def f():\n    b = 1\n    return b\n", "rename"))

    def test_declines_collision_with_existing_name(self):
        # `y` already exists -> renaming x->y would merge two variables
        self.assertIsNone(swap_local_rename("def f(x, y):\n    return x + y\n", "x", "y"))

    def test_declines_nested_scope_shadow(self):
        # `x` also bound in a nested lambda -> ambiguous -> defer
        self.assertIsNone(swap_local_rename("def f(x):\n    g = lambda x: x\n    return x\n", "x", "z"))

    def test_declines_non_single_function(self):
        self.assertIsNone(swap_local_rename("a = x\nb = x\n", "x", "y"))

    def test_declines_keyword_target(self):
        self.assertIsNone(swap_local_rename("def f(x):\n    return x\n", "x", "class"))

    def test_declines_direct_child_nested_class(self):
        # review bug 1: a `class C:` directly in the body defines a distinct binding
        # `wrong` in another code object — must NOT be renamed (declines via nested guard)
        self.assertIsNone(swap_local_rename(
            "def f(wrong):\n    class C:\n        wrong = 1\n    return wrong\n", "wrong", "right"))

    def test_declines_direct_child_nested_def(self):
        self.assertIsNone(swap_local_rename(
            "def f(wrong):\n    def g(wrong):\n        return wrong\n    return wrong\n", "wrong", "right"))

    def test_does_not_rename_param_default(self):
        # review bug 2: default `b=wrong` resolves in the ENCLOSING scope — leave it,
        # rename only the body local
        out = swap_local_rename("def f(a, b=wrong):\n    wrong = a + b\n    return wrong\n", "wrong", "right")
        self.assertEqual(out, ("def f(a, b=wrong):\n    right = a + b\n    return right\n", "rename"))

    def test_does_not_rename_param_annotation(self):
        out = swap_local_rename("def f(a: wrong):\n    wrong = 1\n    return wrong\n", "wrong", "right")
        self.assertEqual(out, ("def f(a: wrong):\n    right = 1\n    return right\n", "rename"))

    def test_dispatch_prefers_rename_for_load_fast(self):
        # LOAD_FAST name divergence with 2 uses -> rename path (not single-swap)
        der = _BC([_Inst(6, "LOAD_FAST", "a"), _Inst(12, "LOAD_FAST", "a")])
        gt = _BC([_Inst(6, "LOAD_FAST", "b"), _Inst(12, "LOAD_FAST", "a")])
        div = extract_operand_divergence(gt, der, 6)
        self.assertEqual(div["opname"], "LOAD_FAST")
        self.assertEqual(div["total"], 2)


class ReviewRegressionTest(unittest.TestCase):
    """Regressions for the 4 adversarial-review findings."""

    def test_negative_literal_not_mismatched(self):
        # `-1` folds to one LOAD_CONST(-1); the inner Constant(1) must NOT be
        # counted, so targeting value 1 hits b's real `1`, not the `1` in `-1`.
        frag = "def f():\n    a = -1\n    b = 1\n    return a + b\n"
        out = swap_literal(frag, 1, 5, 0)
        self.assertEqual(out, ("def f():\n    a = -1\n    b = 5\n    return a + b\n", "unique"))

    def test_inf_right_value_declined(self):
        # repr(inf)=='inf' compiles to a NAME, not a literal -> must decline.
        self.assertIsNone(swap_literal("def f():\n    return 1.5\n", 1.5, float("inf"), 0))
        self.assertIsNone(swap_literal("def f():\n    return 1.5\n", 1.5, float("nan"), 0))

    def test_comprehension_ordinal_declined(self):
        # Two '1' constants inside a comprehension; ordinal disambiguation is
        # unreliable (bytecode order != AST order) -> decline.
        frag = "def g(y):\n    return [1 for _ in y if 1 in _]\n"
        self.assertIsNone(swap_literal(frag, 1, 2, 1))

    def test_const_inline_requires_self_reference(self):
        der = _BC([_Inst(14, "LOAD_GLOBAL", "SCALE")])
        gt = _BC([_Inst(14, "LOAD_CONST", "")])
        gt._insts[0].argval = 2
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 14}
        # legitimate `base = SCALE` (not a self-reference) must NOT be rewritten
        self.assertIsNone(const_inline_candidate(gt, der, "base = SCALE\nreturn base\n", ctx))
        # the self-reference placeholder IS handled
        out = const_inline_candidate(gt, der, "SCALE = SCALE\n", ctx)
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "SCALE = 2\n")


class ConstToNameTest(unittest.TestCase):
    """const_placeholder_to_named_load: DER LOAD_CONST placeholder that GT names
    as a variable load. Reads the name verbatim from GT bytecode, localizes the
    placeholder constant in source by bytecode ordinal, rewrites it to a Name."""

    def _ctx(self, offset=10, message="Different bytecode"):
        return {"pylingual_failed_result": {"message": message}, "failed_offset": offset}

    def test_positive_rewrites_const_to_name(self):
        # GT loads local `x`; DER emitted a placeholder LOAD_CONST None. Exactly one
        # None constant in source -> rewrite that None to a Name `x`.
        frag = "def f():\n    x = compute()\n    return None\n"
        der = _BC([_Inst(10, "LOAD_CONST", "None", None)])
        gt = _BC([_Inst(10, "LOAD_FAST", "x", "x")])
        out = const_to_name_candidate(gt, der, frag, self._ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out["text"], "def f():\n    x = compute()\n    return x\n")
        self.assertEqual(out["kind"], "const_to_name")
        self.assertEqual(out["opname"], "LOAD_CONST")

    def test_declines_ambiguous_two_none_constants(self):
        # Two None constants in source but only one None LOAD_CONST in DER -> the
        # bytecode ordinal cannot be uniquely mapped to a single source Constant.
        frag = "def f():\n    a = None\n    return None\n"
        der = _BC([_Inst(10, "LOAD_CONST", "None", None)])
        gt = _BC([_Inst(10, "LOAD_FAST", "x", "x")])
        self.assertIsNone(const_to_name_candidate(gt, der, frag, self._ctx()))

    def test_declines_non_identifier_gt(self):
        # GT operand is not a valid identifier (tuple literal (0, 2) -> the
        # _euler_to_rotation tuple case) -> defer, never emit an invalid Name.
        frag = "def f():\n    x = compute()\n    return None\n"
        der = _BC([_Inst(10, "LOAD_CONST", "None", None)])
        gt = _BC([_Inst(10, "LOAD_FAST", "(0, 2)", (0, 2))])
        self.assertIsNone(const_to_name_candidate(gt, der, frag, self._ctx()))

    def test_declines_gt_is_also_const(self):
        # GT is LOAD_CONST too -> same-opcode literal swap owned by
        # extract_operand_divergence, not this operator.
        frag = "def f():\n    return None\n"
        der = _BC([_Inst(10, "LOAD_CONST", "None", None)])
        gt = _BC([_Inst(10, "LOAD_CONST", "5", 5)])
        self.assertIsNone(const_to_name_candidate(gt, der, frag, self._ctx()))

    def test_declines_wrong_message_or_no_offset(self):
        frag = "def f():\n    return None\n"
        der = _BC([_Inst(10, "LOAD_CONST", "None", None)])
        gt = _BC([_Inst(10, "LOAD_FAST", "x", "x")])
        self.assertIsNone(const_to_name_candidate(
            gt, der, frag, self._ctx(message="Different control flow")))
        self.assertIsNone(const_to_name_candidate(
            gt, der, frag, self._ctx(offset=None)))

    def test_declines_tail_return_feeder(self):
        # DER's LOAD_CONST None immediately feeds RETURN_VALUE at the tail; GT tail
        # is `LOAD_FAST y; RETURN_VALUE`. That belongs to tail_return_candidate, so
        # const_to_name must decline (no double-fire at the function tail).
        frag = "def f():\n    y = compute()\n    return None\n"
        der = _BC([_Inst(10, "LOAD_CONST", "None", None), _Inst(12, "RETURN_VALUE", "")])
        gt = _BC([_Inst(10, "LOAD_FAST", "y", "y"), _Inst(12, "RETURN_VALUE", "")])
        self.assertIsNone(const_to_name_candidate(gt, der, frag, self._ctx()))


if __name__ == "__main__":
    unittest.main()
