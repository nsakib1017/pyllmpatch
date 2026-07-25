"""Investigation for a suspected 3.15 deterministic-operator "opcode coverage" gap
(2026-07-22): the leaf/instruction operators (utils/semantic_operators/leaf.py,
literal_form.py, unary_op.py, container.py) match GT vs derived instructions by
OPNAME STRING, never by numeric opcode -- and a full diff of the 3.15 opmap
(utils/py315_support/opcode_315_data.json) against xdis's own 3.14 opname table shows
3.15 introduces ZERO opcode NAMES beyond 3.14's non-specialized set (the "removed" names
are all Tier-2 adaptive-specialization forms that never appear in a compiled .pyc). The
one nominal 3.14+ delta the operators could plausibly miss, LOAD_FAST_BORROW, is rewritten
to LOAD_FAST by pylingual's own `replace_borrow` bytecode patch before ANY operator ever
sees an instruction (applied in both `pylingual.equivalence_check.compare_pyc` and
`utils.pyc_code_object_distance.load_editable_bytecode_from_pyc`), so it is moot in
practice.

A corpus audit of 120 real 3.15 semantic-repair failures (196 "Different bytecode"
targets, results/experiment_outputs/*/shard*/**/result.json) found the low firing rate is
NOT an opcode-coverage gap:
  * 89% (175/196) have no PyLingual `failed_offset` at all because the GT and derived
    INSTRUCTION COUNTS DIFFER -- a raw structural/length mismatch that cannot be a
    same-opcode operand swap regardless of any coverage set.
  * Of the 21 that DO have an offset, leaf's dispatch already fires on 7 (33%) -- proving
    same-opcode swaps (BINARY_OP, LOAD_CONST, ...) already work on 3.15's renumbered
    opcodes today.
  * The remaining 14 are uniformly CROSS-OPCODE divergences (a LOAD_FAST_LOAD_FAST
    superinstruction fusion vs a lone LOAD_FAST, a conditional-jump-sense flip, a
    STORE_FAST/STORE_DEREF scoping difference, a constant-folded literal vs a runtime
    BINARY_OP, a missed `super()` call, ...) that leaf.py's documented design correctly
    and intentionally defers (same-opcode swap only) -- no opcode-name-set extension can
    turn a cross-opcode structural divergence into a safe single-token swap.

This mirrors a documented prior investigation for the general 3.10-3.14 hard tail (see
tests/test_jump_sense_disambiguation.py and the `post-llm-det-and-operator-ceiling`
project memory): "deterministic operator coverage is MAXED... adding leaf/jump operators
will NOT move it... structural/decompiler-bound." This file pins that finding for 3.15
specifically with executable evidence, both positive (real 3.15 bytecode DOES dispatch)
and negative (the real cross-opcode shapes found in the corpus correctly defer, so a
future well-intentioned-but-unsafe "coverage expansion" cannot silently regress this).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual"))

import utils.py315_support as py315
from utils.semantic_operators.leaf import extract_operand_divergence, leaf_value_candidate
from utils.semantic_operators.jump import _is_pure_jump_flip, collapse_dead_pass_else_candidate


def _uv_compile(version: str, src: Path, out: Path) -> bool:
    """Mirrors tests/test_py315_support.py's helper: compile with a uv-managed
    interpreter, skipping gracefully (return False) if uv/the interpreter is unavailable."""
    env = {**os.environ, "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache")}
    try:
        r = subprocess.run(
            ["uvx", "--no-config", "--python", version, "python", "-c",
             f"import py_compile;py_compile.compile({str(src)!r},cfile={str(out)!r})"],
            env=env, capture_output=True, timeout=240,
        )
    except Exception:
        return False
    return out.exists()


class _Inst:
    def __init__(self, offset, opname, argrepr, argval=None):
        self.offset, self.opname, self.argrepr = offset, opname, argrepr
        self.argval = argval


class _BC:
    def __init__(self, insts):
        self._insts = insts

    def __iter__(self):
        return iter(self._insts)


# =========================================================================
# Positive evidence: leaf's same-opcode dispatch DOES fire on REAL, uv-compiled 3.15
# bytecode (magic 3666, HAVE_ARGUMENT=41, fully renumbered opcode table) -- not just on
# opcode NAME strings in a hand-built fixture.
# =========================================================================

class RealPython315LeafDispatchTest(unittest.TestCase):
    def _compare(self, gt_src: str, der_src: str):
        py315.install()
        from pylingual.equivalence_check import compare_pyc

        d = Path(tempfile.mkdtemp())
        (d / "gt.py").write_text(gt_src)
        (d / "der.py").write_text(der_src)
        if not (_uv_compile("3.15", d / "gt.py", d / "gt.pyc")
                and _uv_compile("3.15", d / "der.py", d / "der.pyc")):
            self.skipTest("uv-managed Python 3.15 unavailable")
        return compare_pyc(d / "gt.pyc", d / "der.pyc")

    def _find_diverging(self, results, qualname_contains: str):
        for tr in results:
            if not tr.success and tr.message == "Different bytecode" and qualname_contains in tr.names():
                return tr
        return None

    def test_binary_op_swap_fires_on_real_315_bytecode(self):
        gt_src = "def f(x):\n    return x - 1\n"
        der_src = "def f(x):\n    return x + 1\n"
        results = self._compare(gt_src, der_src)
        tr = self._find_diverging(results, "f")
        self.assertIsNotNone(tr, "expected a 'Different bytecode' divergence in f()")
        self.assertIsNotNone(tr.failed_offset)

        div = extract_operand_divergence(tr.bc_a, tr.bc_b, int(tr.failed_offset))
        self.assertIsNotNone(div, "same-opcode BINARY_OP swap should dispatch on real 3.15 bytecode")
        self.assertEqual(div["kind"], "binary_op")
        self.assertEqual((div["wrong"], div["right"]), ("+", "-"))

        ctx = {"pylingual_failed_result": {"message": tr.message}, "failed_offset": tr.failed_offset}
        candidate = leaf_value_candidate(tr.bc_a, tr.bc_b, der_src, ctx)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["text"], gt_src)

    def test_load_const_literal_swap_fires_on_real_315_bytecode(self):
        gt_src = "def f():\n    y = 'A'\n    return y\n"
        der_src = "def f():\n    y = 'B'\n    return y\n"
        results = self._compare(gt_src, der_src)
        tr = self._find_diverging(results, "f")
        self.assertIsNotNone(tr, "expected a 'Different bytecode' divergence in f()")
        self.assertIsNotNone(tr.failed_offset)

        div = extract_operand_divergence(tr.bc_a, tr.bc_b, int(tr.failed_offset))
        self.assertIsNotNone(div, "same-opcode LOAD_CONST swap should dispatch on real 3.15 bytecode")
        self.assertEqual(div["kind"], "literal")
        self.assertEqual((div["wrong"], div["right"]), ("B", "A"))

        ctx = {"pylingual_failed_result": {"message": tr.message}, "failed_offset": tr.failed_offset}
        candidate = leaf_value_candidate(tr.bc_a, tr.bc_b, der_src, ctx)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["text"], gt_src)


# =========================================================================
# Negative evidence (regression guard): the REAL cross-opcode divergence shapes found in
# the 3.15 corpus audit correctly DEFER. None of these can be fixed by adding an opcode
# name to a coverage set -- they are genuine structural/content differences. Pinning them
# guards against a future "just widen the coverage set" change silently producing a wrong
# (data-corrupting) edit.
# =========================================================================

class RealWorldCrossOpcodeDivergencesDeferTest(unittest.TestCase):
    def test_load_fast_load_fast_superinstruction_vs_single_load_defers(self):
        # Real case: <module>.MSFReader.__iter__ (3.15 corpus). The 3.13+ compiler fuses
        # two ADJACENT LOAD_FAST loads into one LOAD_FAST_LOAD_FAST; here the derived
        # source references an extra variable ("modifications") GT's code does not at
        # this position -- a genuine content difference, not an operand swap.
        der = _BC([_Inst(94, "LOAD_FAST_LOAD_FAST", "entry, modifications")])
        gt = _BC([_Inst(94, "LOAD_FAST", "entry")])
        self.assertIsNone(extract_operand_divergence(gt, der, 94))

    def test_conditional_jump_sense_opcodes_are_not_leaf_operand_swaps(self):
        # Real case: <module>._usage_line (3.15 corpus). POP_JUMP_IF_TRUE/FALSE never
        # appear in leaf's dispatch table at all -- jump.py owns the (much narrower)
        # pure-flip idiom instead. Confirmed by design, not a missing opcode name.
        der = _BC([_Inst(38, "POP_JUMP_IF_FALSE", "to 42")])
        gt = _BC([_Inst(38, "POP_JUMP_IF_TRUE", "to 42")])
        self.assertIsNone(extract_operand_divergence(gt, der, 38))

    def test_store_fast_vs_store_deref_scope_difference_defers(self):
        # Real case: <module>.Form.get_definition (3.15 corpus): a FAST-local store on
        # one side, a closure-cell store on the other -- a scoping/capture difference
        # leaf.py explicitly does not attempt to resolve (see dropped_statement.py's
        # docstring: "indistinguishable from an ordinary closure cell-variable capture").
        der = _BC([_Inst(74, "STORE_FAST", "l_fields")])
        gt = _BC([_Inst(74, "STORE_DEREF", "l_fields")])
        self.assertIsNone(extract_operand_divergence(gt, der, 74))

    def test_constant_folded_literal_vs_runtime_binary_op_defers(self):
        # Real case: <module>.mirAffine (3.15 corpus): candidate constant-folded an
        # entire subtraction into a literal (LOAD_SMALL_INT) where GT computes it at
        # runtime (BINARY_OP -). Cross-opcode by construction; not an operand swap.
        der = _BC([_Inst(78, "LOAD_SMALL_INT", "")])
        gt = _BC([_Inst(78, "BINARY_OP", "-")])
        self.assertIsNone(extract_operand_divergence(gt, der, 78))

    def test_missed_super_call_defers(self):
        # Real case: <module>.Spherical.velocity_equation (3.15 corpus): the decompiler
        # rendered a plain CALL where GT uses LOAD_SUPER_ATTR (a missed `super()` call).
        der = _BC([_Inst(8, "CALL", "3 positional")])
        gt = _BC([_Inst(8, "LOAD_SUPER_ATTR", "NULL|self + velocity_equation")])
        self.assertIsNone(extract_operand_divergence(gt, der, 8))


class Py315PureJumpFlipReportedCaseTest(unittest.TestCase):
    """The one REAL "pure" jump-sense flip found in the 3.15 corpus audit (qualname
    `<module>._usage_line`: the GT and derived streams differ in EXACTLY one instruction
    out of 82, with the same jump target). jump.py's `_is_pure_jump_flip` detector
    correctly recognises the shape, but `collapse_dead_pass_else_candidate` correctly
    DEFERS because the derived source is a plain `if options: pieces.append(...)` (no
    else) -- not the narrow `if C: pass else: BODY` idiom the operator can safely
    rewrite without knowing GT's actual (unavailable) source. This shows that even a
    maximally-clean, single-instruction bytecode divergence does not guarantee an
    inferable, safe source-level fix -- confirming the miss is a genuinely different GT
    source shape, not an opcode-coverage gap."""

    def test_pure_flip_detected_but_source_idiom_absent(self):
        der = _BC([
            _Inst(0, "LOAD_FAST", "options"),
            _Inst(38, "POP_JUMP_IF_FALSE", "to 42"),
            _Inst(40, "LOAD_CONST", "'[OPTIONS]'"),
        ])
        gt = _BC([
            _Inst(0, "LOAD_FAST", "options"),
            _Inst(38, "POP_JUMP_IF_TRUE", "to 42"),
            _Inst(40, "LOAD_CONST", "'[OPTIONS]'"),
        ])
        self.assertEqual(_is_pure_jump_flip(gt, der, 38), "POP_JUMP_IF_FALSE")

        fragment = "if options:\n    pieces.append('[OPTIONS]')\n"
        ctx = {"pylingual_failed_result": {"message": "Different bytecode"}, "failed_offset": 38}
        self.assertIsNone(collapse_dead_pass_else_candidate(gt, der, fragment, ctx))


# =========================================================================
# Opcode-name-stability fact check: 3.15 introduces NO opcode NAME beyond 3.14's
# non-specialized set. Falsifies the "3.15-specific renamed/new opcodes" hypothesis as a
# permanent, executable assertion (guards the family sets in opcode_families.py against
# silent drift on a future 3.15 beta/GA re-generation of opcode_315_data.json).
# =========================================================================

class Opcode315NameStabilityTest(unittest.TestCase):
    def test_315_opcode_names_are_a_subset_of_314s(self):
        import xdis.opcodes.opcode_3x.opcode_314 as op314

        data = json.loads((REPO / "utils" / "py315_support" / "opcode_315_data.json").read_text())
        names_315 = set(data["opmap"])
        names_314 = {n for n in op314.opname if n and not n.startswith("<")}
        new_in_315 = names_315 - names_314
        self.assertEqual(
            new_in_315, set(),
            "3.15 introduced opcode name(s) with no 3.14 counterpart -- the leaf/literal_form/"
            "unary_op/opcode_families coverage sets may need updating for: " + ", ".join(sorted(new_in_315)),
        )


if __name__ == "__main__":
    unittest.main()
