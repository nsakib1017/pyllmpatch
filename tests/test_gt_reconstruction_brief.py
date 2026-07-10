"""Unit tests for the GT-bytecode-derived "reconstruction brief" helpers added to
utils/reattach_source_code_object.py (TDD step 1 — written before the functions exist).

These build REAL ground-truth code-object fixtures by compiling tiny snippets in-process
to a temp .pyc and loading them via the same loaders reattach uses
(index_bytecodes_by_qualname -> EditableBytecode, index_code_objects_by_qualname -> raw
code object). No LLM, no GPU, no network.

The four functions under test are GT-bytecode-only (never GT source):
  _gt_control_flow_brief, _gt_signature_and_placement_brief,
  _gt_operand_inventory, _gt_reconstruction_brief.

NOTE (Python 3.10-3.14): assertions use opcode-prefix-family / offset-direction facts
that survive opname drift (POP_JUMP_IF_FALSE vs POP_JUMP_FORWARD_IF_FALSE, RETURN_CONST,
JUMP_BACKWARD, FOR_ITER). Where a reused helper's output is version-shape-dependent (the
try/except handler triviality via structural._handler_skeleton is (False, None) on a real
3.12 compiled `except X: pass` whose handler path ends in RETURN), the test asserts only
the reliably-derivable structural facts (region + depth + protected call).
"""
import os
import py_compile
import tempfile
import unittest
from pathlib import Path

from utils.reattach_source_code_object import (
    _gt_control_flow_brief,
    _gt_operand_inventory,
    _gt_reconstruction_brief,
    _gt_signature_and_placement_brief,
    index_bytecodes_by_qualname,
    index_code_objects_by_qualname,
)


def _compile_snippet(src: str) -> Path:
    """Compile `src` to a temp .pyc and return its path (kept for the process lifetime)."""
    tmpdir = tempfile.mkdtemp(prefix="gt_recon_brief_")
    py_path = os.path.join(tmpdir, "snippet.py")
    with open(py_path, "w") as fh:
        fh.write(src)
    pyc_path = os.path.join(tmpdir, "snippet.pyc")
    py_compile.compile(py_path, cfile=pyc_path, doraise=True)
    return Path(pyc_path)


def _bytecode(src: str, qualname: str):
    return index_bytecodes_by_qualname(_compile_snippet(src))[qualname]


def _code_object(src: str, qualname: str):
    return index_code_objects_by_qualname(_compile_snippet(src))[qualname]


class ControlFlowBriefTest(unittest.TestCase):
    def test_control_flow_brief_labels_if_branch_sense(self):
        bc = _bytecode("def foo(a, b):\n if a > b:\n  return 1\n return 2\n", "<module>.foo")
        brief = _gt_control_flow_brief(bc)
        self.assertIn("branch @", brief)
        self.assertIn("FALSE", brief)
        self.assertIn("jump to @", brief)
        self.assertIn("fall through", brief)

    def test_control_flow_brief_labels_while_loop_backedge(self):
        bc = _bytecode("def foo(a, b):\n while a < b:\n  a += 1\n return 2\n", "<module>.foo")
        brief = _gt_control_flow_brief(bc)
        self.assertIn("loop @", brief)
        self.assertIn("back-edge", brief)
        self.assertIn("JUMP_BACKWARD", brief)

    def test_control_flow_brief_labels_for_loop(self):
        bc = _bytecode("def foo(b):\n for x in range(b):\n  pass\n return 2\n", "<module>.foo")
        brief = _gt_control_flow_brief(bc)
        self.assertIn("for-loop @", brief)
        self.assertIn("FOR_ITER", brief)
        self.assertIn("exits to @", brief)

    def test_control_flow_brief_reports_try_except_with_depth(self):
        bc = _bytecode(
            "import os\ndef foo(x):\n try:\n  os.remove(x)\n except OSError:\n  pass\n",
            "<module>.foo",
        )
        brief = _gt_control_flow_brief(bc)
        self.assertIn("try @", brief)
        self.assertIn("depth", brief)
        # Protected dotted callee is recovered via structural._protected_call_parts.
        self.assertIn("os.remove", brief)

    def test_straight_line_control_flow_brief_is_empty(self):
        bc = _bytecode("def foo():\n return 1\n", "<module>.foo")
        self.assertEqual(_gt_control_flow_brief(bc), "")


class SignatureAndPlacementBriefTest(unittest.TestCase):
    def test_signature_brief_positional_star_and_kwonly_args(self):
        co = _code_object(
            "class C:\n def m(self, a, b, /, c, *args, k, **kw):\n  return a\n",
            "<module>.C.m",
        )
        brief = _gt_signature_and_placement_brief(co, "<module>.C.m")
        self.assertIn("def m(self, a, b, /, c, *args, k, **kw)", brief)

    def test_signature_brief_async_and_generator(self):
        co_ag = _code_object("async def ag():\n yield 1\n", "<module>.ag")
        brief_ag = _gt_signature_and_placement_brief(co_ag, "<module>.ag")
        self.assertIn("async def ag(", brief_ag)
        self.assertIn("generator", brief_ag)

        co_gen = _code_object("def gen():\n yield 2\n", "<module>.gen")
        brief_gen = _gt_signature_and_placement_brief(co_gen, "<module>.gen")
        self.assertIn("generator", brief_gen)

    def test_signature_brief_class_and_nested_members(self):
        src = "class C:\n def m(self):\n  return 1\n"
        co = _code_object(src, "<module>.C")
        sig_brief = _gt_signature_and_placement_brief(co, "<module>.C")
        self.assertIn("class C:", sig_brief)

        bc = _bytecode(src, "<module>.C")
        inv = _gt_operand_inventory(bc)
        self.assertIn("C.m", inv)


class OperandInventoryTest(unittest.TestCase):
    def test_operand_inventory_calls_names_consts(self):
        bc = _bytecode(
            "def foo(b, x):\n range(b)\n os.remove(x)\n return 2\n",
            "<module>.foo",
        )
        inv = _gt_operand_inventory(bc)
        self.assertIn("range", inv)
        self.assertIn("os.remove", inv)
        self.assertIn("consts:", inv)
        self.assertIn("2", inv)


class ReconstructionBriefComposerTest(unittest.TestCase):
    def test_reconstruction_brief_missing_includes_signature_placement_shape(self):
        bc = _bytecode(
            "class Foo:\n def bar(self, a):\n  if a:\n   return 1\n  return 2\n",
            "<module>.Foo.bar",
        )
        brief = _gt_reconstruction_brief(bc, "<module>.Foo.bar", is_missing=True)
        self.assertIn("def bar(self, a)", brief)   # reconstructed signature
        self.assertIn("Foo", brief)                # placement hint names the parent class
        self.assertIn("branch @", brief)           # control-flow shape (body)

    def test_reconstruction_brief_none_object_is_empty(self):
        self.assertEqual(
            _gt_reconstruction_brief(None, "<module>.x", is_missing=True), ""
        )


class ReviewFixesTest(unittest.TestCase):
    """Regression tests for the three adversarial-review findings on the brief."""

    def test_operand_inventory_captures_self_method_call(self):
        # Finding 1: local/closure-based dotted calls (self.log) must appear in `calls:`,
        # while globals (helper) are still captured and bare local args add no noise.
        bc = _bytecode(
            "class C:\n def m(self):\n  self.log(1)\n  helper(self)\n", "<module>.C.m"
        )
        inv = _gt_operand_inventory(bc)
        self.assertIn("self.log", inv)
        self.assertIn("helper", inv)
        # `self` alone (a bare arg to helper(...)) must NOT be listed as a call target.
        calls_line = next((l for l in inv.splitlines() if l.startswith("calls:")), "")
        self.assertNotIn(" self,", calls_line + ",")
        self.assertNotIn(" self ", " " + calls_line.replace("self.log", "") + " ")

    def test_try_handler_does_not_emit_misleading_bare_except(self):
        # Finding 2: when the exception type is not recoverable from bytecode, hedge instead
        # of printing a definitive bare `except:` (which would falsely imply a catch-all).
        bc = _bytecode(
            "def foo(x):\n try:\n  os.remove(x)\n except OSError:\n  pass\n", "<module>.foo"
        )
        brief = _gt_control_flow_brief(bc)
        self.assertIn("try @", brief)
        self.assertIn("os.remove", brief)
        if "except OSError" in brief:
            # Type was recoverable on this version — fine.
            pass
        else:
            # Type not recoverable -> must hedge, never claim a bare catch-all.
            self.assertIn("not recoverable", brief)
            self.assertNotIn("handler `except:`", brief)

    def test_for_loop_is_not_double_labeled_as_while(self):
        # Finding 3: a plain for-loop is labeled once (as a for-loop), never additionally as a
        # while-style back-edge for the same construct.
        bc = _bytecode("def foo(b):\n for x in range(b):\n  pass\n return 2\n", "<module>.foo")
        brief = _gt_control_flow_brief(bc)
        self.assertIn("for-loop @", brief)
        self.assertNotIn("while-style", brief)


if __name__ == "__main__":
    unittest.main()
