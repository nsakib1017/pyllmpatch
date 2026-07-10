"""TDD spec for tools/build_gt_reconstruction_corpus.py (does not exist yet).

This suite pins the GT-bytecode reconstruction corpus builder contract:

- The oracle GT source span is the verbatim training TARGET.
- The capture predicate mirrors ``_format_gt_reconstruction_brief``'s render gate
  (missing target unconditionally; present control-flow only under
  'Different control flow' or 'Different bytecode' with failed_offset is None).
- The emitted user prompt is the brief-ON inference prompt (train-input == inference).
- An emitted row satisfies the finetuner ``build_chat_samples`` contract byte-for-byte.
- Holdout file_hashes are excluded.
- Identical (source_file_hash, qualname, GT-fragment structural hash) targets dedup.

Everything is CPU-only: real snippets compiled with ``py_compile`` to ``.pyc`` and
loaded via editable_bytecode. No GPU, no torch model load, no LLM, no network.

Until tools/build_gt_reconstruction_corpus.py exists, the module-level import below
raises ImportError and the whole file fails at collection -- that is the expected
red state for TDD step 1.
"""
from __future__ import annotations

import ast
import csv
import os
import py_compile
import tempfile
import unittest
from pathlib import Path

# --- Reused production symbols (import-time .env is already populated) ---------
from pipeline.code_object_repair_loop import (
    OracleFragmentFixer,
    build_semantic_repair_messages,
    _replacement_structural_hash,
)
from utils.reattach_source_code_object import (
    _build_target_repair_context,
    index_code_objects_by_qualname,
)

# --- The symbols under test (module does NOT exist yet -> ImportError) ---------
from tools.build_gt_reconstruction_corpus import (
    OracleCaptureFragmentFixer,
    _should_capture_target,
    _emit_corpus_row,
    _load_holdout_hashes,
    _filter_holdout_rows,
)

BRIEF_HEADER = "Reconstruct the missing ground-truth object from its bytecode:"
BRIEF_ENV = "SEMANTIC_RECONSTRUCTION_BRIEF"


def _compile(dir_path: Path, name: str, source: str) -> tuple[Path, Path]:
    py_path = dir_path / f"{name}.py"
    py_path.write_text(source, encoding="utf-8")
    pyc_path = dir_path / f"{name}.pyc"
    py_compile.compile(str(py_path), cfile=str(pyc_path), doraise=True)
    return py_path, pyc_path


def _ast_segment(source: str, func_name: str) -> str:
    node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == func_name)
    return ast.get_source_segment(source, node)


def _replicate_build_chat_samples(records: list[dict]) -> list[dict]:
    """Inline copy of finetuning.build_chat_samples' 3-line contract.

    The finetuner module eagerly imports unsloth/torch, so we replicate its
    accept/skip logic here rather than importing it.
    """
    samples: list[dict] = []
    for record in records:
        prompt_record = record.get("prompt") or {}
        messages = prompt_record.get("messages")
        assistant_text = record.get("replacement_text")
        if not isinstance(messages, list) or len(messages) < 2 or not assistant_text:
            continue
        prompt_messages = [
            {
                "role": str(message.get("role", "")).strip(),
                "content": str(message.get("content", "")),
            }
            for message in messages
            if isinstance(message, dict) and message.get("role") and message.get("content")
        ]
        if len(prompt_messages) < 2:
            continue
        prompt_messages.append(
            {"role": "assistant", "content": str(assistant_text).strip("\n")}
        )
        samples.append({"messages": prompt_messages})
    return samples


# Ground-truth snippet: alpha (present/matched) + gamma (dropped by the decompiler).
GT_SOURCE = (
    "def alpha(x):\n"
    "    return x + 1\n"
    "\n"
    "\n"
    "def gamma(z):\n"
    "    if z > 0:\n"
    "        return z\n"
    "    else:\n"
    "        return -z\n"
)
# Derived snippet: gamma is missing entirely (the semantic divergence to repair).
DERIVED_SOURCE = "def alpha(x):\n    return x + 1\n"

MISSING_Q = "<module>.gamma"
PRESENT_Q = "<module>.alpha"


class _CompiledPairMixin(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.gt_py, self.gt_pyc = _compile(self.dir, "gt", GT_SOURCE)
        self.der_py, self.der_pyc = _compile(self.dir, "der", DERIVED_SOURCE)
        self.gt_objs = index_code_objects_by_qualname(self.gt_pyc)
        self.der_objs = index_code_objects_by_qualname(self.der_pyc)
        # Ensure env starts clean; tests that toggle it restore via addCleanup.
        self._prev_brief = os.environ.pop(BRIEF_ENV, None)
        self.addCleanup(self._restore_brief)

    def _restore_brief(self) -> None:
        if self._prev_brief is None:
            os.environ.pop(BRIEF_ENV, None)
        else:
            os.environ[BRIEF_ENV] = self._prev_brief

    def _missing_context(self) -> dict:
        return _build_target_repair_context(
            qualname=MISSING_Q,
            gt_code_object=self.gt_objs[MISSING_Q],
            derived_code_object=None,
            verification=None,
            rejected_attempts=[],
        )

    def _present_context(self) -> dict:
        return _build_target_repair_context(
            qualname=PRESENT_Q,
            gt_code_object=self.gt_objs[PRESENT_Q],
            derived_code_object=self.der_objs[PRESENT_Q],
            verification=None,
            rejected_attempts=[],
        )


class OracleGtLabelVerbatimTest(_CompiledPairMixin):
    def test_oracle_gt_label_verbatim(self):
        """OracleFragmentFixer returns the exact verbatim GT source span (the TARGET)."""
        fixer = OracleFragmentFixer(self.gt_pyc, self.gt_py)
        for func_name, qualname in (("alpha", PRESENT_Q), ("gamma", MISSING_Q)):
            candidate = fixer.generate_candidate(
                qualname=qualname,
                gt_code_object=None,
                derived_code_object=None,
                derived_source_fragment="",
                repair_context=None,
            )
            self.assertEqual(candidate, _ast_segment(GT_SOURCE, func_name))


class CapturePredicateTest(_CompiledPairMixin):
    def test_capture_predicate_missing_and_controlflow(self):
        # Real compiled pair: missing target captured, matched-success target rejected.
        self.assertTrue(_should_capture_target(self._missing_context()))
        self.assertFalse(_should_capture_target(self._present_context()))

        # Present control-flow message variants (mirror the render gate exactly).
        def ctx(message, failed_offset):
            return {
                "target_is_missing": False,
                "pylingual_failed_result": None if message is None else {"message": message},
                "failed_offset": failed_offset,
                "gt_reconstruction_brief": "some non-empty brief",
            }

        self.assertTrue(_should_capture_target(ctx("Different control flow", None)))
        self.assertTrue(_should_capture_target(ctx("Different bytecode", None)))
        # A real single-offset leaf ("Different bytecode" + int offset) is NOT captured.
        self.assertFalse(_should_capture_target(ctx("Different bytecode", 42)))
        # A PyLingual success (no failed_result) is NOT captured.
        self.assertFalse(_should_capture_target(ctx(None, None)))


class BriefInUserPromptTest(_CompiledPairMixin):
    def test_brief_present_in_user_prompt(self):
        repair_context = self._missing_context()

        os.environ[BRIEF_ENV] = "true"
        messages_on = build_semantic_repair_messages(
            qualname=MISSING_Q,
            gt_code_object=self.gt_objs[MISSING_Q],
            derived_code_object=None,
            derived_source_fragment="",
            repair_context=repair_context,
            telemetry_guidance="",
            action_pattern_guidance="",
        )
        os.environ.pop(BRIEF_ENV, None)
        messages_off = build_semantic_repair_messages(
            qualname=MISSING_Q,
            gt_code_object=self.gt_objs[MISSING_Q],
            derived_code_object=None,
            derived_source_fragment="",
            repair_context=repair_context,
            telemetry_guidance="",
            action_pattern_guidance="",
        )

        self.assertEqual([m["role"] for m in messages_on], ["system", "user"])
        self.assertIn(BRIEF_HEADER, messages_on[1]["content"])
        self.assertNotIn(BRIEF_HEADER, messages_off[1]["content"])


class RowSchemaContractTest(_CompiledPairMixin):
    def test_row_schema_matches_finetuner_contract(self):
        repair_context = self._missing_context()
        os.environ[BRIEF_ENV] = "true"
        messages = build_semantic_repair_messages(
            qualname=MISSING_Q,
            gt_code_object=self.gt_objs[MISSING_Q],
            derived_code_object=None,
            derived_source_fragment="",
            repair_context=repair_context,
            telemetry_guidance="",
            action_pattern_guidance="",
        )
        os.environ.pop(BRIEF_ENV, None)

        gt_fragment = OracleFragmentFixer(self.gt_pyc, self.gt_py).generate_candidate(
            qualname=MISSING_Q,
            gt_code_object=None,
            derived_code_object=None,
            derived_source_fragment="",
            repair_context=None,
        )

        row = _emit_corpus_row(
            source_file_hash="deadbeef",
            qualname=MISSING_Q,
            target_kind="code_object_fragment",
            repair_operation="insert_missing",
            messages=messages,
            gt_fragment=gt_fragment,
        )

        # Top-level finetuner-visible contract.
        self.assertIsInstance(row, dict)
        self.assertIn("prompt", row)
        emitted_messages = row["prompt"]["messages"]
        self.assertIsInstance(emitted_messages, list)
        self.assertGreaterEqual(len(emitted_messages), 2)
        for message in emitted_messages:
            self.assertTrue(str(message["role"]).strip())
            self.assertTrue(str(message["content"]).strip())
        self.assertEqual([m["role"] for m in emitted_messages], ["system", "user"])
        self.assertTrue(row["replacement_text"])
        self.assertEqual(row["replacement_text"], gt_fragment)

        # Audit keys carried for dedup/holdout/telemetry.
        self.assertEqual(row["source_file_hash"], "deadbeef")
        self.assertEqual(row["qualname"], MISSING_Q)
        self.assertEqual(row["target_kind"], "code_object_fragment")
        self.assertEqual(row["repair_operation"], "insert_missing")

        # The finetuner accepts the row and reconstructs the assistant turn verbatim.
        samples = _replicate_build_chat_samples([row])
        self.assertEqual(len(samples), 1)
        conversation = samples[0]["messages"]
        self.assertEqual([m["role"] for m in conversation], ["system", "user", "assistant"])
        self.assertEqual(conversation[-1]["content"], str(gt_fragment).strip("\n"))


class HoldoutExclusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write_holdout_csv(self, name: str, hashes: list[str]) -> Path:
        path = self.dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file_hash", "error_type", "source"])
            for h in hashes:
                writer.writerow([h, "semantic_error", "PyLingual.IO"])
        return path

    def test_holdout_exclusion(self):
        hybrid = self._write_holdout_csv("hybrid100.csv", ["H1", "H2"])
        briefab = self._write_holdout_csv("briefab.csv", ["H3"])

        holdout = _load_holdout_hashes([hybrid, briefab])
        self.assertEqual(holdout, {"H1", "H2", "H3"})

        rows = [
            {"file_hash": "H2", "source": "PyPi"},   # in holdout -> dropped
            {"file_hash": "KEEP", "source": "PyPi"},  # survives
        ]
        survivors = list(_filter_holdout_rows(rows, holdout))
        self.assertEqual([r["file_hash"] for r in survivors], ["KEEP"])
        self.assertNotIn("H2", {r["file_hash"] for r in survivors})


class DedupOnStructuralHashTest(_CompiledPairMixin):
    def test_dedup_on_structural_hash(self):
        sink: list[dict] = []
        seen: set = set()
        fixer = OracleCaptureFragmentFixer(
            gt_pyc=self.gt_pyc,
            gt_source=self.gt_py,
            source_file_hash="deadbeef",
            sink=sink,
            seen=seen,
        )
        repair_context = self._missing_context()
        gt_fragment = OracleFragmentFixer(self.gt_pyc, self.gt_py).generate_candidate(
            qualname=MISSING_Q,
            gt_code_object=None,
            derived_code_object=None,
            derived_source_fragment="",
            repair_context=None,
        )
        expected_structural = _replacement_structural_hash(gt_fragment)

        # Two identical captured targets must collapse to a single row.
        for _ in range(2):
            candidates = fixer.generate_candidates(
                qualname=MISSING_Q,
                gt_code_object=self.gt_objs[MISSING_Q],
                derived_code_object=None,
                derived_source_fragment="",
                repair_context=repair_context,
            )
            # Returns the oracle GT candidate (same dict shape LLMFragmentFixer yields).
            self.assertTrue(candidates)
            self.assertEqual(candidates[0]["text"], gt_fragment)

        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0]["replacement_text"], gt_fragment)
        self.assertEqual(
            _replacement_structural_hash(sink[0]["replacement_text"]), expected_structural
        )


if __name__ == "__main__":
    unittest.main()
