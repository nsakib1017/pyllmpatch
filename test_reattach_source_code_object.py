from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json
from unittest.mock import patch

import utils.file_helpers as file_helpers
from utils.file_helpers import fetch_pyllmpatch_repair_paths, fetch_pyllmpatch_source_path
from pipeline.code_object_repair_loop import build_semantic_repair_messages
from pipeline.code_object_repair_loop import LLMFragmentFixer
from utils.reattach_source_code_object import ReattachError, _announce_semantic_step, _store_semantic_step, compile_source_to_pyc, infer_source_from_pyc


class InferSourceFromPycTest(unittest.TestCase):
    def test_infers_source_from_pycache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "sample.py"
            source.write_text("print('hello')\n", encoding="utf-8")

            pycache_dir = root / "__pycache__"
            pycache_dir.mkdir()
            pyc = pycache_dir / "sample.cpython-310.pyc"
            pyc.write_bytes(b"")

            self.assertEqual(infer_source_from_pyc(pyc), source.resolve())

    def test_infers_source_from_sibling_cpython_pyc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "sample.py"
            source.write_text("print('hello')\n", encoding="utf-8")

            pyc = root / "sample.cpython-310.pyc"
            pyc.write_bytes(b"")

            self.assertEqual(infer_source_from_pyc(pyc), source.resolve())

    def test_compile_source_to_pyc_rejects_syntax_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "broken.py"
            source.write_text("def f(:\n    pass\n", encoding="utf-8")

            with self.assertRaises(ReattachError):
                compile_source_to_pyc(source, None)

    def test_fetch_pyllmpatch_source_path_uses_highest_indented_source_for_non_pypi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prior_pyl = file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL
            prior_pypi = file_helpers.BASE_DIR_PYTHON_FILES_PYPI
            try:
                file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL = root / "pylingual"
                file_helpers.BASE_DIR_PYTHON_FILES_PYPI = root / "pypi"

                hash_dir = file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL / "abc123" / "decompiler_output"
                hash_dir.mkdir(parents=True)
                source0 = hash_dir / "indented_0.py"
                source2 = hash_dir / "indented_2.py"
                source0.write_text("print('a')\n", encoding="utf-8")
                source2.write_text("print('b')\n", encoding="utf-8")

                self.assertEqual(fetch_pyllmpatch_source_path("abc123", "VirusTotal"), source2.resolve())
            finally:
                file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL = prior_pyl
                file_helpers.BASE_DIR_PYTHON_FILES_PYPI = prior_pypi

    def test_fetch_pyllmpatch_source_path_uses_pypi_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prior_pyl = file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL
            prior_pypi = file_helpers.BASE_DIR_PYTHON_FILES_PYPI
            try:
                file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL = root / "pylingual"
                file_helpers.BASE_DIR_PYTHON_FILES_PYPI = root / "pypi"

                hash_dir = file_helpers.BASE_DIR_PYTHON_FILES_PYPI / "def456"
                hash_dir.mkdir(parents=True)
                source = hash_dir / "sample.py"
                source.write_text("print('hello')\n", encoding="utf-8")

                self.assertEqual(fetch_pyllmpatch_source_path("def456", "PyPi"), source.resolve())
            finally:
                file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL = prior_pyl
                file_helpers.BASE_DIR_PYTHON_FILES_PYPI = prior_pypi

    def test_fetch_pyllmpatch_repair_paths_uses_pypi_decompiled_output_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prior_pyl = file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL
            prior_pypi = file_helpers.BASE_DIR_PYTHON_FILES_PYPI
            try:
                file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL = root / "pylingual"
                file_helpers.BASE_DIR_PYTHON_FILES_PYPI = root / "pypi"

                hash_dir = file_helpers.BASE_DIR_PYTHON_FILES_PYPI / "ghi789"
                pycache_dir = hash_dir / "__pycache__"
                derived_cache_dir = hash_dir / "decompiled_output_pylingual" / "__pycache__"
                derived_source_dir = hash_dir / "decompiled_output_pylingual"
                pycache_dir.mkdir(parents=True)
                derived_cache_dir.mkdir(parents=True)

                gt_pyc = pycache_dir / "sample.cpython-310.pyc"
                derived_pyc = derived_cache_dir / "decompiled_sample.cpython-310.pyc"
                derived_source = derived_source_dir / "decompiled_sample.py"
                source = hash_dir / "sample.py"
                gt_pyc.write_bytes(b"")
                derived_pyc.write_bytes(b"")
                derived_source.write_text("print('derived')\n", encoding="utf-8")
                source.write_text("print('source')\n", encoding="utf-8")

                self.assertEqual(
                    fetch_pyllmpatch_repair_paths("ghi789", "PyPi"),
                    (gt_pyc.resolve(), derived_pyc.resolve(), derived_source.resolve()),
                )
            finally:
                file_helpers.BASE_DIR_PYTHON_FILES_PYLINGUAL = prior_pyl
                file_helpers.BASE_DIR_PYTHON_FILES_PYPI = prior_pypi

    def test_module_prompt_includes_instruction_diff(self) -> None:
        messages = build_semantic_repair_messages(
            qualname="<module>",
            gt_code_object=type("CodeObj", (), {"co_name": "<module>", "co_firstlineno": 1, "co_flags": 0})(),
            derived_code_object=type("CodeObj", (), {"co_name": "<module>", "co_firstlineno": 1, "co_flags": 0})(),
            derived_source_fragment="print('derived')\n",
            repair_context={
                "target_kind": "module_statement",
                "qualname": "<module>",
                "localized_line_number": 12,
                "failed_offset": 152,
                "alignment_tag": "replace",
                "pylingual_failed_result": {"message": "Different bytecode"},
                "instruction_diff": "=> gt: idx=1 offset=2 line=1 LOAD_CONST 1\n=> derived: idx=1 offset=2 line=1 FORMAT_VALUE 0",
                "rejected_attempts": [],
            },
        )
        user_prompt = messages[1]["content"]
        self.assertIn("Instruction diff", user_prompt)
        self.assertIn("LOAD_CONST 1", user_prompt)
        self.assertIn("print('derived')", user_prompt)
        self.assertNotIn("Ground-truth top-level module statement", user_prompt)

    def test_llm_fragment_fixer_falls_back_on_whitespace_response(self) -> None:
        fixer = LLMFragmentFixer(provider="Google", model="gemini-2.5-flash-lite")
        with patch("pipeline.code_object_repair_loop.make_llm_call_from_config", return_value={"content": "   \n"}) as mocked_call:
            candidate = fixer.generate_candidate(
                qualname="<module>",
                gt_code_object=type("CodeObj", (), {"co_name": "<module>", "co_firstlineno": 1, "co_flags": 0})(),
                derived_code_object=type("CodeObj", (), {"co_name": "<module>", "co_firstlineno": 1, "co_flags": 0})(),
                derived_source_fragment="print('derived')\n",
                repair_context=None,
            )
        mocked_call.assert_called_once()
        self.assertEqual(candidate, "print('derived')\n")

    def test_llm_prompt_file_includes_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixer = LLMFragmentFixer(provider="Google", model="gemini-2.5-flash-lite")
            fixer.set_prompt_output_dir(Path(tmp_dir) / "prompts")
            with patch("pipeline.code_object_repair_loop.make_llm_call_from_config", return_value={"content": "print('fixed')\n"}) as mocked_call:
                candidate = fixer.generate_candidate(
                    qualname="<module>",
                    gt_code_object=type("CodeObj", (), {"co_name": "<module>", "co_firstlineno": 1, "co_flags": 0})(),
                    derived_code_object=type("CodeObj", (), {"co_name": "<module>", "co_firstlineno": 1, "co_flags": 0})(),
                    derived_source_fragment="print('derived')\n",
                    repair_context=None,
                )
            mocked_call.assert_called_once()
            self.assertEqual(candidate, "print('fixed')")
            prompt_files = list((Path(tmp_dir) / "prompts").glob("*.json"))
            self.assertEqual(len(prompt_files), 1)
            record = json.loads(prompt_files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["response_text"], "print('fixed')")
            self.assertEqual(record["returned_text"], "print('fixed')")

    def test_semantic_repair_appends_step_log_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            log_file = root / "run_log.jsonl"
            with patch("builtins.print") as mocked_print:
                steps: list[dict] = []
                _store_semantic_step(
                    steps,
                    {
                        "step": 1,
                        "iteration": 1,
                        "qualname": "<module>",
                        "repair_operation": "repair_module_statement",
                        "target_score_before": {"combined_distance": 3},
                        "target_score_after": {"combined_distance": 2},
                        "accepted": True,
                        "acceptance_reason": "candidate improved combined distance",
                    },
                    log_file=log_file,
                    run_id="test-run",
                )

            self.assertEqual(len(steps), 1)
            self.assertTrue(log_file.exists())
            records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(any(record.get("stage") == "step" for record in records))
            self.assertTrue(any(record.get("qualname") == "<module>" for record in records if record.get("stage") == "step"))
            self.assertGreater(mocked_print.call_count, 0)

    def test_semantic_repair_announcement_prints_before_llm_call(self) -> None:
        with patch("builtins.print") as mocked_print:
            _announce_semantic_step(
                step_index=3,
                iteration=2,
                qualname="<module>",
                repair_operation="repair_module_statement",
                source_kind="function",
            )
        mocked_print.assert_called_once()
        printed_text = mocked_print.call_args.args[0]
        self.assertIn("starting step 3 iter 2 <module>", printed_text)
        self.assertIn("source_kind=function", printed_text)


if __name__ == "__main__":
    unittest.main()
