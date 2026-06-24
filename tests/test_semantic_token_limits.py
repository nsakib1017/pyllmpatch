from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finetuning.model_finetuner_qwen_codeobject_semantic import filter_samples_by_token_budget
from pipeline import code_object_repair_loop
from pipeline.code_object_repair_loop import LLMFragmentFixer
from utils.reattach_source_code_object import (
    ReattachError,
    _combined_distance_improved,
    compile_source_to_pyc,
    infer_semantic_repair_python_version,
)
from utils.version import PythonVersion


class CodeObj:
    co_name = "sample"
    co_qualname = "sample"
    co_argcount = 0
    co_posonlyargcount = 0
    co_kwonlyargcount = 0
    co_flags = 0
    co_varnames = ()
    co_names = ()
    co_freevars = ()
    co_cellvars = ()
    co_consts = (None,)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(str(message["content"]) for message in messages)

    def __call__(self, text, add_special_tokens=False, truncation=False):
        del add_special_tokens, truncation
        return {"input_ids": text.split()}


class SemanticTokenLimitTest(unittest.TestCase):
    def test_llm_fragment_fixer_skips_oversized_prompt_before_provider_call(self) -> None:
        fixer = LLMFragmentFixer(provider="Google", model="gemini-2.5-flash-lite")
        source = "def sample():\n    return 1\n"

        output = io.StringIO()
        with patch("pipeline.code_object_repair_loop.count_tokens_safe", return_value=70000), patch(
            "pipeline.code_object_repair_loop.make_llm_call_from_config"
        ) as mocked_call:
            with contextlib.redirect_stdout(output):
                candidate = fixer.generate_candidate(
                    qualname="<module>.sample",
                    gt_code_object=CodeObj(),
                    derived_code_object=CodeObj(),
                    derived_source_fragment=source,
                    repair_context={},
                )

        mocked_call.assert_not_called()
        self.assertEqual(candidate, source)
        self.assertTrue(fixer.calls[-1]["skipped_due_to_token_limit"])
        self.assertEqual(fixer.calls[-1]["prompt_token_count"], 70000)
        self.assertIn("skipping LLM call for <module>.sample", output.getvalue())
        self.assertIn("prompt token count 70000 exceeds threshold", output.getvalue())

    def test_missing_target_token_skip_returns_empty_fragment(self) -> None:
        fixer = LLMFragmentFixer(provider="Google", model="gemini-2.5-flash-lite")

        with patch("pipeline.code_object_repair_loop.count_tokens_safe", return_value=70000), patch(
            "pipeline.code_object_repair_loop.make_llm_call_from_config"
        ) as mocked_call:
            candidate = fixer.generate_candidate(
                qualname="<module>.missing",
                gt_code_object=CodeObj(),
                derived_code_object=None,
                derived_source_fragment="def parent():\n    pass\n",
                repair_context={},
            )

        mocked_call.assert_not_called()
        self.assertEqual(candidate, "")
        self.assertTrue(fixer.calls[-1]["skipped_due_to_token_limit"])

    def test_finetuning_filter_drops_chat_templated_samples_over_max_seq_length(self) -> None:
        samples = [
            {"messages": [{"role": "user", "content": "short prompt"}, {"role": "assistant", "content": "ok"}]},
            {"messages": [{"role": "user", "content": "this prompt has too many tokens"}, {"role": "assistant", "content": "ok"}]},
        ]

        kept = filter_samples_by_token_budget(
            samples,
            FakeTokenizer(),
            max_seq_length=4,
            enable_thinking_template=False,
        )

        self.assertEqual(kept, [samples[0]])

    def test_combined_distance_improvement_check(self) -> None:
        self.assertTrue(_combined_distance_improved({"combined_distance": 10}, {"combined_distance": 9}))
        self.assertFalse(_combined_distance_improved({"combined_distance": 10}, {"combined_distance": 10}))
        self.assertFalse(_combined_distance_improved({"combined_distance": 10}, {"combined_distance": 11}))
        self.assertTrue(_combined_distance_improved({"combined_distance": 10}, {"combined_distance": 8}, min_delta=2))
        self.assertFalse(_combined_distance_improved({"combined_distance": 10}, {"combined_distance": 9}, min_delta=2))

    def test_compile_source_to_pyc_uses_target_python_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sample.py"
            source.write_text("value = 1\n", encoding="utf-8")

            with patch("utils.reattach_source_code_object.compile_version") as mocked_compile:
                output_pyc = compile_source_to_pyc(source, None, PythonVersion((3, 14)))

            self.assertEqual(output_pyc.name, "sample.cpython-314.pyc")
            mocked_compile.assert_called_once()
            self.assertEqual(mocked_compile.call_args.args[2], (3, 14))

    def test_infer_semantic_repair_python_version_rejects_mismatched_pyc_versions(self) -> None:
        with patch(
            "utils.reattach_source_code_object._pyc_python_version",
            side_effect=[PythonVersion((3, 14)), PythonVersion((3, 13))],
        ):
            with self.assertRaises(ReattachError):
                infer_semantic_repair_python_version(Path("gt.pyc"), Path("derived.pyc"))

    def test_dataset_loop_defers_preflight_risky_rows_and_processes_easy_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text(
                "file_hash,source,error_type\nhard,PyPi,semantic_error\neasy,PyPi,semantic_error\n",
                encoding="utf-8",
            )
            paths = {}
            for file_hash in ("hard", "easy"):
                gt_source = root / f"{file_hash}_gt.py"
                gt_pyc = root / f"{file_hash}_gt.pyc"
                derived_pyc = root / f"{file_hash}_derived.pyc"
                derived_source = root / f"{file_hash}_derived.py"
                for path in (gt_source, gt_pyc, derived_pyc, derived_source):
                    path.write_text("", encoding="utf-8")
                paths[file_hash] = (gt_source, gt_pyc, derived_pyc, derived_source)

            call_order: list[str] = []

            def fake_preflight(gt_pyc: Path, derived_pyc: Path) -> dict:
                del derived_pyc
                if gt_pyc.name.startswith("hard"):
                    return {
                        "initial_combined_distance": 500,
                        "initial_gt_code_object_count": 10,
                        "initial_derived_code_object_count": 10,
                        "repair_target_count": 4,
                        "missing_target_count": 0,
                        "extra_target_count": 0,
                        "expression_child_parent_target_count": 0,
                    }
                return {
                    "initial_combined_distance": 5,
                    "initial_gt_code_object_count": 1,
                    "initial_derived_code_object_count": 1,
                    "repair_target_count": 1,
                    "missing_target_count": 0,
                    "extra_target_count": 0,
                    "expression_child_parent_target_count": 0,
                }

            class FakeLoop:
                def run(self, **kwargs):
                    file_hash = Path(kwargs["gt_pyc"]).name.split("_", 1)[0]
                    call_order.append(file_hash)
                    return {
                        "steps": [],
                        "pylingual_verification": {"all_equal": True},
                        "gt_pyc": str(kwargs["gt_pyc"]),
                        "derived_pyc": str(kwargs["derived_pyc"]),
                        "derived_source": str(kwargs["derived_source"]),
                        "initial_summary": {
                            "combined_distance": 5,
                            "gt_code_object_count": 1,
                            "derived_code_object_count": 1,
                        },
                        "final_summary": {
                            "combined_distance": 0,
                            "gt_code_object_count": 1,
                            "derived_code_object_count": 1,
                        },
                        "repair_targets": ["<module>.easy"],
                        "sample_timed_out": False,
                        "sample_timeout_reached": False,
                        "sample_hard_timeout_reached": False,
                        "sample_timeout_checkpoint_count": 0,
                        "sample_timeout_action": None,
                        "sample_timeout_reason": None,
                        "sample_timeout_best_combined_distance": 0,
                        "sample_timeout_best_improvement_reason": None,
                    }

            with patch.object(
                code_object_repair_loop,
                "fetch_pyllmpatch_source_path",
                side_effect=lambda file_hash, source: paths[file_hash][0],
            ), patch.object(
                code_object_repair_loop,
                "fetch_pyllmpatch_repair_paths",
                side_effect=lambda file_hash, source: paths[file_hash][1:],
            ), patch.object(
                code_object_repair_loop, "_semantic_repair_preflight", side_effect=fake_preflight
            ), patch.object(
                code_object_repair_loop, "OracleFragmentFixer", return_value=object()
            ), patch.object(
                code_object_repair_loop, "CodeObjectRepairLoop", return_value=FakeLoop()
            ):
                result = code_object_repair_loop.run_dataset_repair_loop(
                    fixer_name="oracle",
                    dataset_path=dataset_path,
                    output_dir=root / "out",
                    process_easy_cases_first=True,
                    defer_preflight_risky_samples=True,
                    preflight_max_repair_targets=2,
                    preflight_max_initial_combined_distance=200,
                )

            self.assertEqual(call_order, ["easy"])
            self.assertEqual(result["processed_rows"], 2)
            self.assertEqual(result["repaired_rows"], 1)
            self.assertEqual(result["deferred_rows"], 1)
            results_text = Path(result["results_csv"]).read_text(encoding="utf-8")
            deferred_text = Path(result["deferred_csv"]).read_text(encoding="utf-8")
            self.assertIn("easy", results_text)
            self.assertNotIn("hard,PyPi,semantic_error,True", results_text)
            self.assertIn("hard,PyPi,semantic_error,preflight", deferred_text)

    def test_dataset_loop_logs_hard_timeout_kept_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text("file_hash,source,error_type\nabc123,PyPi,semantic_error\n", encoding="utf-8")
            gt_source = root / "gt.py"
            gt_pyc = root / "gt.pyc"
            derived_pyc = root / "derived.pyc"
            derived_source = root / "derived.py"
            for path in (gt_source, gt_pyc, derived_pyc, derived_source):
                path.write_text("", encoding="utf-8")

            captured_kwargs = {}

            class FakeLoop:
                def run(self, **kwargs):
                    captured_kwargs.update(kwargs)
                    return {
                        "steps": [],
                        "pylingual_verification": None,
                        "gt_pyc": str(kwargs["gt_pyc"]),
                        "derived_pyc": str(kwargs["derived_pyc"]),
                        "derived_source": str(kwargs["derived_source"]),
                        "initial_summary": {
                            "combined_distance": 10,
                            "gt_code_object_count": 1,
                            "derived_code_object_count": 1,
                        },
                        "final_summary": {
                            "combined_distance": 5,
                            "gt_code_object_count": 1,
                            "derived_code_object_count": 1,
                        },
                        "repair_targets": [],
                        "sample_timed_out": True,
                        "sample_timeout_reached": True,
                        "sample_hard_timeout_reached": True,
                        "sample_timeout_checkpoint_count": 0,
                        "sample_timeout_action": "stopped_at_hard_timeout_kept_best_result",
                        "sample_timeout_reason": "sample exceeded hard timeout 10800 seconds; keeping best result",
                        "sample_timeout_best_combined_distance": 5,
                        "sample_timeout_best_improvement_reason": "accepted target",
                    }

            output = io.StringIO()
            with patch.object(code_object_repair_loop, "fetch_pyllmpatch_source_path", return_value=gt_source), patch.object(
                code_object_repair_loop,
                "fetch_pyllmpatch_repair_paths",
                return_value=(gt_pyc, derived_pyc, derived_source),
            ), patch.object(code_object_repair_loop, "OracleFragmentFixer", return_value=object()), patch.object(
                code_object_repair_loop, "CodeObjectRepairLoop", return_value=FakeLoop()
            ):
                with contextlib.redirect_stdout(output):
                    result = code_object_repair_loop.run_dataset_repair_loop(
                        fixer_name="oracle",
                        dataset_path=dataset_path,
                        output_dir=root / "out",
                        sample_timeout_seconds=0,
                        sample_hard_timeout_seconds=10800,
                    )

            self.assertIsNone(captured_kwargs["sample_timeout_deadline"])
            self.assertIsNotNone(captured_kwargs["sample_hard_timeout_deadline"])
            self.assertEqual(result["repaired_rows"], 1)
            self.assertEqual(result["timed_out_rows"], 1)
            self.assertEqual(result["hard_timed_out_rows"], 1)
            self.assertIn("hard timeout reached after combined-distance improvement; kept best result", output.getvalue())

            log_entries = [
                json.loads(line)
                for line in Path(result["run_log"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(any(entry.get("sample_hard_timeout_reached") is True for entry in log_entries))


if __name__ == "__main__":
    unittest.main()
