from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import code_object_repair_loop


def _fake_loop_result(gt_pyc: Path, derived_pyc: Path, derived_source: Path) -> dict:
    """Minimal CodeObjectRepairLoop.run() return shape the aggregator needs."""
    return {
        "steps": [],
        "pylingual_verification": {"all_equal": True},
        "gt_pyc": str(gt_pyc),
        "derived_pyc": str(derived_pyc),
        "derived_source": str(derived_source),
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
        "repair_targets": ["foo"],
        "sample_timed_out": False,
        "sample_timeout_reached": False,
        "sample_hard_timeout_reached": False,
        "sample_timeout_checkpoint_count": 0,
        "sample_timeout_action": None,
        "sample_timeout_reason": None,
        "sample_timeout_best_combined_distance": 0,
        "sample_timeout_best_improvement_reason": None,
    }


class PreflightHasModuleBodyTargetTest(unittest.TestCase):
    def test_true_when_module_target_present(self) -> None:
        has = code_object_repair_loop._preflight_has_module_body_target
        self.assertTrue(has({"repair_targets": ["<module>"]}))
        self.assertTrue(has({"repair_targets": ["<module>", "foo"]}))

    def test_false_when_no_module_target(self) -> None:
        has = code_object_repair_loop._preflight_has_module_body_target
        self.assertFalse(has({"repair_targets": ["foo", "bar"]}))
        self.assertFalse(has({"repair_targets": []}))
        self.assertFalse(has({}))
        self.assertFalse(has(None))


class SkipModuleBodyRepairLoopTest(unittest.TestCase):
    def _run_loop(
        self,
        targets_by_hash: dict[str, list[str]],
        *,
        skip_module_body_repair: bool,
        process_easy_cases_first: bool = True,
    ) -> tuple[list[str], dict, str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            header = "file_hash,source,error_type\n"
            body = "".join(f"{h},PyPi,semantic_error\n" for h in targets_by_hash)
            dataset_path.write_text(header + body, encoding="utf-8")

            paths: dict[str, tuple[Path, Path, Path, Path]] = {}
            for file_hash in targets_by_hash:
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
                file_hash = gt_pyc.name.split("_", 1)[0]
                targets = targets_by_hash[file_hash]
                return {
                    "initial_combined_distance": 5,
                    "initial_gt_code_object_count": 1,
                    "initial_derived_code_object_count": 1,
                    "repair_target_count": len(targets),
                    "missing_target_count": 0,
                    "extra_target_count": 0,
                    "expression_child_parent_target_count": 0,
                    "repair_targets": targets,
                }

            class FakeLoop:
                def run(self, **kwargs):
                    file_hash = Path(kwargs["gt_pyc"]).name.split("_", 1)[0]
                    call_order.append(file_hash)
                    return _fake_loop_result(
                        kwargs["gt_pyc"], kwargs["derived_pyc"], kwargs["derived_source"]
                    )

            with patch.object(
                code_object_repair_loop,
                "fetch_pyllmpatch_gt_source_path",
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
                    process_easy_cases_first=process_easy_cases_first,
                    skip_module_body_repair=skip_module_body_repair,
                )

            results_text = Path(result["results_csv"]).read_text(encoding="utf-8")
            deferred_text = Path(result["deferred_csv"]).read_text(encoding="utf-8")
            return call_order, result, results_text, deferred_text

    def test_module_only_file_is_deferred_not_repaired(self) -> None:
        call_order, result, _results, deferred_text = self._run_loop(
            {"mod": ["<module>"]}, skip_module_body_repair=True
        )
        self.assertEqual(call_order, [])
        self.assertEqual(result["deferred_rows"], 1)
        self.assertEqual(result["repaired_rows"], 0)
        self.assertIn("module_body_repair", deferred_text)

    def test_mixed_module_and_leaf_file_is_deferred(self) -> None:
        call_order, result, _results, deferred_text = self._run_loop(
            {"mixed": ["<module>", "foo"]}, skip_module_body_repair=True
        )
        self.assertEqual(call_order, [])
        self.assertEqual(result["deferred_rows"], 1)
        self.assertIn("module_body_repair", deferred_text)

    def test_leaf_only_file_is_repaired_when_flag_on(self) -> None:
        call_order, result, _results, _deferred = self._run_loop(
            {"leaf": ["foo"]}, skip_module_body_repair=True
        )
        self.assertEqual(call_order, ["leaf"])
        self.assertEqual(result["repaired_rows"], 1)
        self.assertEqual(result["deferred_rows"], 0)

    def test_module_file_is_repaired_when_flag_off(self) -> None:
        call_order, result, _results, _deferred = self._run_loop(
            {"mod": ["<module>"]}, skip_module_body_repair=False
        )
        self.assertEqual(call_order, ["mod"])
        self.assertEqual(result["repaired_rows"], 1)
        self.assertEqual(result["deferred_rows"], 0)

    def test_skip_computes_preflight_on_demand_without_easy_first(self) -> None:
        # easy-first off -> preflight not precomputed; the skip must compute it on demand.
        call_order, result, _results, deferred_text = self._run_loop(
            {"mod": ["<module>"]},
            skip_module_body_repair=True,
            process_easy_cases_first=False,
        )
        self.assertEqual(call_order, [])
        self.assertEqual(result["deferred_rows"], 1)
        self.assertIn("module_body_repair", deferred_text)


if __name__ == "__main__":
    unittest.main()
