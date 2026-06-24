from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.code_object_repair_loop import CodeObjectRepairLoop
from utils.generate_bytecode import compile_version


GT_FUNCTION = """\
def choose(value: int) -> int:
    total: int = value
    values = [item for item in range(3)]
    try:
        if total >= 10:
            return total - 1 + len(values)
        return total + 1 + len(values)
    finally:
        total += 0
"""


DERIVED_FUNCTION = """\
def choose(value: int) -> int:
    total: int = value
    values = [item for item in range(3)]
    try:
        if total >= 10:
            return total + 5 + len(values)
        return total - 5 + len(values)
    finally:
        total += 0
"""


HEADER = """\
VALUE: int = 7
"""


class MockSemanticFixer:
    def __init__(self, fixed_fragment: str) -> None:
        self.fixed_fragment = fixed_fragment
        self.calls: list[dict[str, Any]] = []

    def set_prompt_output_dir(self, output_dir: Path | None) -> None:
        self.prompt_output_dir = None if output_dir is None else str(output_dir)

    def generate_candidates(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> list[dict[str, Any]]:
        return [self._candidate(qualname, derived_source_fragment, repair_context)]

    def generate_candidate(
        self,
        *,
        qualname: str,
        gt_code_object: Any,
        derived_code_object: Any,
        derived_source_fragment: str,
        repair_context: dict | None = None,
    ) -> str:
        self._record_call(qualname, derived_source_fragment, repair_context)
        return self.fixed_fragment

    def _candidate(
        self,
        qualname: str,
        derived_source_fragment: str,
        repair_context: dict | None,
    ) -> dict[str, Any]:
        self._record_call(qualname, derived_source_fragment, repair_context)
        return {
            "candidate_index": 0,
            "strategy": "mock_ground_truth_fragment",
            "text": self.fixed_fragment,
            "selected_action_types": ["mocked_llm_output"],
        }

    def _record_call(
        self,
        qualname: str,
        derived_source_fragment: str,
        repair_context: dict | None,
    ) -> None:
        self.calls.append(
            {
                "qualname": qualname,
                "derived_fragment_length": len(derived_source_fragment),
                "target_kind": None if repair_context is None else repair_context.get("target_kind"),
            }
        )


def _source_text(function_text: str) -> str:
    return HEADER + "\n\n" + textwrap.dedent(function_text).strip() + "\n\n"


def _compile_source(source_path: Path, pyc_path: Path, version: tuple[int, int]) -> None:
    pyc_path.parent.mkdir(parents=True, exist_ok=True)
    compile_version(source_path, pyc_path, version)


def _run_version(version: tuple[int, int], base_dir: Path) -> dict[str, Any]:
    version_label = f"{version[0]}.{version[1]}"
    version_tag = f"{version[0]}{version[1]}"
    case_dir = base_dir / f"py{version_tag}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    gt_source = case_dir / "sample_gt.py"
    derived_source = case_dir / "sample_derived.py"
    gt_source.write_text(_source_text(GT_FUNCTION), encoding="utf-8")
    derived_source.write_text(_source_text(DERIVED_FUNCTION), encoding="utf-8")

    pycache_dir = case_dir / "__pycache__"
    gt_pyc = pycache_dir / f"sample_gt.cpython-{version_tag}.pyc"
    derived_pyc = pycache_dir / f"sample_derived.cpython-{version_tag}.pyc"
    _compile_source(gt_source, gt_pyc, version)
    _compile_source(derived_source, derived_pyc, version)

    fixer = MockSemanticFixer(textwrap.dedent(GT_FUNCTION).strip() + "\n")
    output_dir = case_dir / "repair"
    result = CodeObjectRepairLoop(fixer).run(
        gt_pyc=gt_pyc,
        derived_pyc=derived_pyc,
        derived_source=derived_source,
        output_dir=output_dir,
        verify_with_pylingual=True,
        verify_each_step_with_pylingual=True,
        reject_non_improving_candidates=True,
        max_iterations=2,
    )

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    initial_distance = result.get("initial_summary", {}).get("combined_distance")
    final_distance = result.get("final_summary", {}).get("combined_distance")
    pylingual_all_equal = (result.get("pylingual_verification") or {}).get("all_equal")
    accepted_steps = [step for step in result.get("steps", []) if step.get("accepted")]
    final_pyc = str(result.get("final_pyc") or "")
    success = (
        result.get("target_python_version") == version_label
        and initial_distance is not None
        and int(initial_distance) > 0
        and final_distance == 0
        and pylingual_all_equal is True
        and bool(accepted_steps)
        and f"cpython-{version_tag}" in final_pyc
        and len(fixer.calls) > 0
    )

    return {
        "version": version_label,
        "success": success,
        "initial_combined_distance": initial_distance,
        "final_combined_distance": final_distance,
        "pylingual_all_equal": pylingual_all_equal,
        "accepted_step_count": len(accepted_steps),
        "mock_fixer_call_count": len(fixer.calls),
        "target_python_version": result.get("target_python_version"),
        "final_pyc": final_pyc,
        "result_json": str(result_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate semantic repair plumbing across Python bytecode versions using mocked fixer output."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "semantic_repair_version_validation",
    )
    parser.add_argument(
        "--versions",
        nargs="*",
        default=["3.10", "3.11", "3.12", "3.13", "3.14"],
        help="Python versions to validate, e.g. 3.10 3.14",
    )
    args = parser.parse_args()

    versions = []
    for value in args.versions:
        major, minor = value.split(".", 1)
        versions.append((int(major), int(minor)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = []
    for version in versions:
        try:
            summary = _run_version(version, args.output_dir)
        except Exception as exc:
            summary = {
                "version": f"{version[0]}.{version[1]}",
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        summaries.append(summary)
        if not summary.get("success"):
            failures.append(summary)
        status = "PASS" if summary.get("success") else "FAIL"
        print(
            f"{status} {summary['version']}: "
            f"{summary.get('initial_combined_distance')} -> {summary.get('final_combined_distance')}, "
            f"pylingual={summary.get('pylingual_all_equal')}"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
