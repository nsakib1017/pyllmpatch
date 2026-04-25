from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from typing import Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYLINGUAL_ROOT = REPO_ROOT / "pylingual"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYLINGUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PYLINGUAL_ROOT))

from utils.generate_bytecode import CompileError, compile_version
from utils.map_source_code_objects import MappingError, map_source_to_pyc
from utils.pyc_code_object_distance import (
    compare_code_object_distances,
    load_editable_bytecode_from_pyc,
    summarize_results,
    validate_input,
)

class ReattachError(RuntimeError):
    pass


FragmentFixer = Callable[[str, Any, Any, str], str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Locate a mapped source code object, optionally replace its source span, "
            "compile with Python 3.10, and optionally compare the resulting .pyc."
        )
    )
    parser.add_argument("source_path", type=Path, help="Path to the source .py file")
    parser.add_argument("pyc_path", type=Path, help="Path to the corresponding .pyc file used for mapping")
    parser.add_argument("qualname", type=str, help="Mapped source qualname to extract or replace")
    parser.add_argument(
        "--replacement-file",
        type=Path,
        default=None,
        help="Path to a file containing replacement source for the mapped span",
    )
    parser.add_argument(
        "--replacement-text",
        type=str,
        default=None,
        help="Literal replacement source for the mapped span",
    )
    parser.add_argument(
        "--output-source",
        type=Path,
        default=None,
        help="Path to write the updated source file. Required when replacing.",
    )
    parser.add_argument(
        "--output-pyc",
        type=Path,
        default=None,
        help="Optional path for the compiled output .pyc. Defaults to __pycache__ next to output source.",
    )
    parser.add_argument(
        "--compare-pyc",
        type=Path,
        default=None,
        help="Optional reference .pyc path to compare against after compilation",
    )
    parser.add_argument(
        "--comparison-json-out",
        type=Path,
        default=None,
        help="Optional path to save the comparison summary as JSON",
    )
    parser.add_argument(
        "--strict-map",
        action="store_true",
        help="Require the mapper to have no unmatched rows while locating the target.",
    )
    return parser


def _load_text(path: Path) -> str:
    return path.expanduser().resolve().read_text(encoding="utf-8")


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    if not text.endswith(("\n", "\r")):
        offsets.append(len(text))
    return offsets


def _span_to_indices(
    text: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    if start_line < 1 or end_line < 1 or start_line > len(lines) or end_line > len(lines):
        raise ReattachError("Mapped span is outside the source file bounds")

    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    start_index = offsets[start_line - 1] + start_col
    end_index = offsets[end_line - 1] + end_col
    return start_index, end_index


def _find_target_row(source_path: Path, pyc_path: Path, qualname: str, strict_map: bool) -> dict:
    rows = map_source_to_pyc(source_path, pyc_path, strict=strict_map)
    candidates = [
        row
        for row in rows
        if row["row_type"] == "source_to_pyc" and row["source_qualname"] == qualname
    ]
    if not candidates:
        raise ReattachError(f"No mapped source code object found for qualname: {qualname}")
    if len(candidates) > 1:
        raise ReattachError(f"Qualname is ambiguous across {len(candidates)} rows: {qualname}")
    return candidates[0]


def extract_source_segment(source_text: str, target_row: dict) -> str:
    start_index, end_index = _span_to_indices(
        source_text,
        int(target_row["source_lineno"]),
        int(target_row["source_col_offset"]),
        int(target_row["source_end_lineno"]),
        int(target_row["source_end_col_offset"]),
    )
    return source_text[start_index:end_index]


def replace_source_segment(source_text: str, target_row: dict, replacement_text: str) -> str:
    start_index, end_index = _span_to_indices(
        source_text,
        int(target_row["source_lineno"]),
        int(target_row["source_col_offset"]),
        int(target_row["source_end_lineno"]),
        int(target_row["source_end_col_offset"]),
    )
    return source_text[:start_index] + replacement_text + source_text[end_index:]


def compile_source_to_pyc(source_path: Path, output_pyc: Path | None) -> Path:
    source_path = source_path.expanduser().resolve()
    if output_pyc is None:
        pycache_dir = source_path.parent / "__pycache__"
        pycache_dir.mkdir(parents=True, exist_ok=True)
        output_pyc = pycache_dir / f"{source_path.stem}.cpython-310.pyc"
    output_pyc = output_pyc.expanduser().resolve()
    output_pyc.parent.mkdir(parents=True, exist_ok=True)
    prior_uv_cache_dir = os.environ.get("UV_CACHE_DIR")
    os.environ["UV_CACHE_DIR"] = str((Path("/tmp") / "uv-cache").resolve())
    try:
        compile_version(source_path, output_pyc, (3, 10))
    except CompileError as exc:
        raise ReattachError(f"Python 3.10 compilation failed:\n{exc}") from exc
    finally:
        if prior_uv_cache_dir is None:
            os.environ.pop("UV_CACHE_DIR", None)
        else:
            os.environ["UV_CACHE_DIR"] = prior_uv_cache_dir
    return output_pyc


def run_comparison(compiled_pyc: Path, compare_pyc: Path) -> dict:
    results = compare_code_object_distances(validate_input(compare_pyc), validate_input(compiled_pyc))
    return summarize_results(results)


def infer_source_from_pyc(pyc_path: Path) -> Path:
    pyc_path = validate_input(pyc_path)
    if pyc_path.parent.name == "__pycache__":
        stem = pyc_path.name.split(".cpython-", 1)[0]
        candidate = pyc_path.parent.parent / f"{stem}.py"
        if candidate.is_file():
            return candidate.resolve()
    candidate = pyc_path.with_suffix(".py")
    if candidate.is_file():
        return candidate.resolve()
    raise ReattachError(f"Could not infer source file from pyc path: {pyc_path}")


def _qualname_depth(qualname: str) -> int:
    return qualname.count(".")


def _has_selected_ancestor(qualname: str, selected: set[str]) -> bool:
    parts = qualname.split(".")
    for idx in range(1, len(parts) - 1):
        ancestor = ".".join(parts[: idx + 1])
        if ancestor in selected:
            return True
    return False


def select_repair_targets(distance_rows: list[dict]) -> list[str]:
    candidates = [
        row["gt_name"]
        for row in distance_rows
        if row["status"] == "matched"
        and row["combined_distance"] > 0
        and row["gt_name"]
        and row["derived_name"]
        and row["gt_name"] == row["derived_name"]
        and row["gt_name"] != "<module>"
    ]
    ordered = sorted(set(candidates), key=lambda name: (_qualname_depth(name), name))
    selected: list[str] = []
    selected_set: set[str] = set()
    for qualname in ordered:
        if _has_selected_ancestor(qualname, selected_set):
            continue
        selected.append(qualname)
        selected_set.add(qualname)
    return selected


def index_code_objects_by_qualname(pyc_path: Path) -> dict[str, Any]:
    bytecode_root = load_editable_bytecode_from_pyc(validate_input(pyc_path))
    return {bc.name: bc.codeobj for bc in bytecode_root.iter_bytecodes()}


def run_pylingual_verification(gt_pyc: Path, candidate_pyc: Path) -> dict:
    from pylingual.equivalence_check import compare_pyc

    results = compare_pyc(validate_input(gt_pyc), validate_input(candidate_pyc))
    serialized_results = [
        {
            "success": result.success,
            "message": result.message,
            "names": result.names(),
            "failed_line_number": result.failed_line_number,
            "failed_offset": result.failed_offset,
        }
        for result in results
    ]
    return {
        "all_equal": all(result["success"] for result in serialized_results),
        "results": serialized_results,
    }


def _count_pylingual_successes(verification: dict | None) -> int | None:
    if verification is None:
        return None
    return sum(1 for result in verification["results"] if result["success"])


def _should_accept_candidate(
    previous_summary: dict,
    candidate_summary: dict,
    previous_verification: dict | None,
    candidate_verification: dict | None,
) -> tuple[bool, str]:
    previous_distance = int(previous_summary["combined_distance"])
    candidate_distance = int(candidate_summary["combined_distance"])
    if candidate_distance < previous_distance:
        previous_successes = _count_pylingual_successes(previous_verification)
        candidate_successes = _count_pylingual_successes(candidate_verification)
        if previous_successes is not None and candidate_successes is not None and candidate_successes < previous_successes:
            return False, "combined distance improved but PyLingual success count regressed"
        return True, "combined distance improved"

    if candidate_verification is not None and previous_verification is not None:
        previous_successes = _count_pylingual_successes(previous_verification)
        candidate_successes = _count_pylingual_successes(candidate_verification)
        if candidate_successes is not None and previous_successes is not None and candidate_successes > previous_successes:
            return True, "PyLingual success count improved"

    return False, "candidate did not improve combined distance or PyLingual success count"


def repair_mismatching_code_objects(
    gt_pyc: Path,
    derived_pyc: Path,
    derived_source: Path,
    *,
    output_dir: Path | None = None,
    fragment_fixer: FragmentFixer | None = None,
    strict_map: bool = True,
    verify_with_pylingual: bool = True,
    verify_each_step_with_pylingual: bool = True,
    reject_non_improving_candidates: bool = True,
) -> dict:
    gt_pyc = validate_input(gt_pyc)
    derived_pyc = validate_input(derived_pyc)
    derived_source = derived_source.expanduser().resolve()
    gt_source = infer_source_from_pyc(gt_pyc)

    if output_dir is None:
        output_dir = derived_source.parent / f"{derived_source.stem}_repair_pipeline"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fragments_dir = output_dir / "fragments"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    pyc_dir = output_dir / "__pycache__"
    pyc_dir.mkdir(parents=True, exist_ok=True)

    initial_rows = compare_code_object_distances(gt_pyc, derived_pyc)
    initial_summary = summarize_results(initial_rows)
    initial_pylingual_verification = run_pylingual_verification(gt_pyc, derived_pyc) if verify_with_pylingual else None
    repair_targets = select_repair_targets(initial_rows)
    gt_code_objects = index_code_objects_by_qualname(gt_pyc)

    current_source = derived_source
    current_pyc = derived_pyc
    current_summary = initial_summary
    current_pylingual_verification = initial_pylingual_verification
    steps: list[dict] = []

    gt_source_text = _load_text(gt_source)
    for index, qualname in enumerate(repair_targets, start=1):
        target_row = _find_target_row(current_source, current_pyc, qualname, strict_map=strict_map)
        current_text = _load_text(current_source)
        extracted_before = extract_source_segment(current_text, target_row)
        current_code_objects = index_code_objects_by_qualname(current_pyc)
        gt_code_object = gt_code_objects.get(qualname)
        derived_code_object = current_code_objects.get(qualname)
        if gt_code_object is None:
            raise ReattachError(f"No ground-truth code object found for qualname: {qualname}")
        if derived_code_object is None:
            raise ReattachError(f"No derived code object found for qualname: {qualname}")
        if fragment_fixer is None:
            gt_row = _find_target_row(gt_source, gt_pyc, qualname, strict_map=strict_map)
            replacement_text = extract_source_segment(gt_source_text, gt_row)
        else:
            replacement_text = fragment_fixer(
                qualname,
                gt_code_object,
                derived_code_object,
                extracted_before,
            )
        fragment_path = fragments_dir / f"{index:02d}_{qualname.replace('<', '').replace('>', '').replace('.', '_')}.pyfrag"
        fragment_path.write_text(replacement_text, encoding="utf-8")
        updated_text = replace_source_segment(current_text, target_row, replacement_text)

        next_source = output_dir / f"step{index}_{derived_source.stem}.py"
        next_source.write_text(updated_text, encoding="utf-8")
        next_pyc = pyc_dir / f"{next_source.stem}.cpython-310.pyc"
        compile_source_to_pyc(next_source, next_pyc)

        step_rows = compare_code_object_distances(gt_pyc, next_pyc)
        step_summary = summarize_results(step_rows)
        step_pylingual_verification = (
            run_pylingual_verification(gt_pyc, next_pyc)
            if verify_with_pylingual and verify_each_step_with_pylingual
            else None
        )
        accepted, acceptance_reason = _should_accept_candidate(
            current_summary,
            step_summary,
            current_pylingual_verification,
            step_pylingual_verification,
        )
        if not reject_non_improving_candidates:
            accepted = True
            acceptance_reason = "candidate retained without acceptance filtering"
        steps.append(
            {
                "step": index,
                "qualname": qualname,
                "fragment_path": str(fragment_path),
                "output_source": str(next_source),
                "output_pyc": str(next_pyc),
                "gt_code_object_name": getattr(gt_code_object, "co_name", None),
                "derived_code_object_name": getattr(derived_code_object, "co_name", None),
                "extracted_before": extracted_before,
                "replacement_text": replacement_text,
                "summary": step_summary,
                "pylingual_verification": step_pylingual_verification,
                "accepted": accepted,
                "acceptance_reason": acceptance_reason,
            }
        )
        if accepted:
            current_source = next_source
            current_pyc = next_pyc
            current_summary = step_summary
            if step_pylingual_verification is not None:
                current_pylingual_verification = step_pylingual_verification

    final_rows = compare_code_object_distances(gt_pyc, current_pyc)
    final_summary = summarize_results(final_rows)
    pylingual_verification = current_pylingual_verification
    if verify_with_pylingual and pylingual_verification is None:
        pylingual_verification = run_pylingual_verification(gt_pyc, current_pyc)

    return {
        "gt_source": str(gt_source),
        "gt_pyc": str(gt_pyc),
        "derived_source": str(derived_source),
        "derived_pyc": str(derived_pyc),
        "repair_targets": repair_targets,
        "initial_summary": initial_summary,
        "initial_pylingual_verification": initial_pylingual_verification,
        "final_source": str(current_source),
        "final_pyc": str(current_pyc),
        "final_summary": final_summary,
        "pylingual_verification": pylingual_verification,
        "steps": steps,
    }


def main() -> int:
    args = build_parser().parse_args()

    source_path = args.source_path.expanduser().resolve()
    pyc_path = validate_input(args.pyc_path)
    target_row = _find_target_row(source_path, pyc_path, args.qualname, strict_map=args.strict_map)
    source_text = _load_text(source_path)
    extracted = extract_source_segment(source_text, target_row)

    print("Matched row:")
    print(json.dumps(target_row, indent=2))
    print("\nExtracted source:")
    print(extracted)

    replacement_text = None
    if args.replacement_file is not None and args.replacement_text is not None:
        raise ReattachError("Use only one of --replacement-file or --replacement-text")
    if args.replacement_file is not None:
        replacement_text = _load_text(args.replacement_file)
    elif args.replacement_text is not None:
        replacement_text = args.replacement_text

    if replacement_text is None:
        return 0

    if args.output_source is None:
        raise ReattachError("--output-source is required when replacing source")

    output_source = args.output_source.expanduser().resolve()
    output_source.parent.mkdir(parents=True, exist_ok=True)
    updated_text = replace_source_segment(source_text, target_row, replacement_text)
    output_source.write_text(updated_text, encoding="utf-8")
    print(f"\nUpdated source written to: {output_source}")

    compiled_pyc = compile_source_to_pyc(output_source, args.output_pyc)
    print(f"Compiled Python 3.10 .pyc: {compiled_pyc}")

    if args.compare_pyc is not None:
        summary = run_comparison(compiled_pyc, args.compare_pyc.expanduser().resolve())
        print("\nComparison summary:")
        print(json.dumps(summary, indent=2))
        if args.comparison_json_out is not None:
            out_path = args.comparison_json_out.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Comparison JSON written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
