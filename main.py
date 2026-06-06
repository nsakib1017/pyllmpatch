from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project entrypoint.")
    subparsers = parser.add_subparsers(dest="command")

    syntax = subparsers.add_parser("syntactic-repair", help="Run the existing syntax-repair experiment pipeline")
    syntax.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional source filter for the syntax dataset (for example VirusTotal, pylingual, or PyPi)",
    )
    syntax.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for the syntax dataset after filtering",
    )

    repair_loop = subparsers.add_parser(
        "semantic-repair",
        help="Run the per-code-object semantic repair loop",
    )
    repair_loop.add_argument("gt_pyc", type=Path, nargs="?", help="Ground-truth .pyc path")
    repair_loop.add_argument("derived_pyc", type=Path, nargs="?", help="Derived .pyc path")
    repair_loop.add_argument("derived_source", type=Path, nargs="?", help="Derived source .py path")
    repair_loop.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for intermediate repaired files and fragments",
    )
    repair_loop.add_argument(
        "--skip-pylingual-verification",
        action="store_true",
        help="Disable final and per-step PyLingual equivalence checks",
    )
    repair_loop.add_argument(
        "--skip-step-verification",
        action="store_true",
        help="Disable per-step PyLingual checks while keeping final verification enabled",
    )
    repair_loop.add_argument(
        "--keep-non-improving",
        action="store_true",
        help="Retain candidates even when they do not improve the measured state",
    )
    repair_loop.add_argument(
        "--fixer",
        choices=("oracle", "llm"),
        default="oracle",
        help="Fragment fixer backend to use",
    )
    repair_loop.add_argument(
        "--llm-provider",
        type=str,
        default="Google",
        help="Provider from utils.providers for --fixer llm",
    )
    repair_loop.add_argument(
        "--llm-model",
        type=str,
        help="Model name from utils.providers for --fixer llm",
    )
    repair_loop.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum semantic repair iterations over recomputed mismatch targets",
    )
    repair_loop.add_argument(
        "--sample-timeout-seconds",
        type=int,
        default=None,
        help=(
            "Timeout checkpoint for one semantic dataset-mode sample in seconds. "
            "If no combined-distance improvement exists at the checkpoint, skip the sample. "
            "Defaults to SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS or 3600; set 0 to disable."
        ),
    )
    repair_loop.add_argument(
        "--sample-hard-timeout-seconds",
        type=int,
        default=None,
        help=(
            "Absolute hard cap for one semantic dataset-mode sample in seconds. "
            "Defaults to SEMANTIC_REPAIR_SAMPLE_HARD_TIMEOUT_SECONDS or 10800; set 0 to disable."
        ),
    )
    repair_loop.add_argument(
        "--sample-timeout-min-improvement-delta",
        type=int,
        default=None,
        help=(
            "Minimum combined-distance decrease counted as timeout progress. "
            "Defaults to SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA or 1."
        ),
    )
    repair_loop.add_argument(
        "--defer-preflight-risky-samples",
        action="store_true",
        help=(
            "In dataset mode, write samples that fail the bytecode-distance preflight to "
            "semantic_repair_deferred_*.csv instead of processing them."
        ),
    )
    repair_loop.add_argument(
        "--defer-timeout-no-improvement",
        action="store_true",
        help=(
            "In dataset mode, write timeout-without-improvement samples to "
            "semantic_repair_deferred_*.csv instead of the main results CSV."
        ),
    )
    repair_loop.add_argument(
        "--process-easy-cases-first",
        action="store_true",
        help=(
            "In dataset mode, compute preflight metrics and process likely easy samples first "
            "within the selected row set."
        ),
    )
    repair_loop.add_argument(
        "--preflight-max-repair-targets",
        type=int,
        default=None,
        help=(
            "Maximum initial repair targets allowed by --defer-preflight-risky-samples. "
            "Defaults to SEMANTIC_REPAIR_PREFLIGHT_MAX_REPAIR_TARGETS or 2."
        ),
    )
    repair_loop.add_argument(
        "--preflight-max-initial-combined-distance",
        type=int,
        default=None,
        help=(
            "Maximum initial combined distance allowed by --defer-preflight-risky-samples. "
            "Defaults to SEMANTIC_REPAIR_PREFLIGHT_MAX_INITIAL_COMBINED_DISTANCE or 200."
        ),
    )
    repair_loop.add_argument(
        "--preflight-allow-missing-targets",
        action="store_true",
        help="Allow preflight samples with initial missing code-object targets.",
    )
    repair_loop.add_argument(
        "--preflight-allow-extra-targets",
        action="store_true",
        help="Allow preflight samples with initial extra derived code-object targets.",
    )
    repair_loop.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full loop result as JSON",
    )
    repair_loop.add_argument(
        "--dataset-mode",
        action="store_true",
        help="Run semantic repair for semantic_error rows in the env-configured dataset",
    )
    repair_loop.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional dataset CSV override for dataset mode",
    )
    repair_loop.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for dataset mode",
    )
    repair_loop.add_argument(
        "--row-range",
        "--range",
        dest="row_range",
        type=str,
        default=None,
        help="Optional dataset-mode row range after filters, before --limit. Uses zero-based START:END slicing, e.g. 10:20, 10:, :20, or 10.",
    )
    repair_loop.add_argument(
        "--file-hash",
        type=str,
        default=None,
        help="Optional file hash filter for dataset mode",
    )
    repair_loop.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional source filter for dataset mode (for example VirusTotal, pylingual, or PyPi)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "syntactic-repair"

    if command == "syntactic-repair":
        from pipeline.config import load_runtime_config
        from pipeline.runner import run_experiment

        run_experiment(load_runtime_config(), source=args.source, limit=args.limit)
        return

    if command == "semantic-repair":
        from pipeline.code_object_repair_loop import (
            SEMANTIC_REPAIR_PREFLIGHT_MAX_INITIAL_COMBINED_DISTANCE,
            SEMANTIC_REPAIR_PREFLIGHT_MAX_REPAIR_TARGETS,
            SEMANTIC_REPAIR_SAMPLE_HARD_TIMEOUT_SECONDS,
            SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS,
            SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA,
            CodeObjectRepairLoop,
            LLMFragmentFixer,
            OracleFragmentFixer,
            run_dataset_repair_loop,
        )
        from pipeline.config import BASE_DATASET_PATH

        if args.fixer not in {"oracle", "llm"}:
            raise ValueError(f"Unsupported fixer backend: {args.fixer}")

        if args.dataset_mode:
            result = run_dataset_repair_loop(
                fixer_name=args.fixer,
                dataset_path=args.dataset_path or BASE_DATASET_PATH,
                output_dir=args.output_dir,
                limit=args.limit,
                file_hash=args.file_hash,
                source=args.source,
                row_range=args.row_range,
                verify_with_pylingual=not args.skip_pylingual_verification,
                verify_each_step_with_pylingual=not args.skip_step_verification,
                reject_non_improving_candidates=not args.keep_non_improving,
                max_iterations=args.max_iterations,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                sample_timeout_seconds=args.sample_timeout_seconds
                if args.sample_timeout_seconds is not None
                else SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS,
                sample_hard_timeout_seconds=args.sample_hard_timeout_seconds
                if args.sample_hard_timeout_seconds is not None
                else SEMANTIC_REPAIR_SAMPLE_HARD_TIMEOUT_SECONDS,
                sample_timeout_min_improvement_delta=args.sample_timeout_min_improvement_delta
                if args.sample_timeout_min_improvement_delta is not None
                else SEMANTIC_REPAIR_TIMEOUT_MIN_IMPROVEMENT_DELTA,
                defer_preflight_risky_samples=args.defer_preflight_risky_samples,
                defer_timeout_no_improvement=args.defer_timeout_no_improvement,
                process_easy_cases_first=args.process_easy_cases_first,
                preflight_max_repair_targets=args.preflight_max_repair_targets
                if args.preflight_max_repair_targets is not None
                else SEMANTIC_REPAIR_PREFLIGHT_MAX_REPAIR_TARGETS,
                preflight_max_initial_combined_distance=args.preflight_max_initial_combined_distance
                if args.preflight_max_initial_combined_distance is not None
                else SEMANTIC_REPAIR_PREFLIGHT_MAX_INITIAL_COMBINED_DISTANCE,
                preflight_allow_missing_targets=args.preflight_allow_missing_targets,
                preflight_allow_extra_targets=args.preflight_allow_extra_targets,
            )
        else:
            if args.gt_pyc is None or args.derived_pyc is None or args.derived_source is None:
                raise ValueError("gt_pyc, derived_pyc, and derived_source are required unless --dataset-mode is used")
            fixer = (
                OracleFragmentFixer(args.gt_pyc)
                if args.fixer == "oracle"
                else LLMFragmentFixer(provider=args.llm_provider, model=args.llm_model)
            )
            loop = CodeObjectRepairLoop(fixer)
            result = loop.run(
                gt_pyc=args.gt_pyc,
                derived_pyc=args.derived_pyc,
                derived_source=args.derived_source,
                output_dir=args.output_dir,
                verify_with_pylingual=not args.skip_pylingual_verification,
                verify_each_step_with_pylingual=not args.skip_step_verification,
                reject_non_improving_candidates=not args.keep_non_improving,
                max_iterations=args.max_iterations,
            )
        if args.json_out is not None:
            args.json_out.expanduser().resolve().write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result, indent=2, default=str))
        return

    raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
