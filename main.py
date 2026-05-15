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
            "Maximum runtime for one semantic dataset-mode sample in seconds. "
            "Defaults to SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS or 3600; set 0 to disable."
        ),
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
            SEMANTIC_REPAIR_SAMPLE_TIMEOUT_SECONDS,
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
