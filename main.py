from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project entrypoint.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("syntactic-repair", help="Run the existing syntax-repair experiment pipeline")

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
        "--strict-map",
        action="store_true",
        help="Require strict source-to-pyc mapping for span lookup",
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
        choices=("oracle",),
        default="oracle",
        help="Fragment fixer backend to use",
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
        "--file-hash",
        type=str,
        default=None,
        help="Optional file hash filter for dataset mode",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "syntactic-repair"

    if command == "syntactic-repair":
        from pipeline.config import load_runtime_config
        from pipeline.runner import run_experiment

        run_experiment(load_runtime_config())
        return

    if command == "semantic-repair":
        from pipeline.code_object_repair_loop import CodeObjectRepairLoop, OracleFragmentFixer, run_dataset_repair_loop
        from pipeline.config import BASE_DATASET_PATH

        if args.fixer != "oracle":
            raise ValueError(f"Unsupported fixer backend: {args.fixer}")

        if args.dataset_mode:
            result = run_dataset_repair_loop(
                fixer_name=args.fixer,
                dataset_path=args.dataset_path or BASE_DATASET_PATH,
                output_dir=args.output_dir,
                limit=args.limit,
                file_hash=args.file_hash,
                strict_map=args.strict_map,
                verify_with_pylingual=not args.skip_pylingual_verification,
                verify_each_step_with_pylingual=not args.skip_step_verification,
                reject_non_improving_candidates=not args.keep_non_improving,
            )
        else:
            if args.gt_pyc is None or args.derived_pyc is None or args.derived_source is None:
                raise ValueError("gt_pyc, derived_pyc, and derived_source are required unless --dataset-mode is used")
            loop = CodeObjectRepairLoop(OracleFragmentFixer(args.gt_pyc))
            result = loop.run(
                gt_pyc=args.gt_pyc,
                derived_pyc=args.derived_pyc,
                derived_source=args.derived_source,
                output_dir=args.output_dir,
                strict_map=args.strict_map,
                verify_with_pylingual=not args.skip_pylingual_verification,
                verify_each_step_with_pylingual=not args.skip_step_verification,
                reject_non_improving_candidates=not args.keep_non_improving,
            )
        if args.json_out is not None:
            args.json_out.expanduser().resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
