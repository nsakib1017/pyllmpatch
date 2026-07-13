"""Pure row-sharding + per-worker env wiring for the parallel scale run (speedup step 4).

Kept separate from the orchestrator shell so the correctness-critical logic (disjoint/complete
shards; per-worker env isolation) is unit-tested without a GPU or a running server. The
orchestrator (tools/run_scale_parallel.sh) calls `python -m tools.scale_shard shards N W` for the
row ranges and sets each worker's env from `worker_env`.
"""
from __future__ import annotations

import sys


def compute_shards(total: int, workers: int) -> list[str]:
    """Partition ``[0, total)`` into up to ``workers`` contiguous, disjoint, complete row-range
    strings ``"start:end"``. Balanced to within one row (earlier shards absorb the remainder).
    Empty shards are dropped, so ``len(result) == min(workers, total)`` for ``total > 0`` and
    ``[]`` for ``total == 0``. Disjoint + complete is mandatory: SEMANTIC_RESUME's check-then-act
    would double-process a row shared across two workers."""
    if total <= 0 or workers <= 0:
        return []
    workers = min(workers, total)
    base, extra = divmod(total, workers)
    shards: list[str] = []
    start = 0
    for i in range(workers):
        size = base + (1 if i < extra else 0)
        end = start + size
        shards.append(f"{start}:{end}")
        start = end
    return shards


def parse_range(row_range: str) -> tuple[int, int]:
    """Parse ``"start:end"`` -> ``(start, end)``. Mirrors run_scale_repair's ROW_RANGE format."""
    start_s, end_s = str(row_range).split(":")
    return int(start_s), int(end_s)


def worker_env(
    base: dict,
    *,
    worker_idx: int,
    outdir: str,
    server_url: str,
    row_range: str,
    model_name: str = "repair-model",
) -> dict:
    """Overlay a worker's env on ``base`` (never mutating ``base``): its shard, its own OUTDIR
    and per-worker learning-corpus filenames, the shared server URL, resume on, and the
    in-process engine OFF (a worker must never load the 30B — it routes to the shared server)."""
    env = dict(base)
    env["ROW_RANGE"] = row_range
    env["OUTDIR"] = outdir
    env["RESULT"] = f"{outdir}/scale_result.json"
    env["SEMANTIC_VLLM_SERVER_URL"] = server_url
    env["SEMANTIC_VLLM_SERVER_MODEL"] = model_name
    env["SEMANTIC_RESUME"] = "1"
    env["VLLM"] = "0"  # in-process engine off; the shared server holds the only model copy
    env["ACCEPTED_CODE_OBJECT_FILE"] = f"semantic_repair_accepted_code_objects_shard{worker_idx}.jsonl"
    env["ACCEPTED_CODE_OBJECT_TELEMETRY_FILE"] = f"semantic_repair_accepted_case_telemetry_shard{worker_idx}.jsonl"
    return env


def _main(argv: list[str]) -> int:
    # CLI for the orchestrator: `python -m tools.scale_shard shards <total> <workers>`
    if len(argv) >= 4 and argv[1] == "shards":
        for s in compute_shards(int(argv[2]), int(argv[3])):
            print(s)
        return 0
    print("usage: python -m tools.scale_shard shards <total> <workers>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
