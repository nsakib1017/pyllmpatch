# Final Test Plan — Winnable-Subset Semantic Repair Run

**Date:** 2026-07-13
**Status:** infrastructure in place + validated; awaiting user go for the full run.

## Objective

Run the semantic-repair pipeline over the **winnable subset** of the newly-downloaded PyLingual
dataset (3.11–3.15), measure the file-perfect decompilation rate per version, with a resumable,
supervised, live-monitored multi-day run whose outputs land in `results/` per convention.

## Scope: how the 17,662 file×version pairs became 2,624 targets

| Stage | Count | Note |
|---|---|---|
| All decompiler outputs (3.11–3.15) | 17,662 | `stats.json` |
| `Equal` (perfect decompile, no repair) | 8,903 | dropped |
| `Different` | 8,759 | |
| … "semantic" (control-flow/bytecode diff) | 6,805 | my loose filter |
| … of those, **compile cleanly** | 3,411 | 50% — the rest have syntax errors (syntactic-repair territory, excluded) |
| … of those, **winnable** (classifier: not pure decompiler-bound) | **2,624** | 78% of compilable; drops 745 `llm_only` |

**Winnable breakdown (2,624):** 644 fully-deterministic (no GPU) + 1,980 LLM-winnable.
**Per version:** 3.11: 118 · 3.12: 451 · 3.13: 874 · 3.14: 448 · 3.15: 733.

The excluded 745 `llm_only` are control-flow-only / missing-object — historically decompiler-bound
(near-zero fix rate); running them would burn GPU-days for almost no yield.

## Fixability analysis (answer to "how much can we fix")

From `classify_semantic_divergences` over the 3,369 analyzable compilable files:
- **19% fully deterministic** (operators alone, no LLM) — very high expected fix rate.
- **58% need the LLM** with mechanical help (det_plus_easier + mixed).
- **22% decompiler-bound** (`llm_only`) — excluded.

Object-level divergence nature (9,795 objects): control_flow_structural 5,166; literal_value 2,218;
arg_value 1,768; call_shape 1,493; name_ref 1,477; container_construction 1,452; missing_object 710.

## Pipeline

1. **Adapter** (`tools/build_dataset_adapter.py`): materializes `<pyllm_adapter>/<hash>_<ver>/`
   (symlinks to GT `.pyc` + decompiled `.py`, plus the **compiled** derived `.pyc`) so the
   pipeline's `fetch_pyllmpatch_repair_paths` resolves the download. Run CSV: `winnable_run.csv`.
2. **Repair** (parallel `run_scale_parallel.sh`): 1 persistent spec-decode vLLM server + N=8
   CPU workers over disjoint `ROW_RANGE` shards. Config **A+B+C**:
   - A = cross-file batching (server), B = tail-abort (`SEMANTIC_TAIL_DEADLINE=400`),
     C = n-gram spec-decode. Plus post-LLM-det, default-args operator, greedy, max_iter=5, cand=5.
   - Quality-neutral (oracle-gated, greedy); measured **2.92×** over sequential.
3. **Supervisor** (`tools/run_winnable_final.sh`): server persistence + worker restart-on-death +
   memory watchdog + resume, looping until every file has a result or progress stalls.

## Resumability

- `SEMANTIC_RESUME=1`: any file with a `result.json` is skipped on restart.
- Supervisor restarts crashed workers/server automatically (up to `MAX_RESTARTS`).
- Kill anytime → re-run the **same command** → resumes. Survives worker/server crashes and
  mem-watchdog kills. A full **box reboot** kills the supervisor; just re-run the script (resumes).

## Results storage & live monitoring

`results/experiment_outputs/<YYYYMMDDTHHMMSSZ>/` (the repo timestamp convention; a fresh launch
mints the timestamp, resume reuses our own incomplete one via a `.winnable_final` marker):
- `shardN/semantic_repair/pylingual/<hash>_<ver>/result.json` — per-file detail (per shard).
- `live_progress.json` — **updated every 60 s**, safe to read anytime: overall + per-version +
  per-verdict done/perfect, file-perfect rate, throughput (files/hr), ETA (days).
- `summary.json` — durable final aggregate (same shape as prior experiment summaries).
- `supervisor.log`, `parallel.log`, `live.log`, `vllm_serve.log`.

## Time estimate

Winnable 2,624 at the measured A+B+C rate (validated in Phase 1). 644 fully-deterministic files
need no GPU (seconds each). Estimate refined by Phase 1 — target ~**2.5–4 days** at N=8.

## Launch / monitor / stop

```bash
# launch (backgrounded, survives the session):
nohup bash tools/run_winnable_final.sh > results/semantic_repair_winnable_final/nohup.log 2>&1 &

# monitor anytime:
cat results/semantic_repair_winnable_final/live_progress.json

# stop: kill the supervisor pid; re-run the same launch to resume.
```

## Risks & mitigations

- **Box reboot (unified mem spike):** kv-cache-memory-bytes skips the profiling spike;
  gpu-mem-util 0.5; watchdog kills a worker (never the server) on low mem; oom_score_adj. Re-run to resume.
- **Stuck files (never produce a result):** stall-counter stops after `MAX_STALLS` no-progress passes.
- **Batch-numeric quality flips:** greedy + oracle-gated accept/rollback bound it; net-neutral in A/B.
- **Server crash:** supervisor health-checks `/v1/models` and restarts the server.
