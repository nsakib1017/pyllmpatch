# Skip module-body-repair files (semantic dataset runner)

**Date:** 2026-08-01
**Status:** approved (design), pre-implementation
**Scope:** `pipeline/config.py`, `pipeline/code_object_repair_loop.py`, `tools/run_scale_repair.py`

## Problem

In the live malware semantic-repair run, some workers stall for tens of minutes on
files whose repair requires **module-body repair** (a `<module>` qualname target →
`repair_module_statement`). These big, high-distance files keep making tiny
incremental improvements, so the SOFT timeout auto-extends (it only stops on a
stall), and they rarely reach file-perfect. They starve the shard of throughput
(observed: shard4/shard6 with 0 completions while grinding module-body files).

## Goal

Before any repair time is spent on a file, if its preflight shows a `<module>`
repair target, **skip the whole file** and record it as *deferred* (tracked,
revisitable), then move the worker to the next file. Default behavior is
unchanged (flag off).

## Key facts (verified in source)

- The dataset loop already runs `_semantic_repair_preflight(gt_pyc, derived_pyc)`
  on every row when `process_easy_cases_first=True` (the live run has this on).
  It returns `repair_targets: list[str]` of qualnames
  (`code_object_repair_loop.py:2539`, `utils/reattach_source_code_object.py:1643`).
- A module-body-repair file is exactly one whose `repair_targets` contains the
  literal string `"<module>"` (operation mapping at
  `code_object_repair_loop.py:593`). Confirmed against a live `result.json`
  (`repair_targets == ["<module>"]`).
- A "deferred" path already exists for `defer_preflight_risky_samples`: it writes
  a row to `deferred_csv` via `_dataset_deferred_row(...)` and `continue`s
  (`code_object_repair_loop.py:2897`).

## Design

### 1. Config flag
`SEMANTIC_SKIP_MODULE_BODY_REPAIR` in `pipeline/config.py`, bool, default `False`.
Same env-parse shape as the other `SEMANTIC_*` flags.

### 2. Loop parameter
`run_dataset_repair_loop(..., skip_module_body_repair: bool = SEMANTIC_SKIP_MODULE_BODY_REPAIR)`
— mirrors how `preflight_max_repair_targets` defaults to its config constant.
Recorded into the run summary dict alongside the other policy flags.

### 3. Helper
```python
def _preflight_has_module_body_target(preflight: dict | None) -> bool:
    return "<module>" in (preflight or {}).get("repair_targets", [])
```
Small, pure, unit-testable in isolation.

### 4. Skip point (per-file loop)
Right after preflight is resolved (before the timeout setup / repair work, near
the existing `defer_preflight_risky_samples` block ~line 2897):

```
if skip_module_body_repair:
    if preflight is None:
        preflight = _semantic_repair_preflight(gt_pyc, derived_pyc)
    if _preflight_has_module_body_target(preflight):
        write deferred row (defer_stage="module_body_repair",
                            defer_reason="module_body repair target (<module>) present")
        log stage="deferred"
        deferred += 1 ; continue
```

Reuses `_dataset_deferred_row` + the deferred CSV/writer already in scope.
Whole-file skip: a file is skipped if `<module>` appears **at all**, even
alongside non-module targets (a file needs its module body fixed to reach
file-perfect, so repairing only leaf targets cannot make it perfect anyway).

### 5. CLI parity
Add `--skip-module-body-repair` to the argparse block (~line 3327) and thread it
into the `run_dataset_repair_loop(...)` call. The live run drives it via the env
flag, so this is for single-invocation parity.

### 6. Worker wiring
`tools/run_scale_repair.py`: because the loop parameter defaults to the config
constant, exporting `SEMANTIC_SKIP_MODULE_BODY_REPAIR=1` at launch is sufficient.
No signature change needed there beyond confirming it does not override the flag.

## Testing (TDD, mocks only — no LLM, matches suite convention)

Helper unit tests:
- `["<module>"]` → True; `["<module>","foo"]` → True; `["foo"]` → False;
  `None`/`{}` → False.

Loop behavior tests (mock `_semantic_repair_preflight` and the single-file
repair entry point; assert on the deferred CSV and that repair was/ wasn't called):
- flag on + `repair_targets=["<module>"]` → deferred row written, **repair loop
  not invoked**.
- flag on + `["<module>","foo"]` → deferred (whole-file skip).
- flag on + `["foo"]` → processed normally (repair invoked).
- flag **off** + `["<module>"]` → processed normally (no behavior change).

## Rollout to the live run

`run_scale_parallel.sh` hardcodes `REPO=<main checkout>` and `cd "$REPO"`, so the
restart uses the **main checkout's** code, not this worktree's. Applying to the
live run therefore requires the change to reach the main checkout (via the PR /
user action). Restart procedure once applied:

1. Stop orchestrator process group (pgid 923753); leave vLLM (pid 869666) up.
2. Relaunch with `REUSE_SERVER=1 SEMANTIC_RESUME=1 SEMANTIC_SKIP_MODULE_BODY_REPAIR=1`
   plus the same MAX_ITER/CAND/SOFT/HARD as the current launch.
3. Resume skips the done files, re-preflights, and defers module-body files —
   freeing the stalled workers.

## Non-goals

- No change to how module-body repair itself works when the flag is off.
- No change to easy-first ordering or the existing preflight-defer policy.
- Not repairing non-module targets inside a skipped file (whole-file skip by
  decision).
