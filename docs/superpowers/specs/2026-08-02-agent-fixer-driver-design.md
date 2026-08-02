# Single-agent fixer driver (Claude-as-fixer, reused agent)

**Date:** 2026-08-02
**Status:** approved (design)
**Scope:** new `campaign_ops/agent_fixer_driver.py`; one config change in `tools/agent_fixer/claude_serve.py`

## Problem

The semantic-repair pipeline builds a prompt for every code object that is not byte-equal
to GT. Today those prompts are serviced either by the local Qwen model or by a Claude
workflow agent spawned **per file** — which burned the 200-subagent/session cap during the
earlier harvest campaign.

We want one agent, reused for every prompt in a whole campaign, answering purely from its
own knowledge (no tools, no internet), with two sequential attempts per code object.

## What already exists (verified in source — do NOT rebuild)

- `utils/providers.py:41` registers provider `ClaudeAgent`. Its call writes the pipeline's
  real prompt to `$CLAUDE_FIXER_HANDOFF_DIR/req_<id>.json` and **blocks** (poll
  `CLAUDE_FIXER_HANDOFF_POLL`, timeout `CLAUDE_FIXER_HANDOFF_TIMEOUT`) until
  `resp_<id>.json` appears.
- `tools/agent_fixer/claude_serve.py` provides `launch` / `next` / `respond` / `harvest`
  for ONE file, plus persistence to `results/experiment_outputs/claude_fixer_pipeline/`
  and an append-only `run_index.jsonl`.
- Reattachment, recompile, scoring and the PyLingual all-equal check are the pipeline's own.

So the loop "prompt -> fix -> reattach -> compile -> check" already works end to end. The
gap is (a) driving MANY files from one agent and (b) the 2-attempt policy.

## Design

### 1. `campaign_ops/agent_fixer_driver.py`

A campaign state machine over `claude_serve`'s primitives. Commands:

- `init <csv> <work_root>` — read the file list (union-leftover CSV: `file_hash`,
  `gt_pyc`, `derived_source`, `python_version`, `prev_final_distance`), sort ascending by
  `prev_final_distance`, write `campaign.json`.
- `pump <work_root>` — the single command the driving session calls in a loop. Emits JSON:
  - `{"action":"prompt","id":..,"system":..,"user":..,"file_hash":..}` — a pending request
  - `{"action":"advance","harvested":N,"file_hash":..}` — previous file DONE (harvested,
    indexed), next file launched; caller simply calls `pump` again
  - `{"action":"idle"}` — pipeline running, nothing pending yet (caller re-polls)
  - `{"action":"alldone","stats":{...}}` — campaign complete
- `reply <work_root> <id> <fragment_file>` — publish the response (delegates to
  `claude_serve.respond`), unblocking the pipeline.

`campaign.json` records, per file: `status` (pending/running/done/error), `attempts`,
`perfect`, `note`. Append-only progress so the campaign is **resume-safe**.

### 2. The 2-attempt policy is CONFIGURATION, not new code

`--max-iterations 2` is the per-target attempt cap, and the pipeline already feeds rejected
attempts into later prompts, so attempt 2 sees why attempt 1 was rejected.
`SEMANTIC_REPAIR_CANDIDATE_COUNT=1` makes each attempt a single clean fix rather than a
candidate sweep.

`claude_serve.launch` currently passes **no** `--max-iterations` (so it defaults to 1).
Change: accept a `max_iterations` argument and pass it through. Default stays 1 so existing
callers are unaffected.

### 3. Environment for every launched pipeline

```
CLAUDE_FIXER_HANDOFF_DIR=<work>/handoff
SEMANTIC_DETERMINISTIC_OPERATORS=true   # only the true residual reaches the agent
SEMANTIC_REPAIR_CANDIDATE_COUNT=1
CLAUDE_FIXER_HANDOFF_TIMEOUT=1800
```
CPU-only: no local model is loaded, so it runs safely beside a live GPU run.

### 4. The agent

One subagent spawned once, reused for every prompt via SendMessage. It receives the
pipeline's `system` and `user` verbatim and returns ONLY the replacement fragment text.

**Limitation (explicit):** no tool-less agent type is available; the agent is *instructed*
to use no tools and answer from its own knowledge. That is instruction, not enforcement.

## Testing (TDD, no LLM / no GPU / no pipeline)

The driver is a pure state machine over a fake handoff directory:
1. `pump` surfaces a pending `req_*.json` as an `action=prompt` payload.
2. `reply` writes `resp_<id>.json` atomically with the fragment content.
3. The same request is not surfaced twice after a reply.
4. `pump` returns `action=idle` when the pipeline is alive with nothing pending.
5. On DONE, `pump` harvests, marks the file done, and advances to the next file.
6. `alldone` when the file list is exhausted; stats reflect per-file outcomes.
7. `init` sorts by ascending `prev_final_distance`.
8. Resume: re-`init` over an existing `campaign.json` preserves completed files.
9. `claude_serve.launch` passes `--max-iterations` when given (default 1 unchanged).

## Non-goals

- Not changing the repair loop, prompts, scoring, or acceptance gates.
- Not replacing the local-Qwen path; this is an alternative fixer backend.
- Not automating the agent call itself — SendMessage belongs to the driving session, so
  the driver exposes prompts and the session relays them.

## Amendment (2026-08-02): durable persistence + module repair

Two requirements added after the first build:

### 1. Module repairs ARE attempted (confirmed, add a guard)
Single-case `semantic-repair` with `--llm-provider ClaudeAgent` DOES repair `<module>`
targets (the smoke that reached perfect was a `<module>` target). The dataset-mode
`SEMANTIC_SKIP_MODULE_BODY_REPAIR` skip does not apply here and MUST NOT be set by the
driver. Add a regression test asserting the driver never sets that flag.

### 2. Persist result + improved source + improved pyc under results/experiment_outputs
Today `claude_serve` runs each file in an ephemeral work dir and copies only
`result.json`+`prompts/` out — WITHOUT rewriting the paths inside `result.json`, so
`final_source`/`final_pyc` point at deleted job-tmp (verified: `exists=False`). That is why
95 claude_fixer files were unusable in the vLLM union.

Fix: the driver runs each case with its pipeline output dir **directly under**
`results/experiment_outputs/malware_agent_fixer/cases/<hash>/`, so `result.json`,
`prompts/` (the improved `final_source`), `__pycache__/` (the improved `final_pyc`) and
`derived.pyc` all persist with valid absolute paths — same durability as the union runs.
On file completion the driver: (a) backfills `final_pyc` by compiling `final_source` at the
file's version if the pyc is missing, and (b) appends a union-scan-compatible row to
`malware_agent_fixer/run_index.jsonl` (`file_hash, python_version, gt_pyc, final_source,
final_pyc, all_equal, final_combined_distance`). `init` gains an optional `results_root`
(default the above) so tests stay hermetic.

New tests: case dir resolves under results_root; harvest backfills a missing final_pyc;
index row paths exist on disk; driver does not set the module-skip flag.
