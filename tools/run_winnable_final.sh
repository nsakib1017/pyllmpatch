#!/bin/bash
# FINAL RUN — supervised, resumable, live-monitored repair over the winnable subset.
#
# Architecture: ONE persistent spec-decode vLLM server + N sharded CPU workers (A+B+C), wrapped
# in a supervisor loop that resumes (SEMANTIC_RESUME skips files with a result.json) and restarts
# the workers/server on death, until every file has a result or progress stalls. A live aggregator
# writes results/<run>/live_progress.json every minute (safe to read anytime) + summary.json.
#
# Resumability: kill this at any time and re-run the SAME command — done files are skipped, the
# run continues. Survives worker crashes, server crashes, and mem-watchdog kills. (A full BOX
# REBOOT kills the supervisor; just re-run this script — it resumes automatically.)
#
# Results land under results/semantic_repair_winnable_final/ (per-hash result.json trees per shard
# + live_progress.json + summary.json), matching the prior-experiment convention.
set -u
REPO=/home/mxs220189/pylingual_collaboration/pylingual_download/code
VENV=/home/mxs220189/miniconda3/envs/pyllmpatch-vllm
PY="$VENV/bin/python"
cd "$REPO"
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

ADAPTER=/home/mxs220189/pylingual_collaboration/pyllm_adapter
DATASET="${DATASET:-$ADAPTER/winnable_run.csv}"
# Results land under results/experiment_outputs/<YYYYMMDDTHHMMSSZ>/ (the repo's timestamp
# convention). Resume needs a STABLE dir, so: explicit OUTDIR wins; else FRESH=1 mints a new
# timestamp; else auto-resume OUR newest incomplete run (tagged .winnable_final so we never touch
# unrelated experiment_outputs dirs), or mint a new one if none is resumable.
EXP_ROOT="$REPO/results/experiment_outputs"
if [ -z "${OUTDIR:-}" ]; then
  if [ "${FRESH:-0}" = 1 ]; then
    OUTDIR="$EXP_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
  else
    OUTDIR=""
    for d in $(ls -1dt "$EXP_ROOT"/*/ 2>/dev/null); do
      [ -f "${d}.winnable_final" ] || continue
      grep -q "FINAL-RUN-DONE" "${d}supervisor.log" 2>/dev/null && continue
      OUTDIR="${d%/}"; break
    done
    [ -z "$OUTDIR" ] && OUTDIR="$EXP_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
  fi
fi
mkdir -p "$OUTDIR"; touch "$OUTDIR/.winnable_final"
FIXJSON="${FIXJSON:-/home/mxs220189/.claude/jobs/c42cbf21/tmp/fixability_full.json}"
N_WORKERS="${N_WORKERS:-8}"
MAX_RESTARTS="${MAX_RESTARTS:-100}"
MAX_STALLS="${MAX_STALLS:-3}"        # consecutive no-progress passes before giving up on stuck files
LIVE_INTERVAL="${LIVE_INTERVAL:-60}"

# path resolution -> the adapter tree (overrides .env)
export ROOT_FOR_FILES=/home/mxs220189/pylingual_collaboration
export BASE_DIR_PYTHON_FILES_PYLINGUAL=pyllm_adapter
# A+B+C ship config
export SEMANTIC_POST_LLM_DETERMINISTIC=1
export SEMANTIC_TAIL_DEADLINE="${SEMANTIC_TAIL_DEADLINE:-400}"
export MAX_ITER="${MAX_ITER:-5}" CAND_COUNT="${CAND_COUNT:-5}"
export SOFT_TIMEOUT="${SOFT_TIMEOUT:-900}" HARD_TIMEOUT="${HARD_TIMEOUT:-1200}"
export SEMANTIC_DO_SAMPLE=0
SPEC_CONFIG="${SPEC_CONFIG:-{\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}}"

mkdir -p "$OUTDIR"
LOG="$OUTDIR/supervisor.log"
[ -f "$OUTDIR/t0.txt" ] || date +%s > "$OUTDIR/t0.txt"   # persist run start across restarts
T0=$(cat "$OUTDIR/t0.txt")
log(){ echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }

# --- self-documenting metadata: exact code version + full config (written once) ---
export SPEC_CONFIG N_WORKERS DATASET ADAPTER
if [ ! -f "$OUTDIR/run_config.json" ]; then
  GIT_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
  GIT_BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
  GIT_DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  git -C "$REPO" diff HEAD > "$OUTDIR/code_diff.patch" 2>/dev/null || true   # exact uncommitted state (we never commit)
  RUN_ID=$(basename "$OUTDIR") CREATED=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  "$PY" - <<PYEOF
import json, os
cfg = {
  "run_id": os.environ["RUN_ID"], "created_utc": os.environ["CREATED"],
  "description": "Semantic bytecode repair over the WINNABLE subset of the PyLingual 3.11-3.15 dataset (decompiled source recompiled, driven to match the GT .pyc).",
  "code_version": {"git_commit": "$GIT_COMMIT", "git_branch": "$GIT_BRANCH",
                   "uncommitted_files": int("$GIT_DIRTY"), "code_diff": "code_diff.patch"},
  "dataset": {"run_csv": os.environ["DATASET"], "total_files": int("$TOTAL"),
              "source": "/home/mxs220189/pylingual_collaboration/pyllm_final_dataset",
              "adapter": os.environ["ADAPTER"],
              "selection": "winnable = compilable semantic cases NOT pure decompiler-bound (classify_semantic_divergences keeps fully_det_fixable|det_plus_easier|mixed_needs_llm, drops llm_only). 6805 semantic -> 3411 compilable -> 2624 winnable."},
  "model": {"path": os.environ.get("MODEL_PATH", "finetuning/merged_models/unsloth__qwen3-coder-30b-a3b-instruct/run_flywheel1"),
            "served_as": "repair-model", "quantization": "fp8"},
  "inference": {"backend": "vllm serve (OpenAI-compatible, localhost)", "speculative_config": os.environ.get("SPEC_CONFIG"),
                "greedy": os.environ.get("SEMANTIC_DO_SAMPLE", "0") != "1", "n_workers": int(os.environ.get("N_WORKERS", "8"))},
  "repair_config": {"speed_levers": "A=cross-file batching, B=tail-abort, C=ngram spec-decode",
                    "max_iterations": int(os.environ.get("MAX_ITER", "5")), "candidate_count": int(os.environ.get("CAND_COUNT", "5")),
                    "soft_timeout_s": int(os.environ.get("SOFT_TIMEOUT", "900")), "hard_timeout_s": int(os.environ.get("HARD_TIMEOUT", "1200")),
                    "tail_deadline_s": int(os.environ.get("SEMANTIC_TAIL_DEADLINE", "0")),
                    "post_llm_deterministic": os.environ.get("SEMANTIC_POST_LLM_DETERMINISTIC") == "1", "deterministic_prepass": True,
                    "acceptance": "oracle-gated (whole-file combined-distance non-regression / pylingual-success; never weakened)"},
  "how_it_runs": {"launcher": "tools/run_winnable_final.sh",
                  "orchestrator": "tools/run_scale_parallel.sh (1 persistent vllm server + N sharded CPU workers, mem watchdog, oom_score_adj)",
                  "adapter_builder": "tools/build_dataset_adapter.py", "fixability_classifier": "tools/classify_semantic_divergences.py",
                  "live_aggregator": "tools/live_aggregate.py -> live_progress.json + summary.json",
                  "resumable": "SEMANTIC_RESUME=1 (skip files with result.json) + supervisor restart loop",
                  "plan_doc": "docs/superpowers/specs/2026-07-13-final-winnable-run-plan.md"},
}
json.dump(cfg, open(os.path.join("$OUTDIR", "run_config.json"), "w"), indent=2)
PYEOF
  log "wrote run_config.json + code_diff.patch (git=$GIT_COMMIT dirty=$GIT_DIRTY files)"
fi

TOTAL="${TOTAL:-$(( $(wc -l < "$DATASET") - 1 ))}"
log "===== FINAL RUN: winnable subset, $TOTAL files, N=$N_WORKERS -> $OUTDIR ====="

# live aggregator (idempotent; kill any stale one first)
rm -f "$OUTDIR/STOP_AGGREGATE"
pkill -f "live_aggregate.py --base $OUTDIR" 2>/dev/null
nohup "$PY" tools/live_aggregate.py --base "$OUTDIR" --csv "$DATASET" --fixability "$FIXJSON" \
  --interval "$LIVE_INTERVAL" --t0 "$T0" >> "$OUTDIR/live.log" 2>&1 &
LIVE_PID=$!
log "live aggregator pid=$LIVE_PID -> $OUTDIR/live_progress.json"

done_count(){ find "$OUTDIR" -name result.json 2>/dev/null | wc -l; }

restart=0; stall=0; prev=-1
while :; do
  REUSE_SERVER=1 STOP_SERVER=0 BASE_OUT="$OUTDIR" N_WORKERS="$N_WORKERS" TOTAL="$TOTAL" \
    DATASET="$DATASET" SPEC_CONFIG="$SPEC_CONFIG" \
    bash tools/run_scale_parallel.sh >> "$OUTDIR/parallel.log" 2>&1
  d=$(done_count)
  log "pass complete: $d/$TOTAL results (restart=$restart stall=$stall)"
  [ "$d" -ge "$TOTAL" ] && { log "ALL FILES HAVE RESULTS"; break; }
  if [ "$d" -le "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  prev="$d"
  [ "$stall" -ge "$MAX_STALLS" ] && { log "STALLED ($stall passes, no new results) -> stopping; $((TOTAL-d)) files stuck"; break; }
  restart=$((restart+1))
  [ "$restart" -gt "$MAX_RESTARTS" ] && { log "MAX_RESTARTS reached -> stopping"; break; }
  log "resuming in 30s (SEMANTIC_RESUME skips the $d done)"
  sleep 30
done

# finalize: last snapshot + stop aggregator + tear down server
"$PY" tools/live_aggregate.py --base "$OUTDIR" --csv "$DATASET" --fixability "$FIXJSON" --once --t0 "$T0" >> "$OUTDIR/live.log" 2>&1
touch "$OUTDIR/STOP_AGGREGATE"; sleep 2; kill "$LIVE_PID" 2>/dev/null || true
for p in $(pgrep -f "vllm serve" 2>/dev/null); do pgid=$(ps -o pgid= -p "$p" 2>/dev/null|tr -d ' '); [ -n "$pgid" ] && kill -9 -"$pgid" 2>/dev/null; done
log "===== FINAL RUN COMPLETE — $(python3 -c "import json;d=json.load(open('$OUTDIR/summary.json'));print(f\"file-perfect {d['file_perfect']}/{d['done']} ({100*d['file_perfect_rate_of_done']:.1f}% of done)\")" 2>/dev/null) ====="
log "FINAL-RUN-DONE"
