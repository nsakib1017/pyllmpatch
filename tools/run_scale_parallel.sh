#!/bin/bash
# Parallel scale run (speedup step 4): ONE shared vllm-serve holding the single FP8 model copy +
# N CPU-only worker processes over DISJOINT row shards, all pointed at the local server. vLLM
# continuous-batches the N independent files on the one GPU (the cross-file batching the batch-1
# sequential loop leaves on the table). Quality-neutral: each file runs the identical repair loop;
# only LLM transport (in-process -> localhost) and scheduling (interleaved) change.
#
# GB10 unified-memory safety: single model copy in the server (workers never load weights);
# server oom_score_adj=-800, workers=+600; MemAvailable watchdog SIGTERMs the highest-RSS worker
# (resumable via SEMANTIC_RESUME) on CRIT so the box never reboots. Resume-safe throughout.
#
# DRY_RUN=1 prints the serve command, shards, and each worker's launch WITHOUT touching the GPU.
set -u
REPO=/home/mxs220189/pylingual_collaboration/pylingual_download/code
VENV=/home/mxs220189/miniconda3/envs/pyllmpatch-vllm
PY="$VENV/bin/python"
cd "$REPO"
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

MODEL_PATH="${MODEL_PATH:-$REPO/finetuning/merged_models/unsloth__qwen3-coder-30b-a3b-instruct/run_flywheel1}"
DATASET="${DATASET:-$REPO/dataset/pyllmpatch_semantic_hybrid100.csv}"
N_WORKERS="${N_WORKERS:-8}"
TOTAL="${TOTAL:-98}"
PORT="${PORT:-8001}"
SERVED_MODEL="${SERVED_MODEL:-repair-model}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
KV_CACHE_GB="${KV_CACHE_GB:-20}"              # explicit KV size -> vLLM SKIPS the memory-profiling
                                              # dummy forward pass (the ~45GB MoE spike that would
                                              # reboot this unified box). Total = ~31GB weights + KV.
BASE_OUT="${BASE_OUT:-$REPO/results/parallel_run}"
SPEC_CONFIG="${SPEC_CONFIG:-}"                 # e.g. '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
FLOOR_CRIT_GIB="${FLOOR_CRIT_GIB:-12}"
DRY_RUN="${DRY_RUN:-0}"
# repair knobs passed to every worker (ship config: post-LLM-det ON, default-args on, corrective off)
MAX_ITER="${MAX_ITER:-5}"; CAND_COUNT="${CAND_COUNT:-5}"
SOFT_TIMEOUT="${SOFT_TIMEOUT:-900}"; HARD_TIMEOUT="${HARD_TIMEOUT:-1200}"
SEMANTIC_DO_SAMPLE="${SEMANTIC_DO_SAMPLE:-0}"
SEMANTIC_POST_LLM_DETERMINISTIC="${SEMANTIC_POST_LLM_DETERMINISTIC:-1}"
SEMANTIC_TAIL_DEADLINE="${SEMANTIC_TAIL_DEADLINE:-0}"

mkdir -p "$BASE_OUT"
LOG="$BASE_OUT/orchestrator.log"
BASE_URL="http://localhost:$PORT/v1"
log(){ echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }

SERVE_ARGS=(serve "$MODEL_PATH" --served-model-name "$SERVED_MODEL" --port "$PORT"
  --quantization fp8 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN" --kv-cache-memory-bytes "$((KV_CACHE_GB * 1024 * 1024 * 1024))"
  --dtype bfloat16 --seed 0 --trust-remote-code)
[ -n "$SPEC_CONFIG" ] && SERVE_ARGS+=(--speculative-config "$SPEC_CONFIG")

log "===== parallel scale run: N=$N_WORKERS total=$TOTAL dataset=$(basename "$DATASET") dry_run=$DRY_RUN ====="
mapfile -t SHARDS < <("$PY" -m tools.scale_shard shards "$TOTAL" "$N_WORKERS")
log "${#SHARDS[@]} shards: ${SHARDS[*]}"
log "serve cmd: vllm ${SERVE_ARGS[*]}"

WORKER_PIDS=()
launch_worker(){  # idx row_range -> backgrounds a worker AS A CHILD OF THE MAIN SHELL (so `wait`
                  # actually blocks) and appends its pid to the global WORKER_PIDS. MUST be called
                  # directly, never via $(...) — command substitution reparents the child and
                  # breaks `wait` (the whole run would tear down the instant the server is ready).
  local idx="$1" rr="$2" out="$BASE_OUT/shard$1"
  mkdir -p "$out"
  local -a WENV=(
    MODEL_PATH="$MODEL_PATH" DATASET="$DATASET" ROW_RANGE="$rr" OUTDIR="$out"
    RESULT="$out/scale_result.json" VLLM=0 SEMANTIC_RESUME=1
    SEMANTIC_VLLM_SERVER_URL="$BASE_URL" SEMANTIC_VLLM_SERVER_MODEL="$SERVED_MODEL"
    ACCEPTED_CODE_OBJECT_FILE="semantic_repair_accepted_code_objects_shard$idx.jsonl"
    ACCEPTED_CODE_OBJECT_TELEMETRY_FILE="semantic_repair_accepted_case_telemetry_shard$idx.jsonl"
    MAX_ITER="$MAX_ITER" CAND_COUNT="$CAND_COUNT" SOFT_TIMEOUT="$SOFT_TIMEOUT" HARD_TIMEOUT="$HARD_TIMEOUT"
    SEMANTIC_DO_SAMPLE="$SEMANTIC_DO_SAMPLE" SEMANTIC_POST_LLM_DETERMINISTIC="$SEMANTIC_POST_LLM_DETERMINISTIC"
    SEMANTIC_TAIL_DEADLINE="$SEMANTIC_TAIL_DEADLINE"
  )
  if [ "$DRY_RUN" = 1 ]; then
    log "[dry] worker $idx shard=$rr -> $out ; env: ${WENV[*]}"
    return 0
  fi
  env "${WENV[@]}" "$PY" tools/run_scale_repair.py >> "$out/run.log" 2>&1 &
  local wpid=$!
  echo 600 > "/proc/$wpid/oom_score_adj" 2>/dev/null || true
  WORKER_PIDS+=("$wpid")
  log "worker $idx pid=$wpid shard=$rr -> $out"
}

if [ "$DRY_RUN" = 1 ]; then
  for idx in "${!SHARDS[@]}"; do launch_worker "$idx" "${SHARDS[$idx]}"; done
  log "[dry] would health-check $BASE_URL/models then wait on ${#SHARDS[@]} workers, then aggregate."
  log "PARALLEL-DRYRUN-DONE"
  exit 0
fi

# --- 1. start the single vllm server (its own session so the EngineCore orphan dies with it) ---
# REUSE_SERVER=1 reuses an already-healthy server (decouples the ~7min load from cheap worker
# iteration); STOP_SERVER controls teardown (default: stop unless we reused one).
SERVER_LOG="$BASE_OUT/vllm_serve.log"
REUSE_SERVER="${REUSE_SERVER:-0}"
SERVER_PGID=""
if [ "$REUSE_SERVER" = 1 ] && curl -sf "$BASE_URL/models" >/dev/null 2>&1; then
  log "REUSE_SERVER: reusing healthy server at $BASE_URL (not starting/stopping one)"
  STOP_SERVER="${STOP_SERVER:-0}"
  SERVER_PID=$(pgrep -f "vllm serve" | head -1)
else
  STOP_SERVER="${STOP_SERVER:-1}"
  setsid "$VENV/bin/vllm" "${SERVE_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
  sleep 6
  SERVER_PID=$(pgrep -f "vllm serve $MODEL_PATH" | head -1)
  [ -z "$SERVER_PID" ] && SERVER_PID=$(pgrep -f "$SERVED_MODEL" | head -1)
  [ -z "$SERVER_PID" ] && { log "server launch FAILED (no pid) — see $SERVER_LOG"; exit 1; }
  echo -800 > "/proc/$SERVER_PID/oom_score_adj" 2>/dev/null || true
fi
[ -n "$SERVER_PID" ] && SERVER_PGID=$(ps -o pgid= -p "$SERVER_PID" | tr -d ' ')
log "server pid=$SERVER_PID pgid=$SERVER_PGID reuse=$REUSE_SERVER stop_on_exit=$STOP_SERVER"

# --- 2. health-check /v1/models (hard stop, NO HF fallback) ---
ready=0
LOAD_FLOOR_GIB="${LOAD_FLOOR_GIB:-8}"   # during load (no workers yet) a spike must kill the SERVER, not reboot the box
for _ in $(seq 1 120); do   # up to ~20 min for model load
  if curl -sf "$BASE_URL/models" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { log "SERVER DIED during load — see $SERVER_LOG"; exit 1; }
  avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
  if [ "${avail:-999}" -lt "$LOAD_FLOOR_GIB" ]; then
    log "MEM SPIKE during load (avail=${avail}G < ${LOAD_FLOOR_GIB}G) -> killing server to protect the box"
    kill -9 -"$SERVER_PGID" 2>/dev/null; exit 1
  fi
  sleep 10
done
[ "$ready" = 1 ] || { log "SERVER not ready — aborting (no HF fallback)"; kill -9 -"$SERVER_PGID" 2>/dev/null; exit 1; }
log "server READY at $BASE_URL"

# --- 3. launch N workers over disjoint shards (direct calls -> real children -> `wait` works) ---
for idx in "${!SHARDS[@]}"; do
  launch_worker "$idx" "${SHARDS[$idx]}"
done
log "launched ${#WORKER_PIDS[@]} workers: ${WORKER_PIDS[*]}"

# --- 4. memory watchdog: SIGTERM highest-RSS worker (never the server) on CRIT ---
(
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
    if [ "${avail:-999}" -lt "$FLOOR_CRIT_GIB" ]; then
      victim=$(for p in "${WORKER_PIDS[@]}"; do
                 kill -0 "$p" 2>/dev/null && echo "$(awk '/VmRSS/{print $2+0}' /proc/"$p"/status 2>/dev/null) $p"
               done | sort -rn | head -1 | awk '{print $2}')
      [ -n "$victim" ] && { log "WATCHDOG CRIT avail=${avail}G -> SIGTERM worker $victim (resumable)"; kill -TERM "$victim" 2>/dev/null; }
      sleep 15
    fi
    sleep 2
  done
) & WD_PID=$!

# --- 5. wait, (maybe) stop server, aggregate ---
for p in "${WORKER_PIDS[@]}"; do wait "$p" 2>/dev/null; done
kill "$WD_PID" 2>/dev/null || true
if [ "$STOP_SERVER" = 1 ] && [ -n "$SERVER_PGID" ]; then
  log "all workers done; stopping server pgid=$SERVER_PGID"
  kill -TERM -"$SERVER_PGID" 2>/dev/null; sleep 3; kill -9 -"$SERVER_PGID" 2>/dev/null || true
else
  log "all workers done; leaving server running (reuse) pgid=$SERVER_PGID"
fi
log "===== AGGREGATE ====="
"$PY" -m tools.aggregate_shards "$BASE_OUT" | tee -a "$LOG"
log "PARALLEL-ALL-DONE"
