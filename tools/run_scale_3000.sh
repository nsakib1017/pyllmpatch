#!/bin/bash
# Production supervisor for the multi-day 3000-file semantic-repair run on the crash-prone GB10
# unified-memory box. Wraps tools/run_scale_repair.py with:
#   * crash safety      — SEMANTIC_RESUME=1 (skips any file that already has a result.json), so a
#                         reboot/OOM/kill resumes instead of restarting from scratch.
#   * box crash guard   — a PGID mem-watchdog kills the run (not the box) if free RAM dips too low.
#   * auto-restart      — if the run dies without finishing, relaunch (resume) up to MAX_RESTARTS.
#   * FP8 vLLM + oracle + full quality knobs (parity-passed, no quality drop).
#
# Usage (in the isolated vLLM env, GPU free):
#   DATASET=dataset/your_3000.csv ROW_RANGE=0:3000 OUTDIR=results/run3000 \
#     bash tools/run_scale_3000.sh
#
# Env: DATASET, ROW_RANGE, OUTDIR, MODEL_PATH, MAX_ITER, CAND_COUNT, SOFT_TIMEOUT, HARD_TIMEOUT,
#      VLLM (default 1), FLOOR_GIB (watchdog, default 8), MAX_RESTARTS (default 20).
set -u
REPO=/home/mxs220189/pylingual_collaboration/pylingual_download/code
VENV=/home/mxs220189/miniconda3/envs/pyllmpatch-vllm
JT=/home/mxs220189/.claude/jobs/c42cbf21/tmp
cd "$REPO"

OUTDIR="${OUTDIR:-$REPO/results/run3000}"
RESULT="${RESULT:-$OUTDIR/scale_result.json}"
LOG="${LOG:-$OUTDIR/run3000.log}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
FLOOR_GIB="${FLOOR_GIB:-8}"
mkdir -p "$OUTDIR"

export VLLM="${VLLM:-1}"
export SEMANTIC_VLLM_QUANTIZATION="${SEMANTIC_VLLM_QUANTIZATION:-fp8}"
export VLLM_GPU_MEM_FRACTION="${VLLM_GPU_MEM_FRACTION:-0.6}"
export SEMANTIC_RESUME=1
export SEMANTIC_COMPILE_WORKER=1
export DATASET ROW_RANGE OUTDIR RESULT
export MAX_ITER="${MAX_ITER:-5}" CAND_COUNT="${CAND_COUNT:-5}"
export SOFT_TIMEOUT="${SOFT_TIMEOUT:-900}" HARD_TIMEOUT="${HARD_TIMEOUT:-1200}"
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

echo "$(date -u '+%F %T') supervisor START dataset=$DATASET row_range=${ROW_RANGE:-all} outdir=$OUTDIR" | tee -a "$LOG"

attempt=0
while [ "$attempt" -le "$MAX_RESTARTS" ]; do
  attempt=$((attempt+1))
  done_before=$(find "$OUTDIR" -name result.json 2>/dev/null | wc -l)
  echo "$(date -u '+%F %T') === attempt $attempt (resume; $done_before files already done) ===" | tee -a "$LOG"

  # launch the run as its own process group (leader = python) so the watchdog can kill the group
  setsid "$VENV/bin/python" tools/run_scale_repair.py >> "$LOG" 2>&1 &
  sleep 6
  LEADER=$(pgrep -f "run_scale_repair.py" | while read p; do [ "$(cat /proc/$p/comm 2>/dev/null)" = python ] && echo "$p" && break; done)
  if [ -z "$LEADER" ]; then echo "$(date -u '+%F %T') launch failed (no leader); abort" | tee -a "$LOG"; exit 3; fi
  PGID=$(ps -o pgid= -p "$LEADER" 2>/dev/null | tr -d ' ')
  echo 1000 > "/proc/$LEADER/oom_score_adj" 2>/dev/null || true

  # crash guard for this attempt's group
  PGID="$PGID" FLOOR_GIB="$FLOOR_GIB" LOG="$OUTDIR/watchdog.log" nohup bash "$JT/mem_watchdog_pgid.sh" >/dev/null 2>&1 &
  WD=$!

  # wait for the run to exit (or be killed)
  while kill -0 "$LEADER" 2>/dev/null; do sleep 30; done
  kill "$WD" 2>/dev/null || true

  # completion check: did the runner print its terminal marker?
  if grep -q "FILE-PERFECT:" "$LOG" 2>/dev/null; then
    echo "$(date -u '+%F %T') run finished cleanly." | tee -a "$LOG"
    grep "FILE-PERFECT:" "$LOG" | tail -1 | tee -a "$LOG"
    exit 0
  fi

  done_after=$(find "$OUTDIR" -name result.json 2>/dev/null | wc -l)
  echo "$(date -u '+%F %T') attempt $attempt ended without FILE-PERFECT marker (now $done_after done); restarting" | tee -a "$LOG"
  # if an attempt made zero progress twice in a row, something is stuck — back off
  if [ "$done_after" -le "$done_before" ]; then
    echo "$(date -u '+%F %T') no progress this attempt; sleeping 60s before retry" | tee -a "$LOG"
    sleep 60
  fi
done

echo "$(date -u '+%F %T') hit MAX_RESTARTS=$MAX_RESTARTS; giving up. $(find "$OUTDIR" -name result.json | wc -l) files done." | tee -a "$LOG"
exit 1
