"""Scale-run launcher for semantic repair over a large dataset shard (e.g. 3000 files),
tuned for THROUGHPUT WITHOUT ANY QUALITY DROP.

It enables only the quality-neutral speed levers and keeps every quality knob at its full
value, so file-perfect rate is unchanged versus a normal run:

  * Lever A — persistent per-version compile worker (SEMANTIC_COMPILE_WORKER=1): byte-identical
    .pyc output, but the uvx/interpreter startup is paid once per version instead of once per
    candidate compile. Verified safe (tests/test_compile_worker*.py).
  * Lever B — vLLM offline backend (SEMANTIC_INFERENCE_BACKEND=vllm): OPT-IN, left OFF here
    until the GPU smoke-test passes. Set VLLM=1 to turn it on once validated. Falls back to HF
    automatically if vLLM is unavailable.

  Quality knobs held constant (NOT sped-up): CAND_COUNT=5, MAX_ITER=5, full soft/hard timeout
  budget, deterministic prepass FIRST, oracle-gated acceptance (never weakened), det-first
  ordering (process_easy_cases_first).

LOCAL Qwen only (Alibaba provider, runtime-registered merged model) — no external API.

Env knobs: MODEL_PATH, DATASET, SOURCE, ROW_RANGE (e.g. 0:3000), LIMIT, OUTDIR, RESULT,
MAX_ITER, CAND_COUNT, SOFT_TIMEOUT, HARD_TIMEOUT, VLLM (0/1), VLLM_GPU_MEM_FRACTION.
GPU job — run only when the GPU is free (ONE GPU job at a time).
"""
import os
import sys
import json
import time
import glob
import csv
from collections import defaultdict
from pathlib import Path

R = Path("/home/mxs220189/pylingual_collaboration/pylingual_download/code")
sys.path.insert(0, str(R / "pylingual"))
sys.path.insert(0, str(R))

# --- Quality-neutral speed levers ---------------------------------------------------------
os.environ["SEMANTIC_COMPILE_WORKER"] = os.getenv("SEMANTIC_COMPILE_WORKER", "1")  # Lever A (safe)
if os.getenv("VLLM", "0").strip().lower() in ("1", "true", "yes", "on"):
    os.environ["SEMANTIC_LLM_ENGINE"] = "vllm"  # Lever B (opt-in, HF fallback). Distinct from
    # loader's SEMANTIC_INFERENCE_BACKEND (unsloth|transformers) — do NOT set that one here.
    os.environ.setdefault("SEMANTIC_VLLM_GPU_MEM_FRACTION", os.getenv("VLLM_GPU_MEM_FRACTION", "0.6"))

# --- Quality knobs at full value (unchanged) ----------------------------------------------
os.environ["SEMANTIC_DETERMINISTIC_OPERATORS"] = "true"   # deterministic prepass FIRST (always)
os.environ["SEMANTIC_ACCEPTANCE_MODE"] = "oracle"          # oracle-gated, never weakened
os.environ["SEMANTIC_REPAIR_CANDIDATE_COUNT"] = os.getenv("CAND_COUNT", "5")

MODEL_NAME = "qwen3-coder-30b-scale"
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(R / "finetuning/merged_models/unsloth__qwen3-coder-30b-a3b-instruct/run_flywheel1"),
)

from utils import providers  # noqa: E402

if not Path(MODEL_PATH).exists():
    print(f"!! MODEL_PATH does not exist: {MODEL_PATH}", flush=True)
    sys.exit(2)

providers.OPEN_LLM_MODELS.append({
    "provider": "Alibaba",
    "name": MODEL_NAME,
    "token_for_completion": 16384,
    "model_path": MODEL_PATH,
    "tokenizer_path": MODEL_PATH,
    "generation_config": {"max_new_tokens": 2048, "do_sample": False, "temperature": 0.0, "top_p": 1.0},
})

from pipeline.code_object_repair_loop import run_dataset_repair_loop  # noqa: E402

DATASET = Path(os.getenv("DATASET", str(R / "dataset" / "pyllmpatch_semantic_hybrid100.csv")))
SOURCE = os.getenv("SOURCE") or None
ROW_RANGE = os.getenv("ROW_RANGE") or None
LIMIT = int(os.getenv("LIMIT", "0")) or None
OUTDIR = Path(os.getenv("OUTDIR", str(R / "results" / "scale_run")))
RESULT = Path(os.getenv("RESULT", str(OUTDIR / "scale_result.json")))
MAX_ITER = int(os.getenv("MAX_ITER", "5"))
SOFT = int(os.getenv("SOFT_TIMEOUT", "1500"))
HARD = int(os.getenv("HARD_TIMEOUT", "1800"))
OUTDIR.mkdir(parents=True, exist_ok=True)


def aggregate():
    ver = {}
    try:
        ver = {r["file_hash"]: r.get("bytecode_version", "?") for r in csv.DictReader(open(DATASET))}
    except Exception:
        pass
    pv = defaultdict(lambda: {"n": 0, "perfect": 0, "improved": 0})
    seen = set()
    for p in glob.glob(str(OUTDIR / "**" / "result.json"), recursive=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        fh = (d.get("gt_pyc") or "").split("/")[-2]
        if fh in seen:
            continue
        seen.add(fh)
        v = next((vv for hh, vv in ver.items() if hh.startswith(fh[:16]) or fh.startswith(hh[:16])), "?")
        ib = (d.get("initial_summary") or {}).get("combined_distance")
        fb = (d.get("final_summary") or {}).get("combined_distance")
        ae = (d.get("pylingual_verification") or {}).get("all_equal")
        x = pv[v]
        x["n"] += 1
        if isinstance(ib, int) and isinstance(fb, int) and fb < ib:
            x["improved"] += 1
        if ae is True:
            x["perfect"] += 1
    tot = {k: sum(x[k] for x in pv.values()) for k in ("n", "perfect", "improved")}
    out = {
        "config": {"model_path": MODEL_PATH, "dataset": DATASET.name, "source": SOURCE,
                   "row_range": ROW_RANGE, "max_iter": MAX_ITER, "cand": os.environ["SEMANTIC_REPAIR_CANDIDATE_COUNT"],
                   "soft": SOFT, "hard": HARD, "compile_worker": os.environ.get("SEMANTIC_COMPILE_WORKER"),
                   "inference_backend": os.environ.get("SEMANTIC_INFERENCE_BACKEND", "hf")},
        "totals": tot,
        "file_perfect_rate": round(tot["perfect"] / max(tot["n"], 1), 4),
        "per_version": {k: dict(v) for k, v in sorted(pv.items())},
    }
    RESULT.write_text(json.dumps(out, indent=2))
    return out


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] SCALE RUN model={MODEL_PATH}\n"
          f"  dataset={DATASET.name} source={SOURCE} row_range={ROW_RANGE} limit={LIMIT}\n"
          f"  levers: compile_worker={os.environ.get('SEMANTIC_COMPILE_WORKER')} "
          f"inference={os.environ.get('SEMANTIC_INFERENCE_BACKEND', 'hf')} "
          f"cand={os.environ['SEMANTIC_REPAIR_CANDIDATE_COUNT']} max_iter={MAX_ITER} soft={SOFT} hard={HARD}",
          flush=True)
    try:
        run_dataset_repair_loop(
            fixer_name="llm", dataset_path=DATASET, output_dir=OUTDIR,
            limit=LIMIT, source=SOURCE, row_range=ROW_RANGE,
            llm_provider="Alibaba", llm_model=MODEL_NAME, max_iterations=MAX_ITER,
            sample_timeout_seconds=SOFT, sample_hard_timeout_seconds=HARD,
            process_easy_cases_first=True,
        )
    finally:
        out = aggregate()
        print(f"[{time.strftime('%H:%M:%S')}] DONE {time.time()-t0:.0f}s", flush=True)
        print(f"FILE-PERFECT: {out['totals']['perfect']}/{out['totals']['n']} "
              f"= {out['file_perfect_rate']*100:.1f}%", flush=True)
        print(json.dumps(out["per_version"], indent=2), flush=True)


if __name__ == "__main__":
    main()
