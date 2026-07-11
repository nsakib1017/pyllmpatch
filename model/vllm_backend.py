"""Feature-flagged vLLM inference backend for semantic repair.

Enabled with ``SEMANTIC_INFERENCE_BACKEND=vllm``. It serves the SAME local merged Qwen model
via vLLM's OFFLINE, in-process ``LLM`` engine — no HTTP server, no external API, nothing leaves
the box — so it is compliant with the local-Qwen-only rule and quality-neutral (greedy decoding
maps to temperature 0.0, matching the HF ``do_sample=False`` path). vLLM's paged-attention +
optimised kernels make the dominant generation step several times faster with the same tokens.

Design constraints on this box:
  * vLLM is an optional dependency. If it is not installed, or the flag is off, callers fall back
    to the existing HF path. ``is_vllm_available`` / ``VllmUnavailable`` gate that.
  * The GB10 has UNIFIED memory (GPU shares system RAM), so ``gpu_memory_utilization`` MUST be
    capped (``SEMANTIC_VLLM_GPU_MEM_FRACTION``, default conservative) or a load spike can wedge
    the box — the same failure mode as training. The engine is loaded once and cached.

Only ``select_inference_backend`` and ``build_sampling_params`` are exercised by unit tests; the
engine load + generate require a GPU and are covered by the deferred end-to-end smoke test.
"""
from __future__ import annotations

import os
from typing import Any


class VllmUnavailable(RuntimeError):
    """vLLM is not installed or the engine could not be constructed; fall back to HF."""


def select_inference_backend() -> str:
    """Return the active inference *engine*: ``"vllm"`` only when explicitly selected via
    ``SEMANTIC_LLM_ENGINE=vllm``, otherwise ``"hf"`` (the safe default, incl. unknown values).

    This is deliberately a DIFFERENT env var from ``SEMANTIC_INFERENCE_BACKEND`` (which
    ``model/loader.py`` uses to pick unsloth vs plain-transformers for the HF path) — the
    engine choice is orthogonal to how the HF model is loaded."""
    value = os.getenv("SEMANTIC_LLM_ENGINE", "hf").strip().lower()
    return "vllm" if value == "vllm" else "hf"


def _int_or(config: dict, key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _float_or_none(config: dict, key: str) -> float | None:
    if config.get(key) is None:
        return None
    try:
        return float(config[key])
    except (TypeError, ValueError):
        return None


def build_sampling_params(generation_config: dict[str, Any] | None, max_tokens: int) -> dict:
    """Map an HF-style ``generation_config`` to vLLM ``SamplingParams`` kwargs, mirroring
    ``model.inference.call_llm_with_message`` so decoding is equivalent:

      * new-token budget is ``min(max_tokens, config.max_new_tokens or min(max_tokens, 2048))``;
      * ``do_sample`` False (the default) -> ``temperature=0.0`` (greedy argmax, == HF greedy);
      * sampling passes temperature/top_p/top_k/repetition_penalty through, with an absent
        ``top_k`` disabled (vLLM sentinel ``-1``).
    """
    config = dict(generation_config or {})
    max_tokens = int(max_tokens)
    new_tokens = min(max_tokens, _int_or(config, "max_new_tokens", min(max_tokens, 2048)))

    params: dict[str, Any] = {"max_tokens": new_tokens}
    do_sample = bool(config.get("do_sample", False))
    if not do_sample:
        params["temperature"] = 0.0
        return params

    temperature = _float_or_none(config, "temperature")
    params["temperature"] = temperature if temperature is not None else 1.0
    top_p = _float_or_none(config, "top_p")
    if top_p is not None:
        params["top_p"] = top_p
    top_k = config.get("top_k")
    params["top_k"] = _int_or(config, "top_k", -1) if top_k is not None else -1
    repetition_penalty = _float_or_none(config, "repetition_penalty")
    if repetition_penalty is not None:
        params["repetition_penalty"] = repetition_penalty
    return params


def is_vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
    except Exception:
        return False
    return True


# --- Engine load + generate (GPU; import-guarded; covered by the deferred smoke test) ------
_ENGINE = None
_ENGINE_KEY = None


def _gpu_memory_fraction() -> float:
    try:
        return float(os.getenv("SEMANTIC_VLLM_GPU_MEM_FRACTION", "0.6"))
    except (TypeError, ValueError):
        return 0.6


def _int_env(name: str, default):
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def load_vllm_once(model_path: str, max_model_len: int):
    """Construct (once) and cache the offline vLLM engine for ``model_path``. Raises
    ``VllmUnavailable`` if vLLM is missing or the engine cannot be built (so callers fall back
    to HF). gpu_memory_utilization is capped for the unified-memory box."""
    global _ENGINE, _ENGINE_KEY
    key = (str(model_path), int(max_model_len))
    if _ENGINE is not None and _ENGINE_KEY == key:
        return _ENGINE
    if not is_vllm_available():
        raise VllmUnavailable("vllm is not installed")
    try:
        from vllm import LLM

        # On the unified-memory GB10 the 61GB MoE leaves little headroom; bound the KV-cache
        # profiling forward-pass (max_num_batched_tokens) and concurrent sequences (max_num_seqs)
        # so the transient profiling peak can't exhaust system RAM. All env-tunable.
        kwargs = dict(
            model=str(model_path),
            dtype="bfloat16",
            gpu_memory_utilization=_gpu_memory_fraction(),
            max_model_len=int(max_model_len),
            max_num_seqs=_int_env("SEMANTIC_VLLM_MAX_NUM_SEQS", 8),
            enforce_eager=os.getenv("SEMANTIC_VLLM_ENFORCE_EAGER", "false").strip().lower() in ("1", "true", "yes", "on"),
            trust_remote_code=True,
        )
        # On this 121GB UNIFIED box the 57GB bf16 weights + ~45GB MoE forward working set do
        # not fit. FP8 halves the weights (~28GB) so the model + KV cache fit with headroom.
        # FP8 is near-lossless but NOT bit-identical -> callers MUST gate it behind a file-perfect
        # parity check vs the HF path. Off unless SEMANTIC_VLLM_QUANTIZATION is set (e.g. "fp8").
        _quant = os.getenv("SEMANTIC_VLLM_QUANTIZATION", "").strip()
        if _quant:
            kwargs["quantization"] = _quant
        _mnbt = _int_env("SEMANTIC_VLLM_MAX_NUM_BATCHED_TOKENS", None)
        if _mnbt is not None:
            kwargs["max_num_batched_tokens"] = _mnbt
        # Setting an explicit KV-cache size makes vLLM SKIP the memory-profiling dummy forward
        # pass, whose transient activation spike (~45GB for this MoE) can't fit alongside the
        # 57GB weights on the 121GB unified box. Bounds total memory deterministically.
        _kv_gb = _int_env("SEMANTIC_VLLM_KV_CACHE_GB", None)
        if _kv_gb is not None:
            kwargs["kv_cache_memory_bytes"] = _kv_gb * (1024 ** 3)
        engine = LLM(**kwargs)
    except Exception as exc:  # pragma: no cover - requires GPU
        raise VllmUnavailable(f"could not construct vLLM engine: {exc}") from exc
    _ENGINE, _ENGINE_KEY = engine, key
    return engine


def vllm_generate(*, messages, model_path, max_tokens, tokenizer_path=None, generation_config=None) -> str:  # pragma: no cover - requires GPU
    """Offline in-process generation via vLLM, returning the assistant continuation text.
    Signature mirrors ``model.inference.call_llm_with_message`` for a drop-in swap."""
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    engine = load_vllm_once(model_path, max_model_len=int(max_tokens))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    sampling = SamplingParams(**build_sampling_params(generation_config, max_tokens))
    outputs = engine.generate([prompt], sampling)
    return outputs[0].outputs[0].text.strip()
