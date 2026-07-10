from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

_MODEL_CACHE: dict[tuple[str, str | None], tuple[object, object]] = {}
_MODEL_LOAD_FAILURES: dict[tuple[str, str | None], BaseException] = {}

load_dotenv()


def _env_flag(name: str, default: str = "true") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


LOAD_IN_4BIT = _env_flag("LOAD_IN_4BIT", "true")

warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
    module=r"transformers\.modeling_attn_mask_utils",
)
warnings.filterwarnings(
    "ignore",
    message=r"The tokenizer you are loading from '.*' with an incorrect regex pattern:.*",
    category=UserWarning,
)


def _looks_like_local_path(path_value: str) -> bool:
    return path_value.startswith(("/", "./", "../", "~")) or os.sep in path_value


def _validate_model_location(path_value: str, *, label: str) -> None:
    if not _looks_like_local_path(path_value):
        return
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}. Refusing to fall back to Hugging Face Hub for a local path."
        )
    if path.is_dir() and not any(
        (path / filename).exists()
        for filename in (
            "config.json",
            "adapter_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
        )
    ):
        raise FileNotFoundError(
            f"{label} exists but does not look like a model/tokenizer directory: {path}"
        )


def load_model_once(
    *,
    model_path: str,
    tokenizer_path: str | None = None,
    device_map: str = "auto",
    max_tokens: int,
):
    global _MODEL_CACHE

    cache_key = (model_path, tokenizer_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if cache_key in _MODEL_LOAD_FAILURES:
        previous = _MODEL_LOAD_FAILURES[cache_key]
        raise RuntimeError(
            f"Previous model load failed for model_path={model_path!r}, tokenizer_path={tokenizer_path!r}; "
            "not retrying in this process."
        ) from previous

    max_seq_length = min(max(int(max_tokens) + 4096, 12288), 32768)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    auth_kwargs = {"token": hf_token} if hf_token else {}

    # unsloth's for_inference patched attention crashes generation for some architectures
    # (e.g. Qwen2.5-Coder-32B) with a KV-cache broadcast shape error on transformers 5.3.
    # SEMANTIC_INFERENCE_BACKEND=transformers loads a full/merged model via plain transformers
    # + sdpa attention (verified fast + correct with use_cache=True). Default stays unsloth.
    _backend = os.getenv("SEMANTIC_INFERENCE_BACKEND", "unsloth").strip().lower()

    try:
        _validate_model_location(model_path, label="model_path")
        if tokenizer_path:
            _validate_model_location(tokenizer_path, label="tokenizer_path")

        if _backend == "transformers":
            import torch as _torch
            from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
            model = _AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=_torch.bfloat16,
                device_map={"": 0},
                attn_implementation="sdpa",
                **auth_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path or model_path, trust_remote_code=False, **auth_kwargs
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=LOAD_IN_4BIT if "LOAD_IN_4BIT" in globals() else True,
                device_map={"": 0},
                **auth_kwargs,
            )
            if tokenizer_path:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, **auth_kwargs)
    except Exception as exc:
        _MODEL_LOAD_FAILURES[cache_key] = exc
        raise

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if _backend != "transformers":
        FastLanguageModel.for_inference(model)
    model.eval()

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        # KV cache ON for inference: greedy decoding is identical with the cache
        # but O(n) instead of O(n^2). FastLanguageModel.for_inference(model) above
        # sets up cached fast inference — disabling the cache defeated it.
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return _MODEL_CACHE[cache_key]
