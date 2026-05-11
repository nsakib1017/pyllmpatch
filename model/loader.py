from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from finetuning.model_finetuner_chat_templates import LOAD_IN_4BIT
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

_MODEL_CACHE: dict[tuple[str, str | None], tuple[object, object]] = {}
_MODEL_LOAD_FAILURES: dict[tuple[str, str | None], BaseException] = {}

load_dotenv()

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

    try:
        _validate_model_location(model_path, label="model_path")
        if tokenizer_path:
            _validate_model_location(tokenizer_path, label="tokenizer_path")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT if "LOAD_IN_4BIT" in globals() else True,
            device_map=device_map,
            **auth_kwargs,
        )

        if tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, **auth_kwargs)
    except Exception as exc:
        _MODEL_LOAD_FAILURES[cache_key] = exc
        raise

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)
    model.eval()

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.use_cache = False
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return _MODEL_CACHE[cache_key]
