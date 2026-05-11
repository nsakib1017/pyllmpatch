from __future__ import annotations

import os
import warnings

from dotenv import load_dotenv
from finetuning.model_finetuner_chat_templates import LOAD_IN_4BIT
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

_MODEL_CACHE: dict[tuple[str, str | None], tuple[object, object]] = {}

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

    max_seq_length = min(max(int(max_tokens) + 4096, 12288), 32768)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    auth_kwargs = {"token": hf_token} if hf_token else {}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT if "LOAD_IN_4BIT" in globals() else True,
        device_map=device_map,
        **auth_kwargs,
    )

    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, **auth_kwargs)

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
