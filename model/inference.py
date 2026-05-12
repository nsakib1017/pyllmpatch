from model.loader import load_model_once
import torch
from typing import Any

CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def _int_generation_value(config: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _float_generation_value(config: dict[str, Any], key: str) -> float | None:
    if key not in config or config.get(key) is None:
        return None
    try:
        return float(config[key])
    except (TypeError, ValueError):
        return None


def call_llm_with_message(*, messages, model_path, max_tokens, tokenizer_path=None, generation_config=None) -> str:
    generation_config = dict(generation_config or {})
    model, tokenizer = load_model_once(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens,
    )

    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = CHAT_TEMPLATE

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.use_cache = False
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_max_new_tokens = min(
        int(max_tokens),
        _int_generation_value(generation_config, "max_new_tokens", min(int(max_tokens), 2048)),
    )
    do_sample = bool(generation_config.get("do_sample", False))
    generate_kwargs: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "max_new_tokens": gen_max_new_tokens,
        "do_sample": do_sample,
        "use_cache": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        temperature = _float_generation_value(generation_config, "temperature")
        top_p = _float_generation_value(generation_config, "top_p")
        top_k = generation_config.get("top_k")
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if top_k is not None:
            generate_kwargs["top_k"] = _int_generation_value(generation_config, "top_k", 50)
    repetition_penalty = _float_generation_value(generation_config, "repetition_penalty")
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
