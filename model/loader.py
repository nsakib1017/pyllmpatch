# model/loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL = None
_TOKENIZER = None

def load_model_once(
    *,
    model_path: str,
    device_map: str = "auto",
):
    """
    Load the merged model and tokenizer exactly once.
    """
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,  # ✅ FIX
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer

    return _MODEL, _TOKENIZER
