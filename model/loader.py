# model/loader.py
from finetuning.model_finetuner_chat_templates import LOAD_IN_4BIT
import torch
from unsloth import FastLanguageModel

_MODEL = None
_TOKENIZER = None

def load_model_once(
    *,
    model_path: str,
    device_map: str = "auto",
    max_tokens: int
):
    """
    Load the merged model and tokenizer exactly once.
    """
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_tokens,
        dtype=None,
        load_in_4bit=True,
        device_map={"": "cuda:0"},
    )

    # model.eval()
    FastLanguageModel.for_inference(model)

    _MODEL = model
    _TOKENIZER = tokenizer

    return _MODEL, _TOKENIZER
