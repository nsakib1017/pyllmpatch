# model/inference.py
from model.loader import load_model_once
import contextlib
import os
import sys

# SYSTEM_PROMPT = (
#     "You are an expert Python programmer and code repair specialist. "
#     "Fix ONLY Python syntax errors. "
#     "Make the minimal changes required. "
#     "Return ONLY the corrected code."
# )


def call_llm_with_message(*, messages, model_path) -> str:
    """
    Run inference using the merged model.
    """

    model, tokenizer = load_model_once(model_path=model_path,)

    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     {
    #         "role": "user",
    #         "content": (
    #             f"Error message:\n{error_message}\n\n"
    #             f"Buggy code:\n{buggy_code}"
    #         ),
    #     },
    # ]


    inputs = tokenizer.apply_chat_template( messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=8012)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def align_python_code(*, messages) -> str:
    """
    Run inference using the merged model.
    """

    model, tokenizer = load_model_once(model_path=MODEL_PATH,)

    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     {
    #         "role": "user",
    #         "content": (
    #             f"Error message:\n{error_message}\n\n"
    #             f"Buggy code:\n{buggy_code}"
    #         ),
    #     },
    # ]


    inputs = tokenizer.apply_chat_template( messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=8012)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
