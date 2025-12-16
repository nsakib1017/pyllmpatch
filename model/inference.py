# model/inference.py
import torch
from model.loader import load_model_once

MODEL_PATH = "/home/diogenes/pylingual_colaboration/pylingual_download/code/finetuning/merged_model_qwen2.5_coder_7b_instruct"

# SYSTEM_PROMPT = (
#     "You are an expert Python programmer and code repair specialist. "
#     "Fix ONLY Python syntax errors. "
#     "Make the minimal changes required. "
#     "Return ONLY the corrected code."
# )

def fix_python_syntax(*, messages) -> str:
    """
    Run inference using the merged model.
    """

    model, tokenizer = load_model_once(
        model_path=MODEL_PATH,
    )

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


    inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=8012)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
