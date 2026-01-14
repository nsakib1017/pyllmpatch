# model/inference.py
from model.loader import load_model_once

def call_llm_with_message(*, messages, model_path) -> str:
    """
    Run inference using the merged model.
    """

    model, tokenizer = load_model_once(model_path=model_path,)
    inputs = tokenizer.apply_chat_template( messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=8012)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
