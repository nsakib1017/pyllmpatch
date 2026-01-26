from openai import OpenAI
from google import genai
from google.genai import types
from typing import List, Optional
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

class Colors:
    """Class to hold ANSI color codes for styling terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

LLM_MODELS = [
    {'provider': 'OpenAI', 'name': 'gpt-4o', 'token_for_completion': 16384},
    {'provider': 'DeepSeek', 'name': 'deepseek-chat', 'token_for_completion': 8192},
    {'provider': 'DeepSeek', 'name': 'deepseek-reasoner', 'token_for_completion': 8192},
    {'provider': 'Google', 'name': 'gemini-2.5-pro', 'token_for_completion': 65536},
    {'provider': 'Google', 'name': 'gemini-2.5-flash-lite', 'token_for_completion': 65536},
    {'provider': 'Google', 'name': 'gemini-2.5-flash', 'token_for_completion': 65536}
]

OPEN_LLM_MODELS =  [
    {'provider': 'Alibaba', 'name': 'qwen2.5-coder-7b', 'token_for_completion': 16384, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/merged_model_qwen2.5_coder_7b_instruct"},
    {'provider': 'Alibaba', 'name': 'qwen2.5-coder-32b', 'token_for_completion': 32768, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/merged_model_qwen2.5_coder_32b_instruct_15_epochs"},
    {'provider': 'IBM', 'name': 'granite-4.0-1b', 'token_for_completion': 16384, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/granite-4.0-1b/run_1769129689/checkpoint-700"},
    {'provider': 'MistralAI', 'name': 'mistral-7b-instruct', 'token_for_completion': 8192, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/unsloth/mistral-7b-instruct-v0.3/run_1769076402"},
    {'provider': 'DeepSeek', 'name': 'DeepSeek-R1-0528-Qwen3-8B', 'token_for_completion': 32768, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/DeepSeek-R1-0528-Qwen3-8B/run_1769125567/checkpoint-300"},
    {'provider': 'Microsoft', 'name': 'phi-4', 'token_for_completion': 16384, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/Phi-4-unsloth-bnb-4bit/run_1769294194/checkpoint-300"},
    {'provider': 'OpenAI', 'name': 'gpt-oss-20b', 'token_for_completion': 32768, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/gpt-oss-20b/run_1769125385/checkpoint-800"},
    {'provider': 'Meta', 'name': 'llama-3.3-70b', 'token_for_completion': 32768, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/Llama-3.3-70B-Instruct-bnb-4bit/run_1769058870/checkpoint-200"},
    # {'provider': 'Google', 'name': 'gemma-3-12b-it-unsloth-bnb-4bit', 'token_for_completion': 32768, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/gemma-3-12b-it-unsloth-bnb-4bit/run_1769095208/checkpoint-200"}
]

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") , base_url="https://api.openai.com/v1")
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")  
google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def make_openai_call(prompt, model: str = "gpt-4o", provider: str = "OpenAI"):
        client = openai_client if provider == "OpenAI" else deepseek_client if provider == "DeepSeek" else None
        if not client:
            print("No client configured for the selected provider.")
            return None

        params = {"model": model, "stream": False, "messages": prompt}
        if provider == "DeepSeek":
            params["max_tokens"] = 8192

        try:
            completion = client.chat.completions.create(**params)
            usage = getattr(completion, "usage", None)
            if completion and getattr(completion, "choices", None):
                return {"content": getattr(completion.choices[0].message, "content", None), "usage": usage}
            return None
        except Exception as e:
            print(f"An error occurred during the LLM call: {e}")
            return None


async def make_gemini_call(system_text: str, user_text: str, model: str = "gemini-2.5-flash-lite"):
    try:
        # Put the system prompt as the first content item (SDKs differ on 'system' support).
        budget = -1
        contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=f"[SYSTEM]\n{system_text}")]),
                types.Content(role="user", parts=[types.Part.from_text(text=user_text)]),
            ]

        gen_config = types.GenerateContentConfig(thinking_config = types.ThinkingConfig(thinking_budget=budget,), )

            # Build kwargs once
        kwargs = {"model": model, "contents": contents}
            # Newer SDK uses 'config', older may want 'generate_content_config'
        use_new = hasattr(google_client, "models") and hasattr(google_client.models, "generate_content")
        if gen_config is not None:
            if use_new:
                kwargs["config"] = gen_config
            else:
                kwargs["generate_content_config"] = gen_config

        # Synchronous call wrappers
        def _sync_generate_new(_kwargs):
            return google_client.models.generate_content(**_kwargs)

        def _sync_generate_old(_kwargs):
            return google_client.generate_content(**_kwargs)

        # Run the blocking SDK call in a worker thread
        if use_new:
            response = await asyncio.to_thread(_sync_generate_new, kwargs)
        elif hasattr(google_client, "generate_content"):
            response = await asyncio.to_thread(_sync_generate_old, kwargs)
        else:
            print("Google client lacks 'models.generate_content' and 'generate_content'. Check SDK/version & init.")
            return None

        # Usage (if present)
        usage = getattr(response, "usage_metadata", None)
        
        # Extract text robustly
        def _extract_text(resp) -> Optional[str]:
            t = getattr(resp, "text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
            try:
                if getattr(resp, "candidates", None):
                    cand = resp.candidates[0]
                    parts = getattr(cand, "content", None)
                    parts = getattr(parts, "parts", []) if parts else []
                    out = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                    if out:
                        return "\n".join(out).strip()
            except Exception:
                pass
            return None

        return {"content": _extract_text(response), "usage": usage}

    except Exception as e:
        print(f"{Colors.FAIL}   -> An error occurred during the Gemini API call: {e}")
        return None

def make_llm_call(prompt: List[dict], model: str = "gpt-4o", provider: str = "OpenAI") -> Optional[str]:
    # Basic validation
    if not isinstance(prompt, list) or len(prompt) != 2:
        print("Prompt must be a 2-item list: [system, user].")
        return None
    if prompt[0].get("role") != "system" or prompt[1].get("role") != "user":
        print("Prompt[0] must be system and prompt[1] must be user.")
        return None

    system_text = prompt[0].get("content", "") or ""
    user_text   = prompt[1].get("content", "") or ""

    if provider in ["OpenAI", "DeepSeek"]:
        return make_openai_call(prompt, model=model, provider=provider)

    elif provider == "Google":
        return asyncio.run(make_gemini_call(system_text, user_text, model=model))
    else:
        print(f"Unsupported provider: {provider}")
        return None