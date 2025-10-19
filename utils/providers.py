from openai import OpenAI
from google import genai
from google.genai import types
from typing import List, Optional

import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
LLM_MODELS = [
    {'provider': 'OpenAI', 'name': 'gpt-4o', 'token_for_completion': 16384},
    {'provider': 'DeepSeek', 'name': 'deepseek-chat', 'token_for_completion': 8192},
    {'provider': 'DeepSeek', 'name': 'deepseek-reasoner', 'token_for_completion': 8192},
    {'provider': 'Google', 'name': 'gemini-2.5-pro', 'token_for_completion': 65536},
    {'provider': 'Google', 'name': 'gemini-2.5-flash-lite', 'token_for_completion': 65536},
    {'provider': 'Google', 'name': 'gemini-2.5-flash', 'token_for_completion': 65536}
]

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") , base_url="https://api.openai.com/v1")
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")  
google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def make_llm_call(prompt: List[dict], model: str = "gpt-4o", provider: str = "OpenAI") -> Optional[str]:
    # Basic validation
    if not isinstance(prompt, list) or len(prompt) != 2:
        print("Prompt must be a 2-item list: [system, user].")
        return None
    if prompt[0].get("role") != "system" or prompt[1].get("role") != "user":
        print("Prompt[0] must be system and prompt[1] must be user.")
        return None

    system_text = prompt[0].get("content", "") or ""
    user_text   = prompt[1].get("content", "") or ""

    # --- OpenAI / DeepSeek ---
    if provider in ["OpenAI", "DeepSeek"]:
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

    elif provider == "Google":

        import asyncio
        from functools import partial

        try:
            # model_name = "gemini-flash-latest"

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
            # if usage:
            #     print(f"Prompt tokens: {getattr(usage, 'prompt_token_count', None)}")
            #     print(f"Candidates tokens: {getattr(usage, 'candidates_token_count', None)}")
            #     print(f"Total tokens: {getattr(usage, 'total_token_count', None)}")

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
            print(f"An error occurred during the Gemini API call: {e}")
            return None
    else:
        print(f"Unsupported provider: {provider}")
        return None
