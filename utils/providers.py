from openai import OpenAI
from typing import Any, List, Optional
import asyncio
from dotenv import load_dotenv
import os

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

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
    {'provider': 'OpenAI', 'name': 'gpt-5.5-1', 'token_for_completion': 16384},
    {'provider': 'OpenAI', 'name': 'gpt-5.4-mini', 'token_for_completion': 16384},
    {'provider': 'DeepSeek', 'name': 'deepseek-chat', 'token_for_completion': 8192},
    {'provider': 'DeepSeek', 'name': 'deepseek-reasoner', 'token_for_completion': 8192},
    {'provider': 'Google', 'name': 'gemini-2.5-pro', 'token_for_completion': 65536},
    {'provider': 'Google', 'name': 'gemini-2.5-flash-lite', 'token_for_completion': 65536},
    {'provider': 'Google', 'name': 'gemini-2.5-flash', 'token_for_completion': 65536}
]

OPEN_LLM_MODELS =  [
    {
        'provider': 'Alibaba',
        'name': 'qwen2.5-coder-7b',
        'token_for_completion': 8192,
        'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/merged_model_qwen2.5_coder_7b_instruct",
        'tokenizer_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/tokenizer_snapshots/qwen2.5-coder-7b",
    },
    {
        'provider': 'Alibaba',
        'name': 'qwen2.5-coder-32b',
        'token_for_completion': 8192,
        'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth__Qwen2.5-Coder-32B-Instruct-bnb-4bit/run_1778168589/checkpoint-400",
        'tokenizer_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth__Qwen2.5-Coder-32B-Instruct-bnb-4bit/run_1778168589/checkpoint-400",
    },
    {
        'provider': 'Alibaba',
        'name': 'qwen3-coder-30b',
        'token_for_completion': 16384,
        'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/Qwen3-Coder-30B-A3B-Instruct_run_1778647681/run_1778696366",
        'tokenizer_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/Qwen3-Coder-30B-A3B-Instruct_run_1778647681/run_1778696366",
        'generation_config': {
            # Repair fragments are small; a lower cap bounds runaway greedy decoding
            # (greedy still stops at EOS well before this) for faster inference.
            'max_new_tokens': 2048,
            'do_sample': False,
            'temperature': 0.0,
            'top_p': 1.0,
        },
    },
    {
        # Fine-tuned Qwen2.5-Coder-32B (LoRA adapter, --no-merge) on the corrected
        # completion-only + max_length dataset (finetune_corpus_sub.jsonl). Loader resolves
        # base=unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit from adapter_config.json.
        'provider': 'Alibaba',
        'name': 'qwen25-coder-32b-ft',
        # Merged 16-bit model (LoRA merged into base) — the adapter path crashes generation with
        # use_cache=True (KV-cache broadcast bug on transformers 5.3); the merged model runs fast
        # with use_cache=True like any full model. token_for_completion=8192 + eval headroom 2048
        # keeps the prompt gate at 6144 so only in-distribution (<=8192) samples are processed.
        'token_for_completion': 8192,
        'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/qwen25-coder-32b-ft-merged",
        'tokenizer_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/qwen25-coder-32b-ft-merged",
        'generation_config': {
            'max_new_tokens': 2048,
            'do_sample': False,
            'temperature': 0.0,
            'top_p': 1.0,
        },
    },
    {
        'provider': 'IBM',
        'name': 'granite-4.0-1b',
        'token_for_completion': 8192,
        'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/unsloth/granite-4.0-1b/run_1769129689",
        'tokenizer_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/tokenizer_snapshots/granite-4.0-1b",
    },
    {
        'provider': 'MistralAI',
        'name': 'mistral-7b-instruct',
        'token_for_completion': 8192,
        'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/merged_models/unsloth/mistral-7b-instruct-v0.3/run_1769076402",
        'tokenizer_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/tokenizer_snapshots/mistral-7b-instruct",
    },
    {
        'provider': 'DeepSeek',
        'name': 'DeepSeek-R1-0528-Qwen3-8B',
        'token_for_completion': 8192,
        'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/unsloth/DeepSeek-R1-0528-Qwen3-8B/run_1769125567/checkpoint-300",
        'tokenizer_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/tokenizer_snapshots/deepseek-r1-0528-qwen3-8b",
    },
    {
        'provider': 'Microsoft',
        'name': 'phi-4',
        'token_for_completion': 8192,
        'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/unsloth/phi-4-reasoning/run_1769142686/checkpoint-900",
        'tokenizer_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/tokenizer_snapshots/phi-4",
    },
    # {'provider': 'OpenAI', 'name': 'gpt-oss-20b', 'token_for_completion': 32768, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/gpt-oss-20b/run_1769125385/checkpoint-800"},
    # {'provider': 'Meta', 'name': 'codellama-34b', 'token_for_completion': 16384, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/codellama-34b-bnb-4bit/run_1769663638/checkpoint-200"},
    # {'provider': 'Google', 'name': 'gemma-3-12b-it-unsloth-bnb-4bit', 'token_for_completion': 32768, 'model_path': f"{os.getenv('PROJECT_ROOT_DIR')}/finetuning/finetuned_models/unsloth/gemma-3-12b-it-unsloth-bnb-4bit/run_1769095208/checkpoint-200"}
]

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") , base_url="https://ai-for-cybersecurity-east2-resou.services.ai.azure.com/openai/v1")
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")  
google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if genai is not None else None

def make_openai_call(prompt, model: str = "gpt-5.5-1", provider: str = "OpenAI"):
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


# --- Local vLLM server seam (speedup step 3) -------------------------------------------------
# When SEMANTIC_VLLM_SERVER_URL is set, a local-Qwen config routes generation to a LOCAL
# ``vllm serve`` OpenAI-compatible endpoint on localhost instead of building the in-process
# engine. This lets N stateless worker processes share ONE server so vLLM continuous-batches
# independent files on the one GPU. It is transport-only: localhost, same merged weights + same
# tokenizer/chat-template as the offline path, greedy -> temperature 0.0, so decoding matches and
# the local-Qwen-only rule holds (nothing leaves the box). Unset env == the old in-process path.
_LOCAL_VLLM_CLIENTS: dict[str, Any] = {}


def _local_vllm_client(base_url: str):
    """Build (once, cached per base_url) an OpenAI client pointed at a LOCAL vllm-serve endpoint.
    Distinct from the Azure/DeepSeek clients; the api_key is a placeholder vLLM ignores."""
    client = _LOCAL_VLLM_CLIENTS.get(base_url)
    if client is None:
        client = OpenAI(api_key=os.getenv("SEMANTIC_VLLM_SERVER_API_KEY", "EMPTY"), base_url=base_url)
        _LOCAL_VLLM_CLIENTS[base_url] = client
    return client


def call_local_vllm_server(
    messages: List[dict],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    generation_config: dict | None = None,
) -> Optional[dict]:
    """OpenAI-compatible /v1/chat/completions call to a LOCAL vllm-serve endpoint. The server
    applies the same model dir's chat template, so with greedy (temperature 0.0) the output
    matches the offline ``vllm_generate`` path up to vLLM's batch-invariance."""
    from model.vllm_backend import build_sampling_params

    sampling = build_sampling_params(generation_config, max_tokens)
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": sampling["max_tokens"],
        "temperature": sampling.get("temperature", 0.0),
    }
    extra_body: dict[str, Any] = {}
    if "top_p" in sampling:
        params["top_p"] = sampling["top_p"]
    if sampling.get("top_k", -1) != -1:
        extra_body["top_k"] = sampling["top_k"]
    if "repetition_penalty" in sampling:
        extra_body["repetition_penalty"] = sampling["repetition_penalty"]
    if extra_body:
        params["extra_body"] = extra_body

    client = _local_vllm_client(base_url)
    try:
        completion = client.chat.completions.create(**params)
    except Exception as e:  # noqa: BLE001 - surface, let caller record a failed attempt
        print(f"An error occurred during the local vLLM server call: {e}")
        return None
    usage = getattr(completion, "usage", None)
    if completion and getattr(completion, "choices", None):
        content = getattr(completion.choices[0].message, "content", None)
        return {"content": content.strip() if isinstance(content, str) else content, "usage": usage}
    return None


async def make_gemini_call(system_text: str, user_text: str, model: str = "gemini-2.5-flash-lite"):
    try:
        if google_client is None or types is None:
            print("Google GenAI SDK is not available. Install google-genai to use Gemini.")
            return None
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


def find_llm_config(provider: str, model: str | None = None) -> dict[str, Any]:
    models = LLM_MODELS + OPEN_LLM_MODELS
    for candidate in models:
        if candidate["provider"] == provider and (model is None or candidate["name"] == model):
            return candidate
    model_names = ", ".join(f"{item['provider']}:{item['name']}" for item in models)
    requested = f"{provider}:{model}" if model else provider
    raise ValueError(f"Unsupported LLM config: {requested}. Available configs: {model_names}")


def make_gemini_call_from_env(
    system_text: str,
    user_text: str,
    model: str = "gemini-2.5-flash-lite",
) -> Optional[dict]:
    return asyncio.run(make_gemini_call(system_text, user_text, model=model))


def make_llm_call_from_config(messages: List[dict], llm_config: dict[str, Any]) -> Optional[dict]:
    provider = llm_config["provider"]
    model = llm_config["name"]

    if provider in {"OpenAI", "DeepSeek", "Google"}:
        return make_llm_call(messages, model=model, provider=provider)

    if "model_path" in llm_config:
        # Transport swap: a running LOCAL vllm-serve endpoint short-circuits the in-process
        # engine so N worker processes can share one server (cross-file batching). localhost
        # only; unset env falls through to the unchanged in-process path.
        server_url = os.getenv("SEMANTIC_VLLM_SERVER_URL", "").strip()
        if server_url:
            return call_local_vllm_server(
                messages,
                base_url=server_url,
                model=os.getenv("SEMANTIC_VLLM_SERVER_MODEL", "repair-model"),
                max_tokens=llm_config["token_for_completion"],
                generation_config=llm_config.get("generation_config"),
            )

        from model.inference import call_llm_with_message

        content = call_llm_with_message(
            messages=messages,
            model_path=llm_config["model_path"],
            max_tokens=llm_config["token_for_completion"],
            tokenizer_path=llm_config.get("tokenizer_path"),
            generation_config=llm_config.get("generation_config"),
        )
        return {"content": content, "usage": None}

    raise ValueError(f"Unsupported provider config: {provider}:{model}")
