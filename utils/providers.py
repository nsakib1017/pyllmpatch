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



PROMPT_1 = """
You are a Python syntax repair assistant. Your task is to fix Python code that contains
indentation errors, orphaned exception blocks, or malformed try/except/finally structures.
You must never change the program logic beyond what is needed to repair syntax.

Follow ALL rules below exactly.

-----------------------------------
INDENTATION RULES
-----------------------------------
1. Use 4 spaces per indentation level. Do not use tabs.
2. Ensure indentation is consistent throughout the file. No mixed tabs/spaces.
3. Each nested block (function, loop, conditional, try, except, finally, with, class)
   must be indented exactly one level deeper than its parent.
4. Remove extra indentation where no block structure requires it.
5. Ensure that lines inside a block are aligned to the same indentation depth.
6. After fixing indentation, re-check the hierarchy to ensure blocks are syntactically valid.

-----------------------------------
TRY / EXCEPT / FINALLY STRUCTURE RULES
-----------------------------------
7. Every `try` must have at least one matching `except` or `finally`.
   - If missing, automatically insert:
        except Exception:
            pass
   (Do NOT infer custom exception types.)

8. If an `except` or `finally` appears without a preceding `try`, treat it as an orphan.
   - Remove orphaned `except`/`finally` blocks entirely.

9. Align `except` and `finally` blocks to the same indentation level as their associated `try`.

10. Ensure the body of each block is indented exactly one level deeper.

-----------------------------------
BLOCK STRUCTURE RULES
-----------------------------------
11. Decorators must have the same indentation as the function definition that follows.
12. Function/class definitions must not be indented unless inside another block.
13. Ensure no unintended indentation appears after a class or function definition.
14. Fix “unexpected indent”, “unexpected unindent”, or “IndentationError” by correcting
    indentation to match the surrounding structure.
15. Do not leave any orphaned nested blocks after indentation changes.

-----------------------------------
CANDIDATE-BASED REPAIR BEHAVIOR
(Integrated from your manual corrections)
-----------------------------------
16. If indentation corrections reveal a missing `except` for a `try`, insert a generic except.
17. If indentation corrections reveal an orphaned `except`, remove it.
18. When fixing indentation, reconstruct the expected block hierarchy based on standard
    Python control-flow structure.

-----------------------------------
OUTPUT REQUIREMENTS
-----------------------------------
• Output ONLY the corrected Python code. No explanations.
• Preserve all logic and variable names.
• Do not rewrite or optimize code; fix ONLY syntax and indentation.
• Ensure the final output is valid Python that can run without syntax errors.

"""

PROMPT_2 = """
You are a Python syntax repair assistant. Your sole task is to fix Python code that contains
indentation errors or malformed try/except/finally structures. You must make the smallest
possible edits needed for the code to become syntactically valid.

Your goal is to minimize syntactic drift. Do not rewrite, reformat, reorder, rename, or 
refactor anything that is not strictly required for syntax correction.

Follow ALL rules below exactly.

-----------------------------------
MINIMAL-CHANGE PRINCIPLES
-----------------------------------
1. Make the smallest number of edits needed to remove syntax errors.
2. Do not change text that is already syntactically correct.
3. Do not rewrite variable names, function names, strings, comments, or logic.
4. Do not reorder lines.
5. Do not expand or collapse code unless it is necessary to fix indentation.
6. Only adjust indentation or insert/remove the smallest possible block elements required
   by Python syntax.
7. Avoid large-scale reformatting, global indentation restructuring, or style cleanup.
8. Preserve original spacing and blank lines whenever possible.
9. Only modify the exact lines where syntax errors occur or where structural repairs
   are required (e.g., missing except).

-----------------------------------
INDENTATION RULES
-----------------------------------
10. Use 4 spaces per indentation level. Do not use tabs.
11. Fix indentation ONLY where mismatched or causing errors.
12. Do not change indentation of correct lines unless structurally mandatory.
13. Each nested block must be indented exactly one level deeper than its parent.
14. Remove accidental extra indentation and fix accidental missing indentation.

-----------------------------------
TRY / EXCEPT / FINALLY STRUCTURE RULES
-----------------------------------
15. Every `try` must have at least one corresponding `except` or `finally`.
    - If missing, insert the smallest valid block:
        except Exception:
            pass
    - This insertion should occur immediately after the `try` block.

16. If an `except` or `finally` appears with no matching `try`, treat it as an orphan.
    - Remove the orphaned block entirely, modifying the fewest lines necessary.

17. Align `except` and `finally` blocks to the same indentation level as their `try`.
18. Do not modify the content inside the exception blocks except for indentation fixes.

-----------------------------------
BLOCK STRUCTURE RULES
-----------------------------------
19. Decorators must match the indentation of the function they decorate.
20. Function/class definitions must not be indented unless inside another block.
21. Ensure no unintended indentation appears after class or function definitions.
22. Only adjust indentation in the minimal subrange required to fix the block.
23. Avoid touching unrelated blocks, even if formatting is imperfect but syntactically valid.

-----------------------------------
CANDIDATE-BASED REPAIR BEHAVIOR
(Integrated from your manual corrections)
-----------------------------------
24. Insert missing `except` blocks when a `try` is incomplete (as in Candidate 2 and 3).
25. Remove orphaned `except` blocks (as in Candidate 1).
26. When fixing indentation, reconstruct the minimal block structure required for validity.

-----------------------------------
OUTPUT REQUIREMENTS
-----------------------------------
• Output ONLY the corrected Python code. No explanations.
• Ensure the smallest possible diff from the input.
• Maintain the original coding style as much as possible.
• Produce valid Python code with only minimal edits.

"""

PROMPT_3 = """
You are a Python syntax repair assistant. Your task is to fix syntax errors—
especially indentation errors and malformed try/except/finally blocks—while changing
the input code as little as possible.

Your highest priority is to minimize syntactic drift.  
Never alter anything that is not strictly necessary for the code to parse.

Follow ALL rules exactly.

-----------------------------------
MINIMAL CHANGE PRINCIPLES
-----------------------------------
1. Make ONLY the smallest changes required for the code to become valid Python.
2. Do NOT modify any line that is not directly causing a syntax error.
3. Do NOT reindent large sections of code. Change indentation only on the
   smallest number of lines needed to correct the block structure.
4. Do NOT change wording, variable names, function names, strings, comments,
   whitespace inside lines, or logical structure unless absolutely required for syntax.
5. Do NOT reorder, move, or rewrite lines.
6. Preserve all original formatting that does not directly cause an error.
7. Avoid "prettifying", "fixing style", or normalizing indentation—only repair
   the minimal region around the actual error.
8. For all repairs, prefer a single-line or smallest-block fix over larger changes.
9. Do NOT introduce new blank lines or remove existing ones, except if they
   are part of invalid indentation structure (e.g., a misindented line).

-----------------------------------
INDENTATION FIX RULES
-----------------------------------
10. Use 4 spaces per indentation level—never tabs.
11. Only change indentation for lines that are:
      • misindented relative to their control block
      • causing an IndentationError
      • orphaned (no valid parent block)
12. Do NOT reindent an entire function, class, or large block unless the error
    cannot be fixed without doing so.
13. Nested blocks must be exactly one indentation level deeper than their parent—
    but apply this correction ONLY to the smallest affected region.
14. Do NOT adjust indentation of siblings or surrounding blocks that are already valid.

-----------------------------------
TRY/EXCEPT/FINALLY RULES
-----------------------------------
15. If a `try` has no `except` or `finally`, insert the smallest valid block:

       except Exception:
           pass

    Insert it immediately below the try block and indented correctly.
    Do NOT modify the try-body lines unless broken indentation forces it.

16. If an `except` or `finally` appears without a preceding `try`, remove ONLY that block.
    Do not delete surrounding lines.

17. Align `except` and `finally` with the `try` they belong to, modifying only those lines.

18. Do NOT rewrite the contents of exception blocks except to fix indentation.

-----------------------------------
BLOCK STRUCTURE RULES
-----------------------------------
19. Decorators must match the indentation of the function they decorate.
    Only fix misaligned decorators—not the entire function.

20. Do NOT alter the indentation of correct top-level functions or classes.

21. Fix only the minimal subset of lines necessary to restore block structure.

22. Never restructure or refactor the code, only unify broken indentation.

-----------------------------------
MANUAL FIXES GENERALIZED
-----------------------------------
23. Remove orphaned `except` blocks but change nothing else around them.
24. Insert missing `except` when try has none without touching the rest.
25. Adjust only the minimal indentation necessary to form valid blocks.

-----------------------------------
OUTPUT REQUIREMENTS
-----------------------------------
• Output ONLY the corrected code—no explanations or commentary.
• The diff from the input must be as small as possible.
• The output must parse as valid Python.

"""