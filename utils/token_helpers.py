import tiktoken
from .providers import LLM_MODELS, OPEN_LLM_MODELS
from google import genai
from typing import Optional
import os

SYSTEM_PROMPT_FOR_LOCAL = (
    "You are an expert Python programmer and syntax repair specialist.\n"
    "Your task is to fix Python SYNTAX errors only.\n\n"

    "Make ONLY the minimal changes strictly required for the code to parse and compile successfully.\n"
    "Do NOT fix logic errors, runtime errors, or improve code quality unless doing so is absolutely necessary to resolve a syntax error.\n"
    "Do NOT refactor, reformat, rename symbols, or modify any code that is already syntactically valid.\n\n"

    "The provided error message is for reference only and may point to the wrong line. "
    "You must inspect surrounding lines and earlier code to identify the true root cause of any syntax error.\n\n"

    "If there are orphaned 'break', 'continue', or similar statements that cause invalid syntax, "
    "you may add the smallest possible syntactic wrapper (such as a dummy loop) solely to make the code syntactically valid.\n"

    "If missing or incomplete try/except blocks cause syntax errors, "
    "you may add the minimal required structure (including 'pass' where necessary) to restore syntactic correctness.\n\n"

    "If missing or incomplete match statements cause syntax errors, "
    "you may add the minimal required structure a default case (i.e., case _:, including 'pass' where necessary) to restore syntactic correctness.\n\n"

    "You must also fix any OTHER syntax errors present in the snippet beyond the one mentioned in the error message, "
    "as long as they can be resolved with minimal changes.\n\n"

    "Preserve all error-free lines exactly as they appear in the original code, including indentation.\n"
    "The final output must be a syntactically valid Python program.\n\n"

    "Output ONLY the corrected Python code.\n"
    "Do NOT include explanations, reasoning, comments, markdown, or formatting wrappers of any kind.\n"
    "Do NOT add any text before or after the code."
)


SYSTEM_PROMPT_FOR_ROOT_CAUSE_ANALYSIS = (
    "You are an expert Python compiler and syntax error analyst.\n"
    "Your task is to analyze a Python code snippet and identify the ROOT CAUSE of the syntax error "
    "described in the error message.\n"
    "The error description is provided for reference only and may point to an incorrect line, "
    "as Python syntax errors often originate from earlier lines or unclosed constructs.\n"
    "In addition to the reported error, you must also identify and explain any OTHER Python syntax errors "
    "present in the snippet that would prevent it from parsing or compiling successfully.\n"
    "You must reason about parser state, including unclosed parentheses or blocks, invalid indentation, "
    "misplaced control-flow statements, missing delimiters, or structurally invalid constructs.\n"
    "You must look out for missing try/except blocks, orphaned 'break' or 'continue' statements, match statements without cases, and any other syntax issues that would cause the code to fail to parse.\n"
    "Your output must explain WHAT the syntax errors are and WHY they occur, focusing on how the Python "
    "parser interprets the code.\n"
    "Do NOT propose fixes and do NOT output any code.\n"
    "Provide a concise explanation in 5–6 complete sentences.\n"
    "Do NOT include bullet points, markdown, formatting, or any text outside the explanation.\n"
    "Focus strictly on syntactic structure, not runtime behavior or logical correctness."
)


USER_PROMPT_TEMPLATE_ROOT_CAUSE = (
    "Analyze the Python code snippet below to determine the ROOT CAUSE of the syntax error described, "
    "and to identify any OTHER syntax errors present in the snippet.\n\n"
    "Error description (for reference only; the reported line may be incorrect):\n"
    "{error_message}\n\n"
    "Code snippet:\n"
    "{code_snippet}\n\n"
    "Explain what the primary syntax error is and why it occurs, and also explain any additional "
    "syntax errors that would prevent the code from parsing or compiling successfully. "
    "Your explanation must be 5–6 complete sentences. "
    "Do not suggest fixes and do not output any code."
)


USER_PROMPT_TEMPLATE_LOCAL = (
    "Analyze the Python code snippet below and fix all syntax errors.\n\n"
    "Initial error message (for reference only; it may point to the wrong line):\n"
    "{error_message}\n\n"
    "Initial code snippet:\n"
    "{code_snippet}\n\n"
    "Explanation of the root cause of the syntax error (CONTEXT ONLY — do not repeat or paraphrase):\n"
    "{current_explanation}\n\n"
    "Apply minimal syntax-only fixes based on the code and the explanation above.\n"
    "Output ONLY the corrected Python code.\n"
    "If the code contains no syntax errors or the errors cannot be fixed, return the code unchanged."
)


USER_PROMPT_TEMPLATE_LOCAL_WITHOUT_EXPLANATION = (
    "Analyze the Python code snippet below and fix all syntax errors.\n\n"
    "Initial error message (for reference only; it may point to the wrong line):\n"
    "{error_message}\n\n"
    "Initial code snippet:\n"
    "{code_snippet}\n\n"
    "Apply minimal syntax-only fixes based on the code above.\n"
    "Output ONLY the corrected Python code.\n"
    "If the code contains no syntax errors or the errors cannot be fixed, return the code unchanged."
)


def build_chat_messages(
    code_snippet: str,
    error_message: str,
    system_prompt: str,
    user_prompt_template: str,
    current_explanation: Optional[str] = "",
) -> list[dict]:
    format_kwargs = { "error_message": error_message,   "code_snippet": code_snippet,}

    if  len(current_explanation) > 0:
        format_kwargs["current_explanation"] = current_explanation

    user_prompt = user_prompt_template.format(**format_kwargs)

    messages = [
        {
            "role": "system",
            "content": str(system_prompt).strip(),
        },
        {
            "role": "user",
            "content": str(user_prompt).strip(),
        },
    ]

    return messages



SYSTEM_PROMPT = """
You are an expert Python programmer and code repair specialist. Your task is to analyze Python code snippets that may contain syntax errors and provide corrected versions of the code with minimal changes. You must ensure that the corrected code is syntactically valid and adheres to best practices in Python programming.
""".strip()


def get_previous_attempt_result(previuos_response: str, previous_error) -> str:
    return (
        "Additionally, as there has been a previous repair attempt, the response and the result from the attempt will be provided to help guide your current attempt.\n\n"
        
        "## The resulting response of the previous repair attempt was:  ##\n\n"
        f"{previuos_response}\n\n "
        "## And the compilation error encountered was:  ##\n"
        f"{previous_error}\n\n"

        "Keeping in mind the initial code snippet provided and the code snippet from previuos response along with the compilation error resulting from the previous response, make a new attempt to fix the initial snippet, ensuring your response only modifies the erroring lines while keeping the rest of the program the same.\n"
        "You are required to make the absolute minimum edit required to fix the errors within the snippet and then use the original code provided as reference to keep the error free lines unchanged and return the compilable error free python program.\n"
    )



def get_user_prompt(code_snippet: str, error_message: str, retry_attempt=False, previuos_response="", previous_error="") -> str:
    return (
        "Analyze the initial code Snippet below for possible Python syntax errors.\n"
        "It is highly important that you find and fix every syntax error present in the snippet. \n"
        "Apply the modifications required to fix the errors within the snippet.\n"
        "You must ensure that the corrected snippet does not introduce any new syntax errors.\n"

        "## Initial Error Message was:  ##\n"
        f"{error_message}\n\n"
        "## Initial Code Snippet was:   ##\n"
        f"{code_snippet}\n\n"
        
        "Do not add any additional explanations, notes, comments, or metadata in your response.\n"
        "Do not add any  thinking or reasoning steps, just output the corrected code, modified only where syntax errors were identified.\n"
        "Do not add any ```markdown```, ``python ```, ``` ```, or other formatting wrappers.\n"
        "If no error is not present in the snippet or cannot be fixed within it, return the snippet unchanged.\n"
        "Return only the complete, corrected code snippet (no explanations, no markdown, no extra text).\n"
        "The error message from the initial compilation effort is provided for your reference.\n\n"

        + (get_previous_attempt_result(previuos_response, previous_error) if retry_attempt else "") +  
        
        "Find and fix all syntax errors you can identify in the initial snippet, make the absolutely necessary modifications to ensure the code is syntactically correct and compilable.\n"
        "You will use the original code snippet provided as reference to keep the error free lines unchanged, so that the line differences are as minimal as possible.\n"
        "Your final output must be compilable, free from all sorts of syntax errors and must be as close to the original snippet as possible (i.e., without unnecessary deletions, additions or modifications of error free lines) and with only the necessary changes made to fix the syntax errors.\n"        

    )


# --- ANSI Color Codes for Terminal Output ---
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



def count_tokens_safe(text: Optional[str], provider, model_name) -> int:
    try:
        return count_tokens(text or "", provider, model_name)
    except Exception:
        return len(text or "")

def count_tokens(text: str, encoding_name: str = "cl100k_base", model_name = "gemini-2.5-flash-lite") -> int:
    if not text:
        return 0
    
    if encoding_name == LLM_MODELS[3]['provider']:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.count_tokens(model=model_name, contents=text)
        return response.total_tokens
    else:
        try:
            # Get the encoding for the specified model
            encoding = tiktoken.get_encoding(encoding_name)
            # Encode the text and count the number of tokens
            num_tokens = len(encoding.encode(text))
            return num_tokens
        except Exception as e:
            # print(f"{Colors.WARNING}    -> Correct encoding not found for model {model_name} with encoding {encoding_name}. Trying default encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
            # Encode the text and count the number of tokens
            num_tokens = len(encoding.encode(text))
            return num_tokens
