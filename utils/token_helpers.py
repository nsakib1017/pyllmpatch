import tiktoken
from .providers import LLM_MODELS, OPEN_LLM_MODELS
from google import genai
from typing import Optional
import os

SYSTEM_PROMPT_FOR_LOCAL = (
    "You are an expert Python programmer and code repair specialist. "
    "Your task is to correct Python SYNTAX errors only.\n"
    "Make ONLY the minimal changes strictly required for the code to parse and compile successfully.\n"
    "Do NOT fix logic errors, runtime errors, or improve code quality unless required to resolve a syntax error.\n"
    "Do NOT refactor, reformat, or modify any code that is already syntactically valid.\n"
    "If there are orphaned break, continue, statements without the corresponding loop you can add a dummy one to address this invalid syntax.\n"
    "You must always look for other syntax errors that are PRESENT beyond the one mentioned in the error message, and fix them with minimal modification to the snippet.\n",
    "If there are missing try/except blocks that are causing syntax errors, you can add some pass statement in the added block to fix the error and make the snippet syntactically valid.\n",
    "Preserve all error-free lines exactly as they appear in the original code.\n"
    "The final output must be a syntactically valid Python program.\n"
    "Output ONLY the corrected Python code.\n"
    "You can add missing try/except blocks with necessary/missing codes to fix the error described.\n",
    "You can add colons, parentheses, quotes, or other syntax elements as needed \n"
    "Do NOT include explanations, reasoning, comments, markdown, or formatting wrappers of any kind.\n"
    "Do NOT add any text before or after the code.\n"
)

SYSTEM_PROMPT_FOR_LOCAL_ALIGNMENT = (
    "You are an expert Python programmer and code alignment specialist. "
    "You will be given an ORIGINAL Python code snippet that contains the correct indentation and whitespace context, "
    "and a PATCHED Python code snippet that fixes syntax errors but may have incorrect or missing indentation.\n"
    "Your task is to apply the indentation and leading whitespace structure from the ORIGINAL code snippet to the PATCHED code snippet.\n"
    "You must preserve the exact indentation levels (spaces or tabs) from the ORIGINAL code wherever possible.\n"
    "For lines that were modified or newly introduced in the PATCHED snippet, infer their indentation by matching the logical parent block "
    "and surrounding indentation levels in the ORIGINAL snippet.\n"
    "Make ONLY the minimal indentation changes strictly required for the code to parse and compile successfully.\n"
    "Do NOT change code logic, control flow, identifiers, or line ordering.\n"
    "Do NOT refactor, reformat, or modify any code other than leading whitespace.\n"
    "Preserve all error-free lines exactly as they appear in the ORIGINAL code.\n"
    "Assume indentation levels in lines ABOVE the error locations in the ORIGINAL snippet are reliable and must be treated as ground truth.\n"
    "If indentation ambiguity exists, choose the indentation that allows the code to parse successfully while remaining consistent with the ORIGINAL snippet.\n"
    "The final output must be a syntactically valid Python program.\n"
    "Output ONLY the final Python code.\n"
    "Do NOT include explanations, reasoning, comments, markdown, diff markers, or formatting wrappers of any kind.\n"
    "Do NOT add any text before or after the code."
)



USER_PROMPT_TEMPLATE_LOCAL = (
    "Analyze the Python code snippet below and fix all syntax errors.\n\n"
    "Initial error message:\n"
    "{error_message}\n\n"
    "Initial code snippet:\n"
    "{code_snippet}\n\n"
    "If the code contains no syntax errors or the errors cannot be fixed, return the code unchanged."
)

USER_PROMPT_TEMPLATE_LOCAL_ALIGNMENT = (
    "You are given two Python code snippets.\n\n"
    "ORIGINAL code snippet (this defines the correct indentation and whitespace context):\n"
    "{original_code_snippet}\n\n"
    "PATCHED code snippet (this fixes syntax errors but may have incorrect indentation):\n"
    "{modified_code_snippet}\n\n"
    "Your task is to align the PATCHED code snippet so that its indentation and leading whitespace "
    "match the ORIGINAL code snippet as closely as possible.\n"
    "Preserve all indentation from the ORIGINAL snippet for unchanged lines.\n"
    "Infer indentation for modified lines using the surrounding indentation structure in the ORIGINAL snippet.\n"
    "Do NOT modify code logic, content, or structure beyond indentation.\n"
    "Return ONLY the final aligned Python code."
)



def build_chat_messages(
    code_snippet: str,
    error_message: str,
    system_prompt: str,
    user_prompt_template: str,
) -> list[dict]:
    user_prompt = user_prompt_template.format(
        error_message=error_message,
        code_snippet=code_snippet,
    )

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


def build_chat_messages_alignment(
    original_code_snippet: str,
    modified_code_snippet: str,
    system_prompt: str,
    user_prompt_template: str,
) -> list[dict]:
    user_prompt = user_prompt_template.format(
        original_code_snippet=original_code_snippet,
        modified_code_snippet=modified_code_snippet,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        {
            "role": "user",
            "content": user_prompt.strip(),
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
