import tiktoken
from .providers import LLM_MODELS
from google import genai
import os

# SYSTEM_PROMPT = """
# ROLE
# You are an automated Python syntax repair agent. Your only job is to fix syntax errors in a provided code snippet, one previously identified syntax error has been provided for you and there can be multiple other syntax errors.

# CONTEXT & CONSTRAINTS
# - The repaired snippet will be re-inserted into a larger file and bytecode-diffed (PYC) against the original.
# - It is highly important that you find and fix every syntax error present in the snippet, not just the one provided in the error message as there are often multiple syntax errors present.
# - Do not change any behavior, identifiers, or semantics.
# - Do not change any formatting (indentation, whitespace, blank lines, comments).
# - Do not add or remove lines except where absolutely necessary to fix any syntax error all over the snippet.
# - You must always look for other syntax errors that are pPRESENT beyond the one mentioned in the error message, and fix them with minimal modification to the snippet.
# - If the error cannot be fixed within the provided snippet (e.g., dependency outside the chunk), return the snippet unchanged.

# PRIMARY DIRECTIVES (strict priority)
# - Minimal Change: Make the smallest possible edit (ideally 1–2 characters) that resolves the specified syntax error.
# - Preserve Formatting: Keep all original whitespace, indentation, line breaks, and comments exactly as they are.
# - No Logic Changes: Do not refactor, rename, reorder, or “improve” anything.
# - Chunk Scope: Only modify code within the provided snippet if and only if the error is fixable there.
# - Preserving Comments: Do not remove or alter any comments in the code.
# - Do not return any formatting wrappers like ```python ... ``` or markdown.
# - If the specified error is not present in the snippet or cannot be fixed here, look for other syntax errors that are present beyond the one provided and then return the snippet with only modifications made to resolve the syntax errors.
# - Return the whole corrected snippet, not just the changed line or some subset of it

# DISALLOWED
# - No explanations, notes, or metadata.
# - No markdown fences or formatting wrappers.
# - No additional imports or reordering.
# - No quote normalization or whitespace cleanup.
# - No stylistic edits.

# OUTPUT REQUIREMENTS
# - Return the provided code snippet as it was given, only with the modifcation made to fix the syntax error (verbatim except for the minimal fix), do not remove any comments from the originally submitted file as well.
# - Return raw code only (no prose, no fences). Do not add any code fences or formatting wrappers.
# - If the snippet does not contain ANY syntax errors, then return the snippet in it's entirity without any changes.
# """.strip()


# def get_user_prompt(code_snippet: str, error_message: str, previous_error_summary_prompt="") -> str:
#     return (
#         "Analyze the Code Snippet below for Python syntax errors detailed in addition to the one provided in the Error Message.\n"
#         "It is highly important that you find and fix every syntax error present in the snippet, not just the one provided in the error message as there are often multiple syntax errors present."
#         "You must always look for other syntax errors that are present beyond the one mentioned in the error message, and fix them with minimal modification to the snippet.\n"
#         "Apply the absolute minimum edit required to fix the errors within the snippet.\n"
#         "Do not remove existing comments, do not change any formatting (indentation, whitespace, blank lines).\n"
#         "It is absolutely crucial that you do not remove the comments at the beginning of the snippet if they are given (for example, # Bytecode version: <version_info>). These comments are used to extract version information to perform automated compilation, without these the process will fail.\n"
#         "Do not refactor, rename, reorder, or introduce any logical changes. Make the smallest possible edit.\n"
#         "You can refer to the previous error summary and strategies for fixing errors to help guide your repair if it is provided in the Error Patch Strategy section below.\n"
#         "Do not add any ```markdown```, ``python ```, ``` ```, or other formatting wrappers.\n"
#         "Include the original comments and formatting exactly as they are.\n"
#         "If you identify a syntax error that is not mentioned in the Error Message, and are highly confident in your detection, you must fix it as well.\n"
#         "If the error is not present in the snippet or cannot be fixed within it, return the snippet unchanged.\n"
#         "Return only the complete, corrected code snippet (no explanations, no markdown, no extra text).\n\n"
#         "Keep modifying the snippet until you are absolutely confident that it is suitale for compilation without syntax errors or you determine it cannot be fixed within the snippet.\n\n"
        
#         "Error Message:\n"
#         f"{error_message}\n\n"
#         "Code Snippet:\n"
#         f"{code_snippet}"
#         "Error Patch Strategy:\n"
#         f"{previous_error_summary_prompt}"
#     )

SYSTEM_PROMPT_FOR_LOCAL = (
    "You are an expert Python programmer and code repair specialist. "
    "Your task is to correct Python SYNTAX errors only.\n"
    "Make ONLY the minimal changes strictly required for the code to parse and compile successfully.\n"
    "Do NOT fix logic errors, runtime errors, or improve code quality unless required to resolve a syntax error.\n"
    "Do NOT refactor, reformat, or modify any code that is already syntactically valid.\n"
    "Preserve all error-free lines exactly as they appear in the original code.\n"
    "The final output must be a syntactically valid Python program.\n"
    "Output ONLY the corrected Python code.\n"
    "Do NOT include explanations, reasoning, comments, markdown, or formatting wrappers of any kind.\n"
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

def get_previous_attempt_result_chunk(previuos_response: str, previous_error) -> str:
    return (
        "Additionally, as there has been a previous repair attempt of the provided code chunk, the response and the result from the attempt will be provided to help guide your current attempt.\n\n"
        
        "## The resulting response of the previous code chunk repair attempt was:  ##\n\n"
        f"{previuos_response}\n\n "
        "## And the compilation error encountered was:  ##\n"
        f"{previous_error}\n\n"

        "Keeping in mind the initial code chunk provided and the merged code from previuos responsess along with the compilation error resulting from the previous response, make a new attempt to fix the initial code chunk again, ensuring your response only modifies the erroring lines while keeping the rest of the program the same.\n"
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

def get_user_prompt_chunk(chunk_snippet: str, error_message: str, retry_attempt=False, previuos_response="", previous_error="") -> str:
    return (
        "Following is a code chunk extracted from a larger Python file, with multiple syntax errors.\n"
        "Analyze the initial code chunk below for possible Python syntax errors.\n"
        "It is highly important that you find and fix every syntax error present in the chunk. \n"
        "Apply the modifications required to fix the errors within the snippet.\n"
        "It is important to note that this chunk is part of a larger file, so ensure that your corrections maintain the integrity of the code within this context.\n"
        "The corrected chunk will be re-integrated into the larger file.\n"
        "You must ensure that the corrected chunk does not introduce any new syntax errors when placed back into the larger file.\n"
        "If there are dependencies or references to code outside this chunk that affect its syntax, you must still ensure that the chunk is syntactically correct on its own.\n"
        "If there no syntax errors present in the chunk, return the chunk unchanged.\n"
        "The initial error message may reference lines or context outside this chunk, so focus on identifying and fixing syntax errors that are present within the provided chunk itself.\n"
        
        "## Initial Error Message was:  ##\n"
        f"{error_message}\n\n"
        "## Initial Chunk Snippet was:   ##\n"
        f"{chunk_snippet}\n\n"
        
        "Do not add any additional explanations, notes, comments, or metadata in your response.\n"
        "Do not add any  thinking or reasoning steps, just output the corrected code, modified only where syntax errors were identified.\n"
        "Do not add any ```markdown```, ``python ```, ``` ```, or other formatting wrappers.\n"
        "If no error is not present in the code chunk or cannot be fixed within it, return the code chunk unchanged.\n"
        "Return only the complete, corrected code chunk (no explanations, no markdown, no extra text).\n"
        "The error message from the initial compilation effort is provided for your reference.\n\n"

        + (get_previous_attempt_result_chunk(previuos_response, previous_error) if retry_attempt else "") +  
        
        "Find and fix all syntax errors you can identify in the initial code chunk, make the absolutely necessary modifications to ensure the code is syntactically correct and compilable.\n"
        "You will use the original code chunk provided as reference to keep the error free lines unchanged, so that the line differences are as minimal as possible.\n"
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

def count_tokens(text: str, encoding_name: str = "cl100k_base", model_name = "gemini-2.5-flash-lite") -> int:
    if not text:
        return 0
    
    if encoding_name == LLM_MODELS[3]['provider']:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.count_tokens(model=model_name, contents=text)
        return response.total_tokens
    elif encoding_name == LLM_MODELS[6]['provider']:
        # Get the encoding for the specified model
        encoding = tiktoken.get_encoding("cl100k_base")
        # Encode the text and count the number of tokens
        num_tokens = len(encoding.encode(text))
        return num_tokens
    else:
        # Get the encoding for the specified model
        encoding = tiktoken.get_encoding(encoding_name)
        # Encode the text and count the number of tokens
        num_tokens = len(encoding.encode(text))
        return num_tokens

def generate_progress_bar(percentage: float, length: int = 20) -> str:
    """
    Generates a simple text-based progress bar.

    Args:
        percentage: The percentage to display (0 to 100).
        length: The total character length of the bar.

    Returns:
        A string representing the progress bar.
    """
    filled_length = int(length * percentage // 100)
    bar_color = Colors.OKGREEN if percentage < 80 else Colors.WARNING if percentage < 100 else Colors.FAIL
    bar = '█' * filled_length + '-' * (length - filled_length)
    return f"{bar_color}[{bar}]{Colors.ENDC}"

def check_context_windows(content: str, model: dict = LLM_MODELS[0]):    
    model_name = f"{model['provider']} - {model['name']}"
    max_tokens = model['token_for_completion']

    # print(f"User prompt token count: {user_token_count}")
    # print(f"System prompt token count: {sys_prompt_token_count}")
    # print(f"Total token count (user + system): {token_count}")
    print(f"    -> Model: {model_name}, Max tokens for completion: {max_tokens}")
    total_tokens = count_tokens(content, model["provider"], model["name"]) 
    print(f"    -> {model_name} token count for the provided text: {total_tokens}")
    return total_tokens > max_tokens
