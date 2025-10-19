import tiktoken
from .providers import LLM_MODELS


SUMMARY_PROMPT = """

Objective:
You are tasked with analyzing a list of errors from a Python file. Your goal is to summarize the errors and devise general strategies that can be applied to fix the issues and avoid similar errors in the future.

Instructions:
1. Error Summary:
    -Review the provided errors and generate a concise summary of the most common issues in the list.
    -For each error, identify the error type (e.g., syntax, runtime, logical, etc.) and the underlying cause of the error (e.g., incorrect indentation, undefined variables, incompatible types).
2. Categorize Errors:
    -Identify recurring patterns across the errors (e.g., repeated NameError, multiple TypeError, etc.).
3. General Strategies for Fixing Errors:
    -Based on the identified patterns, propose general strategies to fix each category of errors (not just for the specific errors provided, but for other potential errors of the same type).
    -Suggest practices, or techniques that could be implemented to address these errors across the codebase (e.g., using linters, adopting type hinting, improving error handling, etc.).
    -Include preventive strategies to avoid similar errors in the future, such as better coding practices or automation tools.
4. Focus on Actionable Solutions:
    -For each errors, describe practical steps that can be taken to patch this errors (e.g., consistent code style, automated tests, type checking).
    -Highlight any tools or libraries that could help catch these errors early (e.g., flake8, mypy, pytest).
""".strip()


def generate_summary_prompt(errors_list: list) -> str:
    # Convert the errors list to a string format
    errors_str = "\n".join([f"Error {i+1}:\n{error}" for i, error in enumerate(errors_list)])

    # Create the prompt template
    prompt = f"""
    **Errors:**
    {errors_str}
    """

    return prompt


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

SYSTEM_PROMPT = """
You are an expert Python programmer and code repair specialist. Your task is to analyze Python code snippets that may contain syntax errors and provide corrected versions of the code with minimal changes. You must ensure that the corrected code is syntactically valid and adheres to best practices in Python programming.
""".strip()


def get_previous_attempt_result(previuos_response: str, previous_error: str): 
    
    return f"""
            Additionally, as there has been a previous repair attempt, the response and the result from the attempt will be provided to help guide your current attempt.\n
            The resulting response of the previous repair attempt was:\n
            {previuos_response} \n\n 

            And the compilation error encountered was: \n
            {previous_error} \n\n

            Keeping in mind the initial code snippet provided initially and the code snippet from previuos response and the compilation error resulting from the previous response, make a new attempt to fix the initial snippet, ensuring your response only modifies the erroring lines while keeping the rest of the program the same.\n
            You are required to make the absolute minimum edit required to fix the errors within the snippet and then use the original code provided as reference to keep the error free lines unchanged and return the compilable error free python program.\n
        """


def get_user_prompt(code_snippet: str, error_message: str, retry_attempt=False, previuos_response="", previous_error="") -> str:
    return (
        "Analyze the initial code Snippet below for possible Python syntax errors.\n"
        "It is highly important that you find and fix every syntax error present in the snippet. \n"
        "Apply the absolute minimum edit required to fix the errors within the snippet.\n"

        "Initial Error Message was:\n"
        f"{error_message}\n\n"
        "Initial Code Snippet was:\n"
        f"{code_snippet}\n\n"
        
        "Do not add any  thinking or reasoning steps, just output the corrected code, modified only where syntax errors were identified.\n"
        "Do not add any ```markdown```, ``python ```, ``` ```, or other formatting wrappers.\n"
        "If no error is not present in the snippet or cannot be fixed within it, return the snippet unchanged.\n"
        "Return only the complete, corrected code snippet (no explanations, no markdown, no extra text).\n\n"
        "The error message from the initial compilation effort is provided for your reference.\n"

        + (get_previous_attempt_result(previuos_response, previous_error) if retry_attempt else "") +  
        
        "Find and fix all syntax errors you can identify in the initial snippet, make the absolutely necessary modifications to ensure the code is syntactically correct and compilable.\n"
        "Your final output must be compilable, free from all sorts of syntax errors and must be as close to the original snippet as possible (i.e., without unnecessary deletions, additions or modifications of error free lines) and with only the necessary changes made to fix the syntax errors.\n"        

    )



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


# LLM_MODELS = [
#     {'provider': 'OpenAI', 'name': 'gpt-4o', 'token_for_completion': 16384},
#     {'provider': 'DeepSeek', 'name': 'deepseek-chat', 'token_for_completion': 8192},
#     {'provider': 'DeepSeek', 'name': 'deepseek-reasoner', 'token_for_completion': 8192},
#     {'provider': 'Google', 'name': 'gemini-2.5-flash-latest', 'token_for_completion': 32768}
# ]

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

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0
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

def check_context_windows(content: str, error_description: str, model: dict = LLM_MODELS[0]):

    user_prompt = get_user_prompt(content, error_description)
    user_token_count = count_tokens(user_prompt)
    sys_prompt_token_count = count_tokens(SYSTEM_PROMPT)
    token_count = user_token_count + sys_prompt_token_count
    
    model_name = f"{model['provider']} - {model['name']}"
    max_tokens = model['token_for_completion']

    # print(f"User prompt token count: {user_token_count}")
    # print(f"System prompt token count: {sys_prompt_token_count}")
    # print(f"Total token count (user + system): {token_count}")
    print(f"    -> Model: {model_name}, Max tokens for completion: {max_tokens}")
    is_too_large = token_count > max_tokens
    return is_too_large
