import re

def detect_context_indent(code: str) -> str:
    indents = []
    for line in code.splitlines():
        if line.strip():
            indent = re.match(r"\s*", line).group(0)
            indents.append(indent)
    if not indents:
        return ""
    # Minimum indentation (context level)
    return min(indents, key=len)


def normalize_indentation(code: str, context_indent: str) -> str:
    if not context_indent:
        return code
    return "\n".join(
        line[len(context_indent):] if line.startswith(context_indent) else line
        for line in code.splitlines()
    )


def reapply_context_indent(code: str, context_indent: str) -> str:
    if not context_indent:
        return code
    return "\n".join(
        context_indent + line if line.strip() else line
        for line in code.splitlines()
    )


def detect_context_indent(code: str) -> str:
    """
    Return the minimum leading indentation shared by all non-empty lines.
    """
    indents = []
    for line in code.splitlines():
        if line.strip():
            indents.append(re.match(r"\s*", line).group(0))
    if not indents:
        return ""
    return min(indents, key=len)


def apply_context_indent(
    original_snippet: str,
    fixed_snippet: str,
) -> str:
    """
    Reattach indentation context from original_snippet
    to fixed_snippet.
    """

    context_indent = detect_context_indent(original_snippet)

    if not context_indent:
        # No contextual indentation → return fixed as-is
        return fixed_snippet

    fixed_lines = fixed_snippet.splitlines()

    reindented_lines = [
        context_indent + line if line.strip() else line
        for line in fixed_lines
    ]

    return "\n".join(reindented_lines)