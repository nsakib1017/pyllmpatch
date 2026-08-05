import re


def detect_context_indent(code: str) -> str:
    """Return the minimum leading indentation shared by all non-empty lines."""
    indents = []
    for line in code.splitlines():
        if line.strip():
            indents.append(re.match(r"\s*", line).group(0))
    return min(indents, key=len) if indents else ""


def normalize_indentation(code: str, context_indent: str) -> str:
    """Strip a shared indentation prefix from each line when present."""
    if not context_indent:
        return code
    return "\n".join(
        line[len(context_indent):] if line.startswith(context_indent) else line
        for line in code.splitlines()
    )


def reapply_context_indent(code: str, context_indent: str) -> str:
    """Reapply a shared indentation prefix to each non-empty line."""
    if not context_indent:
        return code
    return "\n".join(
        context_indent + line if line.strip() else line
        for line in code.splitlines()
    )


def apply_context_indent(original_snippet: str, fixed_snippet: str) -> str:
    """Reattach indentation context from the original snippet to the fixed snippet."""
    return reapply_context_indent(fixed_snippet, detect_context_indent(original_snippet))
