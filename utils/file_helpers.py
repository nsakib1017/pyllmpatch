# Read dataset_summary.csv into a pandas DataFrame
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
# import difflib
from typing import List, Tuple, Optional


INDENTED_RE = re.compile(r"^indented_(\d+)(?:\.[^.]+)?$")  # matches indented_12 or indented_12.py

def highest_indented_file(
    directory: Path | str,
    pattern: str = "indented_*"  # change to "indented_*.py" to require .py
) -> Optional[Tuple[Path, int]]:
    """
    Return (path, idx) for the highest-numbered file named like 'indented_{idx}[.ext]'
    in the given directory. Returns None if none found.
    """
    directory = Path(directory)
    best: Optional[Tuple[Path, int]] = None

    for p in directory.glob(pattern):
        if not p.is_file():
            continue
        m = INDENTED_RE.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        if best is None or n > best[1]:
            best = (p, n)
    return best

def norm_str(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return None if s == "" or s.lower() == "nan" else s

def read_file(file_path: Path) -> Optional[str]:
    if file_path:
        try:
            # Open the file, read its content, and print it
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return content
        except FileNotFoundError:
            print(f"Error: File not found at the specified path: {file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None
    else:
        print("No valid file path provided.")
        return None


from pathlib import Path
from typing import Optional

from pathlib import Path
from typing import Optional, Tuple

def fetch_context_lines(
    file_path: Path,
    line_number: int,
    context: int = 10,
) -> Optional[Tuple[str, int, int]]:
    """
    Fetch up to `context` lines above and below the given line number (1-based).

    Returns:
        (context_text, start_line_number, end_line_number)
        or None if file/content is invalid.
    """

    content = read_file(file_path)
    if content is None:
        return None

    lines = content.splitlines()

    # Convert to 0-based index
    idx = line_number - 1

    if idx < 0 or idx >= len(lines):
        print(f"Line number {line_number} is out of range.")
        return None

    start_idx = max(0, idx - context)
    end_idx = min(len(lines) - 1, idx + context)

    context_lines = lines[start_idx : end_idx + 1]

    # Convert back to 1-based line numbers
    start_line_no = start_idx + 1
    end_line_no = end_idx + 1

    return "\n".join(context_lines), start_line_no, end_line_no


def reattach_context_lines(
    file_path: Path,
    new_snippet: str,
    start_line: int,
    end_line: int,
    *,
    newline: str = "\n",
) -> Optional[str]:
    """
    Replace lines [start_line, end_line] (1-based, inclusive) in file_path
    with new_snippet.

    Returns the updated full file content as a string,
    or None if the operation fails.
    """

    try:
        original = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return None

    lines = original.splitlines()

    # Validate range
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        print(
            f"Invalid line range: {start_line}–{end_line} "
            f"(file has {len(lines)} lines)"
        )
        return None

    # Convert to 0-based indices
    start_idx = start_line - 1
    end_idx = end_line     # slicing end is exclusive

    # Normalize new snippet into lines
    new_lines = new_snippet.splitlines()

    # Patch
    updated_lines = (
        lines[:start_idx]
        + new_lines
        + lines[end_idx:]
    )

    updated_content = newline.join(updated_lines)

    return updated_content



