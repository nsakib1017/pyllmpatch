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
INDENT_RE = re.compile(r"^(\s*)\S")
DEF_CLASS_PREFIXES = ("def ", "class ", "async def ")

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



def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_def_or_class(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(DEF_CLASS_PREFIXES) and stripped.rstrip().endswith(":")


def strip_base_indent(lines: list[str], base_indent: str) -> list[str]:
    indent_len = len(base_indent)
    stripped = []
    for l in lines:
        if l.startswith(base_indent):
            stripped.append(l[indent_len:])
        else:
            stripped.append(l)
    return stripped



def compute_base_indent(lines: List[str]) -> str:
    """
    Compute the minimum indentation of non-empty lines.
    This is the indentation that must be preserved when reattaching.
    """
    indents = [
        leading_spaces(l)
        for l in lines
        if l.strip()
    ]
    if not indents:
        return ""
    return " " * min(indents)


# -------------------------
# Core extractor
# -------------------------

def fetch_syntax_context(
    file_path: Path,
    error_line: int,
    fallback_upward_lines: int = 100,
) -> Optional[Tuple[str, int, int, str]]:
    """
    Fetch syntax context with indentation normalization.

    Returns:
        (normalized_context_text, start_line, end_line, base_indent)
    """

    content = read_file(file_path)
    if content is None:
        return None

    lines = content.splitlines()
    idx = error_line - 1

    if idx < 0 or idx >= len(lines):
        return None

    error_indent = leading_spaces(lines[idx])

    # -------------------------
    # 1️⃣ Enclosing function/class
    # -------------------------
    start_idx = None
    block_indent = None

    for i in range(idx, -1, -1):
        line = lines[i]
        if not line.strip():
            continue

        indent = leading_spaces(line)

        if indent < error_indent and is_def_or_class(line):
            start_idx = i
            block_indent = indent
            break

    if start_idx is not None:
        end_idx = start_idx + 1
        for j in range(start_idx + 1, len(lines)):
            line = lines[j]
            if not line.strip():
                continue
            if leading_spaces(line) <= block_indent:
                end_idx = j - 1
                break
        else:
            end_idx = len(lines) - 1

        block_lines = lines[start_idx:end_idx + 1]
        base_indent = compute_base_indent(block_lines)

        normalized_lines = strip_base_indent(block_lines, base_indent)

        return (
            "\n".join(normalized_lines), 
            start_idx + 1,
            end_idx + 1,
            base_indent,                
        )

    # -------------------------
    # 2️⃣ Root-level block
    # -------------------------
    start_idx = idx
    for i in range(idx, -1, -1):
        if lines[i].strip() and leading_spaces(lines[i]) == 0:
            start_idx = i
            break

    end_idx = start_idx + 1
    for j in range(start_idx + 1, len(lines)):
        if lines[j].strip() and leading_spaces(lines[j]) == 0:
            end_idx = j - 1
            break
    else:
        end_idx = len(lines) - 1

    expanded_start = max(0, start_idx - fallback_upward_lines)
    block_lines = lines[expanded_start:end_idx + 1]
    base_indent = compute_base_indent(block_lines)

    normalized_lines = strip_base_indent(block_lines, base_indent)

    return (
        "\n".join(normalized_lines),      # 👈 normalized
        expanded_start + 1,
        end_idx + 1,
        base_indent,
    )


def normalize_lines(text: str) -> List[str]:
    return text.splitlines()


def apply_base_indent(lines: List[str], base_indent: str) -> List[str]:
    """
    Apply base indentation to all non-empty lines.
    """
    return [
        (base_indent + l if l.strip() else l)
        for l in lines
    ]


def align_indentation(
    patched_block: str,
    base_indent: str,
) -> str:
    """
    Align indentation of patched_block to match original_block.
    """

    patched_lines = normalize_lines(patched_block)

    # Reapply original indentation
    aligned = apply_base_indent(patched_lines, base_indent)

    return "\n".join(aligned)



def reattach_block(
    file_path: Path,
    start_line: int,
    end_line: int,
    new_block: str,
    backup: bool = False,
) -> None:
    """
    Replace lines [start_line, end_line] with new_block.
    Creates a backup if requested.
    """

    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        backup_path.write_text(content, encoding="utf-8")

    new_lines = new_block.splitlines()

    updated = (
        lines[: start_line - 1]
        + new_lines
        + lines[end_line:]
    )

    file_path.write_text("\n".join(updated) + "\n", encoding="utf-8")