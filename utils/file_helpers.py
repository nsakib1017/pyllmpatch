# Read dataset_summary.csv into a pandas DataFrame
import shutil
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
# import difflib
from typing import List, Tuple, Optional, Any


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

def copy_file(src: Path | str, dst: Path | str) -> None:
    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_csv_file(file_name: str) -> pd.DataFrame:
    script_dir = Path(__file__).parent
    csv_path = script_dir / file_name
    print(csv_path)
    print(f"Trying to read: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    return df


def get_error_word_message_from_content(filepath):
    FILE_LINE_RE = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in (.+))?')
    ERROR_LINE_RE = re.compile(r'^\s*(\w+(?:Error|Exception))(?:\s*:\s*(.*))?$')
    SORRY_LINE_RE = re.compile(r'^\s*Sorry:\s*(\w+(?:Error|Exception))\s*:\s*(.*?)\s*\(([^,]+),\s*line\s*(\d+)\)\s*$')
    error_word = []
    messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            m_file = FILE_LINE_RE.match(s)
            if m_file:
                continue
            m_err = ERROR_LINE_RE.match(s)
            if m_err:
                error_word.append(m_err.group(1))
                messages.append((m_err.group(2) or "").strip())
                break
            m_sorry = SORRY_LINE_RE.match(line)
            if m_sorry:
                error_word.append(m_sorry.group(1))
                messages.append((m_sorry.group(2) or "").strip())
                break
            
    error_word = error_word[0] if len(error_word) > 0 else None
    message = messages[0] if len(messages) > 0 else None

    return error_word, message


def strip_code_fences(payload: Any) -> str:
    """
    Strips the code fences (```python, ~~~, etc.) from a string,
    and handles different formats like None, dict, list, bytes, etc.
    """
    # ---- 1) Normalize to text ------------------------------------------------
    def _normalize(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            # Common LLM shapes
            val = x.get("content") or x.get("text") or ""
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="ignore")
            return str(val) if not isinstance(val, str) else val
        if isinstance(x, list):
            parts = []
            for item in x:
                if isinstance(item, dict):
                    v = item.get("content") or item.get("text") or ""
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", errors="ignore")
                    parts.append(v if isinstance(v, str) else str(v))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(x)

    text = _normalize(payload)

    # ---- 2) Strip fences (regex patterns) ----------------------------------
    fence = r"(?:```|~~~)"
    lang_until_eol = r"[^\r\n]*"  # Match language until end of line
    # Refined regex that also ensures line breaks are handled
    paired_re = re.compile(
        rf"(?P<f>{fence})[ \t]*{lang_until_eol}[ \t]*(?:\r?\n)?"
        rf"(?P<body>.*?)"
        rf"(?:\r?\n)?(?P=f)[ \t]*", re.DOTALL
    )

    # This should strip the code block properly
    text = paired_re.sub(lambda m: m.group("body"), text)

    # ---- 3) Remove leading fence at the start of the string -----------------
    leading_open_re = re.compile(
        rf"^\ufeff?(?:{fence})[ \t]*{lang_until_eol}[ \t]*(?:\r?\n)?",
        re.DOTALL,
    )
    text = leading_open_re.sub("", text, count=1)

    # ---- 4) Remove trailing fence at the end of the string -----------------
    trailing_close_re = re.compile(
        rf"(?:\r?\n)?(?:{fence})[ \t]*\Z",
        re.DOTALL,
    )
    text = trailing_close_re.sub("", text, count=1)

    return text.strip()





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



def build_triple_quote_mask(lines):
    """
    Returns a list[bool] where True means the line is inside a triple-quoted string.
    """
    in_triple = False
    triple_char = None
    mask = [False] * len(lines)

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Count occurrences to handle """ text """ on one line
        for quote in ('"""', "'''"):
            count = stripped.count(quote)
            if count % 2 == 1:
                if not in_triple:
                    in_triple = True
                    triple_char = quote
                elif quote == triple_char:
                    in_triple = False
                    triple_char = None

        mask[i] = in_triple

    return mask

def fetch_syntax_context(
    file_path: Path,
    error_line: int,
) -> Optional[Tuple[str, int, int, int]]:

    content = read_file(file_path)
    if not content:
        return None

    lines = content.splitlines()
    idx = error_line - 1

    if idx < 0 or idx >= len(lines):
        return None

    triple_mask = build_triple_quote_mask(lines)

    def is_anchor(line: str, i: int) -> bool:
        if triple_mask[i]:
            return False
        s = line.lstrip()
        return s.startswith("@") or is_def_or_class(line)

    def is_top_level_anchor(line: str, i: int) -> bool:
        return (
            not triple_mask[i]
            and line.strip()
            and leading_spaces(line) == 0
            and (line.lstrip().startswith("@") or is_def_or_class(line))
        )

    error_indent = leading_spaces(lines[idx])

    # ----------------------------------------------
    # Case 1: Error inside an indented region
    # ----------------------------------------------
    if error_indent > 0:
        start_idx = 0
        for i in range(idx, -1, -1):
            if lines[i].strip() and is_anchor(lines[i], i):
                start_idx = i
                break

        end_idx = len(lines) - 1
        for j in range(idx + 1, len(lines)):
            if is_top_level_anchor(lines[j], j):
                end_idx = j - 1
                break

        block_lines = lines[start_idx:end_idx + 1]
        base_indent = compute_base_indent(block_lines)
        normalized = strip_base_indent(block_lines, base_indent)

        return (
            "\n".join(normalized),
            start_idx + 1,
            end_idx + 1,
            base_indent,
        )

    # ----------------------------------------------
    # Case 2: Global-scope error
    # ----------------------------------------------
    start_idx = 0
    for i in range(idx - 1, -1, -1):
        if is_top_level_anchor(lines[i], i):
            start_idx = i
            break

    end_idx = len(lines) - 1
    for j in range(idx + 1, len(lines)):
        if is_top_level_anchor(lines[j], j):
            end_idx = j - 1
            break

    block_lines = lines[start_idx:end_idx + 1]
    base_indent = compute_base_indent(block_lines)
    normalized = strip_base_indent(block_lines, base_indent)

    return (
        "\n".join(normalized),
        start_idx + 1,
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



def create_file_from_response(
    file_path: Path,
    content: str,
) -> None:
    content = content.splitlines()
    file_path.write_text("\n".join(content) + "\n", encoding="utf-8")