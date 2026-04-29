import ast
import shutil
import io
import pandas as pd
from pathlib import Path
import os
import re
import tokenize
from typing import List, Tuple, Optional, Any
from dotenv import load_dotenv

load_dotenv()

ROOT_FOR_FILES = Path(str(os.getenv("ROOT_FOR_FILES")))
BASE_DIR_PYTHON_FILES_PYLINGUAL = ROOT_FOR_FILES / str(os.getenv("BASE_DIR_PYTHON_FILES_PYLINGUAL"))
BASE_DIR_PYTHON_FILES_PYPI = ROOT_FOR_FILES / str(os.getenv("BASE_DIR_PYTHON_FILES_PYPI"))

INDENTED_RE = re.compile(r"^indented_(\d+)(?:\.[^.]+)?$")  # matches indented_12 or indented_12.py
INDENT_RE = re.compile(r"^(\s*)\S")
BLOCK_HEADER_KEYWORDS = (
    "async def ",
    "def ",
    "class ",
    "if ",
    "elif ",
    "else:",
    "for ",
    "while ",
    "try:",
    "except ",
    "finally:",
    "with ",
    "match ",
    "case ",
)
MAX_SYNTAX_CONTEXT_EXPANSION_STEPS = 3
SYNTAX_ERROR_DELIMITER_HINTS = (
    "unexpected eof",
    "was never closed",
    "unterminated",
    "f-string",
    "single '}' is not allowed",
    "expecting '}'",
    "unmatched ')'",
    "unmatched ']'",
    "unmatched '}'",
)
SYNTAX_ERROR_INDENTATION_HINTS = (
    "indentationerror",
    "expected an indented block",
    "unexpected indent",
    "unindent does not match",
    "expected ':'",
)

def highest_indented_file(
    directory: Path | str,
    pattern: str = "indented_*"
) -> Optional[Tuple[Path, int]]:
    """Return the highest-numbered `indented_*` file in a directory."""
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


def _first_top_level_pyc(hash_dir: Path) -> Optional[Path]:
    candidates = sorted(
        p for p in hash_dir.glob("*.pyc")
        if p.is_file()
    )
    return candidates[0] if candidates else None


def _first_matching_pyc(directory: Path, pattern: str) -> Optional[Path]:
    candidates = sorted(
        p for p in directory.glob(pattern)
        if p.is_file()
    )
    return candidates[0] if candidates else None


def _first_matching_file(directory: Path, pattern: str) -> Optional[Path]:
    candidates = sorted(
        p for p in directory.glob(pattern)
        if p.is_file()
    )
    return candidates[0] if candidates else None


def _choose_source_py(hash_dir: Path) -> Optional[Path]:
    candidates = [
        p for p in hash_dir.iterdir()
        if p.is_file()
        and p.suffix == ".py"
        and not p.name.startswith("decompiled")
    ]

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_decompiled_source_py(out_dir: Path, pyc_file: Path | None = None) -> Optional[Path]:
    if pyc_file is not None:
        expected_name = f"decompiled_{pyc_file.name.replace('.pyc', '.py')}"
        expected_path = out_dir / expected_name
        if expected_path.exists():
            return expected_path

    candidates = list(out_dir.rglob("decompiled_*.py"))
    if not candidates:
        return None

    if pyc_file is not None:
        pyc_stem = pyc_file.stem

        exact_stem_matches = [p for p in candidates if p.stem == f"decompiled_{pyc_stem}"]
        if exact_stem_matches:
            exact_stem_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return exact_stem_matches[0]

        loose_matches = [p for p in candidates if pyc_stem in p.stem]
        if loose_matches:
            loose_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return loose_matches[0]

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def fetch_pyllmpatch_pyc_paths(file_hash: str, source: str) -> Tuple[Optional[Path], Optional[Path]]:
    file_hash = norm_str(file_hash)
    source = norm_str(source)

    if not file_hash or not source:
        return None, None

    is_pypi = source == "PyPi"
    hash_dir = (BASE_DIR_PYTHON_FILES_PYPI if is_pypi else BASE_DIR_PYTHON_FILES_PYLINGUAL) / file_hash

    if not hash_dir.exists():
        return None, None

    if is_pypi:
        original_pyc = _first_matching_file(hash_dir / "__pycache__", "*.cpython-310.pyc")
        indented_pyc = _first_matching_file(
            hash_dir / "decompiled_output_pylingual" / "__pycache__",
            "decompiled_*.cpython-310.pyc",
        )
        return original_pyc, indented_pyc

    original_pyc = _first_top_level_pyc(hash_dir)
    indented_dir = hash_dir / "decompiler_output"
    highest_indented = highest_indented_file(indented_dir, "indented_*.pyc")
    indented_pyc = highest_indented[0] if highest_indented else None

    return original_pyc, indented_pyc


def fetch_pyllmpatch_source_path(file_hash: str, source: str) -> Optional[Path]:
    file_hash = norm_str(file_hash)
    source = norm_str(source)

    if not file_hash or not source:
        return None

    is_pypi = source == "PyPi"
    hash_dir = (BASE_DIR_PYTHON_FILES_PYPI if is_pypi else BASE_DIR_PYTHON_FILES_PYLINGUAL) / file_hash

    if not hash_dir.exists():
        return None

    if is_pypi:
        return _choose_source_py(hash_dir)

    indented_dir = hash_dir / "decompiler_output"
    highest_indented = highest_indented_file(indented_dir, "indented_*.py")
    return highest_indented[0] if highest_indented else None


def fetch_pyllmpatch_repair_paths(file_hash: str, source: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    file_hash = norm_str(file_hash)
    source = norm_str(source)

    if not file_hash or not source:
        return None, None, None

    gt_pyc, derived_pyc = fetch_pyllmpatch_pyc_paths(file_hash, source)

    is_pypi = source == "PyPi"
    hash_dir = (BASE_DIR_PYTHON_FILES_PYPI if is_pypi else BASE_DIR_PYTHON_FILES_PYLINGUAL) / file_hash
    if not hash_dir.exists():
        return gt_pyc, derived_pyc, None

    if is_pypi:
        derived_source = _find_decompiled_source_py(hash_dir / "decompiled_output_pylingual", derived_pyc)
        return gt_pyc, derived_pyc, derived_source

    indented_dir = hash_dir / "decompiler_output"
    highest_indented = highest_indented_file(indented_dir, "indented_*.py")
    derived_source = highest_indented[0] if highest_indented else None
    return gt_pyc, derived_pyc, derived_source

def read_file(file_path: Path) -> Optional[str]:
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
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
    df = pd.read_csv(csv_path, dtype={"bytecode_version": str})
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
    """Strip markdown code fences from common LLM response shapes."""
    def _normalize(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
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

    fence = r"(?:```|~~~)"
    lang_until_eol = r"[^\r\n]*"
    paired_re = re.compile(
        rf"(?P<f>{fence})[ \t]*{lang_until_eol}[ \t]*(?:\r?\n)?"
        rf"(?P<body>.*?)"
        rf"(?:\r?\n)?(?P=f)[ \t]*", re.DOTALL
    )
    text = paired_re.sub(lambda m: m.group("body"), text)

    leading_open_re = re.compile(
        rf"^\ufeff?(?:{fence})[ \t]*{lang_until_eol}[ \t]*(?:\r?\n)?",
        re.DOTALL,
    )
    text = leading_open_re.sub("", text, count=1)

    trailing_close_re = re.compile(
        rf"(?:\r?\n)?(?:{fence})[ \t]*\Z",
        re.DOTALL,
    )
    text = trailing_close_re.sub("", text, count=1)

    return text.strip()





def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_block_header(line: str) -> bool:
    stripped = line.lstrip()
    return any(stripped.startswith(prefix) for prefix in BLOCK_HEADER_KEYWORDS) and stripped.rstrip().endswith(":")


def is_def_or_class(line: str) -> bool:
    return is_block_header(line)


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
    """Return the minimum indentation of non-empty lines."""
    indents = [
        leading_spaces(l)
        for l in lines
        if l.strip()
    ]
    if not indents:
        return ""
    return " " * min(indents)


def compute_anchor_indent(lines: List[str]) -> str:
    """Return the indentation of the first non-empty line."""
    for line in lines:
        if line.strip():
            return " " * leading_spaces(line)
    return ""


def _syntax_context_result(block_lines: List[str], start_idx: int, end_idx: int) -> tuple[str, int, int, str, str]:
    base_indent = compute_base_indent(block_lines)
    anchor_indent = compute_anchor_indent(block_lines)
    normalized_lines = strip_base_indent(block_lines, base_indent)
    return (
        "\n".join(normalized_lines),
        start_idx + 1,
        end_idx + 1,
        base_indent,
        anchor_indent,
    )



def build_triple_quote_mask(lines):
    """Return a mask showing whether each line is inside a triple-quoted string."""
    in_triple = False
    triple_char = None
    mask = [False] * len(lines)

    for i, line in enumerate(lines):
        stripped = line.strip()

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


def _syntax_context_parses(snippet: str) -> bool:
    try:
        ast.parse(snippet)
        return True
    except SyntaxError:
        return False


def _syntax_error_category(error_description: str | None) -> str:
    if not error_description:
        return "generic"
    lowered = error_description.lower()
    if any(marker in lowered for marker in SYNTAX_ERROR_DELIMITER_HINTS):
        return "delimiter"
    if any(marker in lowered for marker in SYNTAX_ERROR_INDENTATION_HINTS):
        return "indentation"
    return "generic"


def _syntax_context_strategy_order(error_description: str | None) -> list[str]:
    category = _syntax_error_category(error_description)
    prefer_blocks = _prefer_block_localization(error_description)

    if category == "delimiter":
        return ["delimiter", "block", "line"] if prefer_blocks else ["delimiter", "line", "block"]

    if category == "indentation":
        return ["block", "line", "delimiter"] if prefer_blocks else ["line", "block", "delimiter"]

    if prefer_blocks:
        return ["block", "line", "delimiter"]
    return ["line", "block", "delimiter"]


def _syntax_context_for_strategy(
    strategy: str,
    file_path: Path,
    error_line: int,
    error_description: str | None,
    expansion_level: int,
) -> Optional[Tuple[str, int, int, str, str]]:
    if strategy == "delimiter":
        return _delimiter_localization_context(file_path, error_line, error_description, expansion_level)
    if strategy == "block":
        return fetch_based_on_blocks(file_path, error_line, expansion_level)
    if strategy == "line":
        return fetch_based_on_lines(file_path, error_line, expansion_level)
    return None


def _find_string_opener_line(lines: list[str], scan_limit: int, prefer_fstring: bool = False) -> tuple[int, int] | None:
    scan_limit = min(len(lines), max(1, scan_limit))

    def _matched_quote_index(line: str, quote: str) -> int:
        idx = line.find(quote)
        if idx == -1:
            return -1
        return idx

    for line_idx in range(scan_limit - 1, -1, -1):
        line = lines[line_idx]
        if prefer_fstring:
            candidates = [
                _matched_quote_index(line, 'f"'),
                _matched_quote_index(line, "f'"),
                _matched_quote_index(line, 'F"'),
                _matched_quote_index(line, "F'"),
            ]
            opener_col = max(candidates)
            if opener_col != -1:
                return line_idx + 1, opener_col

        for quote in ('"""', "'''"):
            opener_col = _matched_quote_index(line, quote)
            if opener_col != -1:
                return line_idx + 1, opener_col

        for quote in ('"', "'"):
            opener_col = _matched_quote_index(line, quote)
            if opener_col != -1:
                return line_idx + 1, opener_col

    return None


def fetch_based_on_lines(
    file_path: Path,
    error_line: int,
    expansion_level: int = 0,
    fallback_upward_lines: int = 100,
) -> Optional[Tuple[str, int, int, str, str]]:
    """Return a syntax context window around an error line."""

    content = read_file(file_path)
    if content is None:
        return None

    lines = content.splitlines()
    idx = error_line - 1

    if idx < 0 or idx >= len(lines):
        return None

    error_indent = leading_spaces(lines[idx])

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
        return _syntax_context_result(block_lines, start_idx, end_idx)

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

    extra = max(0, int(expansion_level))
    expanded_start = max(0, start_idx - fallback_upward_lines - extra)
    expanded_end = min(len(lines) - 1, end_idx + extra)
    block_lines = lines[expanded_start:expanded_end + 1]
    return _syntax_context_result(block_lines, expanded_start, expanded_end)




def fetch_based_on_blocks(
    file_path: Path,
    error_line: int,
    expansion_level: int = 0,
) -> Optional[Tuple[str, int, int, str, str]]:

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

        extra = max(0, int(expansion_level))
        start_idx = max(0, start_idx - extra)
        end_idx = min(len(lines) - 1, end_idx + extra)
        block_lines = lines[start_idx:end_idx + 1]
        return _syntax_context_result(block_lines, start_idx, end_idx)

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

    extra = max(0, int(expansion_level))
    start_idx = max(0, start_idx - extra)
    end_idx = min(len(lines) - 1, end_idx + extra)
    block_lines = lines[start_idx:end_idx + 1]
    return _syntax_context_result(block_lines, start_idx, end_idx)


def _delimiter_localization_context(
    file_path: Path,
    error_line: int,
    error_description: str | None = None,
    expansion_level: int = 0,
) -> Optional[Tuple[str, int, int, str, str]]:
    content = read_file(file_path)
    if not content:
        return None

    lines = content.splitlines()
    if not lines:
        return None

    opener_to_closer = {"(": ")", "[": "]", "{": "}"}
    closer_to_opener = {v: k for k, v in opener_to_closer.items()}
    stack: list[tuple[str, int, int]] = []
    issue_kind: str | None = None
    issue_line: int | None = None

    try:
        for tok in tokenize.generate_tokens(io.StringIO(content).readline):
            if tok.type != tokenize.OP:
                continue
            if tok.string in opener_to_closer:
                stack.append((tok.string, tok.start[0], tok.start[1]))
                continue
            if tok.string in closer_to_opener:
                if stack and stack[-1][0] == closer_to_opener[tok.string]:
                    stack.pop()
                else:
                    issue_kind = "unmatched_closer"
                    issue_line = tok.start[0]
                    break
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        message = str(exc)
        lowered = message.lower()
        if "multi-line string" in lowered:
            issue_kind = "unterminated_string"
        elif "multi-line statement" in lowered:
            issue_kind = "unterminated_statement"

    if issue_kind is None and error_description:
        lowered = error_description.lower()
        if "f-string" in lowered or "expecting '}'" in lowered or "single '}' is not allowed" in lowered:
            issue_kind = "unterminated_string"
        elif any(marker in lowered for marker in ("unexpected eof", "was never closed", "unterminated")):
            issue_kind = "unterminated_statement"

    if issue_kind == "unmatched_closer" and issue_line is not None:
        extra = max(0, int(expansion_level))
        start_idx = max(0, issue_line - 3 - extra)
        end_idx = min(len(lines) - 1, issue_line + 1 + extra)
        block_lines = lines[start_idx:end_idx + 1]
        return _syntax_context_result(block_lines, start_idx, end_idx)

    if issue_kind in {"unterminated_string", "unterminated_statement"}:
        scan_limit = min(len(lines), max(1, error_line))
        prefer_fstring = bool(error_description and "f-string" in error_description.lower())
        opener = _find_string_opener_line(lines, scan_limit, prefer_fstring=prefer_fstring)
        opener_line = opener[0] if opener is not None else None
        if opener_line is not None:
            extra = max(0, int(expansion_level))
            start_idx = max(0, opener_line - 1 - extra)
            end_idx = min(len(lines) - 1, max(error_line, opener_line) + 1 + extra)
            block_lines = lines[start_idx:end_idx + 1]
            return _syntax_context_result(block_lines, start_idx, end_idx)

    if not stack:
        return None

    idx = max(0, min(len(lines) - 1, error_line - 1))
    candidate_openers = [item for item in stack if item[1] <= error_line]
    opener = candidate_openers[-1] if candidate_openers else stack[-1]
    extra = max(0, int(expansion_level))
    start_idx = max(0, opener[1] - 1 - extra)

    end_idx = len(lines) - 1
    for j in range(idx + 1, len(lines)):
        if lines[j].strip() and leading_spaces(lines[j]) == 0 and is_block_header(lines[j]):
            end_idx = j - 1
            break

    if end_idx < start_idx:
        end_idx = len(lines) - 1

    end_idx = min(len(lines) - 1, end_idx + extra)

    block_lines = lines[start_idx:end_idx + 1]
    return _syntax_context_result(block_lines, start_idx, end_idx)


def _prefer_block_localization(error_description: str | None) -> bool:
    if not error_description:
        return False
    lowered = error_description.lower()
    if any(marker in lowered for marker in SYNTAX_ERROR_INDENTATION_HINTS):
        return True
    return any(
        marker in lowered
        for marker in (
            "unexpected eof",
            "was never closed",
            "unterminated",
            "f-string",
            "single '}' is not allowed",
            "expecting '}'",
            "unmatched ')'",
            "unmatched ']'",
            "unmatched '}'",
        )
    )


def fetch_syntax_context(
    file_path: Path,
    error_line: int,
    error_description: str | None = None,
    expansion_level: int = 0,
) -> Optional[Tuple[str, int, int, str, str]]:
    if error_line is None:
        return None

    start_expansion = max(0, int(expansion_level))
    max_expansion = start_expansion + MAX_SYNTAX_CONTEXT_EXPANSION_STEPS
    strategy_order = _syntax_context_strategy_order(error_description)
    best_context: Optional[Tuple[str, int, int, str, str]] = None
    best_span: Optional[int] = None

    for extra_expansion in range(start_expansion, max_expansion + 1):
        for strategy in strategy_order:
            context = _syntax_context_for_strategy(strategy, file_path, error_line, error_description, extra_expansion)
            if context is None:
                continue

            span = max(0, context[2] - context[1])
            if _syntax_context_parses(context[0]):
                return context

            if best_span is None or span < best_span:
                best_context = context
                best_span = span

    return best_context
    

def normalize_lines(text: str) -> List[str]:
    return text.splitlines()


def apply_base_indent(lines: List[str], base_indent: str) -> List[str]:
    """Apply a shared indentation prefix to non-empty lines."""
    return [
        (base_indent + l if l.strip() else l)
        for l in lines
    ]


def align_indentation(
    patched_block: str,
    base_indent: str,
    anchor_indent: str | None = None,
) -> str:
    """Align a patched block to the original base indentation."""

    patched_lines = normalize_lines(patched_block)
    indent_prefix = anchor_indent if anchor_indent is not None else base_indent
    aligned = apply_base_indent(patched_lines, indent_prefix)
    return "\n".join(aligned)



def reattach_block(
    file_path: Path,
    start_line: int,
    end_line: int,
    new_block: str,
    backup: bool = False,
) -> None:
    """Replace a line range in a file with a new block."""

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



LIST_OF_LOG_FILES = {
    "gemini-flash": {
        "flash":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20251110T012915Z/daf95c71075048e1b3458c3c109344fd/run_log_daf95c71075048e1b3458c3c109344fd.jsonl",
        "flash-lite":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20251022T022513Z/e2648c12511c48558c29f4c5300aa6fe/run_log_e2648c12511c48558c29f4c5300aa6fe.jsonl",
        # "pro":f"{os.getenv('PROJECT_ROOT_DIR')}/logs/run_log_9c729f3ab91c42f39b74e51fd102ebf2.jsonl", 
        },
    "qwen-7b": {
        "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T220209Z/d622ad5599af48dba5cf4d3435eb0545/run_log_d622ad5599af48dba5cf4d3435eb0545_with_config_0.jsonl",
        "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T220209Z/59ab85724c794122905a946c15baa405/run_log_59ab85724c794122905a946c15baa405_with_config_1.jsonl",
        "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T220209Z/b9c48dc9529b4cc4812337f5cd92f047/run_log_b9c48dc9529b4cc4812337f5cd92f047_with_config_2.jsonl",
        },
    "qwen-32b": {
        "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260129T025137Z/294cecf0e82049b58bc599cf48d0622b/run_log_294cecf0e82049b58bc599cf48d0622b_with_config_0.jsonl",
        "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260129T025137Z/3ef782858fa24527bb5a4755d31d169d/run_log_3ef782858fa24527bb5a4755d31d169d_with_config_1.jsonl",
        "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260129T025137Z/4ce329173d8e4307876d20bf4f7b5f33/run_log_4ce329173d8e4307876d20bf4f7b5f33_with_config_2.jsonl",
        },
    # "deepseek-r1": {
    #     "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T212519Z/f52db77b7dc349a3bbc2615801c94643/run_log_f52db77b7dc349a3bbc2615801c94643_with_config_0.jsonl",
    #     "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T212519Z/f16e38fb574f41fb90efe88e5af39907/run_log_f16e38fb574f41fb90efe88e5af39907_with_config_1.jsonl",
    #     "config_2":"",
    #     },
    "granite": {
        "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T175015Z/b3f723500e5c47168ee6d851e8aee71f/run_log_b3f723500e5c47168ee6d851e8aee71f_with_config_0.jsonl",
        "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T175015Z/0a2b899e13b54024af5d59612a2e9802/run_log_0a2b899e13b54024af5d59612a2e9802_with_config_1.jsonl",
        "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T175015Z/b55502ff8969402cb2810cc957dde2b9/run_log_b55502ff8969402cb2810cc957dde2b9_with_config_2.jsonl",
        },
    "mistral": {
        "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T191624Z/3ea156e177624726b401d3c54a475539/run_log_3ea156e177624726b401d3c54a475539_with_config_0.jsonl",
        "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T191624Z/156b448efa6541378f9a1b5b46c4c6b3/run_log_156b448efa6541378f9a1b5b46c4c6b3_with_config_1.jsonl",
        "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T191624Z/b66bcdd98ecb466ab7734d9f65b1bd68/run_log_b66bcdd98ecb466ab7734d9f65b1bd68_with_config_2.jsonl",
        },
    "phi-4": {
        "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260125T155805Z/22aa93a4cc4f4d17ba63c2688b170f54/run_log_22aa93a4cc4f4d17ba63c2688b170f54_with_config_0.jsonl",
        "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260125T155805Z/67ac1a2443ab4ac8bfe7bba365801501/run_log_67ac1a2443ab4ac8bfe7bba365801501_with_config_1.jsonl",
        "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260125T155805Z/8cc795eb35404e8e8da0d6ec87eb7032/run_log_8cc795eb35404e8e8da0d6ec87eb7032_with_config_2.jsonl",
        },
    "delete-everything": {
        "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260306T175101Z/27e9ae6ac6724779acd77324edda51df/run_log_27e9ae6ac6724779acd77324edda51df_with_config_0.jsonl",
        "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260306T175101Z/13d82a4a8a384595a80f7c64a10d6554/run_log_13d82a4a8a384595a80f7c64a10d6554_with_config_1.jsonl",
        "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260306T175101Z/538f140b7e734a4d9970a0599fada635/run_log_538f140b7e734a4d9970a0599fada635_with_config_2.jsonl",
        },
    }
