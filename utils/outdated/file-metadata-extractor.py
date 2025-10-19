#!/usr/bin/env python3
"""
skeleton_to_csv_with_wrapped_names.py

Generate a CSV (via pandas) from a Python file **without compiling**.

Emits rows for:
- Top-level classes:
    definition: "class <Name>: ..."   (angle brackets wrap names)
    body:       immediate lines under the class (indent == class_indent + 4)
                PLUS flattened headers of methods/nested-classes
                (NOT their bodies)
- Functions / Methods (at any indent):
    definition: "<Owner>.<func>(...):" if inside a class, else "<func>(...):"
    body:       full function/method body (until dedent)
- Top-level "other" blocks:
    definition: "other"
    body:       contiguous top-level non-header content (imports, constants,
                module docstring, if __name__ == '__main__': blocks, etc.)

All rows include:
- symbols: JSON list of identifiers & called names found in the body
- start_line / end_line: 1-based, inclusive

Notes:
- Relies on indentation (no AST parse). The file should be consistently indented.
"""

import argparse, json, keyword, re
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd

KEYWORDS = set(keyword.kwlist)
NOISE_NAMES = {"self", "cls", "__class__", "__name__", "__file__"}

IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
CALL_RE  = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")

def normalize_tabs(s: str) -> str:
    return s.replace("\t", "    ")

def leading_spaces(s: str) -> int:
    i = 0
    while i < len(s) and s[i] == " ":
        i += 1
    return i

def strip_comment(s: str) -> str:
    return s.split("#", 1)[0] if "#" in s else s

def remove_string_literals(s: str) -> str:
    out, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch in ("'", '"'):
            q = ch
            if i+2 < n and s[i:i+3] == q*3:
                i += 3
                while i+2 < n and s[i:i+3] != q*3:
                    i += 1
                i += 3 if i+2 < n else 0
            else:
                i += 1
                while i < n and s[i] != q:
                    if i < n and s[i] == "\\": i += 1
                    i += 1
                i += 1 if i < n else 0
        else:
            out.append(ch); i += 1
    return "".join(out)

def is_header_start(s: str) -> bool:
    return s.startswith("def ") or s.startswith("class ") or s.startswith("async def ")

def collect_header(lines: List[str], start: int) -> Tuple[str, int]:
    """
    Returns (flattened_header_text, header_end_idx) where header_end_idx is 0-based index
    of the header's last physical line.
    """
    hdr_lines, bal, i = [], 0, start
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        hdr_lines.append(raw.strip())
        tmp = strip_comment(remove_string_literals(raw))
        bal += tmp.count("(") - tmp.count(")")
        if raw.strip().endswith(":") and bal <= 0:
            break
        i += 1
    return " ".join(h.strip() for h in hdr_lines).strip(), i

def collect_block(lines: List[str], after_hdr_idx: int, parent_indent: int, for_class: bool=False) -> Tuple[str, int]:
    """
    Collect a block following a header.

    Returns:
        (body_text, block_end_idx)
        block_end_idx is 0-based index of the last line that belongs to this block.
        If there is no body, block_end_idx will be after_hdr_idx-1 (the header line itself).
    """
    body_lines: List[str] = []
    i = after_hdr_idx
    min_child: Optional[int] = None

    while i < len(lines):
        raw = lines[i].rstrip("\n")
        norm = normalize_tabs(raw)
        stripped = norm.strip()
        indent = leading_spaces(norm)

        if not stripped:
            if min_child is not None:
                body_lines.append(raw)
            i += 1
            continue

        # dedent → block ends
        if min_child is not None and indent <= parent_indent:
            break

        if min_child is None:
            # First non-empty line sets the child indent
            if indent <= parent_indent:
                i += 1
                continue
            min_child = indent

        if for_class:
            # Only include immediate children (indent == parent_indent + 4)
            if indent == parent_indent + 4:
                if is_header_start(stripped):
                    hdr, j = collect_header(lines, i)
                    body_lines.append(hdr)  # record header only
                    i = j + 1
                    continue
                body_lines.append(raw)
                i += 1
                continue
            # Deeper content (e.g., method bodies) is skipped in class "body"
            i += 1
            continue
        else:
            # Function/method bodies: include everything until dedent
            body_lines.append(raw)
            i += 1

    # If no child content was found, end is the header line
    end_idx = (i - 1) if min_child is not None else (after_hdr_idx - 1)
    return "\n".join(body_lines).rstrip(), end_idx

def collect_other_block(lines: List[str], start: int) -> Tuple[str, int]:
    """
    Collect a top-level 'other' block from `start` (0-based), continuing until the
    next top-level header or EOF. Includes any indented lines (e.g., bodies of
    if __name__ == "__main__": blocks).
    """
    body_lines: List[str] = []
    i = start
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        norm = normalize_tabs(raw)
        stripped = norm.strip()
        indent = leading_spaces(norm)

        if stripped and indent == 0 and is_header_start(stripped):
            break  # stop before next top-level header

        body_lines.append(raw)
        i += 1

    return "\n".join(body_lines).rstrip(), (i - 1)

def extract_symbols(body: str, header: str) -> List[str]:
    if not body: return []
    cleaned = remove_string_literals("\n".join(strip_comment(ln) for ln in body.splitlines()))
    calls = set(CALL_RE.findall(cleaned))
    idents = set(IDENT_RE.findall(cleaned))
    idents -= KEYWORDS
    idents -= NOISE_NAMES
    own = None
    hs = header.strip()
    if hs.startswith("def ") or hs.startswith("async def "):
        own = hs.split(None,1)[1].split("(",1)[0]
    elif hs.startswith("class "):
        own = hs.split(None,1)[1].split("(",1)[0].rstrip(":")
    if own: idents.discard(own)
    for c in list(calls):
        base = c.split(".",1)[0]
        if base and base not in KEYWORDS:
            idents.add(base)
    return sorted({*idents, *calls})

def extract_class_name(header: str) -> Optional[str]:
    s = header.strip()
    if not s.startswith("class "): return None
    return s.split(None,1)[1].split("(",1)[0].rstrip(":")

def is_function_header(header: str) -> bool:
    s = header.strip()
    return s.startswith("def ") or s.startswith("async def ")

def format_class_header(header: str) -> str:
    """Format as class <Name>: with < > around class name"""
    s = header.strip()
    if not s.startswith("class "): return header
    rest = s[len("class "):]
    name = rest.split("(",1)[0].rstrip(":")
    suffix = rest[len(name):]
    return f"class <{name}>{suffix}"

def format_function_header(header: str, owner: Optional[str]) -> str:
    """Return '<Owner>.<func>(...):' or '<func>(...):' (owner is nearest enclosing class)"""
    s = header.strip()
    if s.startswith("async def "):
        name_args = s[len("async def "):]
    elif s.startswith("def "):
        name_args = s[len("def "):]
    else:
        return header
    fname, rest = name_args.split("(",1)
    fname = fname.strip()
    if owner:
        return f"<{owner}>.<{fname}>({rest}"
    return f"<{fname}>({rest}"

def build_dataframe_with_wrapped_names(source_text: str) -> pd.DataFrame:
    lines = source_text.splitlines(True)
    entries: List[Dict[str,object]] = []
    # Stack tracks ANY enclosing class (top-level or nested) so methods get correct owner
    class_stack: List[Dict[str,object]] = []

    i = 0
    while i < len(lines):
        # Pop classes whose block has ended
        while class_stack and class_stack[-1]["end_idx"] is not None and i > class_stack[-1]["end_idx"]:
            class_stack.pop()

        raw = lines[i].rstrip("\n")
        norm = normalize_tabs(raw)
        stripped = norm.strip()
        indent = leading_spaces(norm)
        indent0 = (indent == 0)

        if not stripped:
            i += 1
            continue

        if is_header_start(stripped):
            header, j = collect_header(lines, i)
            header_indent = leading_spaces(normalize_tabs(lines[i]))

            # CLASS: push for ownership; emit a row ONLY if top-level
            if header.startswith("class "):
                # Compute class body (for class row) and detect where the class ends
                cls_body, cls_end = collect_block(lines, j+1, header_indent, for_class=True)
                cls_name = extract_class_name(header) or ""
                # Push to stack so methods inside get the correct owner
                class_stack.append({"name": cls_name, "end_idx": cls_end})

                if indent0:
                    entries.append({
                        "definition": format_class_header(header),
                        "body": cls_body,
                        "symbols": json.dumps(extract_symbols(cls_body, header), ensure_ascii=False),
                        "start_line": i + 1,
                        "end_line":   cls_end + 1,
                    })

                # Continue scanning inside the class body (methods etc.)
                i = j + 1
                continue

            # FUNCTION / METHOD: always emit with FULL body; owner from nearest class on stack
            if is_function_header(header):
                func_body, func_end = collect_block(lines, j+1, header_indent, for_class=False)
                owner = class_stack[-1]["name"] if class_stack else None
                defn = format_function_header(header, owner)
                entries.append({
                    "definition": defn,
                    "body": func_body,
                    "symbols": json.dumps(extract_symbols(func_body, header), ensure_ascii=False),
                    "start_line": i + 1,
                    "end_line":   func_end + 1,
                })
                i = func_end + 1
                continue

        # Top-level non-header: collect an "other" block (only when not inside any class)
        if indent0 and not class_stack:
            # Start of a non-header top-level chunk
            if stripped and not is_header_start(stripped):
                other_body, other_end = collect_other_block(lines, i)
                entries.append({
                    "definition": "other",
                    "body": other_body,
                    "symbols": json.dumps(extract_symbols(other_body, "other"), ensure_ascii=False),
                    "start_line": i + 1,
                    "end_line":   other_end + 1,
                })
                i = other_end + 1
                continue

        i += 1

    cols = ["definition", "body", "symbols", "start_line", "end_line"]
    return pd.DataFrame(entries, columns=cols)

def main():
    ap = argparse.ArgumentParser(description="CSV with classes, functions/methods (full bodies), top-level 'other', symbols, and line numbers (no compile).")
    ap.add_argument("input", help="Python source file")
    ap.add_argument("-o","--output", help="Output CSV (default: <input>_skeleton.csv)")
    args = ap.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_skeleton.csv")
    src = in_path.read_text(encoding="utf-8", errors="replace")
    df = build_dataframe_with_wrapped_names(src)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote CSV to: {out_path}")

if __name__ == "__main__":
    main()
