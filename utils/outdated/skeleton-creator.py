#!/usr/bin/env python3
"""
make_skeleton_textual.py

Create a skeletal .txt from a Python file WITHOUT compiling or using ast.
Keeps:
  - class and (async) def headers (spanning multiple lines)
  - variable declarations at any indentation (simple: name = ..., name: T = ..., name: T)
Removes:
  - all other logic

Usage:
  python make_skeleton_textual.py path/to/file.py
  python make_skeleton_textual.py path/to/file.py -o out.txt
"""

from pathlib import Path
import argparse

def is_identifier(token: str) -> bool:
    return token.isidentifier()

def leading_spaces(s: str) -> int:
    i = 0
    while i < len(s) and s[i] == ' ':
        i += 1
    return i

def normalize_spaces(s: str) -> str:
    # collapse internal tabs to spaces for safety (without changing indentation count)
    return s.replace('\t', '    ')

def is_def_or_class_start(line: str) -> bool:
    ls = line.lstrip()
    return ls.startswith("def ") or ls.startswith("class ") or ls.startswith("async def ")

def collect_header(lines, i):
    """
    Collect a possibly multi-line class/def header starting at line i.
    We keep consuming lines while parentheses are unbalanced or the
    header hasn't reached a ':' yet.
    """
    header = []
    balance = 0
    saw_colon = False
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        header.append(raw)
        # crude paren balance (ignore strings/comments for simplicity)
        for ch in raw:
            if ch == '(':
                balance += 1
            elif ch == ')':
                balance -= 1
        if raw.strip().endswith(":"):
            saw_colon = True
        # continue if still inside parens or no ':' yet
        if balance <= 0 and saw_colon:
            break
        i += 1
    return " ".join(h.strip() for h in header), i

def looks_like_assignment(line: str) -> bool:
    """
    Heuristic: treat as assignment if it has a single '=' that is not part of
    '==', '>=', '<=', '!=', ':=' and line is not an import or def/class.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    if s.startswith(("def ", "class ", "async def ", "import ", "from ")):
        return False
    # reject comparisons and walrus
    if "==" in s or ">=" in s or "<=" in s or "!=" in s or ":=" in s:
        return False
    # require a '='
    if "=" not in s:
        # allow annotation-only: "x: T"
        if ":" in s:
            left = s.split(":", 1)[0].strip()
            return is_identifier(left)
        return False
    # Ensure it's an actual assignment '='
    eq_index = s.find("=")
    if eq_index == -1:
        return False
    # Simple guard: left side shouldn’t contain obvious structure
    left = s[:eq_index].strip()
    if not left or any(c in left for c in "([{") or "." in left:
        # skip tuple unpacking, slicing, attributes (self.x), etc.
        return False
    return True

def extract_decl_lines(line: str):
    """
    From a line that looks like an assignment or annotation, produce
    one or more declaration strings like:
      name = ...
      name: T = ...
    Handles simple chained assigns: a = b = 0  -> 'a = ...', 'b = ...'
    Ignores non-simple LHS.
    """
    s = line.strip()
    out = []

    # Annotation-only: "x: T"  (no '=' anywhere)
    if "=" not in s and ":" in s:
        left, ann = s.split(":", 1)
        name = left.strip()
        if is_identifier(name):
            out.append(f"{name}: {ann.strip()} = ...")
        return out

    # Assignment case: split on first '=' to get the LHS chain
    if "=" in s:
        before_eq, _after = s.split("=", 1)
        # for chained assigns, split by '=' and take each left segment's last identifier
        chain = [seg.strip() for seg in before_eq.split("=")]
        for seg in chain:
            # keep only the last token (to avoid things like 'global x = 1' -> 'x')
            tokens = seg.replace(",", " ").split()
            if not tokens:
                continue
            candidate = tokens[-1]
            # strip possible trailing colon annotation marker in weird cases
            candidate = candidate.rstrip(":")
            if is_identifier(candidate):
                out.append(f"{candidate} = ...")
        return out

    return out

def build_skeleton_text(source: str) -> str:
    lines = source.splitlines(True)  # keep newlines for indexing
    skeleton_lines = []

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        norm = normalize_spaces(raw)
        indent = leading_spaces(norm)
        stripped = norm.strip()

        if not stripped:
            i += 1
            continue

        # Capture defs/classes (multi-line signatures)
        if is_def_or_class_start(stripped):
            header, j = collect_header(lines, i)
            # Keep original indentation of the first line
            skeleton_lines.append(" " * indent + header.strip())
            i = j + 1
            continue

        # Capture simple variable declarations anywhere
        if looks_like_assignment(norm):
            decls = extract_decl_lines(norm)
            for d in decls:
                skeleton_lines.append(" " * indent + d)
            i += 1
            continue

        # Otherwise ignore (business logic)
        i += 1

    # Post-process: de-duplicate consecutive duplicates and trim trailing blanks
    deduped = []
    prev = None
    for ln in skeleton_lines:
        if ln != prev:
            deduped.append(ln)
        prev = ln

    # Ensure newline at end
    return "\n".join(ln.rstrip() for ln in deduped if ln.strip()) + "\n"

def main():
    ap = argparse.ArgumentParser(description="Create a skeletal .txt from a Python file without compiling.")
    ap.add_argument("input", help="Path to the Python source file")
    ap.add_argument("-o", "--output", help="Output .txt (default: <input>_skeleton.txt)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_skeleton.txt")

    src = in_path.read_text(encoding="utf-8", errors="replace")
    skel = build_skeleton_text(src)
    out_path.write_text(skel, encoding="utf-8")
    print(f"Wrote skeleton to: {out_path}")

if __name__ == "__main__":
    main()
