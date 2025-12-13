#!/usr/bin/env python3
import argparse


def get_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def compute_base_indent(old_lines, start_idx: int, end_idx: int) -> int:
    block = old_lines[start_idx:end_idx]
    indents = [
        get_indent(l)
        for l in block
        if l.strip()
    ]
    if indents:
        return min(indents)
    # fallback: use indent of previous line or 0
    if start_idx > 0:
        return get_indent(old_lines[start_idx - 1])
    return 0


def compute_snippet_relative(snippet_lines):
    raw_indents = []
    for line in snippet_lines:
        if line.strip():
            raw_indents.append(get_indent(line))
        else:
            raw_indents.append(0)

    if any(raw_indents):
        min_indent = min(raw_indents)
    else:
        min_indent = 0

    rel_indents = [i - min_indent for i in raw_indents]
    stripped = [line.lstrip(" ") for line in snippet_lines]

    return stripped, rel_indents


def apply_patch(old_lines, start_line: int, end_line: int, snippet_text: str):
    start_idx = start_line - 1
    end_idx = end_line

    base_indent = compute_base_indent(old_lines, start_idx, end_idx)
    snippet_lines = snippet_text.split("\n")
    stripped_lines, rel_indents = compute_snippet_relative(snippet_lines)

    new_block = []
    for line, rel in zip(stripped_lines, rel_indents):
        if line.strip():
            final_indent = base_indent + rel
            new_block.append(" " * final_indent + line + "\n")
        else:
            new_block.append("\n")

    return old_lines[:start_idx] + new_block + old_lines[end_idx:]


def main():
    parser = argparse.ArgumentParser(
        description="Apply a snippet patch while preserving snippet-relative indentation and aligning to the original file."
    )
    parser.add_argument("source", help="Original Python file")
    parser.add_argument("output", help="Patched output Python file")
    parser.add_argument("--start", type=int, required=True, help="Start line (1-based, inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End line (1-based, inclusive)")
    parser.add_argument("--snippet", required=True, help="Path to file containing patched snippet")

    args = parser.parse_args()

    with open(args.source, "r", encoding="utf-8") as f:
        old_lines = f.readlines()

    with open(args.snippet, "r", encoding="utf-8") as f:
        snippet_text = f.read()

    new_file = apply_patch(old_lines, args.start, args.end, snippet_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(new_file)

    print(f"[+] Patch applied → {args.output}")


if __name__ == "__main__":
    main()
