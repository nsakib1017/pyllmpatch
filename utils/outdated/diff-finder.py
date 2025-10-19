#!/usr/bin/env python3
"""
diff_filter_comment_only.py — unified diff that keeps original line numbers
but hides changes that are only in comments.

- Works on Python files (treats `#` as comment starter).
- Drops pure comment-line changes and inline-comment-only edits.
- Preserves file headers (---/+++) and hunk headers (@@ ... @@).
- If a hunk contains only comment-only diffs, it is omitted.

Usage:
  python diff_filter_comment_only.py old.py new.py [out.txt]
"""

import sys
import difflib
import re
from typing import List, Tuple

ANN_LINE_RE = re.compile(r'^\s*#\s*Line\s+\d+\s*$')  # e.g., "# Line 123"

def code_portion(s: str) -> str:
    """
    Return the code portion of a line by stripping inline Python comments.
    Examples:
      "  x = 1  # note" -> "x = 1"
      "  # only comment" -> "" (empty)
    """
    # Remove everything after first '#'
    before_hash = s.split('#', 1)[0]
    return before_hash.rstrip().lstrip()

def is_comment_only(s: str) -> bool:
    """
    True if the line is a pure comment (after stripping whitespace) or
    matches an annotation like '# Line 123'.
    """
    stripped = s.strip()
    if not stripped:
        return False  # empty isn't "comment", but will be handled via code_portion
    if stripped.startswith('#'):
        return True
    if ANN_LINE_RE.match(stripped):
        return True
    return False

def parse_unified_hunks(lines: List[str]) -> List[List[str]]:
    """
    Split a unified diff (as produced by difflib.unified_diff) into hunks,
    keeping file headers outside. Returns a list of hunk lists (including the
    '@@ ... @@' header and subsequent lines up to next hunk/file header).
    """
    hunks = []
    cur = []
    for line in lines:
        if line.startswith('@@ '):
            if cur:
                hunks.append(cur)
            cur = [line]
        elif line.startswith('--- ') or line.startswith('+++ '):
            # file headers handled by caller; end current hunk if any
            if cur:
                hunks.append(cur)
                cur = []
        else:
            if cur:
                cur.append(line)
    if cur:
        hunks.append(cur)
    return hunks

def filter_hunk_only_code_changes(hunk: List[str]) -> List[str]:
    """
    Given a single hunk (starting with '@@ ... @@'), return a filtered hunk where
    only code changes remain:
      - Keep the hunk header.
      - Within the hunk, pair '-' and '+' runs and drop pairs that are identical
        after stripping comments; also drop single '-' or '+' lines that are
        comment-only (code portion empty).
    """
    if not hunk or not hunk[0].startswith('@@ '):
        return hunk[:]  # safety: not a hunk, return unchanged

    header = hunk[0]
    body = hunk[1:]

    # Collect contiguous change lines (n=0 so we shouldn't see ' ' context lines,
    # but we guard anyway).
    dels: List[str] = []
    adds: List[str] = []
    filtered_body: List[str] = []

    def flush_pairs():
        nonlocal dels, adds, filtered_body
        i = j = 0
        while i < len(dels) or j < len(adds):
            dline = dels[i] if i < len(dels) else None
            aline = adds[j] if j < len(adds) else None

            # Helpers to extract raw content (without diff prefix)
            def raw(x: str) -> str:
                return x[1:] if x and (x.startswith('-') or x.startswith('+') or x.startswith(' ')) else (x or "")

            # Decide if lines are comment-only or code-different
            if dline is not None and aline is not None:
                dcode = code_portion(raw(dline))
                acode = code_portion(raw(aline))
                d_comment_only = (not dcode) and is_comment_only(raw(dline))
                a_comment_only = (not acode) and is_comment_only(raw(aline))

                # If code portions are exactly equal, treat as comment-only change → drop
                if dcode == acode:
                    i += 1
                    j += 1
                    continue

                # If one side is pure comment and other has no code change, drop it
                if d_comment_only and not acode:
                    i += 1
                    j += 1
                    continue
                if a_comment_only and not dcode:
                    i += 1
                    j += 1
                    continue

                # Otherwise, it's a real code change → keep both lines
                filtered_body.append(dline)
                filtered_body.append(aline)
                i += 1
                j += 1

            elif dline is not None:
                # Pure deletion with no paired addition
                dcode = code_portion(raw(dline))
                if dcode:  # has code → keep
                    filtered_body.append(dline)
                else:
                    # comment-only deletion → drop
                    pass
                i += 1

            else:  # aline is not None
                acode = code_portion(raw(aline))
                if acode:  # has code → keep
                    filtered_body.append(aline)
                else:
                    # comment-only insertion → drop
                    pass
                j += 1

        dels = []
        adds = []

    # Walk through hunk body, grouping '-' and '+' runs, flushing when we hit anything else
    for ln in body:
        if ln.startswith('-'):
            dels.append(ln)
        elif ln.startswith('+'):
            adds.append(ln)
        else:
            # context or meta (rare with n=0). Flush current run, then keep line as-is.
            flush_pairs()
            # Optional: if you want to *never* show context, skip adding ln.
            # Here we skip context to satisfy "only display differing lines".
            # filtered_body.append(ln)  # <-- keep if you want context
    flush_pairs()

    if not filtered_body:
        return []  # drop entire hunk if nothing but comments changed

    return [header] + filtered_body

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python diff_filter_comment_only.py <old_file> <new_file> [out_file]")
        sys.exit(1)

    old_file, new_file = sys.argv[1], sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) == 4 else None

    with open(old_file, 'r', encoding='utf-8') as f:
        old_lines = f.readlines()
    with open(new_file, 'r', encoding='utf-8') as f:
        new_lines = f.readlines()

    # Make a unified diff over originals (keeps true line numbers in hunk headers)
    raw_diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=old_file, tofile=new_file,
        lineterm=""
    ))

    # Separate file headers and hunks
    out_lines: List[str] = []
    # Always keep file headers if present
    for ln in raw_diff:
        if ln.startswith('--- ') or ln.startswith('+++ '):
            out_lines.append(ln)

    # Parse hunks and filter
    hunks = parse_unified_hunks(raw_diff)
    for h in hunks:
        filtered = filter_hunk_only_code_changes(h)
        if filtered:
            out_lines.extend(filtered)

    text = "\n".join(out_lines)
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text + ("\n" if text and not text.endswith("\n") else ""))
        print(f"Wrote filtered diff to {out_file}")
    else:
        print(text)

if __name__ == "__main__":
    main()
