#!/usr/bin/env python3
"""
label_blocks_in_original.py

Given:
  - a Python source file
  - its corresponding *_blocks.csv (with columns: order, start_line, end_line)

Produce:
  - a .txt file that is the ORIGINAL file text with START/END markers
    appended inline at the end of the start/end lines for each block.

Markers appear inline like:
  <code of start line> ---START BLOCK <order>---
  ...
  <code of end line>   ---END BLOCK <order>---

Notes:
- We annotate the original text; we do NOT reconstruct from CSV bodies.
- We preserve each line's original newline (LF / CRLF), appending markers
  right before that newline (or at the true end if the line has no newline).
- If blocks touch (prev end_line + 1 == next start_line), there’s no conflict
  because markers land on different lines. If a single-line block occurs,
  we append START then END on that same line (in that order).
- If different blocks both map to the same physical line, we append END markers
  before START markers for that line (to mirror the original tie-break rule).
"""

from pathlib import Path
import argparse
import sys
import warnings
from collections import defaultdict
import pandas as pd


def _split_line_keep_newline(s: str) -> tuple[str, str]:
    """Return (content_without_trailing_newline, trailing_newline_sequence)."""
    if s.endswith("\r\n"):
        return s[:-2], "\r\n"
    if s.endswith("\n") or s.endswith("\r"):
        return s[:-1], s[-1]
    return s, ""


def _validate_and_prepare(df: pd.DataFrame, n_lines: int) -> pd.DataFrame:
    required = {"order", "start_line", "end_line"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"blocks_csv missing required columns: {sorted(missing)}")

    # Ensure numeric and sane
    for col in ["order", "start_line", "end_line"]:
        df[col] = pd.to_numeric(df[col], errors="raise", downcast="integer")

    bad = df[df["start_line"] > df["end_line"]]
    if not bad.empty:
        raise ValueError(f"Rows with start_line > end_line:\n{bad}")

    # Clip to file bounds with warning (1-based line numbers)
    oob = df[(df["start_line"] < 1) | (df["end_line"] > n_lines)]
    if not oob.empty:
        warnings.warn(
            "Some blocks exceed file bounds and will be clipped:\n" + oob.to_string(index=False)
        )
    df["start_line"] = df["start_line"].clip(1, n_lines)
    df["end_line"]   = df["end_line"].clip(1, n_lines)

    # Overlap check (touching is allowed; true overlaps are flagged)
    tmp = df[["start_line", "end_line"]].sort_values(["start_line", "end_line"])
    prev_end = None
    for start, end in tmp.itertuples(index=False, name=None):
        if prev_end is not None and start <= prev_end and start != prev_end + 1:
            raise ValueError(
                f"Overlapping blocks detected: previous ends at {prev_end}, next starts at {start}. "
                "Touching (prev_end == next_start - 1) is allowed."
            )
        prev_end = max(prev_end, end) if prev_end is not None else end

    # Keep user-given order stable for labels
    return df.sort_values("order", kind="mergesort")


def label_blocks_in_original(source_file: str, blocks_csv: str, out_txt: str | None = None) -> Path:
    """
    Append START/END BLOCK markers inline at the ends of the start/end lines
    (as given by blocks.csv) in the original file text.
    """
    src_path = Path(source_file)
    src_text = src_path.read_text(encoding="utf-8", errors="replace")
    lines = src_text.splitlines(keepends=True)  # keep exact newlines per line

    df = pd.read_csv(blocks_csv)
    df = _validate_and_prepare(df, n_lines=len(lines))

    # Collect markers per 0-based line index
    starts_at: dict[int, list[int]] = defaultdict(list)
    ends_at: dict[int, list[int]] = defaultdict(list)

    for _, row in df.iterrows():
        order = int(row["order"])
        s_idx = int(row["start_line"]) - 1  # 0-based
        e_idx = int(row["end_line"]) - 1
        starts_at[s_idx].append(order)
        ends_at[e_idx].append(order)

    # For deterministic output, sort order ids at each line
    for idx in list(starts_at.keys()):
        starts_at[idx].sort()
    for idx in list(ends_at.keys()):
        ends_at[idx].sort()

    # Build new lines by appending markers before each line's newline
    new_lines: list[str] = []
    for i, line in enumerate(lines):
        content, nl = _split_line_keep_newline(line)

        line_starts = starts_at.get(i, [])
        line_ends   = ends_at.get(i, [])

        markers: list[str] = []

        # Single-line blocks on this line: append START then END for each (in ascending order)
        singles = [o for o in line_starts if o in line_ends]
        for o in singles:
            markers.append(f"---START BLOCK {o}---")
            markers.append(f"---END BLOCK {o}---")

        # For different blocks that both hit this line, mirror original tie-break:
        # append ENDs first (for blocks ending here), then STARTs (for blocks starting here).
        ends_only   = [o for o in line_ends   if o not in singles]
        starts_only = [o for o in line_starts if o not in singles]

        for o in ends_only:
            markers.append(f"---END BLOCK {o}---")
        for o in starts_only:
            markers.append(f"---START BLOCK {o}---")

        if markers:
            # Add a space before the first marker if the line has any content
            sep = " " if content and not content.endswith(" ") else ""
            content = f"{content}{sep}{' '.join(markers)}"

        new_lines.append(content + nl)

    labeled_text = "".join(new_lines)

    out_path = Path(out_txt) if out_txt else src_path.with_name(f"{src_path.stem}_labeled.txt")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(labeled_text)

    print(f"Wrote labeled file to: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Append block markers inline at the start/end lines using *_blocks.csv line ranges."
    )
    ap.add_argument("source", help="Path to the original Python source file")
    ap.add_argument("blocks_csv", help="Path to the *_blocks.csv file (must have order, start_line, end_line)")
    ap.add_argument("-o", "--output", help="Output labeled .txt path (default: <source>_labeled.txt)")
    args = ap.parse_args()

    try:
        label_blocks_in_original(args.source, args.blocks_csv, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
