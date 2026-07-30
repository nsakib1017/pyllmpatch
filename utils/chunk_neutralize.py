"""Force a decompiled file to parse by isolating damage at TOP-LEVEL CHUNK granularity.

WHY THIS EXISTS (the fifth design, and the first with a real guarantee).
Four earlier attempts repaired one LINE at a time -- close the string, close the bracket, rename the
target, neutralise the line. A ten-file smoke test showed every one of them stopping on the same
thing: `expected 'except' or 'finally'`, `unexpected indent`, `invalid syntax`. Those are BLOCK-level
defects. A missing `try:` three levels up, or a suite whose indentation is wrong for twenty lines,
cannot be fixed by editing the single line the parser happened to point at, so a line-local repair
loop provably cannot converge on them.

This module changes the unit of repair. A Python module is a sequence of top-level chunks, and a
chunk that parses standalone at column zero still parses when concatenated with its neighbours.
So: split into chunks, keep every chunk that parses, and replace any chunk that cannot be repaired
with a string literal holding its exact text. Concatenating parseable chunks yields a parseable
module -- that is a guarantee, not a hope, and it is why this design terminates with a result rather
than with a budget exhausted.

FIDELITY -- read this before quoting any number produced with it. A neutralised chunk no longer
executes. Its bytes survive as data, nothing is deleted, and the marker makes it greppable, but the
code is gone in every sense that matters to behaviour. A file rescued here has been MADE ANALYSABLE,
not repaired. `chunks_neutralised` and `lines_in_neutralised_chunks` exist so that distinction can be
enforced downstream instead of being taken on trust.
"""
from __future__ import annotations

import ast
import re

__all__ = ["force_parseable", "top_level_chunks", "CHUNK_MARKER"]

CHUNK_MARKER = "# pyllmpatch: unparsed-chunk"

_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_OPEN = "([{"
_CLOSE = ")]}"


def _mask(line):
    masked = _STRING.sub(lambda m: " " * len(m.group(0)), line)
    hash_pos = masked.find("#")
    return masked[:hash_pos] if hash_pos != -1 else masked


def top_level_chunks(lines):
    """Split physical lines into top-level chunks. Returns ``[(start, end)]`` inclusive.

    A new chunk begins at a non-blank, non-indented line that is not inside an open bracket and not a
    backslash continuation. Everything indented under it belongs to it.
    """
    chunks = []
    start = None
    depth = 0
    continued = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        starts_here = (
            stripped
            and depth == 0
            and not continued
            and not raw[:1].isspace()
        )
        if starts_here:
            if start is not None:
                chunks.append((start, i - 1))
            start = i
        elif start is None and stripped:
            start = i  # leading junk (e.g. a stray indented line) forms its own chunk
        masked = _mask(raw)
        for ch in masked:
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth = max(0, depth - 1)
        continued = masked.rstrip().endswith("\\")
    if start is not None:
        chunks.append((start, len(lines) - 1))
    return chunks


def _parses(text):
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def _neutralise_chunk(chunk_lines):
    """Render a chunk as a single string-literal statement, padded to keep the line count stable.

    ``repr`` escapes everything, so no quoting inside the payload can break out -- important when the
    chunk contains arbitrary malware text. Padding preserves line numbering, which keeps later
    bytecode-to-source mapping comparable with the original file.
    """
    payload = "\n".join(chunk_lines)
    out = [f"{payload!r}  {CHUNK_MARKER}"]
    out.extend([""] * (len(chunk_lines) - 1))
    return out


def force_parseable(source, repair=None):
    """Return source guaranteed to parse, plus a ledger of what had to be neutralised.

    ``repair`` is an optional ``callable(text) -> text`` tried on a broken chunk before giving up;
    the caller passes the line-local closure pass so cheap defects are still fixed properly and only
    genuinely unrepairable chunks lose their code.
    """
    ledger = {
        "chunks": 0,
        "chunks_neutralised": 0,
        "lines_in_neutralised_chunks": 0,
        "lines_deleted": 0,
        "parses": False,
    }
    if not source.strip():
        ledger["parses"] = True
        return source, ledger

    lines = source.splitlines()
    if _parses(source):
        ledger["parses"] = True
        ledger["chunks"] = len(top_level_chunks(lines))
        return source, ledger

    out = []
    for start, end in top_level_chunks(lines):
        ledger["chunks"] += 1
        chunk = lines[start : end + 1]
        text = "\n".join(chunk)
        if _parses(text):
            out.extend(chunk)
            continue
        if repair is not None:
            try:
                fixed = repair(text + "\n")
            except Exception:
                fixed = None
            if fixed and _parses(fixed):
                fixed_lines = fixed.splitlines()
                # Keep the line count stable so numbering still lines up with the original.
                if len(fixed_lines) < len(chunk):
                    fixed_lines += [""] * (len(chunk) - len(fixed_lines))
                out.extend(fixed_lines)
                continue
        out.extend(_neutralise_chunk(chunk))
        ledger["chunks_neutralised"] += 1
        ledger["lines_in_neutralised_chunks"] += len(chunk)

    text = "\n".join(out) + ("\n" if source.endswith("\n") else "")
    ledger["parses"] = _parses(text)
    return text, ledger
