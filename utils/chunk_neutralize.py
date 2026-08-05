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
    payload = "\n".join(chunk_lines)
    out = [f"{payload!r}  {CHUNK_MARKER}"]
    out.extend([""] * (len(chunk_lines) - 1))
    return out


def force_parseable(source, repair=None):
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
