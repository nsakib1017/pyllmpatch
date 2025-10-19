from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

# ---------- Data model ----------
@dataclass
class TopObj:
    kind: str        # "imports" | "function" | "class" | "global"
    name: str        # name or "" for imports/global
    start: int       # [start, end) absolute char indices in source
    end: int
    text: str

# ---------- Helpers ----------
_DEF_RE   = re.compile(r"^(async\s+def|def)\b")
_CLASS_RE = re.compile(r"^class\b")
_IMP_RE   = re.compile(r"^(from\s+\S+\s+import\b|import\b)")

def _line_offsets(src: str) -> List[int]:
    offs, total = [0], 0
    for ln in src.splitlines(True):
        total += len(ln); offs.append(total)
    return offs

def _leading_ws(s: str) -> str:
    i = 0
    while i < len(s) and s[i] in (" ", "\t"): i += 1
    return s[:i]

def _next_nonblank(lines: List[str], start: int) -> Optional[int]:
    for i in range(start, len(lines)):
        if lines[i].strip(): return i
    return None

def _scan_line_state(line: str, in_str: bool, q: str, esc: bool, depth: int) -> Tuple[bool, str, bool, int]:
    """Update (in_str, quote, esc, paren_depth) by scanning a single line."""
    j, L = 0, len(line)
    while j < L:
        ch = line[j]
        nxt3 = line[j:j+3]
        if in_str:
            if esc: esc = False
            elif ch == "\\" and q in ("'", '"'): esc = True
            elif q in ("'''", '"""'):
                if nxt3 == q: j += 2; in_str = False
            else:
                if ch == q: in_str = False
        else:
            if ch in "([{": depth += 1
            elif ch in ")]}": depth = max(0, depth - 1)
            elif ch in ("'", '"'):
                if nxt3 == ch*3: in_str = True; q = ch*3; j += 2
                else: in_str = True; q = ch
        j += 1
    return in_str, q, esc, depth


def extract_top_level_objects_lenient(src: str) -> List[TopObj]:
    """
    Detect top-level imports (grouped), def/async def (with decorators),
    class (with decorators), and global blocks. No ast; robust to broken code.
    """
    lines = src.splitlines(True)  # keep EOLs
    lo = _line_offsets(src)
    items: List[TopObj] = []

    n = len(lines)
    i = 0
    in_str, q, esc, depth = False, "", False, 0

    def abs_at(line_idx: int, col: int) -> int:
        return lo[line_idx] + col

    while i < n:
        line = lines[i]
        # update state for this line so we only consider headers at true top level
        in_str, q, esc, depth = _scan_line_state(line, in_str, q, esc, depth)

        # skip blank lines
        if not line.strip():
            i += 1
            continue

        # Consider only true top-level (column 0) and not inside string/paren
        if _leading_ws(line) == "" and depth == 0 and not in_str:
            lstr = line.lstrip()

            # 1) Decorators + next def/class
            if lstr.startswith("@"):
                deco_start = i
                j = i
                while j < n and lines[j].lstrip().startswith("@"):
                    # keep scanning state anyway
                    in_str, q, esc, depth = _scan_line_state(lines[j], in_str, q, esc, depth)
                    j += 1
                if j < n:
                    hdr = lines[j].lstrip()
                    if _DEF_RE.match(hdr) or _CLASS_RE.match(hdr):
                        # capture until next top-level header/import (or EOF)
                        k = j + 1
                        while k < n:
                            nxt = lines[k]
                            in_str, q, esc, depth = _scan_line_state(nxt, in_str, q, esc, depth)
                            if _leading_ws(nxt) == "" and depth == 0 and not in_str:
                                nl = nxt.lstrip()
                                if nl.startswith("@") or _DEF_RE.match(nl) or _CLASS_RE.match(nl) or _IMP_RE.match(nl):
                                    break
                            k += 1
                        s = abs_at(deco_start, 0); e = abs_at(k-1, len(lines[k-1]))
                        if _DEF_RE.match(hdr):
                            name = hdr.split("def", 1)[1].split("(", 1)[0].strip()
                            items.append(TopObj("function", name, s, e, src[s:e]))
                        else:
                            name = hdr.split("class", 1)[1].split("(", 1)[0].strip()
                            items.append(TopObj("class", name, s, e, src[s:e]))
                        i = k
                        continue

            # 2) Function or class without decorators
            if _DEF_RE.match(lstr) or _CLASS_RE.match(lstr):
                j = i + 1
                while j < n:
                    nxt = lines[j]
                    in_str, q, esc, depth = _scan_line_state(nxt, in_str, q, esc, depth)
                    if _leading_ws(nxt) == "" and depth == 0 and not in_str:
                        nl = nxt.lstrip()
                        if nl.startswith("@") or _DEF_RE.match(nl) or _CLASS_RE.match(nl) or _IMP_RE.match(nl):
                            break
                    j += 1
                s = abs_at(i, 0); e = abs_at(j-1, len(lines[j-1]))
                if _DEF_RE.match(lstr):
                    name = lstr.split("def", 1)[1].split("(", 1)[0].strip()
                    items.append(TopObj("function", name, s, e, src[s:e]))
                else:
                    name = lstr.split("class", 1)[1].split("(", 1)[0].strip()
                    items.append(TopObj("class", name, s, e, src[s:e]))
                i = j
                continue

            # 3) Imports: group consecutive top-level imports
            if _IMP_RE.match(lstr):
                j = i + 1
                while j < n and _IMP_RE.match(lines[j].lstrip()):
                    j += 1
                s = abs_at(i, 0); e = abs_at(j-1, len(lines[j-1]))
                items.append(TopObj("imports", "", s, e, src[s:e]))
                i = j
                continue

        # 4) Global code: from here until next recognized top-level header/import
        j = i + 1
        while j < n:
            nxt = lines[j]
            in_str, q, esc, depth = _scan_line_state(nxt, in_str, q, esc, depth)
            if _leading_ws(nxt) == "" and depth == 0 and not in_str:
                nl = nxt.lstrip()
                if nl.startswith("@") or _DEF_RE.match(nl) or _CLASS_RE.match(nl) or _IMP_RE.match(nl):
                    break
            j += 1
        s = abs_at(i, 0); e = abs_at(j-1, len(lines[j-1]))
        items.append(TopObj("global", "", s, e, src[s:e]))
        i = j

    items.sort(key=lambda x: x.start)
    return items

def _get_encoder(model: str | None = None, encoding_name: str | None = None):
    try:
        import tiktoken
        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except Exception:
                pass
        if encoding_name:
            return tiktoken.get_encoding(encoding_name)
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def _count_tokens(text: str, enc=None) -> int:
    if enc is None:
        # rough fallback when tiktoken isn't available
        return max(1, len(text) // 4)
    return len(enc.encode(text))

def _pack_objects_to_chunks(objs, max_tokens_per_chunk: int, enc) -> list[str]:
    """First pass: greedy pack (never splits a top-level object)."""
    chunks, cur, cur_tokens = [], [], 0
    for obj in objs:
        t = _count_tokens(obj.text, enc)
        if t > max_tokens_per_chunk:
            if cur:
                chunks.append("".join(cur)); cur, cur_tokens = [], 0
            # Oversized object: emit alone (safer than splitting)
            chunks.append(obj.text)
            continue
        if cur_tokens + t > max_tokens_per_chunk:
            chunks.append("".join(cur))
            cur, cur_tokens = [obj.text], t
        else:
            cur.append(obj.text)
            cur_tokens += t
    if cur:
        chunks.append("".join(cur))
    return chunks

def _merge_adjacent_chunks(chunks: list[str], max_tokens_per_chunk: int, enc) -> list[str]:
    """
    Second pass: greedily fuse neighbors if the combined text still fits.
    Keeps order; never splits content; minimizes chunk count further.
    """
    if not chunks:
        return chunks
    merged: list[str] = []
    carry = chunks[0]
    for nxt in chunks[1:]:
        if _count_tokens(carry + nxt, enc) <= max_tokens_per_chunk:
            carry = carry + nxt
        else:
            merged.append(carry)
            carry = nxt
    merged.append(carry)
    return merged

def chunk_top_level_objects_lenient_merged(
    content: str,
    max_tokens_per_chunk: int,
    model: str | None = None,
    encoding_name: str | None = None,
    headroom_tokens: int = 0,
    extra_merge_passes: int = 1,
) -> list[str]:
    """
    Chunk by top-level objects, then aggressively merge adjacent chunks
    while respecting the token budget.

    Args
    ----
    max_tokens_per_chunk : token budget for the *content* you send per request
    headroom_tokens      : optional reserved tokens (system msgs, tools, etc.). We subtract this
                           from the budget so we never overflow your true per-call allowance.
    extra_merge_passes   : run the merge pass multiple times (usually 1 is enough).
    """
    enc = _get_encoder(model=model, encoding_name=encoding_name)
    effective_budget = max(1, max_tokens_per_chunk - max(0, headroom_tokens))

    # 1) extract objects (imports, top-level def/class, global)
    objs = extract_top_level_objects_lenient(content)

    # 2) first greedy pack (object-safe)
    chunks = _pack_objects_to_chunks(objs, effective_budget, enc)

    # 3) optional merge passes (adjacent only)
    for _ in range(max(0, extra_merge_passes)):
        new_chunks = _merge_adjacent_chunks(chunks, effective_budget, enc)
        if len(new_chunks) == len(chunks):  # no further reduction
            break
        chunks = new_chunks

    return chunks

# Optional verification (if you didn't edit chunks)
def verify_chunks_merge(chunks: list[str], original: str) -> bool:
    return "".join(chunks) == original