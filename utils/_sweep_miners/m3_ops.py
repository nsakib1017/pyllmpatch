from __future__ import annotations
import ast, re

_KW = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)")
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def lead_kw(line: str):
    m = _KW.match(line)
    return m.group(1) if m else None


def _is_blank(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("#")


# 1. align_clause_to_opener  (whitespace-only; preserves every character of code)

CLAUSE_OPENERS = {
    "except": {"try", "except"},
    "finally": {"try", "except", "else"},
    "else": {"if", "elif", "for", "while", "try", "except"},
    "elif": {"if", "elif"},
    "case": {"match", "case"},
}

_HDR_SIMPLE = re.compile(r"^[ \t]*(else|finally|try)[ \t]*:[ \t]*(#.*)?$")
_HDR_ARG = re.compile(r"^[ \t]*(except|elif|case)\b.*:[ \t]*(#.*)?$")


def is_clause_header(line: str) -> bool:
    return bool(_HDR_SIMPLE.match(line) or _HDR_ARG.match(line))


def line_context(lines):
    n = len(lines)
    in_str = [False] * n
    depth0 = [0] * n
    cont = False           # previous line ended with a backslash
    q = None
    depth = 0
    prev_cont = False
    stmt = [False] * n
    for i, s in enumerate(lines):
        in_str[i] = q is not None
        depth0[i] = depth
        stmt[i] = (q is None) and depth == 0 and not prev_cont and bool(s.strip()) \
            and not s.lstrip().startswith("#")
        j = 0
        m = len(s)
        while j < m:
            c = s[j]
            if q:
                if len(q) == 1:
                    # single-quoted strings cannot span lines; treat as closed at EOL
                    pass
                if c == "\\":
                    j += 2
                    continue
                if s.startswith(q, j):
                    j += len(q)
                    q = None
                    continue
                j += 1
                continue
            if c in "\"'":
                if s.startswith(c * 3, j):
                    q = c * 3
                    j += 3
                else:
                    k = j + 1
                    while k < m:
                        if s[k] == "\\":
                            k += 2
                            continue
                        if s[k] == c:
                            k += 1
                            break
                        k += 1
                    j = k
                continue
            if c == "#":
                break
            if c in _OPEN:
                depth += 1
            elif c in _CLOSE:
                depth = max(0, depth - 1)
            j += 1
        if q and len(q) == 1:
            q = None
        prev_cont = s.rstrip().endswith("\\") and q is None
    return in_str, depth0, stmt


def local_orphan(lines, i, kws) -> bool:
    I = indent_of(lines[i])
    for j in range(i - 1, -1, -1):
        s = lines[j]
        if _is_blank(s):
            continue
        ind = indent_of(s)
        if ind > I:
            continue
        return not (ind == I and lead_kw(s) in kws and is_block_header(s, lead_kw(s)))
    return True


def find_opener(lines, i, kws):
    min_ind = None
    for j in range(i - 1, -1, -1):
        s = lines[j]
        if _is_blank(s):
            continue
        ind = indent_of(s)
        kw = lead_kw(s)
        if kw in kws and (min_ind is None or ind < min_ind) and is_block_header(s, kw):
            return j, ind
        min_ind = ind if min_ind is None else min(min_ind, ind)
        if min_ind == 0 and j > 0:
            # nothing above can enclose us any more except indent-0 headers,
            # which are still reachable, so keep scanning.
            pass
    return None


def _has_toplevel_colon(text: str) -> bool:
    depth = 0
    q = None
    i = 0
    while i < len(text):
        c = text[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if text.startswith(q, i):
                i += len(q)
                q = None
                continue
            i += 1
            continue
        if c in "\"'":
            q = c * 3 if text.startswith(c * 3, i) else c
            i += len(q)
            continue
        if c == "#":
            break
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == ":" and depth == 0:
            return True
        i += 1
    return False


def is_block_header(line: str, kw: str) -> bool:
    body = line.split("#", 1)[0].rstrip()
    if body.endswith(":"):
        return True
    return _has_toplevel_colon(body)


def _shift_block(lines, i, I, delta):
    end = i + 1
    while end < len(lines):
        t = lines[end]
        if t.strip() and indent_of(t) <= I:
            break
        end += 1
    out = list(lines)
    for k in range(i, end):
        t = out[k]
        if not t.strip():
            continue
        if delta < 0:
            if indent_of(t) < -delta:
                return None
            out[k] = t[-delta:]
        else:
            out[k] = " " * delta + t
    return out


def align_clause_to_opener(lines, err):
    try:
        in_str, depth0, stmt = line_context(lines)
        cand_idx = _candidate_indices(lines, err)
        for i in cand_idx:
            if in_str[i] or not stmt[i]:
                continue
            s = lines[i]
            kw = lead_kw(s)
            if kw not in CLAUSE_OPENERS or not is_clause_header(s):
                continue
            # a clause that ALREADY has a legal owner at its own indent is
            # correct Python -- never move it.
            if not local_orphan(lines, i, CLAUSE_OPENERS[kw]):
                continue
            I = indent_of(s)
            found = find_opener(lines, i, CLAUSE_OPENERS[kw])
            if not found:
                continue
            _, T = found
            if T == I:
                continue
            out = _shift_block(lines, i, I, T - I)
            if out is not None and out != lines:
                return out
        return None
    except Exception:
        return None


def _candidate_indices(lines, err):
    ln = getattr(err, "lineno", None)
    if not ln:
        return range(len(lines))
    i = ln - 1
    order = [i]
    for d in range(1, 40):
        if i + d < len(lines):
            order.append(i + d)
        if i - d >= 0:
            order.append(i - d)
    return [k for k in order if 0 <= k < len(lines)]


# 2. orphan_else_to_if_true  (in-place keyword rewrite; line count unchanged)

_ELSE_LINE = re.compile(r"^([ \t]*)else[ \t]*:(.*)$")
_ELIF_LINE = re.compile(r"^([ \t]*)elif[ \t](.*)$")


def orphan_else_to_if_true(lines, err):
    try:
        in_str, depth0, stmt = line_context(lines)
        for i in _candidate_indices(lines, err):
            if in_str[i] or not stmt[i]:
                continue
            s = lines[i]
            kw = lead_kw(s)
            if kw not in ("else", "elif") or not is_clause_header(s):
                continue
            if not local_orphan(lines, i, CLAUSE_OPENERS[kw]):
                continue
            out = list(lines)
            if kw == "else":
                m = _ELSE_LINE.match(s)
                if not m:
                    continue
                out[i] = f"{m.group(1)}if True:{m.group(2)}"
            else:
                m = _ELIF_LINE.match(s)
                if not m:
                    continue
                out[i] = f"{m.group(1)}if {m.group(2)}"
            if out != lines:
                return out
        return None
    except Exception:
        return None


# 3. invalid_lhs_rename  (generalised literal_lhs_rename)

def _top_level_assign_pos(text: str):
    depth = 0
    q = None
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if text.startswith(q, i):
                i += len(q)
                q = None
                continue
            i += 1
            continue
        if c in "\"'":
            for t in (c * 3,):
                if text.startswith(t, i):
                    q = t
                    i += 3
                    break
            else:
                q = c
                i += 1
            continue
        if c == "#":
            return None
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth < 0:
                return None
        elif c == "=" and depth == 0:
            if i + 1 < n and text[i + 1] == "=":
                i += 2
                continue
            if i > 0 and text[i - 1] in "=!<>+-*/%&|^~:@":
                i += 1
                continue
            return i
        i += 1
    return None


def _mangle(target: str):
    t = target.strip()
    prefix = ""
    if t.startswith("self."):
        prefix, t = "self.", t[5:]
    m = re.sub(r"\W", "_", t, flags=re.UNICODE)
    if not m:
        return None
    if m[0].isdigit():
        m = "_" + m
    if not m.isidentifier():
        return None
    return prefix + m


def invalid_lhs_rename(lines, err):
    try:
        in_str, depth0, stmt = line_context(lines)
        for i in _candidate_indices(lines, err):
            if in_str[i] or not stmt[i] or depth0[i] != 0:
                continue
            s = lines[i]
            if not s.strip() or s.lstrip().startswith("#"):
                continue
            pos = _top_level_assign_pos(s)
            if pos is None:
                continue
            lhs = s[:pos]
            ind = len(lhs) - len(lhs.lstrip(" \t"))
            target = lhs.strip()
            if not target:
                continue
            try:
                ast.parse(target + " = None")
                continue           # already a legal target -> not our case
            except SyntaxError:
                pass
            except Exception:
                continue
            new = _mangle(target)
            if not new:
                continue
            out = list(lines)
            out[i] = s[:ind] + new + s[pos:]
            if out != lines:
                return out
        return None
    except Exception:
        return None


# 4. insert_missing_open_paren  (adds only balancing punctuation)

def _paren_excess(text: str):
    depth = 0
    excess = 0
    q = None
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if text.startswith(q, i):
                i += len(q)
                q = None
                continue
            i += 1
            continue
        if c in "\"'":
            if text.startswith(c * 3, i):
                q = c * 3
                i += 3
            else:
                q = c
                i += 1
            continue
        if c == "#":
            break
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            if depth == 0:
                excess += 1
            else:
                depth -= 1
        i += 1
    if q:
        return None
    return excess


def insert_missing_open_paren(lines, err):
    try:
        msg = (getattr(err, "msg", "") or "").lower()
        if "unmatched" not in msg and "does not match" not in msg:
            return None
        in_str, depth0, stmt = line_context(lines)
        for i in _candidate_indices(lines, err)[:8]:
            if in_str[i] or not stmt[i]:
                continue
            s = lines[i]
            ex = _paren_excess(s)
            if not ex:
                continue
            pos = _top_level_assign_pos(s)
            if pos is not None:
                j = pos + 1
                while j < len(s) and s[j] == " ":
                    j += 1
            else:
                j = indent_of(s)
            out = list(lines)
            out[i] = s[:j] + "(" * ex + s[j:]
            if out != lines:
                return out
        return None
    except Exception:
        return None


# 5. synthesize_try_for_orphan_handler  (ADDS a line -> not code-preserving)

def synthesize_try_for_orphan_handler(lines, err):
    try:
        in_str, depth0, stmt = line_context(lines)
        for i in _candidate_indices(lines, err):
            if in_str[i] or not stmt[i]:
                continue
            s = lines[i]
            kw = lead_kw(s)
            if kw not in ("except", "finally") or not is_clause_header(s):
                continue
            I = indent_of(s)
            # only a real `try` (or an already-attached except-chain) can own us
            if not local_orphan(lines, i, {"try", "except"}) or find_opener(lines, i, {"try"}):
                continue
            j = i - 1
            while j >= 0 and _is_blank(lines[j]):
                j -= 1
            if j < 0:
                continue
            # walk back over the whole preceding statement (its suite is deeper)
            start = j
            while start > 0 and indent_of(lines[start]) > I:
                start -= 1
            if indent_of(lines[start]) != I:
                continue
            out = lines[:start] + [" " * I + "try:"] + \
                [(" " * 4 + t if t.strip() else t) for t in lines[start:i]] + lines[i:]
            return out
        return None
    except Exception:
        return None


# 6. close_truncated_line  (per-line closer; appends only terminators)

def _scan_line(text: str):
    stack = []
    q = None
    qstart = -1
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if text.startswith(q, i):
                i += len(q)
                q = None
                continue
            i += 1
            continue
        if c in "\"'":
            qstart = i
            if text.startswith(c * 3, i):
                q = c * 3
                i += 3
            else:
                q = c
                i += 1
            continue
        if c == "#" and not stack:
            break
        if c in _OPEN:
            stack.append(c)
        elif c in _CLOSE:
            if stack and stack[-1] == _CLOSE[c]:
                stack.pop()
        i += 1
    return stack, q, qstart


ALL_OPS = [
    align_clause_to_opener,
    orphan_else_to_if_true,
    invalid_lhs_rename,
    insert_missing_open_paren,
    synthesize_try_for_orphan_handler,
]


def close_truncated_line(lines, err):
    """The decompiler stops emitting a long constant sequence mid-literal,
    leaving one physical line that ends inside a string and/or inside open
    brackets.  Terminate that ONE line in place: close the string (guarding an
    odd trailing backslash) then close its brackets.  Appends terminators only;
    every character the decompiler did emit is kept."""
    try:
        for i in _candidate_indices(lines, err)[:6]:
            s = lines[i]
            if not s.strip():
                continue
            stack, q, _ = _scan_line(s)
            # A bracket-only imbalance is ordinary multi-line code. The reliable
            # truncation signature is a line that ENDS inside a single-quoted
            # string (those can never legally span a line).
            if not q or len(q) != 1:
                continue
            tail = s
            add = ""
            if q:
                nbs = len(tail) - len(tail.rstrip("\\"))
                if nbs % 2:
                    add += "\\"
                add += q
            add += "".join(_OPEN[o] for o in reversed(stack))
            if not add:
                continue
            out = list(lines)
            out[i] = s + add
            if out != lines:
                return out
        return None
    except Exception:
        return None
