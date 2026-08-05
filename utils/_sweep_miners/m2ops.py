import ast, re
from dataclasses import dataclass

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {")": "(", "]": "[", "}": "{"}
_TRIPLE = ("'''", '"""')


@dataclass
class Err:
    msg: str
    lineno: int | None
    offset: int | None


def probe(src):
    try:
        compile(src, "<s>", "exec")
        return None
    except SyntaxError as e:
        return Err(e.msg or "", e.lineno, e.offset)
    except Exception as e:
        return Err("NONSYNTAX:" + type(e).__name__, None, None)


def advanced(before, after):
    if after is None:
        return True
    if before.lineno is None or after.lineno is None:
        return False
    if after.lineno > before.lineno:
        return True
    return after.lineno == before.lineno and (after.offset or 0) > (before.offset or 0)


def _lws(s):
    return s[: len(s) - len(s.lstrip(" \t"))]


def _ind(s):
    return len(s) - len(s.lstrip(" \t"))


def _scan_line(text):
    stack, i, n, q = [], 0, len(text), None
    while i < n:
        ch = text[i]
        if q:
            if ch == "\\":
                i += 2
                continue
            if text.startswith(q, i):
                i += len(q)
                q = None
                continue
            i += 1
            continue
        if ch == "#":
            break
        if ch in "'\"":
            trip = ch * 3
            if text.startswith(trip, i):
                q, i = trip, i + 3
            else:
                q, i = ch, i + 1
            continue
        if ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            if stack and stack[-1] == _CLOSE[ch]:
                stack.pop()
            else:
                stack.append(ch)          # surplus closer marker
        i += 1
    return stack, q


def _split_top_level_assign(text):
    depth, i, n, q = 0, 0, len(text), None
    while i < n:
        ch = text[i]
        if q:
            if ch == "\\":
                i += 2
                continue
            if text.startswith(q, i):
                i += len(q)
                q = None
                continue
            i += 1
            continue
        if ch in "'\"":
            trip = ch * 3
            if text.startswith(trip, i):
                q, i = trip, i + 3
            else:
                q, i = ch, i + 1
            continue
        if ch == "#":
            return None
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
        elif ch == "=" and depth == 0:
            if text[i + 1: i + 2] == "=":
                i += 2
                continue
            if i and text[i - 1] in "=!<>+-*/%&|^@:":
                i += 1
                continue
            return text[:i], text[i:]
        i += 1
    return None


def _valid_target(txt):
    txt = txt.strip()
    if not txt:
        return False
    try:
        tree = ast.parse(txt + " = 0")
    except Exception:
        return False
    # AnnAssign matters: `client: str = expr` is a legal ANNOTATED binding whose target
    # is `client`.  Accepting only ast.Assign renamed those to `_dc_N`, destroying a
    # real variable name on 8 already-valid files.
    return bool(tree.body) and isinstance(tree.body[0], (ast.Assign, ast.AnnAssign))


_KEYWORD_HEAD = re.compile(
    r"^\s*(if|elif|else|for|while|with|try|except|finally|def|class|return|yield|raise|"
    r"import|from|assert|del|global|nonlocal|lambda|pass|break|continue|async|await|@)\b")


# shared: carry-state scan -> which physical lines START a logical line
_CONT_TAIL = tuple(",([{+-*/%|&^=:~<>")


def _scan_from(text, stack, q):
    stack = list(stack)
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if q:
            if ch == "\\":
                i += 2
                continue
            if text.startswith(q, i):
                i += len(q)
                q = None
                continue
            i += 1
            continue
        if ch == "#":
            break
        if ch in "'\"":
            trip = ch * 3
            if text.startswith(trip, i):
                q, i = trip, i + 3
            else:
                q, i = ch, i + 1
            continue
        if ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            if stack and stack[-1] == _CLOSE[ch]:
                stack.pop()
            else:
                stack.append(ch)
        i += 1
    return stack, q


def _next_nonblank(lines, i):
    for j in range(i + 1, len(lines)):
        if lines[j].strip():
            return lines[j]
    return None


def _looks_truncated(body, nxt):
    if nxt is None:
        return True
    head = nxt.lstrip()[:1]
    if head in (")", "]", "}", ","):
        return False
    return _ind(nxt) <= _ind(body)


def line_starts(src):
    lines = src.splitlines()
    stack, q, out = [], None, []
    for i, line in enumerate(lines):
        out.append(not stack and q is None)
        stack, q = _scan_from(line, stack, q)
        if q in _TRIPLE:
            continue
        if (stack or q) and _looks_truncated(line, _next_nonblank(lines, i)):
            stack, q = [], None          # truncated: the statement ends here
        elif q is not None:
            q = None                     # single-quoted strings never span lines
    return out


# H1  close_unclosed_lines            GLOBAL, append-only
def close_unclosed_lines(src, err=None):
    lines = src.splitlines(keepends=True)
    out, changed = list(lines), False
    stack, q = [], None
    plain = src.splitlines()
    for i, line in enumerate(lines):
        body = line.rstrip("\n")
        if not body.strip() or body.rstrip().endswith("\\"):
            stack, q = _scan_from(body, stack, q)
            continue
        at_start = not stack and q is None
        nstack, nq = _scan_from(body, stack, q)
        if nq in _TRIPLE:
            stack, q = nstack, nq
            continue
        if not nq and not any(c in _OPEN for c in nstack):
            stack, q = nstack, nq
            continue
        if not at_start:
            # Interior line of a genuine multi-line literal (`'content': '',` inside a
            # hand-written dict).  Only a line that BEGINS a logical line can be the
            # truncation point -- without this guard such lines get a stray `}` appended.
            stack, q = nstack, nq
            continue
        if not _looks_truncated(body, _next_nonblank(plain, i)):
            stack, q = nstack, nq
            continue
        closers = "".join(_OPEN[c] for c in reversed(nstack) if c in _OPEN)
        out[i] = body + (nq or "") + closers + "\n"
        changed = True
        stack, q = [], None
    return "".join(out) if changed else None


# H2  invalid_lhs_rename              GLOBAL, rename-only
def invalid_lhs_rename(src, err=None):
    lines = src.splitlines(keepends=True)
    starts = line_starts(src)
    out, changed = list(lines), False
    for i, line in enumerate(lines):
        body = line.rstrip("\n")
        if not body.strip() or _KEYWORD_HEAD.match(body):
            continue
        if i < len(starts) and not starts[i]:
            continue                      # continuation line, not a statement
        sp = _split_top_level_assign(body)
        if not sp:
            continue
        lhs, rest = sp
        if "lambda" in lhs:
            continue                      # `lambda x=1: ...` -- that '=' is a default
        if _valid_target(lhs):
            continue
        out[i] = f"{_lws(line)}_dc_{i}{rest}\n"
        changed = True
    return "".join(out) if changed else None


# H3  orphan_clause_header            GLOBAL, SYNTHESIZES one header line
_CLAUSE = re.compile(r"^(\s*)(except\b|finally\s*:|else\s*:|elif\b)")
_HDR_FOR_ELSE = re.compile(r"^\s*(if\b|elif\b|for\b|while\b|try\s*:|async\s+for\b|except\b|else\s*:)")
_EXC_HDR = re.compile(r"^\s*(try\s*:|except\b|else\s*:)")


def orphan_clause_header(src, err=None):
    lines = src.splitlines(keepends=True)
    edits = []
    for i, line in enumerate(lines):
        m = _CLAUSE.match(line)
        if not m:
            continue
        I = len(m.group(1))
        kind = m.group(2).rstrip(": ").strip()
        j, owner = i - 1, None
        while j >= 0:
            s = lines[j]
            if not s.strip() or s.lstrip().startswith("#"):
                j -= 1
                continue
            d = _ind(s)
            if d <= I:
                owner = (d, s)
                break
            j -= 1
        if owner is None:
            continue
        oind, os_ = owner
        if oind == I:
            if kind in ("except", "finally") and _EXC_HDR.match(os_):
                continue
            if kind in ("else", "elif") and _HDR_FOR_ELSE.match(os_):
                continue
        k, start = i - 1, i
        while k >= 0:
            s = lines[k]
            if not s.strip() or s.lstrip().startswith("#"):
                k -= 1
                continue
            if _ind(s) < I:
                break
            start = k
            k -= 1
        if start >= i:
            continue
        run = [x for x in lines[start:i] if x.strip()]
        if not run or min(_ind(x) for x in run) < I:
            continue
        header = "try:" if kind in ("except", "finally") else "if True:"
        edits.append((start, i, I, header, min(_ind(x) for x in run) == I))
    if not edits:
        return None
    for start, i, I, header, reindent in sorted(edits, reverse=True):
        block = lines[start:i]
        if reindent:
            block = [("    " + x) if x.strip() else x for x in block]
        lines[start:i] = [" " * I + header + "\n"] + block
    return "".join(lines)


# H4  normalize_indent                GLOBAL, whitespace-only
def _code_only(body):
    out, i, n, q = [], 0, len(body), None
    while i < n:
        ch = body[i]
        if q:
            if ch == "\\":
                i += 2
                continue
            if body.startswith(q, i):
                out.append(" " * len(q))
                i += len(q)
                q = None
                continue
            i += 1
            continue
        if ch == "#":
            break
        if ch in "'\"":
            trip = ch * 3
            q = trip if body.startswith(trip, i) else ch
            out.append(" " * len(q))
            i += len(q)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _is_block_opener(body):
    return _code_only(body).rstrip().endswith(":")


def normalize_indent(src, err=None):
    lines = src.splitlines(keepends=True)
    starts = line_starts(src)
    out, stack, expect_deeper, changed = [], [0], False, False
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        if i < len(starts) and not starts[i]:
            out.append(line)              # continuation line: indent is free-form
            continue
        d, body = _ind(line), line.lstrip()
        if expect_deeper:
            if d <= stack[-1]:
                d = stack[-1] + 4
            stack.append(d)
        else:
            if d > stack[-1]:
                d = stack[-1]
            while len(stack) > 1 and d < stack[-1]:
                stack.pop()
            if d != stack[-1]:
                d = stack[-1]
        expect_deeper = _is_block_opener(line.rstrip("\n"))
        new = " " * d + body
        if new != line:
            changed = True
        out.append(new)
    return "".join(out) if changed else None


# H5  retype_mismatched_closer        error-driven, 1-char retype
def retype_mismatched_closer(src, err):
    if err is None or err.lineno is None or "does not match opening" not in (err.msg or ""):
        return None
    m = re.search(r"closing parenthesis '(.)' does not match opening parenthesis '(.)'", err.msg)
    if not m:
        return None
    want = _OPEN[m.group(2)]
    lines = src.splitlines(keepends=True)
    ln = err.lineno - 1
    if not (0 <= ln < len(lines)):
        return None
    body = lines[ln].rstrip("\n")
    col = (err.offset or 0) - 1
    if not (0 <= col < len(body)) or body[col] != m.group(1):
        return None
    lines[ln] = body[:col] + want + body[col + 1:] + "\n"
    return "".join(lines)


# H6  drop_surplus_closer             error-driven, deletes 1 delimiter char
def drop_surplus_closer(src, err):
    if err is None or err.lineno is None or "unmatched" not in (err.msg or ""):
        return None
    lines = src.splitlines(keepends=True)
    ln = err.lineno - 1
    if not (0 <= ln < len(lines)):
        return None
    body = lines[ln].rstrip("\n")
    col = (err.offset or 0) - 1
    if not (0 <= col < len(body)) or body[col] not in _CLOSE:
        return None
    lines[ln] = body[:col] + body[col + 1:] + "\n"
    return "".join(lines)


# H7  fill_empty_dict_slot            error-driven, SYNTHESIZES `None`
_EMPTY_VAL = re.compile(r":\s*(?=[,}\]])")
_EMPTY_KEY = re.compile(r"(?<=[{,])\s*:")


def fill_empty_dict_slot(src, err):
    if err is None or err.lineno is None or "dictionary key" not in (err.msg or "").lower():
        return None
    lines = src.splitlines(keepends=True)
    ln = err.lineno - 1
    if not (0 <= ln < len(lines)):
        return None
    body = lines[ln].rstrip("\n")
    new = _EMPTY_VAL.sub(": None", body, count=1)
    if new == body:
        new = _EMPTY_KEY.sub(" None:", body, count=1)
    if new == body:
        return None
    lines[ln] = new + "\n"
    return "".join(lines)


# H8  decorator_call_rejoin           GLOBAL-ish, rejoins a mangled call
_DECOR = re.compile(r"^(\s*)@([A-Za-z_][\w.]*)\s*$")


def decorator_call_rejoin(src, err=None):
    lines = src.splitlines(keepends=True)
    out, changed, i = list(lines), False, 0
    while i < len(lines) - 1:
        m = _DECOR.match(lines[i].rstrip("\n"))
        if not m:
            i += 1
            continue
        nxt = lines[i + 1].rstrip("\n")
        stack, q = _scan_line(nxt)
        surplus = [c for c in stack if c in _CLOSE]
        if q or surplus != [")"]:
            i += 1
            continue
        callee, sp = m.group(2), _split_top_level_assign(nxt)
        if sp:
            out[i + 1] = f"{_lws(nxt)}{sp[0].strip()} = {callee}({sp[1][1:].strip()}\n"
        else:
            out[i + 1] = f"{_lws(nxt)}{callee}({nxt.strip()}\n"
        out[i] = None
        changed = True
        i += 2
    return "".join(x for x in out if x is not None) if changed else None


# H9  strip_juxtaposed_number         error-driven, DELETES a stray token
_JUXT = re.compile(r"(?<=[\w\)\]\}])\s+(\d+)(?=\s*(?:[<>=!+\-*/%|&^]|\)|\]|:|,|$))")


def strip_juxtaposed_number(src, err):
    if err is None or err.lineno is None:
        return None
    lines = src.splitlines(keepends=True)
    ln = err.lineno - 1
    if not (0 <= ln < len(lines)):
        return None
    body = lines[ln].rstrip("\n")
    if _scan_line(body)[1]:
        return None
    new = _JUXT.sub("", body, count=1)
    if new == body:
        return None
    lines[ln] = new + "\n"
    return "".join(lines)


# H10 close_orphan_try                GLOBAL, SYNTHESIZES `except Exception: pass`
_TRY = re.compile(r"^(\s*)try\s*:")
_HANDLER = re.compile(r"^\s*(except|finally)\b")


def close_orphan_try(src, err=None):
    lines = src.splitlines(keepends=True)
    inserts = []
    for i, line in enumerate(lines):
        m = _TRY.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        end, handled = len(lines), False
        for j in range(i + 1, len(lines)):
            st = lines[j].strip()
            if not st or st.startswith("#"):
                continue
            cur = _ind(lines[j])
            if cur > indent:
                continue
            if cur == indent and _HANDLER.match(lines[j]):
                handled = True
            end = j
            break
        if not handled:
            inserts.append((end, indent))
    if not inserts:
        return None
    for pos, indent in sorted(inserts, reverse=True):
        pad = " " * indent
        lines[pos:pos] = [f"{pad}except Exception:\n", f"{pad}    pass\n"]
    return "".join(lines)


# H11 retype_try_else_to_except       GLOBAL, keyword rename
def retype_try_else_to_except(src, err=None):
    lines = src.splitlines(keepends=True)
    out, changed = list(lines), False
    for i, line in enumerate(lines):
        m = _TRY.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        for j in range(i + 1, len(lines)):
            st = lines[j].strip()
            if not st or st.startswith("#"):
                continue
            cur = _ind(lines[j])
            if cur > indent:
                continue
            if cur == indent and re.match(r"^\s*else\s*:", lines[j]):
                out[j] = " " * indent + "except Exception:" + lines[j].rstrip()[
                    lines[j].strip().index(":") + _ind(lines[j]) + 1:] + "\n"
                changed = True
            break
    return "".join(out) if changed else None


# H12 append_missing_colon            GLOBAL, append-only
_COMPOUND = re.compile(r"^\s*(if|elif|else|for|while|with|try|except|finally|def|class|"
                       r"async\s+def|async\s+for|async\s+with)\b")


def append_missing_colon(src, err=None):
    lines = src.splitlines(keepends=True)
    starts = line_starts(src)
    out, changed = list(lines), False
    for i, line in enumerate(lines):
        body = line.rstrip("\n")
        if not body.strip() or not _COMPOUND.match(body):
            continue
        if i < len(starts) and not starts[i]:
            continue
        code = _code_only(body)
        stack, q = _scan_line(body)
        if q or stack or ":" in code:
            # `:` anywhere at top level means the header already has one, possibly with
            # an inline body (`if False: yield`) -- appending a second `:` breaks it.
            continue
        out[i] = body.rstrip() + ":\n"
        changed = True
    return "".join(out) if changed else None


# driver
GLOBALS_ = [close_unclosed_lines, invalid_lhs_rename, decorator_call_rejoin,
            append_missing_colon, retype_try_else_to_except,
            orphan_clause_header, close_orphan_try, normalize_indent]
LOCALS_ = [retype_mismatched_closer, drop_surplus_closer, fill_empty_dict_slot,
           strip_juxtaposed_number]


def pipeline(src, globals_=None, locals_=None, max_rounds=200):
    globals_ = GLOBALS_ if globals_ is None else globals_
    locals_ = LOCALS_ if locals_ is None else locals_
    cur, fired = src, []
    for _ in range(max_rounds):
        changed = False
        for op in globals_:
            try:
                cand = op(cur, None)
            except Exception:
                cand = None
            if cand is not None and cand != cur:
                cur, changed = cand, True
                fired.append(op.__name__)
        before = probe(cur)
        if before is None:
            return cur, True, fired
        for op in locals_:
            try:
                cand = op(cur, before)
            except Exception:
                cand = None
            if cand is None or cand == cur:
                continue
            after = probe(cand)
            if after is None or advanced(before, after):
                cur, changed = cand, True
                fired.append(op.__name__)
                break
        if not changed:
            break
    return cur, probe(cur) is None, fired


def fixpoint(src, ops, max_rounds=200):
    """Error-driven: accept a candidate only if it compiles or strictly advances."""
    cur, fired = src, []
    for _ in range(max_rounds):
        before = probe(cur)
        if before is None:
            return cur, True, fired
        got = False
        for op in ops:
            try:
                cand = op(cur, before)
            except Exception:
                cand = None
            if cand is None or cand == cur:
                continue
            after = probe(cand)
            if after is None or advanced(before, after):
                cur, got = cand, True
                fired.append(op.__name__)
                break
        if not got:
            return cur, False, fired
    return cur, probe(cur) is None, fired
