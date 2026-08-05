import re, ast, keyword, sys

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def _leading_ws(s):
    return s[: len(s) - len(s.lstrip(" \t"))]


def _indent(s):
    return len(_leading_ws(s.expandtabs(8)))


def _parses(src):
    try:
        ast.parse(src)
        return True
    except Exception:
        return False


# N1: nonidentifier_lhs_rename  (generalises literal_lhs_rename)
# `1e:6c:34:93:68:64 = [...]`, `self.3c:ec:ef:44:01:aa = [...]`,
# `7AB5C494-39F5-4941-9163-47F54D6D5016.False = [...]`
_ASSIGN = re.compile(r"^([ \t]*)([^=\n]+?)([ \t]*=[ \t]*)(?!=)(\S[^\n]*)(\n?)$")
# a target made only of "constant-ish" characters: hex/word chunks glued by : - . and digits
_CONSTISH = re.compile(r"^[A-Za-z0-9_.:\-]+$")


# N2: align_dangling_handler  (whitespace only)
_TRY = re.compile(r"^(\s*)try\s*:\s*(#.*)?$")
_HANDLER = re.compile(r"^(\s*)(except\b|finally\s*:)")
_DEDENT_STOP = re.compile(r"^(\s*)(def |class |async def )")


# N4: truncate_partial_element  (DELETES a partial trailing element)
def _bracket_stack(text):
    stack, in_s, q, esc = [], False, "", False
    for ch in text:
        if in_s:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                in_s = False
            continue
        if ch in "'\"":
            in_s, q = True, ch
        elif ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE and stack and stack[-1] == _CLOSE[ch]:
            stack.pop()
    return stack, in_s


def _comma_positions(text):
    res, stack, in_s, q, esc = [], [], False, "", False
    for i, ch in enumerate(text):
        if in_s:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                in_s = False
            continue
        if ch in "'\"":
            in_s, q = True, ch
        elif ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE and stack and stack[-1] == _CLOSE[ch]:
            stack.pop()
        elif ch == ",":
            res.append((i, len(stack)))
    return res


_BLOCK_KW = re.compile(r"^\s*(if|elif|while|for|with|except|async\s+with|async\s+for)\b")


def _close_line(text):
    stack, _ = _bracket_stack(text)
    out = text + "".join(_OPEN[o] for o in reversed(stack))
    if _BLOCK_KW.match(out) and not out.rstrip().endswith(":"):
        out = out.rstrip() + ":"
    return out


def truncate_partial_element(source, error=None):
    try:
        if error is None or error.lineno is None:
            return None
        lines = source.splitlines(keepends=True)
        ln = error.lineno - 1
        if not (0 <= ln < len(lines)):
            return None
        text = lines[ln].rstrip("\n")
        stack, in_s = _bracket_stack(text)
        if not stack and not in_s:
            return None
        commas = _comma_positions(text)
        if not commas:
            return None
        maxd = max(d for _, d in commas)
        for depth in range(maxd, 0, -1):
            cands = [i for i, d in commas if d == depth]
            if not cands:
                continue
            cut = cands[-1]
            new = _close_line(text[:cut])
            if new == text:
                continue
            trial = list(lines)
            trial[ln] = new + "\n"
            cand = "".join(trial)
            if cand != source:
                return cand
        return None
    except Exception:
        return None


# N5: add_missing_block_colon
_HEADER = re.compile(r"^\s*(if|elif|else|while|for|try|except|finally|with|def|class|async\s+def|async\s+with|async\s+for)\b")




def _sanitize_target(t):
    parts = t.split(".")
    out = []
    for p in parts:
        q = re.sub(r"[^0-9A-Za-z_]", "_", p)
        if not q or not (q[0].isalpha() or q[0] == "_") or keyword.iskeyword(q):
            q = "_lit_" + q
        out.append(q)
    return ".".join(out)

def nonidentifier_lhs_rename(source, error=None):
    """Rename an assignment TARGET that can never be a legal Python name.

    PyLingual emits recovered dict keys / MAC addresses / GUIDs in the target position
    (`1e:6c:34:93:68:64 = [...]`, `self.3c:ec:ef:44:01:aa = [...]`). Only the *name token*
    is rewritten -- the right-hand side, which carries all the analytic content, is kept
    byte-for-byte. Fires only when the original target is un-parseable as a target, so it
    can never change the meaning of legal code."""
    try:
        if not source:
            return None
        out, changed = [], False
        for line in source.splitlines(keepends=True):
            m = _ASSIGN.match(line)
            done = False
            if m:
                indent, target, eq, rhs, nl = m.groups()
                t = target.strip()
                if t and _CONSTISH.match(t) and not _parses(t + " = None"):
                    new = _sanitize_target(t)
                    if new != t and _parses(new + " = None"):
                        out.append(f"{indent}{new}{eq}{rhs}{nl}")
                        changed = True
                        done = True
            if not done:
                out.append(line)
        return "".join(out) if changed else None
    except Exception:
        return None

def _logical_lines(lines):
    """Indices of lines that begin a logical line (crudely: non-blank, non-comment)."""
    return [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")]

def _find_unterminated_try(lines, upto):
    """Last `try:` at index < upto whose suite has no matching handler at its own indent."""
    best = None
    for i in range(min(upto, len(lines))):
        m = _TRY.match(lines[i].rstrip("\n"))
        if not m:
            continue
        ind = _indent(lines[i])
        handled = False
        for j in range(i + 1, len(lines)):
            if not lines[j].strip():
                continue
            cur = _indent(lines[j])
            if cur > ind:
                continue
            h = _HANDLER.match(lines[j])
            if cur == ind and h:
                handled = True
            break
        if not handled:
            best = i
    return best

def align_dangling_handler(source, error=None):
    """Re-indent an `except:`/`finally:` clause onto the `try:` it belongs to.

    PyLingual routinely emits the handler at the WRONG indentation (`try:` at column 12,
    its `except Exception:` at column 8, or at column 28). Python then reports
    "expected 'except' or 'finally' block" and the delete-only fallback throws away the
    entire try suite. The repair is pure leading whitespace: no token is added, removed
    or reordered."""
    try:
        if not error or not error.msg or "except" not in error.msg or "finally" not in error.msg:
            return None
        lines = source.splitlines(keepends=True)
        t = _find_unterminated_try(lines, error.lineno or len(lines))
        if t is None:
            return None
        tind = _indent(lines[t])
        # find the next handler line ANYWHERE after the try, at any indent
        h = None
        for j in range(t + 1, len(lines)):
            if not lines[j].strip():
                continue
            if _HANDLER.match(lines[j]):
                h = j
                break
            # do not cross a def/class header that is at or outside the try's indent
            if _DEDENT_STOP.match(lines[j]) and _indent(lines[j]) <= tind:
                break
        if h is None:
            return None
        hind = _indent(lines[h])
        if hind == tind:
            return None                      # already aligned; a different defect
        delta = tind - hind
        # the handler clause = handler line + every following line indented deeper than it
        end = len(lines)
        for j in range(h + 1, len(lines)):
            if not lines[j].strip():
                continue
            if _indent(lines[j]) <= hind and not _HANDLER.match(lines[j]):
                end = j
                break
            if _indent(lines[j]) < hind:
                end = j
                break
        out = list(lines)
        for j in range(h, end):
            if not out[j].strip():
                continue
            cur = _indent(out[j])
            new = max(0, cur + delta)
            out[j] = " " * new + out[j].lstrip()
        return "".join(out)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# N3: absorb_underindented_try_body  (whitespace only)
# ---------------------------------------------------------------------------
def absorb_underindented_try_body(source, error=None):
    """Pull statements that fell OUT of a `try:` suite back into it.

    PyLingual sometimes drops the indentation of the tail of a try suite to the `try:`'s own
    column (or below), so the suite ends early and the later `except:` is orphaned. Every
    statement between the truncation point and the handler is re-indented back to the suite's
    column. Whitespace only -- no statement is added, deleted or moved."""
    try:
        if not error or not error.msg or "except" not in error.msg or "finally" not in error.msg:
            return None
        lines = source.splitlines(keepends=True)
        t = _find_unterminated_try(lines, error.lineno or len(lines))
        if t is None:
            return None
        tind = _indent(lines[t])
        # body indent = indent of first non-blank line after the try:
        b = None
        for j in range(t + 1, len(lines)):
            if lines[j].strip():
                b = j
                break
        if b is None:
            return None
        bind = _indent(lines[b])
        if bind <= tind:
            return None
        # find where the suite breaks (first line with indent <= tind) and the handler after it
        brk = None
        for j in range(b, len(lines)):
            if not lines[j].strip():
                continue
            if _indent(lines[j]) <= tind:
                brk = j
                break
        if brk is None:
            return None
        if _HANDLER.match(lines[brk]):
            return None                     # that's N2's job
        h = None
        for j in range(brk, len(lines)):
            if not lines[j].strip():
                continue
            if _HANDLER.match(lines[j]):
                h = j
                break
            if _DEDENT_STOP.match(lines[j]) and _indent(lines[j]) <= tind:
                break
        if h is None:
            return None
        out = list(lines)
        for j in range(brk, h):
            if not out[j].strip():
                continue
            out[j] = " " * bind + out[j].lstrip()
        # and align the handler itself onto the try
        hind = _indent(out[h])
        if hind != tind:
            delta = tind - hind
            end = len(out)
            for j in range(h + 1, len(out)):
                if not out[j].strip():
                    continue
                if _indent(out[j]) <= hind:
                    end = j
                    break
            for j in range(h, end):
                if not out[j].strip():
                    continue
                out[j] = " " * max(0, _indent(out[j]) + delta) + out[j].lstrip()
        return "".join(out)
    except Exception:
        return None

def add_missing_block_colon(source, error=None):
    """Append the `:` PyLingual dropped from a compound-statement header.

    Fires only when the header's brackets are balanced, the line has no trailing `:` and the
    NEXT non-blank line is indented deeper -- i.e. the file already treats it as a block opener.
    One punctuation character; no statement is added or removed."""
    try:
        if error is None or error.lineno is None:
            return None
        lines = source.splitlines(keepends=True)
        ln = error.lineno - 1
        if not (0 <= ln < len(lines)):
            return None
        text = lines[ln].rstrip("\n")
        if not _HEADER.match(text) or text.rstrip().endswith(":"):
            return None
        stack, in_s = _bracket_stack(text)
        if stack or in_s:
            return None
        nxt = None
        for j in range(ln + 1, len(lines)):
            if lines[j].strip():
                nxt = j
                break
        if nxt is None or _indent(lines[nxt]) <= _indent(lines[ln]):
            return None
        lines[ln] = text.rstrip() + ":\n"
        return "".join(lines)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# N6: drop_extra_closer  (deletes a stray ')' )
# ---------------------------------------------------------------------------
def drop_extra_closer(source, error=None):
    """Remove a single unmatched closing bracket that PyLingual emitted twice.

    `unmatched ')'` on lines like `encrypted_data))`. Deletes exactly one punctuation
    character, never a statement."""
    try:
        if not error or not error.msg or "unmatched" not in error.msg.lower():
            return None
        if error.lineno is None:
            return None
        lines = source.splitlines(keepends=True)
        ln = error.lineno - 1
        if not (0 <= ln < len(lines)):
            return None
        text = lines[ln].rstrip("\n")
        col = (error.offset - 1) if error.offset else None
        if col is None or not (0 <= col < len(text)) or text[col] not in _CLOSE:
            # fall back: strip the last closer on the line
            for k in range(len(text) - 1, -1, -1):
                if text[k] in _CLOSE:
                    col = k
                    break
            else:
                return None
        lines[ln] = text[:col] + text[col + 1:] + "\n"
        return "".join(lines)
    except Exception:
        return None
