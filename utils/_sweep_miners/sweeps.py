import re
from .ops4 import _indent, _leading_ws, _bracket_stack, _OPEN, _CLOSE

_TRY = re.compile(r"^\s*try\s*:\s*(#.*)?$")
_HANDLER = re.compile(r"^\s*(except\b|finally\s*:)")
_POSTINS = re.compile(r"^\s*pass\s*#\s*postinserted\s*$")
_ELSEINS = re.compile(r"^\s*else\s*:\s*#\s*inserted\s*$")
_OPENER = re.compile(r":\s*(#.*)?$")


def _nb(lines, j, step, stop=None):
    while 0 <= j < len(lines):
        if lines[j].strip():
            return j
        j += step
    return None


def _suite_end(lines, i, indent):
    end = len(lines)
    for j in range(i + 1, len(lines)):
        if not lines[j].strip():
            continue
        if _indent(lines[j]) <= indent:
            end = j
            break
    while end > i + 1 and not lines[end - 1].strip():
        end -= 1
    return end


# S3: sweep_fold_postinserted_handler_body   (WHITESPACE ONLY)
def sweep_fold_postinserted_handler_body(source, error=None):
    try:
        lines = source.splitlines(keepends=True)
        out = list(lines)
        changed = False
        for i, l in enumerate(lines):
            if not _HANDLER.match(l):
                continue
            I = _indent(l)
            b = _nb(lines, i + 1, 1)
            if b is None or not _POSTINS.match(lines[b]) or _indent(lines[b]) <= I:
                continue
            end = _suite_end(lines, i, I)
            # the run of lines at indent I after the handler suite, terminated by `else:  # inserted`
            j = end
            run = []
            while j < len(lines):
                if not lines[j].strip():
                    j += 1
                    continue
                if _indent(lines[j]) != I:
                    break
                if _ELSEINS.match(lines[j]):
                    break
                if _HANDLER.match(lines[j]) or _TRY.match(lines[j].rstrip("\n")):
                    break
                run.append(j)
                j += 1
            if not run:
                continue
            nxt = _nb(lines, j, 1)
            if nxt is None or not _ELSEINS.match(lines[nxt]):
                continue                       # require the decompiler's own terminator
            for k in range(run[0], j):
                if out[k].strip():
                    out[k] = "    " + out[k]
                    changed = True
        return "".join(out) if changed else None
    except Exception:
        return None


# S4: sweep_align_handlers   (WHITESPACE ONLY)
def sweep_align_handlers(source, error=None):
    try:
        return _sweep_align_handlers(source, error)
    except Exception:
        return None


def _sweep_align_handlers(source, error=None):
    lines = source.splitlines(keepends=True)
    out = list(lines)
    changed = False
    i = 0
    while i < len(out):
        l = out[i]
        if not _TRY.match(l.rstrip("\n")):
            i += 1
            continue
        T = _indent(l)
        end = _suite_end(out, i, T)
        if end <= i + 1:
            i += 1
            continue
        nxt = _nb(out, end, 1)
        if nxt is None:
            i += 1
            continue
        if _indent(out[nxt]) == T and _HANDLER.match(out[nxt]):
            i += 1
            continue                       # already fine
        # find the first handler below, without crossing another try: at indent <= T
        h = None
        for j in range(end, len(out)):
            if not out[j].strip():
                continue
            if _HANDLER.match(out[j]):
                h = j
                break
            if _TRY.match(out[j].rstrip("\n")) and _indent(out[j]) <= T:
                break
            if re.match(r"^\s*(def |class |async def )", out[j]) and _indent(out[j]) <= T:
                break
        if h is None:
            i += 1
            continue
        H = _indent(out[h])
        if H == T:
            i += 1
            continue
        hend = _suite_end(out, h, H)
        delta = T - H
        for j in range(h, hend):
            if out[j].strip():
                out[j] = " " * max(0, _indent(out[j]) + delta) + out[j].lstrip()
        changed = True
        i += 1
    return "".join(out) if changed else None


# S5: sweep_nonidentifier_lhs  (rename-only; already global in ops4)


def _apply_inserts(lines, inserts):
    ins = {}
    for idx, txt in inserts:
        ins.setdefault(idx, []).append(txt)
    out = []
    for j, l in enumerate(lines):
        if j in ins:
            out.extend(ins[j])
        out.append(l)
    if len(lines) in ins:
        out.extend(ins[len(lines)])
    return out

# ---------------------------------------------------------------------------
# S1: sweep_close_orphan_try   (SYNTHESIZES `except Exception: pass`)
# ---------------------------------------------------------------------------
def sweep_close_orphan_try(source, error=None):
    """Give EVERY handler-less `try:` in the file an `except Exception: pass`.

    Global version of the existing single-shot `complete_orphan_try`. Measured: 205/272 of the
    hard-residual malware files contain a handler-less `try:` and they average ~5 of them, so
    the single-shot operator can never converge -- the parser only reports the first one."""
    try:
        lines = source.splitlines(keepends=True)
        inserts = []
        for i, l in enumerate(lines):
            if not _TRY.match(l.rstrip("\n")):
                continue
            I = _indent(l)
            end = _suite_end(lines, i, I)
            if end <= i + 1:
                continue                      # empty suite -> a different defect
            nxt = _nb(lines, end, 1)
            if nxt is not None and _indent(lines[nxt]) == I and _HANDLER.match(lines[nxt]):
                continue                      # already handled
            pad = " " * I
            inserts.append((end, f"{pad}except Exception:\n{pad}    pass\n"))
        if not inserts:
            return None
        return "".join(_apply_inserts(lines, inserts))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# S2: sweep_insert_missing_try   (SYNTHESIZES a `try:` line)
# ---------------------------------------------------------------------------
def _stmt_start(lines, p, I):
    """First line index of the logical statement that ends at line p (indent I)."""
    s = p
    while s >= 0:
        if _indent(lines[s]) == I:
            st, ins_ = _bracket_stack("".join(lines[s:p + 1]))
            if not st and not ins_:
                return s
        s -= 1
    return p

def sweep_insert_missing_try(source, error=None):
    """Restore the `try:` PyLingual dropped in front of a dangling `except:`/`finally:`.

    Canonical malware damage (188/272 hard-residual files, 1387 occurrences):

        hostname_pc = socket.gethostname()
        except:
            pass  # postinserted

    The handler is emitted but its `try:` never is, so the delete-only fallback throws away the
    protected statement AND the handler. This wraps the immediately-preceding logical statement
    in a `try:`; nothing is deleted, one scaffold line is added per site."""
    try:
        lines = source.splitlines(keepends=True)
        inserts, shifts = [], []
        for i, l in enumerate(lines):
            if not _HANDLER.match(l):
                continue
            I = _indent(l)
            # dangling? scan back inside the same suite for a `try:` at indent I
            found = False
            for j in range(i - 1, -1, -1):
                if not lines[j].strip():
                    continue
                k = _indent(lines[j])
                if k < I:
                    break
                if k == I:
                    if _TRY.match(lines[j].rstrip("\n")):
                        found = True
                    if _HANDLER.match(lines[j]):
                        continue
                    break
            if found:
                continue
            p = _nb(lines, i - 1, -1)
            if p is None or _indent(lines[p]) != I:
                continue
            if _OPENER.search(lines[p].rstrip("\n")):
                continue                       # a block header, not a protectable statement
            s = _stmt_start(lines, p, I)
            pad = " " * I
            inserts.append((s, f"{pad}try:\n"))
            shifts.append((s, p + 1))
        if not inserts:
            return None
        out = list(lines)
        for a, b in shifts:
            for j in range(a, b):
                if out[j].strip():
                    out[j] = "    " + out[j]
        return "".join(_apply_inserts(out, inserts))
    except Exception:
        return None
