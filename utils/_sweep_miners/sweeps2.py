import re
from .ops4 import _indent
from .sweeps import _suite_end, _nb

_FORHDR = re.compile(r"^(\s*)(async\s+for|for)\b")




def toplevel_kw_positions(text, words):
    """Positions of bare keyword tokens at bracket depth 0, outside string literals."""
    res, stack, in_s, q, esc = [], [], False, "", False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_s:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == q: in_s = False
            i += 1; continue
        if ch in "'\"":
            in_s, q = True, ch; i += 1; continue
        if ch in "([{":
            stack.append(ch); i += 1; continue
        if ch in ")]}":
            if stack: stack.pop()
            i += 1; continue
        if not stack and (ch.isalpha() or ch == "_"):
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            w = text[i:j]
            if w in words and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_.")):
                res.append((i, w))
            i = j; continue
        i += 1
    return res

def sweep_split_flattened_for(source, error=None):
    """Re-nest a comprehension that PyLingual emitted as ONE statement header.

        for cpku in range(len(lin)) for cpkuni in lin[nocp].split('|'):
        for _file in files if _file.endswith('.sqlite'):

    PyLingual flattens a nested loop / filtered loop onto a single `for` line, which is never
    legal Python. Splitting at each depth-0 `for`/`if` keyword restores the nesting. Every
    token of the original line is kept -- only newlines and indentation are introduced -- and
    the original suite is shifted right to stay inside the innermost clause."""
    try:
        lines = source.splitlines(keepends=True)
        out = []
        changed = False
        i = 0
        while i < len(lines):
            l = lines[i]
            m = _FORHDR.match(l)
            t = l.rstrip("\n").rstrip()
            if not m or not t.endswith(":"):
                out.append(l); i += 1; continue
            body = t[:-1]
            kws = toplevel_kw_positions(body, {"for", "if"})
            if len(kws) < 2:
                out.append(l); i += 1; continue
            pad = m.group(1)
            cuts = [p for p, _ in kws[1:]]
            segs, prev = [], 0
            for p in cuts:
                segs.append(body[prev:p].rstrip())
                prev = p
            segs.append(body[prev:].rstrip())
            if any(not s.strip() for s in segs):
                out.append(l); i += 1; continue
            for k, s in enumerate(segs):
                out.append(f"{pad}{'    ' * k}{s.strip() if k else s.strip()}:\n")
            hind = _indent(l)
            end = _suite_end(lines, i, hind)
            shift = "    " * len(cuts)
            for j in range(i + 1, end):
                out.append((shift + lines[j]) if lines[j].strip() else lines[j])
            changed = True
            i = end
        return "".join(out) if changed else None
    except Exception:
        return None
