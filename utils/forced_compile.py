"""Recursive-descent forced compilation for decompiled malware (implementer #2).

WHY THIS EXISTS. The chunk-level design (``chunk_neutralize``) neutralises a WHOLE top-level chunk
the moment any line inside it fails to repair. On a file that is one 1544-line ``def rcd():`` that
throws away the entire program over a single broken ``except`` -- 0.5% of the code survives.

THE FIX IS RECURSIVE DESCENT. When a compound statement (def/class/if/for/while/with/try/match) fails
to parse but its HEADER is sound, keep the header and recurse into each clause body. Only the broken
SUB-statement -- at whatever depth it lives -- is neutralised; its siblings and every enclosing header
survive. Recursion is bounded by nesting depth (guarded anyway), so it terminates in O(file size).

TWO DEFECT CLASSES BEYOND PLAIN PARSING:

* The PyLingual "postinserted" bug emits ``except``/``finally``/``else``/``elif`` one indent level too
  deep -- at the try/if BODY indent instead of the header indent. Neutralising it would wipe the
  (often huge) ``else`` body. We instead PROMOTE such an orphan clause back to its opener's indent,
  but only when no nested compound at that level can legitimately own it -- so a nested ``if``'s
  ``else`` is never stolen by the enclosing ``try``.

* ``return`` outside a function and ``continue``/``break`` outside a loop PARSE but do not COMPILE.
  ``ast.parse`` cannot see them; ``compile`` can. The final gate is therefore ``compile``: any
  compile-only error has its single offending statement neutralised until the module compiles.

HARD RULES honoured: never delete a physical line (neutralised code becomes a ``repr`` string padded
with blanks to keep the line count and thus line numbering stable); every byte survives IN the output
and is greppable; each hole is tagged ``# pyllmpatch: unparsed``.
"""
from __future__ import annotations

import ast
import re

from utils.gt_literal_recovery import load_gt_literals, repair_source
from utils.structural_closure import (
    close_source,
    close_unterminated_string,
    close_line_brackets,
)
from utils.chunk_neutralize import force_parseable

__all__ = ["force_compile", "recursive_salvage"]

MARKER = "# pyllmpatch: unparsed"
_MAX_DEPTH = 80

_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OPEN, _CLOSE = "([{", ")]}"

CONT_KW = {"elif", "else", "except", "finally", "case"}
OPENER_KW = {"if", "for", "while", "with", "try", "def", "class", "match", "async"}
# Keywords (and the decorator '@') that can only begin a statement, never continue a bracketed
# expression. Used to break out of a run of UNCLOSED brackets left by a truncated line -- otherwise a
# decompiler-truncated ``{`` or ``(`` swallows the rest of the file into one statement.
STMT_START_KW = OPENER_KW | {
    "import", "from", "return", "raise", "yield", "pass", "break", "continue",
    "global", "nonlocal", "del", "assert", "await", "elif", "else", "except", "finally", "case",
}


# --------------------------------------------------------------------------- primitives
def _mask(line):
    """Blank complete strings and drop a trailing comment, so bracket/colon scans see structure only."""
    m = _STRING.sub(lambda mo: " " * len(mo.group(0)), line)
    h = m.find("#")
    return m[:h] if h != -1 else m


def _indent(line):
    return len(line) - len(line.lstrip())


def _first_word(s):
    m = _WORD.match(s.lstrip())
    return m.group(0) if m else ""


def _is_stmt_start(s):
    """A stripped line that can only BEGIN a statement (keyword-led or a decorator)."""
    return s.startswith("@") or _first_word(s) in STMT_START_KW


def _has_top_level_assign(s):
    """True if the (masked) line carries a statement-level ``=``/aug-assign outside all brackets."""
    m = _mask(s)
    depth = 0
    for i, ch in enumerate(m):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            prev = m[i - 1] if i else " "
            nxt = m[i + 1] if i + 1 < len(m) else " "
            if prev not in "=!<>:" and nxt != "=":
                return True
    return False


def _is_assign_start(s):
    """A name/target-led assignment statement -- used to break out of a truncated bracket run."""
    return bool(s) and (s[0].isalnum() or s[0] == "_") and _has_top_level_assign(s)


def _min_indent(lines):
    nb = [l for l in lines if l.strip()]
    return min((_indent(l) for l in nb), default=0)


def _first_nonblank_idx(lines):
    for i, l in enumerate(lines):
        if l.strip():
            return i
    return None


def _parses(text):
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def _dedent(lines, d):
    return "\n".join(l[d:] if l.strip() else "" for l in lines)


def _parses_as_suite(lines):
    """A suite parses iff, dedented by its minimum indent, it parses as a module."""
    if not any(l.strip() for l in lines):
        return True
    return _parses(_dedent(lines, _min_indent(lines)))


def _cont_for(opener_kw, header_str):
    if opener_kw == "if":
        return {"elif", "else"}
    if opener_kw in ("for", "while"):
        return {"else"}
    if opener_kw == "try":
        return {"except", "else", "finally"}
    if opener_kw == "match":
        return {"case"}
    if opener_kw == "async":
        parts = header_str.split()
        return {"else"} if len(parts) > 1 and parts[1] == "for" else set()
    return set()  # def, class, with


def _neutralise(lines):
    """Render a region as one inert ``repr`` string at its own indent, padded to keep the line count.

    ``repr`` escapes everything, so arbitrary malware text cannot break out of the literal. The bare
    string constant emits no bytecode, the marker is a comment (also no bytecode), and every original
    byte survives inside the literal.
    """
    indent = " " * _min_indent(lines)
    payload = "\n".join(lines)
    out = [f"{indent}{payload!r}  {MARKER}"]
    out.extend([""] * (len(lines) - 1))
    return out


def _count_markers(lines):
    return sum(1 for l in lines if MARKER in l)


# --------------------------------------------------------------------------- statement splitting
def _split_statements(lines, ci):
    """Split a suite into logical statements whose headers sit at indent ``ci``.

    Bracket depth and backslash continuation are respected. A continuation clause keyword
    (elif/else/except/finally/case) at ``ci`` attaches to the previous statement rather than starting
    a new one -- so a whole ``try/except/else`` is one logical statement.
    """
    spans = []
    start = None
    depth = 0
    cont = False
    for i, raw in enumerate(lines):
        s = raw.strip()
        at_ci = bool(s) and not cont and _indent(raw) == ci
        # A boundary when brackets are balanced; also when a truncated sibling left brackets open but
        # this line can only be a fresh statement -- then reset depth so the defect stays local.
        boundary = at_ci and depth == 0
        if at_ci and depth > 0 and (_is_stmt_start(s) or _is_assign_start(s)):
            boundary = True
            depth = 0
        if boundary:
            kw = _first_word(s)
            if kw in CONT_KW and start is not None:
                pass  # continuation clause -> stays with current statement
            else:
                if start is not None:
                    spans.append((start, i - 1))
                start = i
        elif start is None and s:
            start = i
        masked = _mask(raw)
        for ch in masked:
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth = max(0, depth - 1)
        cont = masked.rstrip().endswith("\\")
    if start is not None:
        spans.append((start, len(lines) - 1))
    return spans


def _consume_header(lines, i):
    """From line ``i``, consume the physical lines of a compound header until its ``:`` closes.

    Returns ``(header_idxs, index_after_header)``. A header with no closing colon (broken) returns the
    lines scanned and the end index -- the skeleton check then rejects it.
    """
    depth = 0
    cont = False
    idxs = []
    j = i
    n = len(lines)
    base = _indent(lines[i])
    while j < n:
        # Stop before swallowing a fresh statement at or above the header's indent (truncated header).
        if j > i and lines[j].strip() and not cont and _indent(lines[j]) <= base and _is_stmt_start(lines[j].strip()):
            break
        idxs.append(j)
        masked = _mask(lines[j])
        for ch in masked:
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth = max(0, depth - 1)
        cont = masked.rstrip().endswith("\\")
        if depth == 0 and not cont and masked.rstrip().endswith(":"):
            return idxs, j + 1
        j += 1
    return idxs, j


def _split_clauses(lifted, base):
    """Split a compound statement into ``[(header_idxs, body_idxs)]`` clauses at indent ``base``."""
    n = len(lifted)
    i = _first_nonblank_idx(lifted)
    if i is None or _indent(lifted[i]) != base:
        return None
    clauses = []
    while i < n:
        hdr_idxs, after = _consume_header(lifted, i)
        body_idxs = []
        j = after
        while j < n:
            s = lifted[j].strip()
            if not s:
                body_idxs.append(j)
                j += 1
                continue
            if _indent(lifted[j]) <= base:
                break
            body_idxs.append(j)
            j += 1
        clauses.append((hdr_idxs, body_idxs))
        i = j
        if (
            i < n
            and lifted[i].strip()
            and _indent(lifted[i]) == base
            and _first_word(lifted[i].strip()) in CONT_KW
        ):
            continue
        break
    # Any trailing lines after the clause chain would mean the "statement" was mis-bounded.
    if i < n and any(lifted[k].strip() for k in range(i, n)):
        return None
    return clauses


def _skeleton_parses(lifted, clauses, base):
    """The clause HEADERS, with ``pass`` bodies, must COMPILE together.

    ``compile`` (not just ``ast.parse``) is used so a duplicate bare ``except`` -- accepted by the
    parser but rejected as ``default 'except:' must be last`` -- is caught here and the offending
    trailing clause is dropped, instead of surfacing later and costing the whole compound.
    """
    parts = []
    for hdr_idxs, _body in clauses:
        for hi in hdr_idxs:
            parts.append(lifted[hi][base:] if lifted[hi].strip() else "")
        parts.append("    pass")
    try:
        compile("\n".join(parts), "<skeleton>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


# --------------------------------------------------------------------------- orphan-clause promotion
def _promote_orphans(seg, base, conts):
    """Pull PyLingual's over-indented continuation clauses back to their opener's indent.

    Only clauses at the opener's DIRECT body indent are considered, and only when no nested compound
    already open at that indent can legitimately own them -- so a nested ``if``'s ``else`` is never
    stolen by the enclosing ``try``.
    """
    if not conts:
        return list(seg)
    out = list(seg)
    n = len(out)
    fi = _first_nonblank_idx(out)
    if fi is None:
        return out
    _hdr, after = _consume_header(out, fi)
    body_indent = None
    k = after
    while k < n:
        if out[k].strip():
            ind = _indent(out[k])
            if ind > base:
                body_indent = ind
            break
        k += 1
    if body_indent is None:
        return out

    i = after
    open_conts = set()
    while i < n:
        raw = out[i]
        s = raw.strip()
        if not s:
            i += 1
            continue
        ind = _indent(raw)
        if ind <= base:
            break
        if ind == body_indent:
            kw = _first_word(s)
            if kw in conts and kw not in open_conts:
                delta = body_indent - base
                hj, hafter = _consume_header(out, i)
                for hi in hj:
                    if out[hi].strip():
                        out[hi] = out[hi][delta:]
                j = hafter
                while j < n:
                    if out[j].strip():
                        if _indent(out[j]) <= body_indent:
                            break
                        out[j] = out[j][delta:]
                    j += 1
                break  # first body ended; remaining clauses handled by split + recursion
            if kw in OPENER_KW:
                open_conts = _cont_for(kw, s)
            elif kw in open_conts:
                pass  # continuing a nested compound -- keep its continuation set
            else:
                open_conts = set()
        i += 1
    return out


# --------------------------------------------------------------------------- recursion
def _salvage_suite(lines, depth):
    """Repair a suite: keep every statement that parses, recurse into the ones that do not."""
    if not any(l.strip() for l in lines):
        return list(lines), 0
    if _parses_as_suite(lines):
        return list(lines), 0
    if depth > _MAX_DEPTH:
        return _neutralise(lines), len(lines)
    ci = _min_indent(lines)
    out = list(lines)
    neut = 0
    for a, b in _split_statements(lines, ci):
        seg = lines[a : b + 1]
        if _parses_as_suite(seg):
            continue
        new_seg, sneut = _salvage_statement(seg, depth + 1)
        out[a : b + 1] = new_seg
        neut += sneut
    if not _parses_as_suite(out):
        out, extra = _harden_suite(out, ci)
        neut += extra
    return out, neut


def _salvage_statement(seg, depth):
    """Repair one logical statement. Compound -> recurse; simple/broken-header -> close or neutralise."""
    if _parses_as_suite(seg):
        return list(seg), 0
    fi = _first_nonblank_idx(seg)
    base = _min_indent(seg)
    kw = _first_word(seg[fi].strip())
    if kw in OPENER_KW:
        res = _salvage_compound(seg, base, depth)
        if res is not None:
            return res
        # A cheap line-local header repair (close a truncated string/bracket) before giving up.
        patched = _repair_header_lines(seg)
        if patched is not None:
            res = _salvage_compound(patched, base, depth)
            if res is not None:
                return res
        # Header genuinely unsound -> neutralise the whole compound as one inert unit.
        return _neutralise(seg), len(seg)
    return _close_or_neutralise(seg, base)


def _rewrite_try_to_if(line):
    """``try:`` -> ``if True:`` in place, preserving indent and any trailing comment."""
    m = re.match(r"^(\s*)try\s*:(.*)$", line)
    if not m:
        return line
    # keep every original byte as a comment (comments emit no bytecode, stay greppable)
    return f"{m.group(1)}if True:  {MARKER} was: {line.strip()}"


def _flatten_match(seg):
    """Rewrite a mangled ``match``/``case`` block to sibling ``if True:`` blocks, in place.

    PyLingual routinely emits ``case`` clauses at the ``match`` indent (not nested), a bogus ``pass``
    match body, and an ``else`` a match cannot have -- unrecoverable as a real ``match``. Rewriting the
    ``match`` and every ``case`` header to ``if True:`` keeps every body line executable and the exact
    line count; the block then re-splits into ordinary statements. ``.*:`` takes the final top-level
    colon, so patterns containing ``:`` (``case {'k': v}:``) collapse cleanly.
    """
    out = []
    for line in seg:
        m = re.match(r"^(\s*)(?:match|case)\b.*:(.*)$", line)
        if m and _first_word(line.strip()) in ("match", "case"):
            # keep the match-subject / case-pattern bytes as a comment; they are lost otherwise
            out.append(f"{m.group(1)}if True:  {MARKER} was: {line.strip()}")
        else:
            out.append(line)
    return out


def _repair_header_lines(seg):
    """Close a truncated string / bracket left open ON the header line(s). ``None`` if nothing changed."""
    fi = _first_nonblank_idx(seg)
    if fi is None:
        return None
    hdr_idxs, _after = _consume_header(seg, fi)
    out = list(seg)
    changed = False
    for hi in hdr_idxs:
        line = out[hi]
        newl, _ch = close_unterminated_string(line)
        newl, _n = close_line_brackets(newl)
        if newl != line:
            out[hi] = newl
            changed = True
    return out if changed else None


def _salvage_compound(seg, base, depth):
    """Keep a compound's headers, recurse into each clause body; ``None`` if the header is unsound."""
    fi = _first_nonblank_idx(seg)
    header_str = seg[fi].strip()
    opener_kw = _first_word(header_str)
    conts = _cont_for(opener_kw, header_str)

    # A broken ``match`` is almost always mis-indented cases the real clause model cannot represent;
    # flatten it to ``if True:`` blocks and re-split as an ordinary suite so the case bodies survive.
    if opener_kw == "match":
        flattened = _flatten_match(seg)
        if flattened != seg:
            return _salvage_suite(flattened, depth + 1)

    lifted = _promote_orphans(seg, base, conts)
    clauses = _split_clauses(lifted, base)
    if clauses is None:
        return None

    # A ``try:`` whose except/finally the decompiler dropped cannot compile. With no handler the block
    # is behaviourally just its body (exceptions propagate either way), so rewrite the header to
    # ``if True:`` in place -- every body line stays executable and the line count is unchanged.
    if opener_kw == "try" and not any(
        _first_word(lifted[hi].strip()) in ("except", "finally")
        for hdr_idxs, _b in clauses
        for hi in hdr_idxs
    ):
        fi2 = _first_nonblank_idx(lifted)
        rewritten = _rewrite_try_to_if(lifted[fi2])
        if rewritten != lifted[fi2]:
            lifted[fi2] = rewritten
            opener_kw = "if"
            conts = _cont_for("if", rewritten.strip())
            clauses = _split_clauses(lifted, base)
            if clauses is None:
                return None

    # Keep the longest valid PREFIX of clauses; neutralise the trailing clauses that break the header
    # sequence -- a duplicate ``else``, a non-last bare ``except``, decompiler garbage. Dropping a
    # suffix leaves the neutralised strings AFTER the compound, so they read as sibling statements.
    # The full list is tried first (the common case) because an intermediate prefix such as a lone
    # ``try:`` is itself invalid.
    if _skeleton_parses(lifted, clauses, base):
        n_keep = len(clauses)
    else:
        n_keep = 0
        for k in range(len(clauses) - 1, 0, -1):
            if _skeleton_parses(lifted, clauses[:k], base):
                n_keep = k
                break
    if n_keep == 0:
        return None  # even the opener header is unsound

    out = list(lifted)
    neut = 0
    for k, (hdr_idxs, body_idxs) in enumerate(clauses):
        if k >= n_keep:
            span = hdr_idxs + body_idxs
            a, b = min(span), max(span)
            out[a : b + 1] = _neutralise(lifted[a : b + 1])
            neut += b - a + 1
            continue
        body = [lifted[j] for j in body_idxs]
        if any(l.strip() for l in body):
            new_body, bneut = _salvage_suite(body, depth + 1)
            for j, nl in zip(body_idxs, new_body):
                out[j] = nl
            neut += bneut
        elif not body_idxs:
            # A clause with no body lines at all -- give it an inline ``pass`` (adds no line).
            last = hdr_idxs[-1]
            if out[last].rstrip().endswith(":"):
                out[last] = out[last] + " pass"
    if _parses_as_suite(out):
        return out, neut
    return None


def _close_or_neutralise(seg, base):
    """Line-local closure on a simple statement; fall back to neutralising the whole statement.

    ``close_source`` may partially rescue a multi-line simple statement -- close a truncated string or
    bracket, rename an illegal ``108 =`` target, and stringify only the lines it truly cannot parse
    (its neutralisation marker is the same one we use, so the accounting stays consistent). Anything it
    neutralises keeps every byte via ``repr`` just as ours does.
    """
    dedented = _dedent(seg, base)
    fixed = None
    try:
        fixed, _led = close_source(dedented + "\n")
    except Exception:
        fixed = None
    if fixed and _parses(fixed):
        fl = fixed.splitlines()
        reind = [(" " * base + l if l.strip() else "") for l in fl]
        if len(reind) <= len(seg):
            reind += [""] * (len(seg) - len(reind))
            return reind, _count_markers(reind)
        # closure grew the line count (inserted a body/handler); keep the line rule and neutralise.
    return _neutralise(seg), len(seg)


def _span_containing(lines, ci, lineno):
    """Top-level statement span of this suite that contains 1-indexed ``lineno`` (else the last)."""
    idx = min(max(lineno - 1, 0), len(lines) - 1)
    spans = _split_statements(lines, ci)
    for a, b in spans:
        if a <= idx <= b:
            return a, b
    return spans[-1] if spans else None


def _harden_suite(lines, ci):
    """Safety net: neutralise the offending top-level statement until the suite parses."""
    out = list(lines)
    neut = 0
    for _ in range(len(out) + 50):
        if _parses_as_suite(out):
            break
        d = _min_indent(out)
        try:
            ast.parse(_dedent(out, d))
            break
        except SyntaxError as exc:
            span = _span_containing(out, ci, exc.lineno or len(out))
        except Exception:
            break
        if span is None:
            break
        a, b = span
        seg = out[a : b + 1]
        if all(MARKER in l or not l.strip() for l in seg):
            break
        out[a : b + 1] = _neutralise(seg)
        neut += b - a + 1
    return out, neut


def recursive_salvage(text):
    """Return ``(parseable_text, lines_neutralised)`` via recursive descent. Line count is preserved."""
    if not text.strip():
        return text, 0
    lines = text.splitlines()
    if _parses(text):
        return text, 0
    out, neut = _salvage_suite(lines, 0)
    result = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    if not _parses(result):
        # Provable fallback: chunk-level neutralisation always yields a parseable module.
        result, led = force_parseable(text, repair=lambda t: close_source(t)[0])
        neut = int(led.get("lines_in_neutralised_chunks", neut) or 0)
    return result, neut


# --------------------------------------------------------------------------- compile hardening
def _logical_line_span(lines, idx):
    """Physical span of the logical line at ``idx`` (extends over bracket / backslash continuation)."""
    start = idx
    while start > 0:
        prev = _mask(lines[start - 1])
        depth = 0
        for ch in prev:
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth = max(0, depth - 1)
        if depth > 0 or prev.rstrip().endswith("\\"):
            start -= 1
        else:
            break
    end = start
    depth = 0
    while end < len(lines):
        m = _mask(lines[end])
        for ch in m:
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth = max(0, depth - 1)
        if depth > 0 or m.rstrip().endswith("\\"):
            end += 1
        else:
            break
    return start, min(end, len(lines) - 1)


def _compound_end(lines, a, lvl):
    """Last line index of the compound opened at ``a`` (indent ``lvl``): its body and clause chain."""
    b = a
    j = a + 1
    n = len(lines)
    while j < n:
        s = lines[j].strip()
        if not s:
            j += 1
            continue
        ind = _indent(lines[j])
        if ind > lvl:
            b = j
            j += 1
            continue
        if ind == lvl and _first_word(s) in CONT_KW:
            b = j
            j += 1
            continue
        break
    return b


def _harden_candidates(lines, idx):
    """Spans to neutralise for a compile error at ``idx``, smallest first.

    A simple statement (``return``/``continue``) is just its logical line. A clause header
    (``except``/``else``/...) or opener escalates to the WHOLE compound it belongs to, so neutralising
    never leaves a header dangling (which would break the parse the compile stage relies on).
    """
    cands = [_logical_line_span(lines, idx)]
    il = _indent(lines[idx])
    kw = _first_word(lines[idx].strip())
    # A clause header -> the compound opener at the same indent that owns it -> its full span.
    if kw in CONT_KW or kw in OPENER_KW:
        a = idx
        owner = -1
        while a >= 0:
            s = lines[a].strip()
            if s and _indent(lines[a]) == il and _first_word(s) in OPENER_KW:
                owner = a
                break
            if s and _indent(lines[a]) < il:
                break
            a -= 1
        if owner >= 0:
            cands.append((owner, _compound_end(lines, owner, il)))
    # Ancestor compounds, from nearest to module level.
    prev_il = il
    anchor = idx
    while prev_il > 0:
        a = anchor - 1
        found = -1
        while a >= 0:
            s = lines[a].strip()
            if s and _indent(lines[a]) < prev_il:
                if _first_word(s) in OPENER_KW:
                    found = a
                break
            a -= 1
        if found < 0:
            break
        lvl = _indent(lines[found])
        cands.append((found, _compound_end(lines, found, lvl)))
        anchor = found
        prev_il = lvl
    # de-dup, keep smallest-first order
    seen = set()
    out = []
    for sp in cands:
        if sp not in seen:
            seen.add(sp)
            out.append(sp)
    return out



class _ContextFinder(ast.NodeVisitor):
    """One AST pass finding statements that PARSE but do not COMPILE: ``return``/``yield`` outside a
    function, ``break``/``continue`` outside a loop. Replaces the per-error recompile loop (which was
    O(n^2)) with a single O(n) walk."""

    def __init__(self):
        self.bad = set()
        self._func = 0
        self._loop = 0

    def _in_function(self, node):
        prev = self._loop
        self._func += 1
        self._loop = 0  # a loop does not cross a function boundary
        self.generic_visit(node)
        self._func -= 1
        self._loop = prev

    visit_FunctionDef = _in_function
    visit_AsyncFunctionDef = _in_function

    def _in_loop(self, node):
        self._loop += 1
        self.generic_visit(node)
        self._loop -= 1

    visit_For = _in_loop
    visit_AsyncFor = _in_loop
    visit_While = _in_loop

    def visit_Return(self, node):
        if not self._func and node.lineno:
            self.bad.add(node.lineno)
        self.generic_visit(node)

    def visit_Yield(self, node):
        if not self._func and getattr(node, "lineno", None):
            self.bad.add(node.lineno)
        self.generic_visit(node)

    def visit_Break(self, node):
        if not self._loop and node.lineno:
            self.bad.add(node.lineno)

    def visit_Continue(self, node):
        if not self._loop and node.lineno:
            self.bad.add(node.lineno)


def _neutralise_structural(text):
    """Neutralise every structurally-illegal statement in ONE pass. Returns (text, count)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, 0
    finder = _ContextFinder()
    finder.visit(tree)
    if not finder.bad:
        return text, 0
    lines = text.splitlines()
    count = 0
    for ln in finder.bad:
        i = ln - 1
        if 0 <= i < len(lines) and lines[i].strip() and MARKER not in lines[i]:
            lines[i] = _neutralise([lines[i]])[0]
            count += 1
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), count


def _host_compiles(text):
    try:
        compile(text, "<forced>", "exec")
        return True
    except Exception:
        return False


def _whole_file_literal(src):
    """Ultimate always-compiles fallback: the entire file as one string-literal statement.

    Trivially compiles at every version and preserves every byte (recoverable via ast.literal_eval).
    Padded with blank lines so the physical line count is unchanged. Only reached when nothing else
    makes the file compile -- code preservation is 0% here, which is exactly what the ledger records."""
    n = src.count("\n")
    body = f"{src!r}  {MARKER}-whole-file"
    return body + ("\n" * n if n else "\n")


def _compile_harden(text):
    """Neutralise the smallest span that fixes each compile-only error until the module compiles.

    ``ast.parse`` already accepts ``text``; anything ``compile`` still rejects is a structural error it
    alone can see (``return`` outside a function, ``continue`` outside a loop, a duplicate bare
    ``except``, ...). The host compiler is a faithful proxy for these version-stable rules on 3.10-3.13
    malware. Candidate spans are tried smallest-first, and only one that keeps the module PARSING is
    accepted, so a clause header is never half-neutralised.
    """
    neut = 0
    trailing_nl = text.endswith("\n")
    text, structural = _neutralise_structural(text)   # one O(n) pass kills the quadratic case
    neut += structural
    for _ in range(300):
        try:
            compile(text, "<forced>", "exec")
            break
        except SyntaxError as exc:
            lineno = exc.lineno
            if not lineno:
                break
            lines = text.splitlines()
            idx = min(max(lineno - 1, 0), len(lines) - 1)
            applied = False
            for a, b in _harden_candidates(lines, idx):
                seg = lines[a : b + 1]
                if all(MARKER in l or not l.strip() for l in seg):
                    continue
                trial = list(lines)
                trial[a : b + 1] = _neutralise(seg)
                trial_text = "\n".join(trial) + ("\n" if trailing_nl else "")
                if _parses(trial_text) and trial_text != text:
                    text = trial_text
                    neut += b - a + 1
                    applied = True
                    break
            if not applied:
                break
        except Exception:
            break
    return text, neut


# --------------------------------------------------------------------------- public entry point
def force_compile(source_text, gt_pyc_path, version):
    """Force decompiled ``source_text`` to COMPILE at its own version, keeping maximal live code.

    Returns ``(text, ledger)`` where ``ledger['lines_in_neutralised_chunks']`` counts the original
    physical lines turned into inert string literals. The output always has the same line count as the
    input.
    """
    staged = source_text
    if gt_pyc_path:
        try:
            index = load_gt_literals(gt_pyc_path)
            staged, _stats = repair_source(source_text, index)
        except Exception:
            staged = source_text

    text, neut = recursive_salvage(staged)
    text, neut2 = _compile_harden(text)
    neut += neut2

    fallback = None
    if not _host_compiles(text):
        # Anything left that still will not compile: the chunk-level neutraliser (always parses),
        # then a final structural pass.
        text = force_parseable(text, repair=lambda t: close_source(t)[0])[0]
        text, extra = _neutralise_structural(text)
        neut += extra
        fallback = "chunk"
    if not _host_compiles(text):
        # Ultimate guarantee: the whole file as one string literal. Compiles at every version, every
        # byte preserved; 0% live code, which the ledger makes explicit.
        text = _whole_file_literal(source_text)
        neut = source_text.count("\n")
        fallback = "whole-file"

    ledger = {
        "lines_in_neutralised_chunks": int(neut),
        "parses": _parses(text),
        "compiles": _host_compiles(text),   # authoritative -- never inferred
        "fallback": fallback,               # None | "chunk" | "whole-file"
    }
    return text, ledger
