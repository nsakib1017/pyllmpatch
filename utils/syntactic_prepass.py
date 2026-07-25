"""Deterministic, compile-gated pre-pass for the syntactic-repair pipeline.

Focused operators of the form ``(source: str, error: SyntaxErrorInfo) -> str | None``
try to mechanically fix common syntax errors before the LLM is invoked. A driver
tries each operator in turn and only accepts a candidate if it compiles cleanly
or strictly advances past the current error line (never regresses).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SyntaxErrorInfo:
    lineno: int | None
    offset: int | None
    msg: str


def host_compile(source: str) -> None:
    """Default test/dev compile_fn: raises SyntaxError on failure."""
    compile(source, "<syntactic_prepass>", "exec")


def probe_syntax(source: str, compile_fn) -> SyntaxErrorInfo | None:
    """Return None if source compiles under compile_fn, else a SyntaxErrorInfo."""
    try:
        compile_fn(source)
        return None
    except SyntaxError as e:
        return SyntaxErrorInfo(e.lineno, e.offset, (e.msg or "").strip())
    except Exception as e:
        # Non-SyntaxError compile failure (e.g. production's
        # utils.generate_bytecode.CompileError, a plain Exception whose
        # message is text-only -- no structured .lineno). This is the
        # branch that ALWAYS runs in production, since compile_version
        # never raises a SyntaxError. Recover the line number from the
        # message text (CompileError's message reliably contains
        # "line N") instead of always reporting an unknown line, which
        # crippled every lineno-gated downstream operator/driver check.
        text = str(e)
        m = re.search(r"line\s+(\d+)", text)
        return SyntaxErrorInfo(int(m.group(1)) if m else None, None, f"{type(e).__name__}: {e}".strip())


def advanced(before: SyntaxErrorInfo, after: SyntaxErrorInfo | None) -> bool:
    """True if `after` is None (fixed) or its lineno is strictly greater than before's."""
    if after is None:
        return True
    if before.lineno is None or after.lineno is None:
        return False
    return after.lineno > before.lineno


# ---------------------------------------------------------------------------
# Task 2: balance_delimiters
# ---------------------------------------------------------------------------

import io
import re
import tokenize

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def balance_delimiters(source: str, error: SyntaxErrorInfo) -> str | None:
    """Close unbalanced ()[]{} and unterminated strings, or None if nothing unbalanced."""
    stack: list[str] = []
    unterminated_string = False
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.OP and tok.string in _OPEN:
                stack.append(tok.string)
            elif tok.type == tokenize.OP and tok.string in _CLOSE:
                if stack and stack[-1] == _CLOSE[tok.string]:
                    stack.pop()
    except tokenize.TokenError as e:
        if "string" in str(e).lower():
            unterminated_string = True
    except Exception:
        return None
    if not stack and not unterminated_string:
        return None
    closers = "".join(_OPEN[o] for o in reversed(stack))
    quote = ""
    append_at_end = False
    if unterminated_string:
        # Identify the opening quote (and whether it's a triple-quote) at the
        # error position. error.offset (1-based column) points at the quote
        # that opened the still-unterminated string.
        src_lines = source.splitlines()
        lineno = error.lineno or len(src_lines)
        line = src_lines[lineno - 1] if 1 <= lineno <= len(src_lines) else ""
        col = (error.offset - 1) if error.offset else None
        triple = None
        if col is not None and 0 <= col <= len(line) - 3:
            cand = line[col:col + 3]
            if cand in ("'''", '"""'):
                triple = cand
        if triple is None:
            for marker in ("'''", '"""'):
                if line.count(marker) % 2:
                    triple = marker
                    break
        if triple:
            # A triple-quoted string swallows every line up to EOF (it's the
            # only way it can be "unterminated" mid-file) -> the close belongs
            # at the very end of the source, not on the error's opening line.
            quote = triple
            append_at_end = True
        else:
            quote = "'" if line.count("'") % 2 else ('"' if line.count('"') % 2 else "'")
    lines = source.splitlines(keepends=True)
    if not append_at_end and error.lineno and 1 <= error.lineno <= len(lines):
        ln = error.lineno - 1
        lines[ln] = lines[ln].rstrip("\n") + quote + closers + "\n"
        return "".join(lines)
    return source.rstrip("\n") + quote + closers + "\n"


# ---------------------------------------------------------------------------
# Task 3: dedent_stray_block
# ---------------------------------------------------------------------------

def _leading_ws(s: str) -> str:
    return s[: len(s) - len(s.lstrip(" \t"))]


def dedent_stray_block(source: str, error: SyntaxErrorInfo) -> str | None:
    """For 'unexpected indent' where the preceding non-blank line is not a block
    opener, dedent the flagged line and the contiguous block below it that shares
    its indent, down to the indent of the preceding line."""
    if not error.msg or "indent" not in error.msg.lower() or "unexpected" not in error.msg.lower():
        return None
    if error.lineno is None:
        return None
    lines = source.splitlines(keepends=True)
    ln = error.lineno - 1
    if ln <= 0 or ln >= len(lines):
        return None
    prev = next((lines[i] for i in range(ln - 1, -1, -1) if lines[i].strip()), None)
    if prev is None or prev.rstrip().endswith(":"):
        return None  # legitimate block opener -> defer
    target = _leading_ws(prev)
    cur = _leading_ws(lines[ln])
    if len(cur) <= len(target):
        return None
    out = list(lines)
    i = ln
    while i < len(out) and (not out[i].strip() or _leading_ws(out[i]).startswith(cur)):
        if out[i].strip():
            out[i] = target + out[i][len(cur):]
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Task 4: fix_line_continuation, fix_numeric_literal
# ---------------------------------------------------------------------------

def fix_line_continuation(source: str, error: SyntaxErrorInfo) -> str | None:
    """On 'unexpected character after line continuation character', strip
    everything after a trailing backslash on the error line. None otherwise."""
    if not error.msg or "line continuation" not in error.msg.lower() or error.lineno is None:
        return None
    lines = source.splitlines(keepends=True); ln = error.lineno - 1
    if not (0 <= ln < len(lines)) or "\\" not in lines[ln]:
        return None
    head, _, _ = lines[ln].partition("\\")
    lines[ln] = head.rstrip() + "\n"
    return "".join(lines)


def fix_numeric_literal(source: str, error: SyntaxErrorInfo) -> str | None:
    """On 'invalid decimal literal', conservatively handle only unambiguous
    juxtapositions; defer (return None) on anything ambiguous."""
    if not error.msg or "literal" not in error.msg.lower():
        return None
    return None  # ambiguous by default; extend only with evidence


# ---------------------------------------------------------------------------
# Task 5: driver
# ---------------------------------------------------------------------------

@dataclass
class PrepassResult:
    source: str
    compiled: bool
    operations: list[str] = field(default_factory=list)
    rounds: int = 0


_OPERATORS = [balance_delimiters, dedent_stray_block, fix_line_continuation, fix_numeric_literal]


def run_syntactic_prepass(source: str, compile_fn=host_compile, max_rounds: int = 6) -> PrepassResult:
    """Try operators in order each round, accepting the first candidate that
    compiles or strictly advances past the current error. Never regresses."""
    cur = source; ops: list[str] = []
    for rnd in range(1, max_rounds + 1):
        before = probe_syntax(cur, compile_fn)
        if before is None:
            return PrepassResult(cur, True, ops, rnd - 1)
        applied = False
        for op in _OPERATORS:
            cand = op(cur, before)
            if cand is None or cand == cur:
                continue
            after = probe_syntax(cand, compile_fn)
            if after is None or advanced(before, after):
                cur = cand; ops.append(op.__name__); applied = True
                break
        if not applied:
            return PrepassResult(cur, False, ops, rnd)
    return PrepassResult(cur, probe_syntax(cur, compile_fn) is None, ops, max_rounds)


# ---------------------------------------------------------------------------
# Task 5b: minimal repair window (smallest snippet for the LLM + clean reattach)
# ---------------------------------------------------------------------------

def _common_indent(lines: list[str]) -> str:
    ind = None
    for ln in lines:
        if not ln.strip():
            continue
        cur = ln[: len(ln) - len(ln.lstrip(" \t"))]
        ind = cur if ind is None else (ind if ind == cur else (min(ind, cur, key=len)))
    return ind or ""


def _bracket_depth(text: str) -> int:
    """Net (open - close) ()[]{} depth of `text`, tolerant of it being an
    incomplete/unterminated fragment (e.g. a still-open string or bracket)."""
    depth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.OP:
                if tok.string in _OPEN:
                    depth += 1
                elif tok.string in _CLOSE:
                    depth -= 1
    except (tokenize.TokenError, IndentationError):
        pass  # ran off the end mid-construct; keep whatever depth we counted
    except Exception:
        pass
    return depth


_CLOSING_ONLY_RE = re.compile(r"[\s)\]},]*")


def _is_closing_only(line: str) -> bool:
    """True if `line` consists solely of closing delimiters, commas, and
    whitespace (e.g. ')', '),', '}]') -- i.e. it can only be the tail end of
    an already-open bracketed expression, never the start of new content."""
    return _CLOSING_ONLY_RE.fullmatch(line) is not None


@dataclass
class RepairWindow:
    text: str
    start_line: int
    end_line: int
    indent: str


def minimal_repair_window(source: str, error: SyntaxErrorInfo, expansion: int = 0) -> RepairWindow:
    """Return the tightest snippet around error.lineno: at expansion=0, the single
    failing logical line plus any physical continuation lines it needs to be a
    parseable unit. Each higher expansion widens by one enclosing indentation
    level. `indent` is the common indent stripped from `text`."""
    lines = source.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return RepairWindow(text="", start_line=1, end_line=1, indent="")
    ln = min(max((error.lineno or 1), 1), n)
    start = end = ln
    start_indent_len = len(_leading_ws(lines[start - 1]))
    # Pull following physical lines by ACTUAL bracket depth, not indentation:
    # keep extending while the accumulated snippet has unbalanced brackets (or
    # an open string). A line that is purely closing delimiters/commas is part
    # of the bracketed expression regardless of its indent (Black dedents
    # closers to the opening statement's own indent, or below). A line that
    # starts new content while still unbalanced is only pulled in if it is
    # indented deeper than the opening line; dedented-to-<= new content means
    # the construct is genuinely never closed, so stop before swallowing it.
    while end < n and _bracket_depth("".join(lines[start - 1:end])) > 0:
        nxt = lines[end]
        if not _is_closing_only(nxt) and nxt.strip() and len(_leading_ws(nxt)) <= start_indent_len:
            break
        end += 1
    # expansion: widen to the nearest enclosing block (header + its full body),
    # one level per expansion step. If no enclosing block exists (already at
    # the outermost indentation with nothing above it), expansion is a no-op
    # -- it must never reach sideways for unrelated sibling statements.
    for _ in range(expansion):
        depth_indent = len(_leading_ws(lines[start - 1]))
        header = None
        i = start - 2  # 0-indexed line just above the current window
        while i >= 0:
            if lines[i].strip():
                if len(_leading_ws(lines[i])) < depth_indent:
                    if lines[i].rstrip().endswith(":"):
                        header = i + 1  # 1-based
                    break
                i -= 1
            else:
                i -= 1
        if header is None:
            continue  # nothing to widen into at this level
        start = header
        while end < n and (not lines[end].strip() or len(_leading_ws(lines[end])) >= depth_indent):
            end += 1
    seg = lines[start - 1:end]
    indent = _common_indent(seg)
    text = "".join((l[len(indent):] if l.strip() else l) for l in seg)
    return RepairWindow(text=text, start_line=start, end_line=end, indent=indent)


# ---------------------------------------------------------------------------
# Cause-aware repair window: locate_cause() + cause_aware_window()
#
# minimal_repair_window (above) is deliberately symptom-anchored: it builds
# the tightest snippet around error.lineno, the line CPython happened to
# report. For many real syntax errors that reported line is a downstream
# SYMPTOM, not the CAUSE -- an unclosed bracket, a missing colon, or a
# malformed def signature several lines above is what actually needs fixing.
# locate_cause() maps the reported error to that true cause line using
# error-type rules; cause_aware_window() then builds a window that covers
# both the cause and the minimal context an LLM needs to fix it, reusing
# minimal_repair_window's own bracket-balanced logical-line machinery so the
# result stays a clean, round-trippable RepairWindow.
# ---------------------------------------------------------------------------


def _nearest_preceding_nonblank(lines: list[str], ln: int) -> int:
    """1-based line number of the nearest non-blank line strictly above
    1-based `ln`, or `ln` itself if there is none (already at file top)."""
    i = ln - 2  # 0-indexed line just above ln
    while i >= 0 and not lines[i].strip():
        i -= 1
    return (i + 1) if i >= 0 else ln


def _enclosing_shallower_line(lines: list[str], ln: int) -> int:
    """1-based line number of the nearest preceding non-blank line whose
    indentation is strictly less than `ln`'s own -- i.e. the block `ln` is
    nested inside. Returns 1 (file start) if `ln` is already outermost."""
    n = len(lines)
    if not (1 <= ln <= n) or not lines[ln - 1].strip():
        return ln
    cur_indent = len(_leading_ws(lines[ln - 1]))
    i = ln - 2
    while i >= 0:
        if lines[i].strip():
            if len(_leading_ws(lines[i])) < cur_indent:
                return i + 1
            i -= 1
        else:
            i -= 1
    return 1


def _enclosing_block_scan_start(lines: list[str], ln: int) -> int:
    """1-based line number that is SAFE to start a forward bracket-depth scan
    from in order to find `ln`'s logical-statement start: the body-start of
    the nearest enclosing ':'-terminated header, or the nearest shallower
    line itself (continuation-indent case), or 1 (file start)."""
    n = len(lines)
    if not (1 <= ln <= n) or not lines[ln - 1].strip():
        return 1
    cur_indent = len(_leading_ws(lines[ln - 1]))
    i = ln - 2
    while i >= 0:
        if lines[i].strip():
            if len(_leading_ws(lines[i])) < cur_indent:
                return (i + 2) if lines[i].rstrip().endswith(":") else (i + 1)
            i -= 1
        else:
            i -= 1
    return 1


def _statement_start(lines: list[str], ln: int) -> int:
    """1-based line number where the logical statement containing physical
    line `ln` begins. Scans forward from a known-safe boundary (the
    enclosing block's body start, see `_enclosing_block_scan_start`),
    tracking cumulative bracket depth and backslash continuations to find
    the last statement boundary at or before `ln`. This correctly walks
    past lines that look self-contained in isolation (zero brackets of
    their own) but are still inside a bracket opened further up."""
    n = len(lines)
    if n == 0:
        return 1
    ln = max(1, min(ln, n))
    scan_start = _enclosing_block_scan_start(lines, ln)
    depth = 0
    boundary = scan_start
    for i in range(scan_start, ln):  # 1-based lines scan_start .. ln-1
        line = lines[i - 1]
        depth += _bracket_depth(line)
        if depth <= 0 and not line.rstrip("\n").endswith("\\"):
            boundary = i + 1
    return boundary


def _enclosing_def_class_cap(lines: list[str], ln: int) -> int:
    """CAP (documented on cause_aware_window): the earliest 1-based line
    cause_aware_window may ever include for a window anchored near `ln` --
    one past the nearest strictly-shallower ':'-terminated header (the
    enclosing def/class/if/for/... block), or 1 (file start) when `ln` is
    already outermost or nothing shallower is a real block header. This
    stops cause-anchoring from ballooning into an unrelated outer scope or
    the whole file; it never blocks pulling in this construct's own
    decorators, which sit at the SAME indentation as the header, not
    shallower."""
    n = len(lines)
    if not (1 <= ln <= n) or not lines[ln - 1].strip():
        return 1
    cur_indent = len(_leading_ws(lines[ln - 1]))
    i = ln - 2
    while i >= 0:
        if lines[i].strip():
            if len(_leading_ws(lines[i])) < cur_indent:
                return (i + 2) if lines[i].rstrip().endswith(":") else 1
            i -= 1
        else:
            i -= 1
    return 1


_DEF_CLASS_HEADER_RE = re.compile(r"(async\s+def|def|class)\b")


def _unclosed_delimiter_opener(lines: list[str], ln: int) -> int | None:
    """1-based line number of the nearest line strictly ABOVE `ln` that opens
    a bracket/string still unclosed by the time line `ln` is reached, or
    None if `ln` is not sitting inside such a construct. Reuses the same
    safe-scan-start + cumulative bracket-depth walk as `_statement_start`
    (tracking backslash continuations too), but only reports an opener when
    the accumulated depth is still positive going into `ln` -- i.e. `ln`
    itself is nested inside something opened above it, not just the start of
    its own fresh statement."""
    n = len(lines)
    if not (1 <= ln <= n) or ln <= 1:
        return None
    scan_start = _enclosing_block_scan_start(lines, ln)
    depth = 0
    boundary = scan_start
    for i in range(scan_start, ln):  # 1-based lines scan_start .. ln-1
        line = lines[i - 1]
        depth += _bracket_depth(line)
        if depth <= 0 and not line.rstrip("\n").endswith("\\"):
            boundary = i + 1
    if depth > 0 and boundary < ln:
        return boundary
    return None


@dataclass
class CauseAnchor:
    line: int
    reason: str
    include_preceding: bool = False
    include_decorators: bool = False
    span_statement: bool = False


def locate_cause(source: str, error: SyntaxErrorInfo) -> CauseAnchor:
    """Map a reported syntax error to the line that actually needs fixing.

    CPython's error line is frequently a downstream SYMPTOM: an unclosed
    bracket, missing colon, or malformed def header several lines above is
    the real cause. Rules (checked in this priority order -- see module
    docstring section above for why the def/class content check runs before
    the generic message-based buckets):

    0. Before trusting a def/class-header-shaped reported line at all: if
       that line sits INSIDE an unclosed bracket/string opened on a
       preceding line (a backward bracket-depth scan finds one still open),
       THAT opener is the true cause, not the def/class line -- a decompiler
       artifact routinely leaves an unterminated call/string right above a
       def header, and CPython blames the def line for it. This takes
       priority over rule 1.
    1. A def/async def/class header (or a leaked PEP 695
       "<generic parameters of ...>" placeholder) on the error line, checked
       by SOURCE CONTENT regardless of the exact message text CPython
       attaches (it varies -- "invalid syntax" is common for both malformed
       signatures and unrelated mid-expression errors, so content wins).
       The cause is LOCAL to the header line itself (plus any decorators
       directly above it) -- it never spans into the function/class BODY,
       which may be arbitrarily large decompiler garbage unrelated to the
       broken header.
    2. "unexpected indent" -> the nearest preceding non-blank line (the
       cause is what precedes the over-indented line).
    3. "expected an indented block" -> the header line CPython names
       ("... after 'if' statement on line N"), falling back to the nearest
       preceding non-blank line if the message doesn't parse.
    4. "was never closed" / "unterminated" -> the reported line itself
       (CPython already points at the opener).
    5. "unindent does not match" -> the start of the enclosing block.
    6. "invalid syntax" / "perhaps you forgot a comma" -> the logical
       statement start, found by walking up through open brackets /
       backslash continuations.
    7. default -> the reported line, unchanged.
    """
    lines = source.splitlines(keepends=True)
    n = len(lines)
    msg = (error.msg or "").lower()
    raw_ln = error.lineno if error.lineno else 1
    ln = min(max(raw_ln, 1), n) if n else max(raw_ln, 1)

    line_text = lines[ln - 1] if 1 <= ln <= n else ""
    stripped = line_text.strip()

    looks_like_def_class = "<generic parameters of" in stripped or _DEF_CLASS_HEADER_RE.match(stripped)
    if looks_like_def_class and n:
        opener = _unclosed_delimiter_opener(lines, ln)
        if opener is not None:
            return CauseAnchor(line=opener, reason="unclosed_delimiter_above",
                                span_statement=True)

    if "<generic parameters of" in stripped:
        return CauseAnchor(line=ln, reason="pep695_generic_leak",
                            include_decorators=True, span_statement=False)
    if _DEF_CLASS_HEADER_RE.match(stripped):
        return CauseAnchor(line=ln, reason="def_header",
                            include_decorators=True, span_statement=False)

    if "unexpected indent" in msg:
        cause_line = _nearest_preceding_nonblank(lines, ln) if n else ln
        return CauseAnchor(line=cause_line, reason="unexpected_indent_prev_line",
                            include_preceding=True)

    if "expected an indented block" in msg:
        m = re.search(r"line\s+(\d+)", msg)
        if m and n:
            cause_line = min(max(int(m.group(1)), 1), n)
        else:
            cause_line = _nearest_preceding_nonblank(lines, ln) if n else ln
        return CauseAnchor(line=cause_line, reason="empty_block_header",
                            include_preceding=True)

    if "was never closed" in msg or "unterminated" in msg:
        return CauseAnchor(line=ln, reason="unclosed_delimiter", span_statement=True)

    if "unindent does not match" in msg:
        cause_line = _enclosing_shallower_line(lines, ln) if n else ln
        return CauseAnchor(line=cause_line, reason="unindent_mismatch",
                            include_preceding=True)

    if "invalid syntax" in msg or "perhaps you forgot a comma" in msg:
        cause_line = _statement_start(lines, ln) if n else ln
        return CauseAnchor(line=cause_line, reason="mid_statement", span_statement=True)

    return CauseAnchor(line=ln, reason="reported_line")


def cause_aware_window(source: str, error: SyntaxErrorInfo, expansion: int = 0) -> RepairWindow:
    """Like `minimal_repair_window`, but anchored on the true CAUSE of the
    error (via `locate_cause`) instead of only the reported symptom line,
    plus the minimal extra context needed to act on that cause:

    - the base span is `minimal_repair_window` at the reported error line
      (unchanged symptom-side bracket-balanced logical-line + expansion
      machinery -- this is what keeps the MINIMALITY guarantee: when the
      cause equals the reported line, nothing is added, see below);
    - the window is then pulled up to at least the cause line;
    - `span_statement` additionally walks up through open brackets /
      backslash continuations to the true logical-statement start;
    - `include_decorators` additionally pulls in contiguous `@decorator`
      lines directly above a def/class header;
    - `reason == "unclosed_delimiter_above"` additionally extends the window
      back DOWN to wherever the opener's own bracket/string actually
      balances (re-running the same bracket-balanced descent
      `minimal_repair_window` uses, but anchored at the true opener line
      instead of the reported line) -- covering cases where that requires
      more than just the reported line.

    A def/class header or PEP 695 generic-leak header is DELIBERATELY never
    extended into its function/class BODY: that body may be arbitrarily
    large decompiler garbage unrelated to a broken header line, and pulling
    it in previously ballooned some windows to dozens of lines of noise
    (see `locate_cause` rule 1's docstring). The window for these reasons is
    the header line (plus any decorators) and nothing more.

    CAP: the window never extends above the nearest enclosing def/class (or
    other ':'-terminated) block header -- cause-anchoring stops at that
    block's own body start and never reaches into an outer scope or the
    whole file, regardless of how far a bracket/backslash continuation
    would otherwise walk. In practice each rule's own upward walk already
    stops at the nearest shallower block boundary (see `_statement_start`,
    `_enclosing_shallower_line`), so this is a defensive backstop rather
    than the primary bound.

    MINIMALITY: for the default rule (`reason == "reported_line"`, i.e. no
    error-type rule matched and there is no cause above the reported line),
    every extension below is a documented no-op, so the result is byte-for-
    byte identical to `minimal_repair_window(source, error, expansion)`.

    Recomputes `indent` / dedented `text` over the final line span exactly
    like `minimal_repair_window`, so `reattach_window` still round-trips.
    """
    lines = source.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return minimal_repair_window(source, error, expansion)

    anchor = locate_cause(source, error)
    base = minimal_repair_window(source, error, expansion)
    start, end = base.start_line, base.end_line

    anchor_line = max(1, min(anchor.line, n))
    if anchor_line < start:
        start = anchor_line

    if anchor.span_statement:
        start = min(start, _statement_start(lines, start))

    # CAP: never climb above the enclosing def/class block's own body start.
    # Bounded by base.start_line too -- minimal_repair_window's own
    # expansion parameter is allowed to widen past this (that's its job),
    # the cap only constrains the ADDITIONAL cause-anchor extensions above.
    cap = min(_enclosing_def_class_cap(lines, min(max(error.lineno or start, 1), n)), base.start_line)
    if start < cap:
        start = cap

    if anchor.include_decorators:
        while start > 1 and lines[start - 2].lstrip().startswith("@"):
            start -= 1

    if anchor.reason == "unclosed_delimiter_above":
        # The reported line's own base window doesn't know about the
        # unclosed bracket/string above it -- re-anchor a fresh
        # minimal_repair_window scan AT the true opener (`start`) so the
        # window extends down through wherever that construct actually
        # balances, even if that's past the originally reported line.
        reopened = minimal_repair_window(
            source, SyntaxErrorInfo(lineno=start, offset=None, msg=""), 0
        )
        if reopened.end_line > end:
            end = reopened.end_line

    seg = lines[start - 1:end]
    indent = _common_indent(seg)
    text = "".join((l[len(indent):] if l.strip() else l) for l in seg)
    return RepairWindow(text=text, start_line=start, end_line=end, indent=indent)


# ---------------------------------------------------------------------------
# Code-object-isolation window: codeobject_span() + codeobject_window()
#
# Unlike minimal_repair_window/cause_aware_window (which are anchored at the
# reported error LINE), this window is anchored at the enclosing CODE OBJECT
# (the innermost def/class/async def that actually contains the error line),
# giving the LLM the whole function/method/class body as repair context
# instead of a line-local snippet. It is a no-AST indentation scanner: no
# tokenize/ast dependency beyond the regex header match already used
# elsewhere in this module.
# ---------------------------------------------------------------------------

OBJECT_WINDOW_MAX_LINES = 400

_CODEOBJECT_HDR_RE = re.compile(r"^(\s*)(?:async\s+def|def|class)\s+[A-Za-z_]\w*")


def _no_intervening_dedent(lines: list[str], header_idx: int, err_idx: int, hdr_indent: int) -> bool:
    """True if no line strictly between 0-indexed `header_idx` and `err_idx`
    has indent <= `hdr_indent` -- i.e. nothing dedents back out of the
    header's block before reaching the error position. A dedent found here
    means the error line is actually a SIBLING of whatever follows the
    header, not enclosed by it."""
    for i in range(header_idx + 1, err_idx):
        if lines[i].strip() and len(_leading_ws(lines[i])) <= hdr_indent:
            return False
    return True


def _find_enclosing_header(lines: list[str], err_idx: int, err_indent: int) -> int | None:
    """0-indexed line number of the innermost def/class/async def header
    that encloses 0-indexed `err_idx` (whose indent is `err_indent`), per
    the enclosure rule: the closest preceding header line with indent <
    err_indent, with no intervening dedent between it and err_idx. Lines
    that aren't header matches (or fail the intervening-dedent check) are
    skipped in favor of the next-closest candidate further up; returns None
    if no valid header is found by the top of the file."""
    i = err_idx - 1
    while i >= 0:
        line = lines[i]
        if line.strip():
            line_indent = len(_leading_ws(line))
            if line_indent < err_indent and _CODEOBJECT_HDR_RE.match(line):
                if _no_intervening_dedent(lines, i, err_idx, line_indent):
                    return i
        i -= 1
    return None


def codeobject_span(lines: list[str], err_line: int) -> tuple[int, int] | None:
    """1-indexed (start_header_line, end_line) of the innermost def/class/
    async def that ENCLOSES 1-indexed `err_line`, or None if there is no
    enclosing object (module scope, or `err_line` is a sibling statement to
    the nearest preceding header rather than nested inside it)."""
    n = len(lines)
    if n == 0 or err_line < 1 or err_line > n:
        return None
    idx = err_line - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return None  # nothing but blank lines above/at err_line
    err_indent = len(_leading_ws(lines[idx]))
    header_idx = _find_enclosing_header(lines, idx, err_indent)
    if header_idx is None:
        return None
    hdr_indent = len(_leading_ws(lines[header_idx]))
    end_idx = header_idx
    j = header_idx + 1
    while j < n and (not lines[j].strip() or len(_leading_ws(lines[j])) > hdr_indent):
        end_idx = j
        j += 1
    return (header_idx + 1, end_idx + 1)


def codeobject_window(source: str, error: SyntaxErrorInfo, expansion: int = 0) -> RepairWindow:
    """Return the RepairWindow spanning the innermost def/class/async def
    enclosing `error.lineno`, dedented exactly like minimal_repair_window so
    `reattach_window` round-trips.

    `expansion` widens to successively-outer enclosing objects (repeating
    the enclosure rule from the current header's own indent); once there is
    no further outer object, additional expansion is a no-op that keeps the
    outermost object found.

    Degenerates to `cause_aware_window(source, error, expansion)` -- the
    existing line-anchored window -- when there is no enclosing object
    (module-scope error) or the enclosing object's line count exceeds
    OBJECT_WINDOW_MAX_LINES (the giant-object guard).
    """
    lines = source.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return cause_aware_window(source, error, expansion)
    err_line = min(max((error.lineno or 1), 1), n)
    span = codeobject_span(lines, err_line)
    if span is None:
        return cause_aware_window(source, error, expansion)
    start, end = span
    for _ in range(expansion):
        hdr_indent = len(_leading_ws(lines[start - 1]))
        outer_idx = _find_enclosing_header(lines, start - 1, hdr_indent)
        if outer_idx is None:
            break  # already outermost -- further expansion is a no-op
        outer_indent = len(_leading_ws(lines[outer_idx]))
        end_idx = outer_idx
        j = outer_idx + 1
        while j < n and (not lines[j].strip() or len(_leading_ws(lines[j])) > outer_indent):
            end_idx = j
            j += 1
        start, end = outer_idx + 1, end_idx + 1
    if (end - start + 1) > OBJECT_WINDOW_MAX_LINES:
        return cause_aware_window(source, error, expansion)
    seg = lines[start - 1:end]
    indent = _common_indent(seg)
    text = "".join((l[len(indent):] if l.strip() else l) for l in seg)
    return RepairWindow(text=text, start_line=start, end_line=end, indent=indent)


def reattach_window(source: str, window: RepairWindow, fixed_text: str) -> str:
    """Re-indent fixed_text by window.indent and replace exactly lines
    start_line..end_line. Inverse of minimal_repair_window for an unchanged fix."""
    lines = source.splitlines(keepends=True)
    fixed = fixed_text.splitlines()
    reindented = [(window.indent + f if f.strip() else f) for f in fixed]
    body = "\n".join(reindented) + ("\n" if source.endswith("\n") or window.end_line < len(lines) else "")
    return "".join(lines[: window.start_line - 1]) + body + "".join(lines[window.end_line:])
