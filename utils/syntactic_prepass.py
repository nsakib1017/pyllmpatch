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


_SURRENDER_MARKERS = re.compile(r"<mask_\d+>|<Code\d+ code object>")

_MIN_TRUNCATED_LITERAL_CHARS = 80


def is_truncated_literal_line(line) -> bool:
    """True when `line` looks like a decompiler-TRUNCATED literal rather than a merely unbalanced one.

    PyLingual caps constant sequences at 51 elements and stops mid-literal, leaving a long line with
    a dangling opening quote inside a still-open bracket. Closing that literal makes the file
    compile while silently discarding the remaining elements -- a repair that scores as
    "genuine" (no lines deleted) but is not faithful. Callers use this to LABEL such repairs, so the
    genuine rate can be split into faithful vs lossy instead of over-reporting fidelity.

    Signature (measured on the malware corpus): a long line whose quote count is odd (an unclosed
    string) AND whose brackets are still net-open. Never raises."""
    if not line:
        return False
    text = str(line).rstrip("\n")
    if len(text) < _MIN_TRUNCATED_LITERAL_CHARS:
        return False
    if (text.count("'") + text.count('"')) % 2 == 0:
        return False
    depth = 0
    for ch in text:
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
    return depth > 0


_LONG_LITERAL = re.compile(r"('([^'\\\n]|\\.){%d,}'|\"([^\"\\\n]|\\.){%d,}\")")
_ELIDE_TOKEN = "__PYLLM_PAYLOAD_%d__"


def elide_long_string_literals(source, max_literal_chars=2000):
    """Hide oversized string-literal payloads behind short placeholders.

    The lines that make a decompiled malware file un-windowable are almost always embedded payloads
    (`encrypted_data = '<744,000 chars of base64>'`): syntactically trivial, never the cause of the
    syntax error, but large enough that the whole file is discarded for exceeding the generation
    budget. Eliding them shrinks the window enough to repair the file.

    Returns `(elided_source, mapping)` where `mapping` maps each placeholder back to the ORIGINAL
    literal text, so `restore_elided_literals` reproduces the input byte-for-byte -- the payload is
    preserved, merely hidden from the model. Returns `(source, {})` when nothing is oversized.
    Never raises."""
    try:
        if not source:
            return source, {}
        pattern = re.compile(_LONG_LITERAL.pattern % (max_literal_chars, max_literal_chars))
        mapping = {}

        def _swap(m):
            token = _ELIDE_TOKEN % len(mapping)
            literal = m.group(0)
            mapping[token] = literal
            # keep the same quote style so the result is still a valid string literal
            quote = literal[0]
            return f"{quote}{token}{quote}"

        return pattern.sub(_swap, source), mapping
    except Exception:
        return source, {}


def restore_elided_literals(source, mapping):
    """Put the original payloads back, undoing `elide_long_string_literals`. Never raises."""
    try:
        if not source or not mapping:
            return source
        out = source
        for token, literal in mapping.items():
            quote = literal[0]
            out = out.replace(f"{quote}{token}{quote}", literal)
        return out
    except Exception:
        return source


def restore_and_verify(source, mapping):
    """`restore_elided_literals` plus proof that it worked. Returns `(restored, ok)`.

    Elision is only safe if every placeholder comes back. A repair pass can rename, split, or
    delete one -- and if it does, the payload is unrecoverable. Writing the file anyway would put
    `__PYLLM_PAYLOAD_0__` where hundreds of kilobytes of base64 used to be: output that still
    compiles, still reports zero deleted lines, and would score as a genuine repair while being
    silently corrupt. `ok` is False whenever any placeholder is unaccounted for, so the caller can
    fail the file instead of shipping it. Never raises."""
    try:
        if not mapping:
            return source, True
        if not source:
            return source, False
        restored = restore_elided_literals(source, mapping)
        for token, literal in mapping.items():
            # The payload must be BACK. Checking only that the token is gone would pass vacuously
            # when a repair pass deleted or renamed the placeholder -- precisely the cases where
            # the payload is lost.
            if literal not in restored:
                return restored, False
            if token in restored:
                return restored, False   # a second occurrence survived unrestored
        return restored, True
    except Exception:
        return source, False


_TRY = re.compile(r"^(\s*)try\s*:")
_HANDLER = re.compile(r"^\s*(except|finally)\b")
_LITERAL_LHS = re.compile(r"^(\s*)(\d+)(\s*=\s*)(?!=)")


def complete_orphan_try(source, error=None):
    """Give a handler-less `try:` an `except Exception: pass` instead of deleting its whole body.

    The dominant structural defect in decompiled malware: 67% of files that only compiled via
    line-deletion contain a `try:` whose except/finally the decompiler never emitted. The fallback's
    only recourse is to delete the try AND its entire suite -- a median of 243 real lines across
    this corpus. Synthesizing the two missing scaffold lines keeps every original statement.

    This ADDS code, so it is not a faithful reconstruction: the caller must report such files in a
    dedicated "scaffolded" bucket (below genuine, above deletion), never folded into the genuine
    rate. Returns the rewritten source, or None when there is no orphan `try:`. Never raises."""
    try:
        if not source:
            return None
        lines = source.splitlines(keepends=True)
        for i, line in enumerate(lines):
            m = _TRY.match(line)
            if not m:
                continue
            indent = len(m.group(1))
            end = len(lines)
            handled = False
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()
                if not stripped:
                    continue
                cur = len(lines[j]) - len(lines[j].lstrip())
                if cur > indent:
                    continue                      # still inside the try suite
                if cur == indent and _HANDLER.match(lines[j]):
                    handled = True
                end = j
                break
            if handled:
                continue
            pad = " " * indent
            patch = [f"{pad}except Exception:\n", f"{pad}    pass\n"]
            return "".join(lines[:end] + patch + lines[end:])
        return None
    except Exception:
        return None


def literal_lhs_rename(source, error=None):
    """Rewrite `72 = expr` (a literal on the left of `=`) to `_lit_72 = expr`.

    The decompiler sometimes loses a binding's name and emits the constant in its place, producing
    a line Python can never parse; 13% of delete-only malware files contain one. Deletion throws
    away the whole statement including its right-hand side -- usually the analytically interesting
    part (the call chain). Renaming the target keeps the expression intact.

    Only NUMERIC targets are rewritten: they cannot be valid assignment targets under any Python
    version, so the rewrite can never alter the meaning of otherwise-legal code. Returns the
    rewritten source, or None when no literal target is present. Never raises."""
    try:
        if not source:
            return None
        out, changed = [], False
        for line in source.splitlines(keepends=True):
            m = _LITERAL_LHS.match(line)
            if m:
                out.append(f"{m.group(1)}_lit_{m.group(2)}{m.group(3)}{line[m.end():]}")
                changed = True
            else:
                out.append(line)
        return "".join(out) if changed else None
    except Exception:
        return None


_COMPLETE_NUMBER = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_COMPLETE_STRING = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"")


def recover_truncated_literal(line, gt_sequences):
    """Rebuild a decompiler-TRUNCATED constant sequence from the ground truth's own constants.

    PyLingual stops emitting a constant sequence after 51 elements, leaving a dangling opening
    quote inside an unclosed bracket. Closing that literal blind (what `balance_delimiters` does)
    compiles but DISCARDS the remaining elements, so the repair scores "genuine" while being
    unfaithful. The complete sequence survives in the GT `.pyc`, so this recovery is exact.

    `line` is the truncated source line; `gt_sequences` is an iterable of tuples/lists/frozensets
    of constants harvested from the GT code-object tree. Returns the completed line, or None when
    recovery is not safe: no dangling quote, no GT sequence extending the parsed prefix, or MORE
    THAN ONE candidate (emitting the wrong constants would look faithful while being wrong, which
    is worse than leaving the line alone). Never raises.
    """
    try:
        if not line or not gt_sequences:
            return None
        text = str(line).rstrip("\n")

        # A truncated literal ends with an unclosed quote: odd quote count on the line.
        if (text.count("'") + text.count('"')) % 2 == 0:
            return None

        prefix_elements = [m.group(1) if m.group(1) is not None else m.group(2)
                           for m in _COMPLETE_STRING.finditer(text)]
        if not prefix_elements:
            # Numeric payloads (`[98, 97, 115, 101, ...]`) are common in packed malware and carry
            # no quotes at all, so the string scanner finds nothing for them.
            prefix_elements = [int(n) for n in _COMPLETE_NUMBER.findall(text)]
        if not prefix_elements:
            return None

        def _candidates(prefix):
            """GT sequences that extend `prefix`. Ordered containers must match positionally;
            a frozenset has NO source order, so positional matching would reject the very
            sequence we are looking for -- those match by containment instead."""
            found = []
            for seq in gt_sequences:
                try:
                    items = list(seq)
                except TypeError:
                    continue
                if len(items) <= len(prefix):
                    continue  # nothing was lost -- not the truncation case
                if not all(isinstance(x, (str, int, bytes)) for x in items):
                    continue
                if isinstance(seq, frozenset) or isinstance(seq, set):
                    if set(prefix) <= set(items):
                        rest = [x for x in items if x not in set(prefix)]
                        found.append(list(prefix) + rest)   # keep the source's observed order
                elif items[:len(prefix)] == list(prefix):
                    found.append(items)
            return found

        matches = _candidates(prefix_elements)
        if not matches and len(prefix_elements) > 1:
            # The decompiler may have stopped MID-element, so the last thing we parsed is a
            # fragment rather than a real element; retry without it. The candidate must still
            # extend BEYOND what the source already shows -- otherwise this "recovers" a literal
            # that was never truncated, which is how the retry first went wrong.
            matches = [m for m in _candidates(prefix_elements[:-1])
                       if len(m) > len(prefix_elements)]
        if not matches:
            return None
        if len(matches) > 1:
            # Ambiguous: prefer a single longest candidate (the most complete recovery). Only
            # decline when even that is tied, since emitting the wrong constants would look
            # faithful while being wrong.
            longest = max(len(m) for m in matches)
            top = [m for m in matches if len(m) == longest]
            if len(top) != 1:
                return None
            matches = top

        full = matches[0]
        head = text[:text.index(_COMPLETE_STRING.search(text).group(0))]
        depth = 0
        for ch in head:
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth -= 1
        if depth <= 0:
            return None  # the elements are not inside an open bracket

        closers = ""
        stack = []
        for ch in head:
            if ch in _OPEN:
                stack.append(ch)
            elif ch in _CLOSE and stack:
                stack.pop()
        for opener in reversed(stack):
            closers += _OPEN[opener]

        return head + ", ".join(repr(x) for x in full) + closers
    except Exception:
        return None


def splice_truncated_literals(source, error, gt_sequences):
    """Whole-file wrapper around `recover_truncated_literal`: rebuild ONE truncated constant
    sequence from the ground truth's own constants.

    Repairs a single line per call so the driver re-probes between steps -- a file can carry
    several truncated payloads, and splicing them blind in one pass would hide a bad candidate
    behind a good one. The line the parser is complaining about is tried first (that is the one
    blocking compilation); otherwise every line the truncation signature matches is tried in order.

    Returns the rewritten source, or None when nothing was recovered. Never raises.
    """
    try:
        if not source or not gt_sequences:
            return None
        lines = str(source).splitlines(keepends=True)
        if not lines:
            return None

        order = []
        lineno = getattr(error, "lineno", None)
        if lineno and 1 <= int(lineno) <= len(lines):
            order.append(int(lineno) - 1)
        order.extend(i for i in range(len(lines)) if i not in order)

        for index in order:
            line = lines[index]
            if not is_truncated_literal_line(line):
                continue
            recovered = recover_truncated_literal(line, gt_sequences)
            if not recovered:
                continue
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = recovered.rstrip("\n") + newline
            return "".join(lines)
        return None
    except Exception:
        return None


def has_decompiler_surrender_markers(source) -> bool:
    """True when the decompiler explicitly reported it never recovered some content.

    `<mask_N>` and `<CodeNNN code object>` are PyLingual's own surrender markers: the bytes behind
    them were never turned into source. No prompt, operator, or amount of sampling can invent them,
    so spending the full retry budget on such a file is wasted compute (measured: 16.8% of the
    delete-only set). Never raises."""
    if not source:
        return False
    return bool(_SURRENDER_MARKERS.search(str(source)))


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


# Ordered by FIDELITY, most faithful first. `literal_lhs_rename` only rewrites targets that could
# never be valid Python, so it is as safe as the existing operators. `complete_orphan_try` is LAST
# because it is the only operator that SYNTHESIZES code: it must never pre-empt a transform that
# repairs the file without adding anything.
_OPERATORS = [balance_delimiters, dedent_stray_block, fix_line_continuation, fix_numeric_literal,
              literal_lhs_rename, complete_orphan_try]

# Decompiled malware carries many independent defects per file (a median of 243 deleted lines in the
# delete-only set), and each round fixes at most one. Six rounds stops long before the operators run
# out of work -- measured: they keep making progress on 70% of non-parsing files well past round 6.
_DEFAULT_PREPASS_ROUNDS = 24


def _operators_for(gt_sequences):
    """The operator list, with the GT literal splice FIRST when ground truth is available.

    Position is the whole point. `balance_delimiters` also makes a truncated-literal file compile,
    by closing the bracket and discarding every element the decompiler never emitted -- a repair
    that deletes no source line and so scores "genuine" while being unfaithful. The splice restores
    those elements exactly. Whichever operator fires first wins, so the splice must precede it.
    With no ground truth the list is the shipped one, unchanged.
    """
    if not gt_sequences:
        return _OPERATORS

    def splice(source, error):
        return splice_truncated_literals(source, error, gt_sequences)

    splice.__name__ = "splice_truncated_literals"
    return [splice] + _OPERATORS


def run_syntactic_prepass(source: str, compile_fn=host_compile, max_rounds: int = _DEFAULT_PREPASS_ROUNDS, gt_sequences=None) -> PrepassResult:
    """Try operators in order each round, accepting the first candidate that
    compiles or strictly advances past the current error. Never regresses.

    `gt_sequences` are constant sequences harvested from the ground-truth `.pyc`
    (`gt_syntactic_context.harvest_constant_sequences`). When supplied they enable the truncated-
    literal splice ahead of every other operator; when omitted the behaviour is the shipped one."""
    cur = source; ops: list[str] = []
    operators = _operators_for(gt_sequences)
    for rnd in range(1, max_rounds + 1):
        before = probe_syntax(cur, compile_fn)
        if before is None:
            return PrepassResult(cur, True, ops, rnd - 1)
        applied = False
        for op in operators:
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
