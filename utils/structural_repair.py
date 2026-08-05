from __future__ import annotations

import ast
import re
from typing import Callable, NamedTuple


class StructOp(NamedTuple):
    name: str
    fn: Callable[[str, object], str | None]
    inserts_code: bool


_HEADER = re.compile(r"^(\s*)(if|elif|else|for|while|with|try|except|finally|def|class|async)\b.*:\s*(#.*)?$")
_CLAUSE = re.compile(r"^(\s*)(else|elif|except|finally)\b")
_INDENT = re.compile(r"^(\s*)")

# `EXPR for VAR in ITER:` -- a for-loop the decompiler flattened into comprehension shape. The
# trailing colon is what gives it away: a real comprehension never ends a statement with one.
_FLAT_FOR = re.compile(r"^(\s*)(?P<body>\S.*?)\s+for\s+(?P<var>[\w, ]+?)\s+in\s+(?P<iter>.+?):\s*(#.*)?$")
# `for V in ITER for V in [EXPR]:` -- the same damage with the loop header duplicated.
_DOUBLE_FOR = re.compile(
    r"^(\s*)for\s+(?P<v1>[\w, ]+?)\s+in\s+(?P<it1>.+?)\s+for\s+(?P<v2>[\w, ]+?)\s+in\s+\[(?P<expr>.+?)\]\s*:\s*(#.*)?$"
)


def _parses(source) -> bool:
    try:
        ast.parse(source)
        return True
    except Exception:
        return False


def _indent_of(line) -> int:
    return len(_INDENT.match(line).group(1).expandtabs(8))


def give_headerless_block_a_body(source, error=None):
    try:
        return _give_headerless_block_a_body(source, error)
    except Exception:
        return None


def _give_headerless_block_a_body(source, error=None):
    if not source or _parses(source):
        return None
    lines = source.splitlines(keepends=True)
    if not lines:
        return None
    out = []
    changed = False
    for index, line in enumerate(lines):
        out.append(line)
        header = _HEADER.match(line.rstrip("\n"))
        if not header:
            continue
        # find the next line with content
        nxt = None
        for j in range(index + 1, len(lines)):
            if lines[j].strip():
                nxt = lines[j]
                break
        if nxt is None:
            continue
        # A CLAUSE is never a body, however it is indented. PyLingual emits
        #     except Exception:
        #         else:  # inserted
        # and an indent test alone reads that `else:` as the handler's body, so the operator
        # declined the exact shape it exists for. Measured cost: the dominant form of the
        # 168-file dangling-clause residual.
        if _indent_of(nxt) > _indent_of(line) and not _CLAUSE.match(nxt):
            continue  # a real body is present
        # No body. Only act when what follows is a clause or a dedent -- otherwise the file is
        # damaged in some other way and this is not our defect.
        if not (_CLAUSE.match(nxt) or _indent_of(nxt) <= _indent_of(line)):
            continue
        pad = _INDENT.match(line).group(1)
        newline = "\n" if line.endswith("\n") else ""
        out.append(f"{pad}    pass{newline}")
        changed = True
    if not changed:
        return None
    return "".join(out)


def renest_flattened_for(source, error=None):
    try:
        if not source or _parses(source):
            return None
        lines = source.splitlines(keepends=True)
        out, changed = [], False
        for line in lines:
            stripped = line.rstrip("\n")
            m = _FLAT_FOR.match(stripped)
            if not m or _DOUBLE_FOR.match(stripped):
                out.append(line)
                continue
            body = m.group("body").strip()
            # Guard: a bracketed comprehension is a real expression, not this defect.
            if body.startswith(("[", "{", "(")) or body.endswith(("[", "{", "(", ",")):
                out.append(line)
                continue
            pad = m.group(1)
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{pad}for {m.group('var').strip()} in {m.group('iter').strip()}:{nl}")
            out.append(f"{pad}    {body}{nl}")
            changed = True
        if not changed:
            return None
        return "".join(out)
    except Exception:
        return None


def reduce_doubled_for(source, error=None):
    try:
        if not source or _parses(source):
            return None
        lines = source.splitlines(keepends=True)
        out, changed = [], False
        for line in lines:
            m = _DOUBLE_FOR.match(line.rstrip("\n"))
            if not m or m.group("v1").strip() != m.group("v2").strip():
                out.append(line)
                continue
            pad = _INDENT.match(line).group(1)
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{pad}for {m.group('v1').strip()} in {m.group('it1').strip()}:{nl}")
            out.append(f"{pad}    {m.group('expr').strip()}{nl}")
            changed = True
        if not changed:
            return None
        return "".join(out)
    except Exception:
        return None


# A line that begins a NEW statement -- used to decide that a preceding unclosed bracket was never
# meant to continue. A genuine multi-line expression never does this at the same or lower indent.
_NEW_STATEMENT = re.compile(
    r"^\s*(?:(?:if|elif|else|for|while|with|try|except|finally|def|class|async|return|import|"
    r"from|raise|assert|pass|break|continue|global|nonlocal|del|yield)\b|[A-Za-z_]\w*\s*(?:=[^=]|\.|\())"
)


# PyLingual's leaked internal name for a PEP-695 generic-parameter scope, and the dot-prefixed
# synthetic parameters it emits alongside them (`.defaults`, `.type_params`).
_LEAKED_GENERIC = re.compile(r"<generic parameters of ([A-Za-z_][\w]*)>")
_DOT_PARAM = re.compile(r"(?<=[(,\s])\.([A-Za-z_]\w*)")


def normalize_leaked_generic_params(source, error=None):
    try:
        if not source or _parses(source):
            return None
        if not _LEAKED_GENERIC.search(source):
            return None
        out = _LEAKED_GENERIC.sub(lambda m: "_generic_%s" % m.group(1), source)
        # `.defaults` / `.type_params` appear only in the signatures these markers head, so the
        # rename is scoped to lines that carried a marker.
        fixed = []
        for line in out.splitlines(keepends=True):
            if "_generic_" in line and _DOT_PARAM.search(line):
                line = _DOT_PARAM.sub(lambda m: "_" + m.group(1), line)
            fixed.append(line)
        result = "".join(fixed)
        return result if result != source else None
    except Exception:
        return None


# A run of open brackets directly before a statement keyword. `((((( if` is not valid Python at any
# nesting depth -- an open bracket may be followed by an expression, never by a statement keyword --
# so the run is provably surplus. Requires 2+ brackets so a legitimate `(if_expr)` cannot match.
_STATEMENT_KEYWORDS = (
    "if", "elif", "else", "for", "while", "with", "try", "except", "finally", "def", "class",
    "return", "import", "from", "raise", "assert", "pass", "break", "continue", "global",
    "nonlocal", "del", "yield", "lambda", "async", "await",
)
_SURPLUS_LEAD = re.compile(
    r"^(?P<indent>\s*)(?P<junk>[(\[{]{2,})(?P<rest>(?:%s)\b.*)$" % "|".join(_STATEMENT_KEYWORDS)
)


def strip_surplus_leading_delimiters(source, error=None):
    try:
        if not source or _parses(source):
            return None
        lines = source.splitlines(keepends=True)
        out, changed = [], False
        for line in lines:
            m = _SURPLUS_LEAD.match(line.rstrip("\n"))
            if not m:
                out.append(line)
                continue
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{m.group('indent')}{m.group('rest')}{nl}")
            changed = True
        if not changed:
            return None
        return "".join(out)
    except Exception:
        return None


# A `for` clause stranded on its own line, carrying the comprehension's surplus closers and often a
# trailing `if` filter. The comprehension HEAD follows on the next, more-indented line.
_STRANDED_FOR = re.compile(r"^(?P<indent>\s*)(?P<clause>for\s+.+?)\s*$")


_OPEN_BRACKETS = "([{"
_CLOSE_BRACKETS = ")]}"


def _net_open(text):
    depth = 0
    quote = None
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in _OPEN_BRACKETS:
            depth += 1
        elif ch in _CLOSE_BRACKETS:
            depth -= 1
    return depth


def rejoin_split_comprehension(source, error=None):
    try:
        if not source or _parses(source):
            return None
        lines = source.splitlines(keepends=True)
        out, skip, changed = [], False, False
        for index, line in enumerate(lines):
            if skip:
                skip = False
                continue
            stripped = line.rstrip("\n")
            match = _STRANDED_FOR.match(stripped)
            if not match or index + 1 >= len(lines):
                out.append(line)
                continue
            clause = match.group("clause").strip()
            # the clause must carry surplus closers -- that is what marks it as a fragment
            if _net_open(clause) >= 0:
                out.append(line)
                continue
            head_line = lines[index + 1].rstrip("\n")
            head = head_line.strip()
            if not head or _indent_of(head_line) <= _indent_of(line):
                out.append(line)
                continue
            # the head must be an INCOMPLETE expression, i.e. it opens brackets it never closes
            if _net_open(head) <= 0:
                out.append(line)
                continue
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{match.group('indent')}{head} {clause}{nl}")
            skip = True           # the head line has been merged, not dropped
            changed = True
        if not changed:
            return None
        return "".join(out)
    except Exception:
        return None


# Ordered most-faithful first. The surplus-delimiter strip leads because it is the only operator
# here that is provably correct rather than shape-matched, and clearing it can expose the real
# defect underneath; the two re-nesting operators preserve the emitted code and only fix its
# structure; the body-synthesizing one runs last because it is the only one that adds code.
STRUCT_OPS = (
    StructOp("strip_surplus_leading_delimiters", strip_surplus_leading_delimiters, False),
    StructOp("normalize_leaked_generic_params", normalize_leaked_generic_params, False),
    StructOp("rejoin_split_comprehension", rejoin_split_comprehension, False),
    StructOp("reduce_doubled_for", reduce_doubled_for, False),
    StructOp("renest_flattened_for", renest_flattened_for, False),
    StructOp("give_headerless_block_a_body", give_headerless_block_a_body, True),
)


def _unclosed_stack(text):
    """Bracket characters left open in `text`, in order, ignoring string contents."""
    stack = []
    quote = None
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in _OPEN_BRACKETS:
            stack.append(ch)
        elif ch in _CLOSE_BRACKETS and stack:
            stack.pop()
    return stack

def close_unclosed_opener(source, error=None):
    """NOT REGISTERED -- measured NET NEGATIVE (-19 files). Kept for the record only.

    Correct in isolation and fully tested, but it costs 19 malware conversions at ANY position in
    STRUCT_OPS (1030 -> 1011, stage3 61 -> 42). Registering it 2nd let it pre-empt the precise
    operators; moving it LAST did not help either, because `_structural_fixpoint` loops up to 8
    rounds, so its output feeds back to those operators on the next round. Being right about a line
    is not enough when operators compose -- an aggressive rewrite destroys the patterns a precise
    operator needs. Do not re-register without re-measuring the full census.

    Close a bracket the decompiler opened and never closed, on the line that opened it.

    The parser reports these failures far from their cause. A real 3.11 sample blocks on
    `okh = str(th) + '.txt'` -- valid Python -- because an unclosed bracket sits several lines above.
    Every other operator in this module keys on the REPORTED error line, so none of them can reach
    this defect. Measured: 207 of 470 blocked 3.11 files (44%) are unmatched-closer errors, plus an
    unknown share of the "invalid syntax" bucket that is downstream of the same cause.

    The rule is deliberately conservative. A line is only closed when the next non-blank line STARTS
    A NEW STATEMENT at the same or lower indent -- something a genuine multi-line call, list, or
    dict never does, since its continuation lines are either indented further or are bare
    elements. Adds only closing characters: no line is added, removed, or reordered.

    Returns None when nothing matches; never raises."""
    try:
        if not source or _parses(source):
            return None
        lines = source.splitlines(keepends=True)
        out, changed = [], False
        for index, line in enumerate(lines):
            stripped = line.rstrip("\n")
            stack = _unclosed_stack(stripped)
            if not stack:
                out.append(line)
                continue
            nxt = None
            for j in range(index + 1, len(lines)):
                if lines[j].strip():
                    nxt = lines[j]
                    break
            if nxt is None or _indent_of(nxt) > _indent_of(line):
                out.append(line)
                continue
            if not _NEW_STATEMENT.match(nxt):
                out.append(line)
                continue
            closers = "".join(_CLOSE_BRACKETS[_OPEN_BRACKETS.index(ch)] for ch in reversed(stack))
            nl = "\n" if line.endswith("\n") else ""
            out.append(stripped + closers + nl)
            changed = True
        if not changed:
            return None
        return "".join(out)
    except Exception:
        return None
