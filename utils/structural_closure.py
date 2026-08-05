from __future__ import annotations

import ast
import re

__all__ = [
    "close_source",
    "close_open_brackets",
    "close_unterminated_string",
    "NEUTRALISED_MARKER",
]

_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_OPEN = "([{"
_CLOSE = ")]}"
_MATCH = {"(": ")", "[": "]", "{": "}"}
NEUTRALISED_MARKER = "# pyllmpatch: unparsed"

_BLOCK_HEADER = re.compile(
    r"^(\s*)(if|elif|else|for|while|with|try|except|finally|def|class|async\s+def)\b.*:\s*(#.*)?$"
)


def _blank_strings(line):
    return _STRING.sub(lambda m: " " * len(m.group(0)), line)


def close_unterminated_string(line):
    rest = _blank_strings(line)
    for ch in rest:
        if ch in "'\"":
            return line + ch, True
        if ch == "#":
            break
    return line, False


def close_line_brackets(line):
    scan = _blank_strings(line).split("#", 1)[0]
    stack = []
    for ch in scan:
        if ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE and stack and _MATCH[stack[-1]] == ch:
            stack.pop()
    if not stack:
        return line, 0
    return line + "".join(_MATCH[c] for c in reversed(stack)), len(stack)


def close_open_brackets(text):
    stack = []
    for raw in text.splitlines():
        line = _blank_strings(raw)
        line = line.split("#", 1)[0]
        for ch in line:
            if ch in _OPEN:
                stack.append(ch)
            elif ch in _CLOSE and stack and _MATCH[stack[-1]] == ch:
                stack.pop()
    if not stack:
        return text, 0
    closers = "".join(_MATCH[c] for c in reversed(stack))
    if not text.endswith("\n"):
        text += "\n"
    return text + closers + "\n", len(stack)


def _indent_of(line):
    return len(line) - len(line.lstrip())


def _synthesise_handler(lines, lineno):
    idx = min(max(lineno - 1, 0), len(lines) - 1)
    for i in range(idx, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("try") and stripped.rstrip().endswith(":"):
            indent = " " * _indent_of(lines[i])
            insert = i + 1
            while insert < len(lines) and (
                not lines[insert].strip() or _indent_of(lines[insert]) > _indent_of(lines[i])
            ):
                insert += 1
            lines[insert:insert] = [indent + "except Exception:", indent + "    pass"]
            return True
    return False


def _insert_body(lines, lineno):
    idx = min(max(lineno - 1, 0), len(lines) - 1)
    for i in range(idx, -1, -1):
        if _BLOCK_HEADER.match(lines[i]):
            indent = " " * (_indent_of(lines[i]) + 4)
            lines.insert(i + 1, indent + "pass")
            return True
    return False


_HANDLER = re.compile(r"^\s*(except\b|finally\b)")


def _insert_try(lines, lineno):
    idx = lineno - 1
    if not (0 <= idx < len(lines)):
        return False
    # The reported line may be just after the handler; walk back onto it.
    while idx >= 0 and not _HANDLER.match(lines[idx]):
        idx -= 1
        if lineno - idx > 4:
            return False
    if idx < 0:
        return False
    indent = _indent_of(lines[idx])
    # First statement of the suite this handler belongs to.
    start = None
    for j in range(idx - 1, -1, -1):
        if not lines[j].strip():
            continue
        cur = _indent_of(lines[j])
        if cur < indent:
            break
        if cur == indent:
            start = j
        else:
            continue
    if start is None:
        return False
    for j in range(start, idx):
        if lines[j].strip():
            lines[j] = "    " + lines[j]
    lines.insert(start, " " * indent + "try:")
    return True


def _dedent_line(lines, lineno):
    idx = lineno - 1
    if not (0 <= idx < len(lines)):
        return False
    cur = _indent_of(lines[idx])
    for j in range(idx - 1, -1, -1):
        if lines[j].strip():
            target = _indent_of(lines[j])
            if target < cur:
                lines[idx] = " " * target + lines[idx].lstrip()
                return True
            break
    if cur:
        lines[idx] = lines[idx].lstrip()
        return True
    return False


_LITERAL_TARGET = re.compile(r"^(\s*)([0-9][^\s=]*|[^\s=]*[:\-][^\s=]*)(\s*=\s*)(?!=)(.*)$")


def _rename_literal_target(lines, lineno):
    idx = lineno - 1
    if not (0 <= idx < len(lines)):
        return False
    m = _LITERAL_TARGET.match(lines[idx])
    if not m:
        return False
    safe = "_" + re.sub(r"\W", "_", m.group(2).strip())
    lines[idx] = f"{m.group(1)}{safe}{m.group(3)}{m.group(4)}"
    return True


def _fix_assignment_in_condition(lines, lineno):
    idx = lineno - 1
    if not (0 <= idx < len(lines)):
        return False
    line = lines[idx]
    if not re.match(r"\s*(if|elif|while)\b", line):
        return False
    new = re.sub(r"(?<![=!<>])=(?!=)", "==", line, count=1)
    if new == line:
        return False
    lines[idx] = new
    return True


def _strip_continuation(lines, lineno):
    idx = lineno - 1
    if not (0 <= idx < len(lines)) or "\\" not in lines[idx]:
        return False
    lines[idx] = lines[idx].replace("\\", " ")
    return True


def _neutralise_line(lines, lineno, force=False):
    idx = lineno - 1
    if not (0 <= idx < len(lines)):
        return False
    raw = lines[idx]
    if not raw.strip() or raw.strip().startswith("#") or NEUTRALISED_MARKER in raw:
        return False
    # If the line parses on its own, IT is not the defect -- the enclosing block is. Neutralising it
    # destroys a valid statement and leaves the real problem untouched (seen on `pass` lines inside a
    # `match` whose `case` clauses were the actual issue).
    if not force:
        try:
            ast.parse(raw.strip())
            return False
        except SyntaxError:
            pass
    indent = " " * _indent_of(raw)
    if force:
        # Forced neutralisation also normalises indentation to the enclosing context: a stray indent
        # is itself a defect no other handler covers, and the string must land at a legal level.
        for j in range(idx - 1, -1, -1):
            if lines[j].strip():
                prev = lines[j]
                indent = " " * (_indent_of(prev) + (4 if prev.rstrip().endswith(":") else 0))
                break
    body = raw.strip()
    if body.endswith(":"):
        body = body[:-1]
    # The marker costs no bytecode (comments and bare constants are both discarded by the compiler)
    # and makes every hole machine-findable -- for prompt construction, for auditing, and so a later
    # pass can tell "we could not parse this" apart from a string the program genuinely contained.
    lines[idx] = f"{indent}{body!r}  {NEUTRALISED_MARKER}"
    return True


def close_source(source, max_rounds=None):
    ledger = {
        "strings_closed": 0,
        "brackets_closed": 0,
        "handlers_synthesised": 0,
        "bodies_inserted": 0,
        "dedents": 0,
        "tries_inserted": 0,
        "targets_renamed": 0,
        "conditions_fixed": 0,
        "continuations_stripped": 0,
        "lines_neutralised": 0,
        "lines_deleted": 0,
        "rounds": 0,
        "total_edits": 0,
        "parses": False,
        "faithful": True,
    }
    if not source.strip():
        ledger["parses"] = True
        return source, ledger
    if max_rounds is None:
        # Each round resolves at most one defect, and there are at most as many defects as lines.
        max_rounds = source.count("\n") + 200

    # Pass 1 -- close every truncated string lexically, all sites at once, and close any bracket the
    # same line left open (on that line, so it cannot swallow later insertions).
    lines = source.splitlines()
    for i, line in enumerate(lines):
        fixed, changed = close_unterminated_string(line)
        if changed:
            fixed, n = close_line_brackets(fixed)
            lines[i] = fixed
            ledger["strings_closed"] += 1
            ledger["brackets_closed"] += n
    text = "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    # Pass 2 -- reconcile brackets STATEMENT-LOCALLY. Whole-file balancing appends closers at EOF,
    # which puts a following `except` inside an open bracket and sends the try-insertion handler into
    # an infinite wrap. Statement-local repair bounds the blast radius.
    from utils.bracket_reconcile import reconcile_brackets as _reconcile

    text, _bl = _reconcile(text)
    ledger["brackets_closed"] += _bl["total_edits"]

    # Pass 3 -- the parser now reports one structural defect at a time, and each has a
    # deterministic minimal completion, so this terminates rather than searching.
    # Store digests, not copies: retaining every intermediate text costs gigabytes on a 1 MB file
    # across thousands of rounds, which OOM-killed the first full run.
    import hashlib

    seen = set()
    last_state = None
    last_msg = None
    msg_repeats = 0
    stuck = 0

    def _digest(t):
        return hashlib.blake2b(t.encode("utf-8", "replace"), digest_size=16).digest()

    while ledger["rounds"] < max_rounds:
        try:
            ast.parse(text)
            ledger["parses"] = True
            break
        except SyntaxError as exc:
            ledger["rounds"] += 1
            _d = _digest(text)
            if _d in seen:
                break
            seen.add(_d)
            msg = exc.msg or ""
            lines = text.splitlines()
            lineno = exc.lineno or len(lines)
            # Inserting a line shifts every subsequent line number, so a handler that keeps firing
            # uselessly still looks "new" by (line, msg). Count message repeats instead.
            msg_repeats = msg_repeats + 1 if msg == last_msg else 0
            last_msg = msg
            if msg_repeats > 5:
                ok = _neutralise_line(lines, lineno)
                ledger["lines_neutralised"] += int(ok)
                msg_repeats = 0
                if not ok:
                    break
                text = "\n".join(lines) + "\n"
                continue
            state = (lineno, msg)
            if state == last_state:
                # The previous handler reported success but the parser is unmoved. Escalate straight
                # to neutralisation rather than let a no-op handler spin (one file burned 1,752
                # rounds re-synthesising the same `except` clause).
                stuck += 1
                if stuck > 2:
                    break
                ok = _neutralise_line(lines, lineno)
                ledger["lines_neutralised"] += int(ok)
                if not ok:
                    break
                text = "\n".join(lines) + "\n"
                continue
            stuck = 0
            last_state = state
            if "expected 'except' or 'finally' block" in msg:
                ok = _synthesise_handler(lines, lineno)
                ledger["handlers_synthesised"] += int(ok)
            elif "expected an indented block" in msg:
                ok = _insert_body(lines, lineno)
                ledger["bodies_inserted"] += int(ok)
            elif (
                _HANDLER.match(lines[min(max(lineno - 1, 0), len(lines) - 1)] if lines else "")
                and ledger["tries_inserted"] < 50
            ):
                ok = _insert_try(lines, lineno)
                ledger["tries_inserted"] += int(ok)
            elif "unexpected indent" in msg or "unindent does not match" in msg:
                ok = _dedent_line(lines, lineno)
                ledger["dedents"] += int(ok)
            elif "was never closed" in msg or "unmatched" in msg or "does not match" in msg:
                from utils.bracket_reconcile import reconcile_brackets as _rec
                fixed, _b = _rec("\n".join(lines) + "\n")
                ok = _b["total_edits"] > 0 and fixed != "\n".join(lines) + "\n"
                if ok:
                    ledger["brackets_closed"] += _b["total_edits"]
                    lines = fixed.splitlines()
                else:
                    ok = _neutralise_line(lines, lineno)
                    ledger["lines_neutralised"] += int(ok)
            elif "invalid decimal literal" in msg or "cannot assign to literal" in msg:
                ok = _rename_literal_target(lines, lineno)
                ledger["targets_renamed"] += int(ok)
            elif "'=='" in msg or "cannot contain assignment" in msg:
                ok = _fix_assignment_in_condition(lines, lineno)
                ledger["conditions_fixed"] += int(ok)
            elif "line continuation" in msg:
                ok = _strip_continuation(lines, lineno)
                ledger["continuations_stripped"] += int(ok)
            else:
                ok = False
            if not ok:
                # No structural handler applies: preserve the bytes as data and move on.
                ok = _neutralise_line(lines, lineno)
                ledger["lines_neutralised"] += int(ok)
            if not ok:
                # Last resort: neutralise anyway, normalising indentation to the enclosing block.
                ok = _neutralise_line(lines, lineno, force=True)
                ledger["lines_neutralised"] += int(ok)
            if not ok:
                break
            text = "\n".join(lines) + "\n"
        except Exception:
            break

    ledger["total_edits"] = (
        ledger["strings_closed"]
        + ledger["brackets_closed"]
        + ledger["handlers_synthesised"]
        + ledger["bodies_inserted"]
        + ledger["dedents"]
        + ledger["tries_inserted"]
        + ledger["targets_renamed"]
        + ledger["conditions_fixed"]
        + ledger["continuations_stripped"]
        + ledger["lines_neutralised"]
    )
    ledger["faithful"] = ledger["total_edits"] == 0
    return text, ledger
