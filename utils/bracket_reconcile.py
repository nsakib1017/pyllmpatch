"""Statement-local bracket reconciliation and invalid-identifier renaming.

Both repairs come straight from a measured residual taxonomy (see
``results/benchmark_results/FIX_METHODOLOGY.md``): after literal recovery and structural closure,
brackets account for 43% of what still blocks the `failed` malware files and 33% for the deletion
files, and invalid assignment targets for another 9%/18%.

WHY STATEMENT-LOCAL. The previous closure pass balanced brackets using a WHOLE-FILE stack. On this
corpus that is actively harmful: a single mismatched pair mid-file corrupts every depth count after
it, so the pass appends nonsense closers at end of file. It emitted 929 closers across 173 files and
1,105 across 100 -- and "unmatched ')'" then became the second-largest residual, i.e. the repair was
manufacturing the defect it was meant to remove. Reconciling inside one logical statement bounds the
blast radius: a corrupt statement can no longer poison its neighbours.

Three repairs, all local, none of which deletes a line:
  * a closer with nothing open  -> delete that closer (never previously attempted, and the single
    largest gap in the taxonomy);
  * a closer that mismatches the innermost opener -> rewrite it to match;
  * brackets still open at statement end -> append the closers on that line.

FIDELITY. Every edit here INVENTS structure; none of it is recovered from ground truth. Callers must
count these files as "made analysable", never as faithful repairs.
"""
from __future__ import annotations

import re

__all__ = ["reconcile_brackets", "rename_invalid_targets", "logical_statements"]

_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_OPEN = "([{"
_CLOSE = ")]}"
_MATCH = {"(": ")", "[": "]", "{": "}"}
_OPEN_FOR = {v: k for k, v in _MATCH.items()}


def _mask(line):
    """Replace complete strings and any trailing comment with spaces, preserving column positions."""
    masked = _STRING.sub(lambda m: " " * len(m.group(0)), line)
    hash_pos = masked.find("#")
    if hash_pos != -1:
        masked = masked[:hash_pos] + " " * (len(masked) - hash_pos)
    return masked


def logical_statements(lines):
    """Group physical lines into logical statements by bracket depth. Yields ``(start, end)``."""
    groups = []
    depth = 0
    start = 0
    for i, raw in enumerate(lines):
        masked = _mask(raw)
        for ch in masked:
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth = max(0, depth - 1)
        if depth == 0:
            groups.append((start, i))
            start = i + 1
    if start < len(lines):
        groups.append((start, len(lines) - 1))
    return groups


def _reconcile_one(text):
    """Reconcile a single logical statement. Returns ``(new_text, counters)``."""
    counters = {"surplus_deleted": 0, "mismatched_rewritten": 0, "closers_appended": 0}
    masked = _mask(text)
    chars = list(text)
    drop = set()
    stack = []
    for i, ch in enumerate(masked):
        if ch in _OPEN:
            stack.append((ch, i))
        elif ch in _CLOSE:
            if not stack:
                drop.add(i)  # closes nothing -> surplus
                counters["surplus_deleted"] += 1
                continue
            opener, _pos = stack[-1]
            if _MATCH[opener] == ch:
                stack.pop()
            else:
                chars[i] = _MATCH[opener]  # wrong kind -> rewrite to match
                counters["mismatched_rewritten"] += 1
                stack.pop()
    out = "".join(c for i, c in enumerate(chars) if i not in drop)
    if stack:
        closers = "".join(_MATCH[o] for o, _ in reversed(stack))
        counters["closers_appended"] = len(stack)
        out = out.rstrip("\n") + closers
    return out, counters


_CONTINUES = re.compile(r"[,+\-*/%<>=&|^~([{:\\]\s*$")


def _is_continuation(masked_line):
    """True when a line legitimately continues onto the next one.

    A genuine multi-line call ends on a comma, an operator or an opener. A TRUNCATED line ends on a
    value. Without this test, closing an unclosed bracket at statement end lets it swallow the
    following statement -- the same defect that made whole-file balancing harmful.
    """
    return bool(_CONTINUES.search(masked_line.rstrip()))


def reconcile_brackets(source):
    """Reconcile brackets locally, closing a truncated line on that line.

    Balanced source is returned byte-identical -- repairing intact code is damage, not repair.
    """
    ledger = {"surplus_deleted": 0, "mismatched_rewritten": 0, "closers_appended": 0, "total_edits": 0}
    if not source.strip():
        return source, ledger

    lines = source.splitlines()
    out_lines = list(lines)
    for start, end in logical_statements(lines):
        # A multi-line group whose every line legitimately continues is a real construct: leave it.
        genuine = all(_is_continuation(_mask(lines[i])) for i in range(start, end)) if end > start else False
        if genuine:
            chunk = "\n".join(lines[start : end + 1])
            fixed, counts = _reconcile_one(chunk)
            if any(counts.values()):
                for key in counts:
                    ledger[key] += counts[key]
                new = fixed.split("\n")
                while len(new) < (end - start + 1):
                    new.append("")
                out_lines[start : end + 1] = new[: end - start + 1]
            continue
        # Otherwise treat each physical line as its own repair unit.
        for i in range(start, end + 1):
            if _is_continuation(_mask(lines[i])):
                continue
            fixed, counts = _reconcile_one(lines[i])
            if any(counts.values()):
                for key in counts:
                    ledger[key] += counts[key]
                out_lines[i] = fixed.replace("\n", "")

    ledger["total_edits"] = (
        ledger["surplus_deleted"] + ledger["mismatched_rewritten"] + ledger["closers_appended"]
    )
    text = "\n".join(out_lines)
    if source.endswith("\n"):
        text += "\n"
    return text, ledger


# ---------------------------------------------------------------- invalid targets

_ASSIGN = re.compile(r"^(\s*)([^=\n]+?)(\s*=\s*)(?!=)(.*)$")
_IDENT = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
_SANITISE = re.compile(r"\W")


def _looks_like_lost_name(target):
    """A MAC/GUID/hex constant the decompiler emitted where a NAME belonged."""
    t = target.strip()
    if not t or _IDENT.match(t):
        return False
    if any(op in t for op in ("(", ")", "[", "]", "{", "}", ",", '"', "'", " ")):
        return False
    return bool(re.match(r"^[0-9A-Fa-f]", t)) and bool(re.search(r"[:\-]", t))


def rename_invalid_targets(source):
    """Rename assignment targets that are constants, not identifiers.

    ``1e:6c:34:93:68:64 = [...]`` and ``7AB5C494-39F5-...False = [...]`` are MAC addresses and GUIDs
    emitted where a variable name belonged; the parser reports ``invalid decimal literal``. The name
    is already lost, so sanitising it costs no fidelity -- and crucially the right-hand payload is
    preserved verbatim, which deletion would have destroyed.
    """
    ledger = {"renamed": 0}
    if not source.strip():
        return source, ledger
    mapping = {}
    out = []
    for line in source.splitlines():
        m = _ASSIGN.match(line)
        if m and _looks_like_lost_name(m.group(2)):
            original = m.group(2).strip()
            if original not in mapping:
                mapping[original] = "_" + _SANITISE.sub("_", original)
            out.append(f"{m.group(1)}{mapping[original]}{m.group(3)}{m.group(4)}")
            ledger["renamed"] += 1
        else:
            out.append(line)
    if mapping:
        # Same token elsewhere in the file must resolve to the same new name.
        for i, line in enumerate(out):
            for original, replacement in mapping.items():
                if original in line and not line.lstrip().startswith(replacement):
                    out[i] = line.replace(original, replacement)
                    line = out[i]
    text = "\n".join(out)
    if source.endswith("\n"):
        text += "\n"
    return text, ledger
