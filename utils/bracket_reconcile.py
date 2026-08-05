from __future__ import annotations

import re

__all__ = ["reconcile_brackets", "rename_invalid_targets", "logical_statements"]

_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_OPEN = "([{"
_CLOSE = ")]}"
_MATCH = {"(": ")", "[": "]", "{": "}"}
_OPEN_FOR = {v: k for k, v in _MATCH.items()}


def _mask(line):
    masked = _STRING.sub(lambda m: " " * len(m.group(0)), line)
    hash_pos = masked.find("#")
    if hash_pos != -1:
        masked = masked[:hash_pos] + " " * (len(masked) - hash_pos)
    return masked


def logical_statements(lines):
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
    return bool(_CONTINUES.search(masked_line.rstrip()))


def reconcile_brackets(source):
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
    t = target.strip()
    if not t or _IDENT.match(t):
        return False
    if any(op in t for op in ("(", ")", "[", "]", "{", "}", ",", '"', "'", " ")):
        return False
    return bool(re.match(r"^[0-9A-Fa-f]", t)) and bool(re.search(r"[:\-]", t))


def rename_invalid_targets(source):
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
