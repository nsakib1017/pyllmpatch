from __future__ import annotations

import re
from collections import defaultdict

__all__ = [
    "LiteralIndex",
    "find_truncated_lines",
    "repair_source",
    "load_gt_literals",
]

# A complete single-quoted or double-quoted string, honouring backslash escapes.
_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_OPENERS = "{[("
_CLOSER_FOR = {"{": "}", "[": "]", "(": ")"}


def _strip_complete_strings(line):
    return _STRING.sub(lambda m: " " * len(m.group(0)), line)


def find_truncated_lines(source):
    out = []
    for i, line in enumerate(source.splitlines(), start=1):
        rest = _strip_complete_strings(line)
        hash_pos = rest.find("#")
        if hash_pos != -1 and "'" not in rest[:hash_pos] and '"' not in rest[:hash_pos]:
            continue  # a comment, with no dangling quote before it
        q = None
        for ch in rest:
            if ch in "'\"":
                q = ch
                break
        if q is not None:
            out.append((i, q))
    return out


class LiteralIndex:

    def __init__(self, containers=()):
        self._by_element = defaultdict(list)
        self._containers = []
        for c in containers:
            self.add(c)

    def add(self, container):
        try:
            elements = list(container)
        except TypeError:
            return
        if not elements:
            return
        self._containers.append(container)
        for el in elements:
            if isinstance(el, str):
                self._by_element[el].append(container)

    def lookup(self, element):
        return list(self._by_element.get(element, ()))

    def best_match(self, visible):
        visible = [v for v in visible if isinstance(v, str)]
        if not visible:
            return None
        candidates = self.lookup(visible[0])
        best = None
        for c in candidates:
            try:
                members = set(c)
            except TypeError:
                continue
            if not all(v in members for v in visible):
                continue
            if best is None or len(members) > len(set(best)):
                best = c
        return best

    def __len__(self):
        return len(self._containers)


def _render(container, opener):
    if isinstance(container, (set, frozenset)):
        return "{" + ", ".join(repr(x) for x in sorted(container, key=lambda v: (str(type(v)), str(v)))) + "}"
    if opener == "[":
        return repr(list(container))
    if opener == "(":
        return repr(tuple(container))
    return repr(list(container))


def _visible_elements(fragment):
    return [m.group(0)[1:-1] for m in _STRING.finditer(fragment)]


def _repair_line(line, index):
    stripped = _strip_complete_strings(line)
    opener_pos = max((stripped.rfind(o) for o in _OPENERS), default=-1)
    if opener_pos < 0:
        return None, None
    opener = line[opener_pos]
    fragment = line[opener_pos:]
    visible = _visible_elements(fragment)
    if not visible:
        return None, None

    prefix = line[:opener_pos]
    # A container opened inside a still-open call needs that call closed too.
    depth = sum(prefix.count(o) - prefix.count(_CLOSER_FOR[o]) for o in _OPENERS)
    tail = ")" * max(0, depth)
    if re.match(r"\s*(if|elif|while|for|with|try|def|class|else|except|finally)\b", prefix):
        tail += ":"

    container = index.best_match(visible)
    if container is not None:
        return prefix + _render(container, opener) + tail, "recovered"
    # No ground truth: close around what survived. Lossy, but the statement is preserved.
    kept = ", ".join(repr(v) for v in visible)
    return prefix + opener + kept + _CLOSER_FOR[opener] + tail, "lossy"


def repair_source(source, index):
    truncated = find_truncated_lines(source)
    stats = {"truncated": len(truncated), "recovered": 0, "lossy_closed": 0, "unrepairable": 0}
    if not truncated:
        return source, stats

    lines = source.splitlines()
    for lineno, _q in truncated:
        if lineno > len(lines):
            continue
        new_line, kind = _repair_line(lines[lineno - 1], index)
        if kind == "recovered":
            lines[lineno - 1] = new_line
            stats["recovered"] += 1
        elif kind == "lossy":
            lines[lineno - 1] = new_line
            stats["lossy_closed"] += 1
        else:
            stats["unrepairable"] += 1
    return "\n".join(lines) + ("\n" if source.endswith("\n") else ""), stats


def load_gt_literals(pyc_path):
    index = LiteralIndex()
    try:
        from xdis.load import load_module

        loaded = load_module(str(pyc_path))
    except Exception:
        return index
    roots = [obj for obj in loaded if hasattr(obj, "co_code")]
    if not roots:
        return index

    seen = set()

    def walk(code):
        if id(code) in seen:
            return
        seen.add(id(code))
        for const in getattr(code, "co_consts", ()):
            if hasattr(const, "co_code"):
                walk(const)
            elif isinstance(const, (frozenset, set, tuple, list)):
                index.add(const)
            elif isinstance(const, dict):
                index.add(tuple(const.keys()))

    walk(roots[0])
    return index
