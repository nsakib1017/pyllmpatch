"""Mechanically rebuild container literals the decompiler truncated mid-way.

WHY THIS EXISTS. PyLingual sometimes emits a line that stops in the middle of a container literal --
``if host in {'a', 'b', 'c`` -- because the literal was too long for its model. The file then cannot
parse, and the pipeline's only remaining move is the delete-only fallback, which throws the whole
statement away. That is how ~87% of the deletion-classified malware files fail: not one fatal defect,
but a median of 8-9 truncated lines each.

The contents of those literals are CONSTANTS in the ground-truth ``.pyc``. Reading them back is not an
oracle shortcut -- the ``.pyc`` is the decompiler's own input, and recovering a constant from it is
precisely the decompilation that was declined.

DESIGN NOTE (this is the second design; the first one failed).
The obvious approach -- parse, fix the first SyntaxError, re-parse, repeat -- measured 0/41 and 0/50.
It fails because these files hold MANY defects of SEVERAL classes at once: repairing a truncated
literal routinely exposes a *different* defect, sometimes on an EARLIER line that the broken literal
had been masking, so "did the error line advance?" is not a usable progress signal.

So this module does not follow the parser. It scans lexically for every truncated line in one pass,
repairs them all, and only then hands the result to the parser and to the existing operator layer.
Whether a file ultimately compiles is somebody else's decision; this module's contract is narrow:
remove the truncated-literal defect class, and report honestly which lines it could not source from
ground truth.

FIDELITY. Two outcomes are tracked separately and must never be conflated:
  * ``recovered``    -- the full literal came from the GT constant. Faithful.
  * ``lossy_closed`` -- no GT constant matched, so the literal was closed around the elements that
                        survived. The statement is preserved and no line is deleted, but content IS
                        missing. Callers that care about fidelity must treat these as lossy.
"""
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
    """Blank out every *complete* string literal, leaving structure and any dangling opener."""
    return _STRING.sub(lambda m: " " * len(m.group(0)), line)


def find_truncated_lines(source):
    """Return ``[(lineno, quote_char)]`` for lines that open a string and never close it.

    Lexical, not syntactic: these files do not parse, so the parser cannot be the detector. Complete
    strings are blanked first, so an apostrophe inside a finished string -- or inside a comment that
    follows one -- cannot raise a false positive.
    """
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
    """Ground-truth container constants, indexed by every string element they contain.

    Indexing by *every* element (not just the first) matters: the surviving prefix of a truncated
    literal may begin part-way through the container as the decompiler wrote it, and set ordering in
    a ``.pyc`` constant does not match source order at all.
    """

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
        """Smallest container that contains EVERY visible element, else the largest superset.

        Requiring a superset of all visible elements is what keeps this honest: a container that is
        missing one of the elements actually present in the source is a different literal, and
        splicing it would silently rewrite the malware's behaviour.
        """
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
    """Repair one truncated line. Returns ``(new_line, kind)``; kind is 'recovered'|'lossy'|None."""
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
    """Repair EVERY truncated line in one pass.

    Returns ``(new_source, stats)`` with ``stats`` = recovered / lossy_closed / unrepairable.
    Source with no truncated line is returned byte-identical -- repairing intact code is damage.
    """
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
    """Every container constant anywhere in a ground-truth ``.pyc``, as a :class:`LiteralIndex`.

    Large set/list displays compile to ``LOAD_CONST <frozenset|tuple>`` followed by
    ``SET_UPDATE``/``LIST_EXTEND``, so the whole literal is reachable from ``co_consts`` and no
    instruction decoding is required. Raises nothing: an unreadable ``.pyc`` yields an empty index,
    and the caller degrades to lossy closing.
    """
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
