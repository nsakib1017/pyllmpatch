"""Stateful whole-file lexical scanner: the piece every structural transform needs.

Per-line scanning is not enough -- a docstring or a bracketed call spans lines, and treating its
interior as statements is what made the first version of these transforms rewrite valid code.
"""
_OPEN = {'(': ')', '[': ']', '{': '}'}
_CLOSE = {v: k for k, v in _OPEN.items()}


def scan_file(lines):
    """Return per-line records (depth, active_triple_quote, is_backslash_continuation) describing
    the lexical state AT THE START of each physical line, plus a string/comment-blanked copy.

    Damage model: an unterminated SINGLE-quoted string is treated as ending at end-of-line (that is
    exactly what PyLingual's mid-literal truncation produces); only triple quotes carry over."""
    depth = 0
    q = None
    cont = False
    recs = []
    blanked = []
    for L in lines:
        recs.append((depth, q, cont))
        out = []
        i = 0
        s = L
        n = len(s)
        while i < n:
            c = s[i]
            if q:
                if s.startswith(q, i):
                    out.append('0' * len(q))
                    i += len(q)
                    q = None
                    continue
                out.append('0')
                i += 1
                continue
            if c == '#':
                break
            if c in '\'"':
                if s.startswith(c * 3, i):
                    q = c * 3
                    out.append('000')
                    i += 3
                    continue
                j = i + 1
                while j < n:
                    if s[j] == '\\':
                        j += 2
                        continue
                    if s[j] == c:
                        break
                    j += 1
                end = j + 1 if j < n else n
                # NOT spaces: blanking a string to whitespace makes `x: 'T'` look like a block
                # opener (`x:` after rstrip), which inserted spurious `pass` lines on valid code.
                out.append('0' * (end - i))
                i = end
                continue
            if c in _OPEN:
                depth += 1
            elif c in _CLOSE:
                depth = max(0, depth - 1)
            out.append(c)
            i += 1
        blanked.append(''.join(out))
        cont = L.rstrip('\n').endswith('\\')
    return recs, blanked


def statement_spans(lines):
    """[(start, end_exclusive)] logical statements. Blank / comment-only lines are NOT statements
    (CPython emits no INDENT/DEDENT for them) and are attached to the following statement."""
    recs, blanked = scan_file(lines)
    spans = []
    i = 0
    n = len(lines)
    while i < n:
        d, q, cont = recs[i]
        st = lines[i].strip()
        if d == 0 and q is None and not cont and (not st or st.startswith('#')):
            i += 1
            continue
        start = i
        i += 1
        while i < n:
            d, q, cont = recs[i]
            if d == 0 and q is None and not cont:
                break
            i += 1
        spans.append((start, i))
    return spans, recs, blanked
