_OPEN = {'(': ')', '[': ']', '{': '}'}
_CLOSE = {v: k for k, v in _OPEN.items()}


def scan_file(lines):
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
