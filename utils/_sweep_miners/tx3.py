"""Final transform set, all built on the stateful scanner (scanner.statement_spans)."""
import ast, re, keyword
from .scanner import statement_spans, scan_file, _OPEN, _CLOSE

_CLAUSE_KW = re.compile(r'^(\s*)(except\b[^:]*:|else\s*:|finally\s*:|elif\b.*:)(.*)$')
_OPENERS = {
    'except': (re.compile(r'^\s*try\s*:'), re.compile(r'^\s*except\b')),
    'finally': (re.compile(r'^\s*try\s*:'), re.compile(r'^\s*except\b'), re.compile(r'^\s*else\s*:')),
    'else': (re.compile(r'^\s*try\s*:'), re.compile(r'^\s*except\b'), re.compile(r'^\s*(if|elif)\b'),
             re.compile(r'^\s*(for|while)\b'), re.compile(r'^\s*async\s+for\b')),
    'elif': (re.compile(r'^\s*(if|elif)\b'),),
}


def indent_of(s):
    return len(s) - len(s.lstrip(' \t'))


def _kw(line):
    st = line.strip()
    for k in ('except', 'finally', 'elif', 'else'):
        if st.startswith(k):
            return k
    return None


# =========================================================== H1  clause_to_if_true
def clause_to_if_true(source):
    """Rewrite an ORPHAN `else:` / `except X:` / `finally:` HEADER as `if True:`.

    PyLingual constantly emits a clause whose opener it never printed -- typically
    `else:  # inserted` parked one or two levels deeper than the `except:` that already closed.
    The only alternative the pipeline has is deleting the clause AND its entire suite (the branch
    that usually holds the interesting behaviour). Rewriting the four header characters keeps every
    statement and every token of the body.

    Fires only when the clause's same-column predecessor STATEMENT (comments and continuation
    lines skipped) is not a legal opener for that keyword.
    NOT faithful: the branch becomes unconditional, so these files belong in a 'rewritten' bucket,
    not the faithful-genuine rate."""
    lines = source.split('\n')
    spans, _recs, _bl = statement_spans(lines)
    out = list(lines)
    fired = 0
    for si, (a, _b) in enumerate(spans):
        m = _CLAUSE_KW.match(lines[a])
        if not m:
            continue
        k = _kw(lines[a])
        if k is None or k == 'elif':
            continue
        d = indent_of(lines[a])
        peer = None
        for sj in range(si - 1, -1, -1):
            P = lines[spans[sj][0]]
            pd = indent_of(P)
            if pd > d:
                continue
            if pd == d:
                peer = P
            break
        if peer is not None and any(p.match(peer) for p in _OPENERS[k]):
            continue
        out[a] = ' ' * d + 'if True:' + (m.group(3) or '')
        fired += 1
    return '\n'.join(out) if fired else None


# =========================================================== H2  normalize_indentation
def _opens_block(blanked, a, b):
    body = ' '.join(blanked[k] for k in range(a, b))
    return body.rstrip().endswith(':')


def normalize_indentation(source, insert_missing_body=True):
    """Re-anchor every statement to a LEGAL indent column; leading whitespace is the ONLY edit.

    Decompiled malware is full of statements parked where no live block owns the column: a suite
    that resumes between two enclosing levels ("unindent does not match any outer indentation
    level"), a body indented under a statement that opens nothing ("unexpected indent").
    Walks the file with a stack of (original_column -> emitted_column) pairs; a column deeper than
    the innermost live block without a preceding opener snaps back to it, and a column matching no
    live level snaps to the nearest enclosing one. A moved statement takes its whole suite with it,
    so relative structure is preserved. `insert_missing_body` adds the one `pass` a block opener
    needs when its suite was lost -- the only synthesis; turn it off for the pure variant."""
    lines = source.split('\n')
    spans, _recs, blanked = statement_spans(lines)
    if not spans:
        return None
    out = list(lines)
    stack = [(0, 0)]
    pending_open = False
    prev_orig = 0
    inserts = {}
    changed = False
    for (a, b) in spans:
        oi = indent_of(lines[a])
        is_clause = bool(_CLAUSE_KW.match(lines[a]))
        if pending_open:
            if oi > prev_orig and not is_clause:
                # keep the file's OWN indent step (never force 4): on an already-consistent file
                # every column then maps to itself and the transform correctly does not fire.
                stack.append((oi, stack[-1][1] + (oi - prev_orig)))
            elif insert_missing_body:
                inserts.setdefault(a, []).append(' ' * (stack[-1][1] + 4) + 'pass')
                changed = True
        pending_open = False
        while len(stack) > 1 and oi < stack[-1][0]:
            stack.pop()
        new = stack[-1][1]
        delta = new - oi
        if delta:
            changed = True
            for k in range(a, b):
                if not lines[k].strip():
                    continue
                out[k] = ' ' * max(0, indent_of(lines[k]) + delta) + lines[k].lstrip(' \t')
        if _opens_block(blanked, a, b):
            pending_open = True
            prev_orig = oi
    if pending_open and insert_missing_body:
        inserts.setdefault(len(lines), []).append(' ' * (stack[-1][1] + 4) + 'pass')
        changed = True
    if not changed:
        return None
    res = []
    for i, L in enumerate(out):
        for p in inserts.get(i, ()):
            res.append(p)
        res.append(L)
    for p in inserts.get(len(lines), ()):
        res.append(p)
    return '\n'.join(res)


def normalize_indentation_nosynth(source):
    return normalize_indentation(source, insert_missing_body=False)


# =========================================================== H3  bad_lhs_rename
_SANITIZE = re.compile(r'\W+')


def _split_top_assign(code, raw):
    """Split `raw` at the first top-level '=' located using the string/comment-blanked `code`."""
    depth = 0
    for i, c in enumerate(code):
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == '=' and depth == 0:
            if code[i + 1:i + 2] == '=':
                continue
            if i and code[i - 1] in '=!<>+-*/%&|^@:~':
                continue
            return raw[:i], raw[i:]
    return None


def bad_lhs_rename(source):
    """Rename an UNPARSEABLE assignment target to a sanitized identifier, keeping the RHS intact.

    PyLingual loses a binding's name and prints one of the CONSTANTS in its place:
      `1e:6c:34:93:68:64 = [...]`   (MAC address)
      `7AB5C494-39F5-4941-9163-47F54D6D5016.False = [...]`   (UUID)
      `self.3c:ec:ef:44:01:aa = [...]`,  `42:01:0a:96:00:22.00:e0:4c:b8:7a:58 = {...}`
    Deleting the line throws away the whole blacklist the malware is checking against.
    Generalises the shipped numeric-only `literal_lhs_rename`; gated on ast: fires only when the
    original target does NOT parse as a target and the sanitized one does, so it can never touch
    legal code."""
    lines = source.split('\n')
    spans, _recs, blanked = statement_spans(lines)
    out = list(lines)
    fired = 0
    for (a, b) in spans:
        # NOTE: do NOT require b == a+1. The RHS of these lines is usually a TRUNCATED list, so the
        # statement span runs to end-of-file; gating on single-line spans silently skipped 36 of the
        # 50 real targets. The '=' is still located at depth 0 on the statement's first line.
        L = lines[a]
        st = L.lstrip()
        if not st or st[0] in '#@':
            continue
        # A compound one-liner (`else: passwd = None`) has an unparseable "target" too -- skip any
        # statement that starts with a keyword, or the target rename mangles legal code.
        head = re.match(r'[A-Za-z_]\w*', st)
        if head and (keyword.iskeyword(head.group(0))
                     or head.group(0) in ('match', 'case', 'type', 'print', 'exec')):
            continue
        sp = _split_top_assign(blanked[a], L)
        if not sp:
            continue
        lhs, rest = sp
        core = lhs.strip()
        if not core:
            continue
        try:
            ast.parse(core + ' = None')
            continue
        except SyntaxError:
            pass
        except Exception:
            continue
        new = '_lit_' + _SANITIZE.sub('_', core).strip('_')
        try:
            ast.parse(new + ' = None')
        except Exception:
            continue
        out[a] = ' ' * indent_of(L) + new + rest
        fired += 1
    return '\n'.join(out) if fired else None


# =========================================================== H4  close_truncated_lines
_HEADER_KW = re.compile(r'^\s*(if|elif|for|while|with|except|def|class|async|match|case)\b')


def close_truncated_lines(source, synth_body=True):
    """Terminate EVERY decompiler-truncated line in the file in one pass.

    PyLingual hard-cuts a long constant sequence mid-token: `x = ['a', 'b', 'c` . The shipped
    `balance_delimiters` only ever repairs the FIRST such line (it is driven by the compiler's
    error position); these files carry a median of 4 and up to 40. Closes the dangling quote and
    every bracket still open ON THAT LINE. When the truncated statement is a compound HEADER, `:`
    is appended and a `pass` body synthesized if the next line is not deeper (synth_body=False
    measures the no-synthesis variant, which then simply skips headers)."""
    lines = source.split('\n')
    recs, _bl = scan_file(lines)
    out = list(lines)
    inserts = {}
    fired = 0
    for i, L in enumerate(lines):
        s = L.rstrip()
        if len(s) < 80:
            continue
        d0, q0, _c = recs[i]
        if q0 is not None:
            continue                        # inside a triple-quoted string: not truncation
        # rescan this line alone to find what it leaves open
        stack = []
        q = None
        j = 0
        while j < len(s):
            c = s[j]
            if q:
                if c == '\\':
                    j += 2
                    continue
                if c == q:
                    q = None
                j += 1
                continue
            if c == '#':
                break
            if c in '\'"':
                if s.startswith(c * 3, j):
                    q = None
                    j = len(s)
                    break
                q = c
                j += 1
                continue
            if c in _OPEN:
                stack.append(c)
            elif c in _CLOSE:
                if stack and stack[-1] == _CLOSE[c]:
                    stack.pop()
            j += 1
        if q is None or not stack:
            continue                        # not a truncated literal inside an open bracket
        new = s + q + ''.join(_OPEN[c] for c in reversed(stack))
        if _HEADER_KW.match(L):
            if not synth_body:
                continue
            new += ':'
            nxt = next((lines[k] for k in range(i + 1, len(lines)) if lines[k].strip()), None)
            if nxt is None or indent_of(nxt) <= indent_of(L):
                inserts[i] = ' ' * (indent_of(L) + 4) + 'pass'
        out[i] = new
        fired += 1
    if not fired:
        return None
    res = []
    for i, L in enumerate(out):
        res.append(L)
        if i in inserts:
            res.append(inserts[i])
    return '\n'.join(res)


def close_truncated_lines_nosynth(source):
    return close_truncated_lines(source, synth_body=False)


# =========================================================== H4b complete_truncated_statement
def _line_open_state(s):
    """(open_bracket_stack, dangling_quote) for ONE physical line scanned from a clean state."""
    stack = []
    q = None
    j = 0
    n = len(s)
    while j < n:
        c = s[j]
        if q:
            if c == '\\':
                j += 2
                continue
            if c == q:
                q = None
            j += 1
            continue
        if c == '#':
            break
        if c in '\'"':
            if s.startswith(c * 3, j):
                return stack, None          # triple quote: not the truncation shape
            q = c
            j += 1
            continue
        if c in _OPEN:
            stack.append(c)
        elif c in _CLOSE:
            if stack and stack[-1] == _CLOSE[c]:
                stack.pop()
        j += 1
    return stack, q


_HDR2 = re.compile(r'^\s*(if|elif|for|while|with|except|def|class|async|match|case)\b')


def complete_truncated_statement(source, allow_body=True, _max=200):
    """Repeat the single-victim repair until the file's brackets rebalance (files carry up to 40
    truncated statements; repairing one per driver round never converged)."""
    cur = source
    for _ in range(_max):
        nxt = _complete_one_truncated_statement(cur, allow_body)
        if nxt is None or nxt == cur:
            break
        cur = nxt
    return cur if cur != source else None


def _complete_one_truncated_statement(source, allow_body=True):
    """Complete a decompiler-TRUNCATED statement with the shortest suffix that actually parses.

    PyLingual hard-cuts a long constant mid-token and the rest of the statement is simply gone:
        badgeList = [{'Name': 'Early_Verified_Bot_Developer', ..., 'E     <- cut here
    Blindly appending the closing quote + brackets (what the shipped `balance_delimiters` does)
    produces `'E'}]`, i.e. a dict key with no value -> "':' expected after dictionary key"; that is
    the single largest residual error class on this corpus. Instead, try a short ordered LADDER of
    completions and keep the first whose statement parses IN ISOLATION:
        quote+closers | quote + ': None' + closers | quote + 'None' + closers | ... (+ ':'/pass for
    a compound header). Detection: a physical line that leaves a quote or brackets open while the
    next non-blank line is NOT deeper -- i.e. the statement visually ended mid-air. Runs over EVERY
    truncated line in the file, not just the one the compiler happens to point at.
    Deletes nothing; the appended text is delimiters plus, at most, a `None` placeholder."""
    lines = source.split('\n')
    # SOUNDNESS GATE: only touch a file whose brackets never rebalance, i.e. the tail of the file
    # was swallowed. Without this the transform fired on legal multi-line statements in hand-written
    # code (measured: 7/573 stdlib files, 6 of them broken).
    recs, _bl = scan_file(lines)
    end_depth, end_q, _ = recs[-1] if recs else (0, None, False)
    tailstack, tailq = _line_open_state(lines[-1]) if lines else ([], None)
    if end_depth + len(tailstack) <= 0 and end_q is None:
        return None
    spans, _r, _b2 = statement_spans(lines)
    victim = None
    for (a, b) in spans:
        if b >= len(lines):
            st, qq = _line_open_state(lines[a].rstrip())
            if st or qq:
                victim = a
            break
    if victim is None:
        return None
    out = list(lines)
    inserts = {}
    fired = 0
    for i in [victim]:
        L = lines[i]
        s = L.rstrip()
        if len(s.strip()) < 20:
            continue
        stack, q = _line_open_state(s)
        if not stack and q is None:
            continue
        nxt = next((lines[k] for k in range(i + 1, len(lines)) if lines[k].strip()), None)
        closers = ''.join(_OPEN[c] for c in reversed(stack))
        base = s + (q or '')
        tails = [closers, ': None' + closers, 'None' + closers, "'x'" + closers,
                 ': None' + closers[1:] if closers else '']
        hdr = bool(_HDR2.match(L))
        best = None
        for t in tails:
            cand = base + t
            probe = cand.lstrip()
            if hdr:
                probe = probe + ':\n    pass'
            try:
                ast.parse(probe)
            except Exception:
                continue
            best = cand + (':' if hdr else '')
            break
        if best is None:
            continue
        if best == s:
            continue
        out[i] = best
        if hdr and allow_body:
            if nxt is None or indent_of(nxt) <= indent_of(L):
                inserts[i] = ' ' * (indent_of(L) + 4) + 'pass'
        elif hdr:
            out[i] = L                       # would need a synthesized body: skip in pure mode
            continue
        fired += 1
    if not fired:
        return None
    res = []
    for i, L in enumerate(out):
        res.append(L)
        if i in inserts:
            res.append(inserts[i])
    return '\n'.join(res)


def complete_truncated_statement_nosynth(source):
    return complete_truncated_statement(source, allow_body=False)


# =========================================================== H5  decorator_to_call
_DECO = re.compile(r'^(\s*)@\s*([A-Za-z_][\w.]*)\s*$')


def decorator_to_call(source):
    """Turn PyLingual's spurious `@name` lines back into the CALL they came from.

    The decompiler prints an outer call as a decorator and strands its `)` on the next statement:
        @urlopen
        x = Request(f'...')).read()      ->     x = urlopen(Request(f'...')).read()
    Fires only when the following statement is not a def/class AND carries at least as many
    SURPLUS `)` as there are stacked `@` lines, so the injected `(` are exactly consumed.
    Deletes nothing: each `@name` line becomes the `name(` prefix of the statement."""
    lines = source.split('\n')
    spans, _recs, blanked = statement_spans(lines)
    starts = [a for a, _ in spans]
    out = list(lines)
    drop = set()
    fired = 0
    si = 0
    while si < len(spans):
        a, b = spans[si]
        m = _DECO.match(lines[a])
        if not m or b != a + 1:
            si += 1
            continue
        run = [(a, m.group(2))]
        sj = si + 1
        while sj < len(spans):
            aa, bb = spans[sj]
            mm = _DECO.match(lines[aa])
            if not mm or bb != aa + 1:
                break
            run.append((aa, mm.group(2)))
            sj += 1
        if sj >= len(spans):
            break
        ta, tb = spans[sj]
        if re.match(r'^\s*(def |class |async def )', lines[ta]):
            si = sj + 1
            continue
        code = ' '.join(blanked[k] for k in range(ta, tb))
        depth = 0
        surplus = 0
        for c in code:
            if c in _OPEN:
                depth += 1
            elif c in _CLOSE:
                if depth:
                    depth -= 1
                else:
                    surplus += 1
        if surplus < len(run):
            si = sj
            continue
        names = [n for _, n in run]
        prefix = ''.join(n + '(' for n in names)
        sp = _split_top_assign(blanked[ta], lines[ta])
        ind = indent_of(lines[ta])
        if sp and tb == ta + 1:
            lhs, rest = sp
            out[ta] = ' ' * ind + lhs.strip() + ' = ' + prefix + rest[1:].lstrip()
        else:
            out[ta] = ' ' * ind + prefix + lines[ta].lstrip()
        for k, _ in run:
            drop.add(k)
        fired += 1
        si = sj + 1
    if not fired:
        return None
    return '\n'.join(L for k, L in enumerate(out) if k not in drop)


# =========================================================== H6  complete_orphan_try (whole file)
_TRY = re.compile(r'^\s*try\s*:')
_HANDLER = re.compile(r'^\s*(except|finally)\b')


def complete_all_orphan_try(source):
    """Give EVERY handler-less `try:` in the file an `except Exception: pass`.

    The shipped `complete_orphan_try` repairs one try per driver round and the round budget runs
    out; these files carry up to 30. SYNTHESIZES two lines per orphan; deletes nothing."""
    lines = source.split('\n')
    spans, _recs, _bl = statement_spans(lines)
    starts = [a for a, _ in spans]
    inserts = {}
    for si, (a, _b) in enumerate(spans):
        if not _TRY.match(lines[a]):
            continue
        d = indent_of(lines[a])
        end = len(lines)
        handled = False
        for sj in range(si + 1, len(spans)):
            P = lines[spans[sj][0]]
            pd = indent_of(P)
            if pd > d:
                continue
            if pd == d and _HANDLER.match(P):
                handled = True
            end = spans[sj][0]
            break
        if handled:
            continue
        inserts.setdefault(end, []).extend([' ' * d + 'except Exception:', ' ' * d + '    pass'])
    if not inserts:
        return None
    res = []
    for i, L in enumerate(lines):
        for p in inserts.get(i, ()):
            res.append(p)
        res.append(L)
    for p in inserts.get(len(lines), ()):
        res.append(p)
    return '\n'.join(res)
