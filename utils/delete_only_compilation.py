from __future__ import annotations

import io
import re
import tokenize
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


CompileOracle = Callable[[str, str, str, Optional[Any]], Dict[str, Any]]
LineExtractor = Callable[[str], Optional[int]]
ErrWordMsgParser = Callable[[str], Tuple[Optional[str], Optional[str]]]


@dataclass
class DeletionAttempt:
    iteration: int
    error_type: str
    error_msg: str
    error_lineno: int
    error_end_lineno: int
    deleted_start: int
    deleted_end: int
    reason: str
    remaining_lines: int
    escalate_level: int
    bucket: str


def delete_lines_until_compilable_with_oracle(
    py_file_content: str,
    compile_check: CompileOracle,
    extract_line_number: LineExtractor,
    get_error_word_message_from_content: ErrWordMsgParser,
    *,
    out_py_path: str,
    out_pyc_path: str,
    version: Optional[Any] = None,
    base_window: int = 0,
    max_iters: int = 20_000,
    max_deleted_ratio: float = 0.95,
    min_remaining_lines: int = 1,
    err_txt_path: Optional[str] = None,
    debug_escalation: bool = False,
) -> Tuple[str, List[DeletionAttempt], Dict[str, Any]]:
    """Try to reach a compilable file by deleting increasingly broad line spans."""
    lines = _split_keepends(py_file_content)
    original_line_count = len(lines)
    attempts: List[DeletionAttempt] = []
    seen_counts: Dict[Tuple[str, int], int] = {}
    escalate_level = 0

    recent_linenos: deque[int] = deque(maxlen=8)
    recent_buckets: deque[str] = deque(maxlen=8)

    def parse_error(err_desc: str) -> Tuple[str, str, Optional[int]]:
        err_desc = (err_desc or "").strip()

        err_word = None
        msg = None
        ln = None

        if err_txt_path:
            try:
                with open(err_txt_path, "w", encoding="utf-8") as f:
                    f.write(err_desc or "Unknown error")
                err_word, msg = get_error_word_message_from_content(err_txt_path)
                if msg:
                    ln = extract_line_number(msg)
            except Exception:
                err_word, msg, ln = None, None, None

        if ln is None and err_desc:
            ln = extract_line_number(err_desc)

        if not isinstance(ln, int):
            ln = None

        return err_word or "CompileError", (msg or err_desc or "").strip(), ln

    last_compile_res: Dict[str, Any] = {"is_compiled": False, "error_description": None}

    for it in range(1, max_iters + 1):
        src = "".join(lines)
        last_compile_res = compile_check(src, out_py_path, out_pyc_path, version)

        if bool(last_compile_res.get("is_compiled", False)):
            break

        if len(lines) <= min_remaining_lines:
            break

        deleted_ratio = 1.0 - (len(lines) / max(1, original_line_count))
        if deleted_ratio > max_deleted_ratio:
            break

        err_desc = (last_compile_res.get("error_description") or "").strip()
        err_word, msg, lineno = parse_error(err_desc)

        if lineno is None:
            lineno = _last_significant_line(lines)

        n = len(lines)
        lineno = _clamp(lineno, 1, n)
        end_lineno = lineno

        bucket = _bucketize(msg)
        recent_linenos.append(lineno)
        recent_buckets.append(bucket)

        sig = (msg, lineno)
        seen_counts[sig] = seen_counts.get(sig, 0) + 1

        if bucket == "BLOCK":
            if seen_counts[sig] in (2, 4, 6):
                escalate_level = min(3, escalate_level + 1)
        else:
            if seen_counts[sig] in (3, 6, 10):
                escalate_level = min(3, escalate_level + 1)

        if len(recent_linenos) == recent_linenos.maxlen:
            span = max(recent_linenos) - min(recent_linenos)
            if span <= 3:
                if bucket == "BLOCK" and escalate_level < 2:
                    escalate_level = 2
                else:
                    escalate_level = min(3, escalate_level + 1)

        if len(recent_buckets) == recent_buckets.maxlen:
            if len(set(recent_buckets)) <= 2:
                if bucket == "BLOCK" and escalate_level < 2:
                    escalate_level = 2
                else:
                    escalate_level = min(3, escalate_level + 1)

        if debug_escalation:
            span_dbg = (
                (max(recent_linenos) - min(recent_linenos))
                if len(recent_linenos) == recent_linenos.maxlen
                else None
            )
            uniq_b_dbg = len(set(recent_buckets)) if len(recent_buckets) == recent_buckets.maxlen else None
            print(
                f"[del-only] it={it} bucket={bucket} esc={escalate_level} "
                f"sig_count={seen_counts[sig]} ln={lineno} span={span_dbg} uniq_b={uniq_b_dbg} msg={msg!r}"
            )

        pseudo = _PseudoError(msg=msg, lineno=lineno, end_lineno=end_lineno, bucket=bucket)

        candidates = _generate_candidates(
            lines,
            pseudo,
            base_window=base_window,
            escalate_level=escalate_level,
        )

        chosen: Optional[Tuple[int, int, str]] = None
        chosen_applied = False

        smallest: Optional[Tuple[int, int, str]] = None
        smallest_len = 10**18

        for (s, e, reason) in candidates:
            if s < 1 or e > len(lines) or s > e:
                continue

            span_len = e - s + 1
            if span_len < smallest_len:
                smallest = (s, e, reason + " (fallback smallest)")
                smallest_len = span_len

            trial_lines = _apply_deletion(lines, s, e)
            trial_src = "".join(trial_lines)
            trial_res = compile_check(trial_src, out_py_path, out_pyc_path, version)

            if bool(trial_res.get("is_compiled", False)):
                chosen = (s, e, reason + " (compiles)")
                lines = trial_lines
                last_compile_res = trial_res
                chosen_applied = True
                break

            trial_desc = (trial_res.get("error_description") or "").strip()
            tln = extract_line_number(trial_desc) if trial_desc else None
            if isinstance(tln, int) and (trial_desc, tln) != (msg, lineno):
                chosen = (s, e, reason + " (changes error)")
                lines = trial_lines
                last_compile_res = trial_res
                chosen_applied = True
                break

        if not chosen_applied:
            if smallest is None:
                break
            chosen = smallest
            s, e, _ = chosen
            lines = _apply_deletion(lines, s, e)

        s, e, reason = chosen
        attempts.append(
            DeletionAttempt(
                iteration=it,
                error_type=err_word,
                error_msg=msg,
                error_lineno=lineno,
                error_end_lineno=end_lineno,
                deleted_start=s,
                deleted_end=e,
                reason=reason,
                remaining_lines=len(lines),
                escalate_level=escalate_level,
                bucket=bucket,
            )
        )

    final_text = "".join(lines)
    last_compile_res = compile_check(final_text, out_py_path, out_pyc_path, version)
    return final_text, attempts, last_compile_res


class _PseudoError:
    def __init__(self, msg: str, lineno: int, end_lineno: int, bucket: str):
        self.msg = msg
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.bucket = bucket


def _bucketize(err_msg: str) -> str:
    m = (err_msg or "").lower()
    if any(s in m for s in ["was never closed", "unexpected eof", "unterminated", "eol while scanning"]):
        return "UNCLOSED"
    if any(s in m for s in ["expected ':'", "expected an indented block", "unexpected indent", "unindent does not match", "indentation"]):
        return "BLOCK"
    if "invalid syntax" in m:
        return "GENERIC"
    return "OTHER"


def _split_keepends(text: str) -> List[str]:
    return text.splitlines(keepends=True) or ["\n"]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


_ONLY_COMMENT_OR_WS_RE = re.compile(r"^\s*(#.*)?$")


def _is_blank_or_comment(line: str) -> bool:
    return bool(_ONLY_COMMENT_OR_WS_RE.match(line))


def _last_significant_line(lines: List[str]) -> int:
    last = len(lines)
    while last > 1 and _is_blank_or_comment(lines[last - 1]):
        last -= 1
    return last


def _apply_deletion(lines: List[str], start: int, end: int) -> List[str]:
    return lines[: start - 1] + lines[end:]


_HEADER_RE = re.compile(r"^\s*(if|elif|else|for|while|def|class|try|except|finally|with|match|case)\b")
_DANGLING_CLAUSE_RE = re.compile(r"^\s*(except|elif|else|finally|case)\b")


def _prev_line_has_backslash_cont(lines: List[str], lineno: int) -> bool:
    if lineno <= 1:
        return False
    prev = lines[lineno - 2].rstrip("\n")
    return prev.rstrip().endswith("\\")


def _line_has_backslash_cont(lines: List[str], lineno: int) -> bool:
    if lineno < 1 or lineno > len(lines):
        return False
    cur = lines[lineno - 1].rstrip("\n")
    return cur.rstrip().endswith("\\")


def _find_nearest_header_above(lines: List[str], lineno: int, max_lookback: int = 80) -> Optional[int]:
    lineno = _clamp(lineno, 1, len(lines))
    lo = max(1, lineno - max_lookback)
    for ln in range(lineno, lo - 1, -1):
        if _is_blank_or_comment(lines[ln - 1]):
            continue
        if _HEADER_RE.match(lines[ln - 1]):
            return ln
    return None


def _find_unclosed_construct_start(lines: List[str]) -> Optional[int]:
    src = "".join(lines)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError as te:
        if len(te.args) >= 2 and isinstance(te.args[1], tuple) and len(te.args[1]) >= 1:
            line_no = te.args[1][0]
            if isinstance(line_no, int) and line_no >= 1:
                return _clamp(line_no, 1, len(lines))
        return None
    except Exception:
        return None

    opens = {"(": ")", "[": "]", "{": "}"}
    closes = {")": "(", "]": "[", "}": "{"}
    stack: List[Tuple[str, int]] = []

    for tok in toks:
        _, tstr, (sline, _), _, _ = tok
        if tstr in opens:
            stack.append((tstr, sline))
        elif tstr in closes:
            if stack and stack[-1][0] == closes[tstr]:
                stack.pop()
            else:
                return _clamp(sline, 1, len(lines))

    if stack:
        _, start_line = stack[-1]
        return _clamp(start_line, 1, len(lines))
    return None


def _find_statement_chunk(lines: List[str], lineno: int) -> Tuple[int, int]:
    n = len(lines)
    lineno = _clamp(lineno, 1, n)
    start = lineno
    end = lineno

    while start > 1 and _prev_line_has_backslash_cont(lines, start):
        start -= 1
    while end < n and _line_has_backslash_cont(lines, end):
        end += 1

    br_start = _find_unclosed_construct_start(lines[:end])
    if br_start is not None and br_start <= lineno:
        start = min(start, br_start)
        end = min(n, max(end, lineno + 2))
    return start, end


def _indent_prefix_width(line: str) -> int:
    i = 0
    for ch in line:
        if ch == " " or ch == "\t":
            i += 1
        else:
            break
    return i


def _find_indented_region(lines: List[str], lineno: int) -> Optional[Tuple[int, int]]:
    """Return a contiguous indented region rooted at `lineno`."""
    if lineno < 1 or lineno > len(lines):
        return None

    base_indent = _indent_prefix_width(lines[lineno - 1])
    if base_indent == 0:
        return None

    start = lineno
    end = lineno

    for j in range(lineno + 1, len(lines) + 1):
        s = lines[j - 1]
        if _is_blank_or_comment(s):
            end = j
            continue
        ind = _indent_prefix_width(s)
        if ind >= base_indent:
            end = j
        else:
            break

    return (start, end) if end >= start else None


def _dedupe_spans(cands: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    seen = set()
    out = []
    for s, e, r in cands:
        if (s, e) not in seen:
            seen.add((s, e))
            out.append((s, e, r))
    return out


def _generate_candidates(
    lines: List[str],
    err: _PseudoError,
    *,
    base_window: int,
    escalate_level: int,
) -> List[Tuple[int, int, str]]:
    n = len(lines)
    lineno = _clamp(int(err.lineno or 1), 1, n)
    end_lineno = _clamp(int(err.end_lineno or lineno), lineno, n)
    msg_l = (err.msg or "").lower()
    bucket = err.bucket

    cands: List[Tuple[int, int, str]] = []

    if bucket == "UNCLOSED":
        start_line = _find_unclosed_construct_start(lines)
        if start_line is not None:
            cands.append((start_line, start_line, "tokenizer inferred start line"))
            cands.append((max(1, start_line - 1), min(n, start_line + 1), "tokenizer start neighborhood"))
            if escalate_level >= 2:
                cands.append((start_line, n, "tokenizer start to EOF"))

    if bucket == "BLOCK" and "unexpected indent" in msg_l:
        region = _find_indented_region(lines, lineno)
        if region is not None:
            s, e = region
            cands.append((s, e, "indent-region chop (unexpected indent)"))

    cands.append((lineno, end_lineno, "reported span"))
    cands.append((lineno, lineno, "reported line only"))

    if 1 <= lineno <= n and _DANGLING_CLAUSE_RE.match(lines[lineno - 1]):
        cands.insert(0, (lineno, lineno, "dangling clause keyword line"))
        hdr = _find_nearest_header_above(lines, lineno)
        if hdr is not None:
            cands.insert(1, (hdr, hdr, "nearest header above (re-balance blocks)"))

    if _prev_line_has_backslash_cont(lines, lineno):
        cands.insert(0, (max(1, lineno - 1), lineno, "backslash continuation (prev + current)"))
    if _line_has_backslash_cont(lines, lineno):
        cands.insert(0, (lineno, min(n, lineno + 1), "backslash continuation (current + next)"))

    windows = [max(0, base_window), 1, 2, 4]
    if bucket == "BLOCK":
        if escalate_level >= 2:
            windows += [8, 16]
        if escalate_level >= 3:
            windows += [32]
    else:
        if escalate_level >= 2:
            windows += [8]
        if escalate_level >= 3:
            windows += [16]

    for w in windows:
        s = _clamp(lineno - w, 1, n)
        e = _clamp(end_lineno + w, s, n)
        cands.append((s, e, f"window w={w}"))

    if any(s in msg_l for s in ["expected ':'", "expected an indented block", "unexpected indent", "unindent does not match", "indentation", "invalid syntax"]):
        hdr = _find_nearest_header_above(lines, lineno)
        if hdr is not None:
            cands.insert(0, (hdr, hdr, "nearest header above"))
            cands.insert(1, (max(1, hdr - 1), min(n, hdr + 1), "header neighborhood"))

    cs, ce = _find_statement_chunk(lines, lineno)
    cands.append((cs, ce, "statement chunk"))

    cands.append((lineno, n, "error line to EOF"))

    if escalate_level >= 3:
        cands.append((1, lineno, "BOOM: start to error line"))

    return _dedupe_spans(cands)
