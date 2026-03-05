from __future__ import annotations

import io
import os
import re
import tokenize
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------
# 1) Deletion-only tool (uses your injected compile + parsers)
# --------------------------------------------------------------------

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


def delete_lines_until_compilable_with_oracle(
    py_file_content: str,
    compile_check: CompileOracle,
    extract_line_number: LineExtractor,
    get_error_word_message_from_content: ErrWordMsgParser,
    *,
    # paths aligned with your main loop
    out_py_path: str,
    out_pyc_path: str,
    version: Optional[Any] = None,
    base_window: int = 0,
    max_iters: int = 20_000,
    max_deleted_ratio: float = 0.95,
    min_remaining_lines: int = 1,
    err_txt_path: Optional[str] = None,
) -> Tuple[str, List[DeletionAttempt], Dict[str, Any]]:
    """
    Deletion-only repair that is aligned with your pipeline:

    - compile_check is compile_new_pyc(content, py_file_dir, out_file_base_dir, version)
      where py_file_dir is the output .py path and out_file_base_dir is the output .pyc path.

    - We reuse your error parsing by writing error_description to err_txt_path and calling
      get_error_word_message_from_content(err_txt_path). That gives (error_word, message).
      We then use extract_line_number(message) first, then fallback to extract_line_number(error_description).

    Returns:
      fixed_code, deletion_log, last_compile_result
    """
    lines = _split_keepends(py_file_content)
    original_line_count = len(lines)
    attempts: List[DeletionAttempt] = []
    seen_counts: Dict[Tuple[str, int], int] = {}
    escalate_level = 0

    def parse_error(err_desc: str) -> Tuple[str, str, int]:
        """
        Returns (err_word, msg, lineno)
        """
        err_desc = (err_desc or "").strip()

        err_word = None
        msg = None
        ln = None

        if err_txt_path:
            try:
                Path(err_txt_path).parent.mkdir(parents=True, exist_ok=True)
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

        # IMPORTANT: compile_check signature in your project:
        # compile_new_pyc(content, py_file_dir(.py), out_file_base_dir(.pyc), version)
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

        sig = (msg, lineno)
        seen_counts[sig] = seen_counts.get(sig, 0) + 1
        if seen_counts[sig] in (3, 6, 10):
            escalate_level = min(3, escalate_level + 1)

        pseudo = _PseudoError(msg=msg, lineno=lineno, end_lineno=end_lineno)
        candidates = _generate_candidates(lines, pseudo, base_window=base_window, escalate_level=escalate_level)

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
            if isinstance(tln, int) and (trial_desc, tln) != sig:
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
            )
        )

    final_text = "".join(lines)
    # Ensure final compile status is accurate and writes the final outputs
    last_compile_res = compile_check(final_text, out_py_path, out_pyc_path, version)
    return final_text, attempts, last_compile_res


# ---------------- internals ----------------

class _PseudoError:
    def __init__(self, msg: str, lineno: int, end_lineno: int):
        self.msg = msg
        self.lineno = lineno
        self.end_lineno = end_lineno


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


def _looks_like_block_problem(msg_l: str) -> bool:
    return any(s in msg_l for s in (
        "expected ':'",
        "expected an indented block",
        "unexpected indent",
        "unindent does not match",
        "indentation",
        "invalid syntax",
    ))


def _looks_like_unclosed_string(msg_l: str) -> bool:
    return any(s in msg_l for s in (
        "unterminated string",
        "eol while scanning string literal",
        "unterminated triple-quoted string literal",
    ))


def _looks_like_unclosed_bracket_or_eof(msg_l: str) -> bool:
    return any(s in msg_l for s in (
        "unexpected eof while parsing",
        "was never closed",
        "closing parenthesis",
        "closing bracket",
        "closing brace",
    ))


def _find_nearest_header_above(lines: List[str], lineno: int, max_lookback: int = 60) -> Optional[int]:
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

    cands: List[Tuple[int, int, str]] = []
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
    if escalate_level >= 2:
        windows += [8]
    if escalate_level >= 3:
        windows += [16]
    for w in windows:
        s = _clamp(lineno - w, 1, n)
        e = _clamp(end_lineno + w, s, n)
        cands.append((s, e, f"window w={w}"))

    if _looks_like_block_problem(msg_l):
        hdr = _find_nearest_header_above(lines, lineno)
        if hdr is not None:
            cands.insert(0, (hdr, hdr, "nearest header above"))
            cands.insert(1, (max(1, hdr - 1), min(n, hdr + 1), "header neighborhood"))

    if _looks_like_unclosed_string(msg_l) or _looks_like_unclosed_bracket_or_eof(msg_l):
        start_line = _find_unclosed_construct_start(lines)
        if start_line is not None:
            cands.insert(0, (start_line, start_line, "tokenizer inferred start line"))
            cands.insert(1, (max(1, start_line - 1), min(n, start_line + 1), "tokenizer start neighborhood"))
            if escalate_level >= 2:
                cands.append((start_line, n, "tokenizer start to EOF"))

    cs, ce = _find_statement_chunk(lines, lineno)
    cands.append((cs, ce, "statement chunk"))
    cands.append((lineno, n, "error line to EOF"))

    if escalate_level >= 3:
        cands.append((1, lineno, "BOOM: start to error line"))

    return _dedupe_spans(cands)


# --------------------------------------------------------------------
# 2) Small integration snippet: where to call from your main loop
# --------------------------------------------------------------------
"""
In your loop, you already create:
  AFFECTED_FILE_PATH = LOG_BASE / file_hash
  out_py = AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.py"
  out_pyc = AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.pyc"
  err_txt = AFFECTED_FILE_PATH / f"syntax_failed_repaired_{file_name[:-3]}_error.txt"

Add this fallback exactly when max_retries is exhausted (or whenever you want):

    if (max_retries < 0 or elapsed > MAX_EXAMPLE_RUNTIME_SEC) and (not is_compiled):
        # Deletion-only fallback
        fixed_code, del_log, last_res = delete_lines_until_compilable_with_oracle(
            py_file_content=compilation_candidate,
            compile_check=compile_new_pyc,
            extract_line_number=extract_line_number,
            get_error_word_message_from_content=get_error_word_message_from_content,
            out_py_path=str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.py"),
            out_pyc_path=str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.pyc"),
            version=version,
            base_window=1,
            max_iters=5000,
            max_deleted_ratio=0.95,
            min_remaining_lines=1,
            err_txt_path=str(AFFECTED_FILE_PATH / f"syntax_failed_repaired_{file_name[:-3]}_error.txt"),
        )

        # If it compiled, last_res["is_compiled"] is True and the .py/.pyc are already written.
        if last_res.get("is_compiled"):
            is_compiled = True
            log_rec.update({
                "compiled_success": True,
                "path_out": str(AFFECTED_FILE_PATH / f"syntax_repaired_{file_name[:-3]}.py"),
                "delete_only_fallback_used": True,
                "delete_only_deletions": len(del_log),
            })
            # Optional: persist deletion log next to outputs
            with open(str(AFFECTED_FILE_PATH / f"delete_only_log_{file_name[:-3]}.jsonl"), "w", encoding="utf-8") as f:
                for rec in del_log:
                    f.write(str(asdict(rec)) + "\\n")
        else:
            log_rec.update({
                "delete_only_fallback_used": True,
                "delete_only_deletions": len(del_log),
                "delete_only_failed_error": last_res.get("error_description"),
            })

        _append_log(LOG_FILE, log_rec)
        try:
            os.unlink(copy_dir)
        except FileNotFoundError:
            pass
        break
"""