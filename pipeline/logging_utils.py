from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def append_log(path: Path, record: dict[str, Any]) -> None:
    try:
        _annotate_deletions(record)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[logging] failed to append {path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _annotate_deletions(record: dict[str, Any]) -> None:
    """Derive `lines_deleted_vs_original` from the paths already in the record, at log time.

    The runner writes `path_out` from four separate places (prepass-solo, LLM success, retry
    success, delete-only fallback). Computing the audit per-site invites exactly the omission that
    produced the original defect: `delete_only_fallback_used` covered one route only, so 84% of
    repairs labelled "genuine" had actually removed lines. Deriving it here makes the measurement
    structurally unmissable -- every logged outcome carries it.

    `path_in` is the pristine decompiled source (verified byte-identical to the dataset's
    `decompiled_source`). Leaves an explicitly-supplied value alone, and stays silent when either
    path is missing or unreadable rather than fabricating a zero."""
    try:
        if "lines_deleted_vs_original" in record:
            return
        src, out = record.get("path_in"), record.get("path_out")
        if not src or not out:
            return
        if not (os.path.isfile(str(src)) and os.path.isfile(str(out))):
            return
        with open(str(src), errors="replace") as f:
            original = f.read()
        with open(str(out), errors="replace") as f:
            final = f.read()
        record["lines_deleted_vs_original"] = count_deleted_lines(original, final)
    except Exception:
        return


def clean_error_message(error_message: str | None):
    if not error_message:
        return None
    cleaned_message = error_message.lower().strip()
    cleaned_message = re.sub(r"\(.*?\)", "", cleaned_message).strip()
    return cleaned_message


def extract_line_number(error_msg: str | None):
    if not error_msg:
        return None
    m = re.search(r"line\s+(\d+)", error_msg)
    return int(m.group(1)) if m else None


def window_exceeds_token_budget(window_text, max_tokens, count_tokens):
    """True when the repair window is larger than the LLM generation budget and so cannot be
    regenerated in one shot (the fix would be truncated). ``count_tokens`` is a callable
    (str) -> int, the model tokenizer. A window exceeding ``max_tokens`` tokens means the file
    is out of scope for windowed repair: it is DISCARDED (not attempted) and EXCLUDED from the
    analyzability denominator (see pipeline.runner.run_experiment). This subsumes the
    single-blob-line case (a minified payload line tokenizes well past the budget) and also
    catches ordinary-but-large windows. ``max_tokens <= 0`` disables the guard; empty/None
    windows never exceed."""
    if not window_text or max_tokens <= 0:
        return False
    return count_tokens(window_text) > max_tokens


_LIGHT_DELETION_MAX_LINES = 50


def count_deleted_lines(original, final):
    """How many lines of `original` are absent from `final` -- the honest deletion measure.

    The repair pipeline's old notion of "genuine" keyed off ``delete_only_fallback_used``, which
    only tracks the delete-only FALLBACK and is blind to deletions the LLM makes on its own. A
    full-population audit found 38.6% of accepted "genuine" repairs had in fact removed lines
    (median 5, max 236), with the LLM route contaminated 44.5% of the time.

    Compares multisets of stripped non-empty lines, so re-ordering and re-indentation are NOT
    counted as deletion -- only content that actually disappears. Never raises; unknown inputs
    report 0 so an unmeasurable case can never fabricate a deletion."""
    try:
        if not original or not final:
            return 0
        from collections import Counter

        before = Counter(l.strip() for l in str(original).splitlines() if l.strip())
        after = Counter(l.strip() for l in str(final).splitlines() if l.strip())
        return sum((before - after).values())
    except Exception:
        return 0


def classify_repair_outcome(record):
    """Bucket a syntactic run-log record into the outcome actually reported.

    Buckets: "genuine" (compiles, no code removed), "genuine_lossy" (compiles with no lines removed
    but a decompiler-TRUNCATED literal was closed, so content was silently dropped),
    "light_deletion" (<= 50 lines removed), "heavy_deletion", "failed", "out_of_scope"
    (not attempted -- un-windowable).

    Two corrections over classifying on ``delete_only_fallback_used`` alone:
      * the fallback can engage and delete NOTHING (its best-of candidate compiled as-is); such a
        file is a genuine repair, not a deletion (20 files in the malware run were miscounted);
      * a no-deletion repair that closed a truncated literal is not faithful, so it is reported as
        ``genuine_lossy`` rather than inflating the genuine rate.

    Pure and total: unknown/missing fields degrade to "out_of_scope", never raises."""
    if not record:
        return "out_of_scope"
    compiled = record.get("compiled_success")
    if compiled is None:
        return "out_of_scope"
    if compiled is not True:
        return "failed"

    # Only MECHANICAL deletion disqualifies a repair. Lines the LLM removes are repair work -- the
    # decompiler emits genuine garbage (`# postinserted` scaffolding, duplicated statements,
    # orphaned fragments) that a correct fix deletes on purpose. What is not a repair is the
    # delete-only fallback stripping lines until the file happens to parse, so the fallback is what
    # the classification keys on. `lines_deleted_vs_original` refines the SIZE of that mechanical
    # deletion when recorded (it counts every removed line, not just the fallback's own tally);
    # outside the fallback it is diagnostic only and never re-buckets a repair.
    if record.get("delete_only_fallback_used"):
        measured = record.get("lines_deleted_vs_original")
        deletions = measured if measured is not None else (record.get("delete_only_deletions") or 0)
    else:
        deletions = 0
    if record.get("delete_only_fallback_used") and deletions > 0:
        return "light_deletion" if deletions <= _LIGHT_DELETION_MAX_LINES else "heavy_deletion"
    # Synthesized scaffolding (an `except Exception: pass` for a handler-less `try:`) DELETES
    # nothing and preserves the entire original body, so it is a real repair by the same rule --
    # only mechanical deletion disqualifies. The synthesis is still recorded in
    # `deterministic_prepass_operations` so reporting can break it out as a sub-category, but it
    # belongs in the genuine numerator, not in a competing bucket.
    return "genuine_lossy" if record.get("lossy_literal_closure") else "genuine"


def clamp_window_to_budget(lines, error_index, max_tokens, count_tokens):
    """Shrink an oversized repair window around the error line so it fits ``max_tokens``.

    Repair is local, so an over-budget window is not a reason to abandon the file: the window is
    clamped around the error line and the compile-and-retry loop re-localizes to the NEXT error on
    the following iteration, sliding the window across the whole file. Only a file whose ERROR LINE
    ALONE exceeds the budget is genuinely out of scope -- that is the minified/packer one-liner,
    which cannot be windowed at any granularity.

    ``lines`` is the window's lines (each keeping its newline), ``error_index`` indexes the line to
    keep centred, and ``count_tokens`` is a callable (str) -> int. Returns an ``(start, end)``
    half-open line span that always contains ``error_index``, or ``None`` when the file is
    un-windowable (or there are no lines). ``max_tokens <= 0`` disables the clamp.

    Grows symmetrically outward from the error line, alternating sides, so the model sees balanced
    context; stops as soon as the next line on either side would break the budget.
    """
    if not lines:
        return None
    error_index = max(0, min(int(error_index or 0), len(lines) - 1))
    if max_tokens is None or max_tokens <= 0:
        return (0, len(lines))

    if count_tokens("".join(lines)) <= max_tokens:
        return (0, len(lines))

    used = count_tokens(lines[error_index])
    if used > max_tokens:
        return None  # un-windowable: the error line alone busts the budget

    start, end = error_index, error_index + 1
    grow_up = True
    while start > 0 or end < len(lines):
        # Alternate sides so context stays balanced; skip a side once it is exhausted.
        if grow_up and start == 0:
            grow_up = False
        elif not grow_up and end == len(lines):
            grow_up = True

        candidate = lines[start - 1] if grow_up else lines[end]
        cost = count_tokens(candidate)
        if used + cost > max_tokens:
            break
        used += cost
        if grow_up:
            start -= 1
        else:
            end += 1
        grow_up = not grow_up

    return (start, end)


def choose_initial_error(row_error, probe_result=None):
    """Return an error description that carries a resolvable LINE NUMBER.

    The syntactic-repair loop localizes with ``extract_line_number(error_description)``; an
    error with no line number (or None) yields an empty window and no LLM call. Dataset rows
    do not always carry a line-numbered error (e.g. the malware manifest stores only
    "unexpected indent"). ``probe_result`` is a fresh ``compile_new_pyc``-shaped dict
    ({"is_compiled": bool, "error_description": str|None}) for the actual file -- the file is
    the source of truth for its current error. Prefer ``row_error`` when it already has a line
    number; otherwise use the probe's (line-numbered) error; otherwise fall back to
    ``row_error`` unchanged (nothing better available)."""
    if row_error and extract_line_number(row_error) is not None:
        return row_error
    if probe_result and not probe_result.get("is_compiled") and probe_result.get("error_description"):
        return probe_result["error_description"]
    return row_error


def failure_cleanup(
    affected_file_path: Path,
    final_code: str,
    idx: str,
    error_word: str | None,
    error_message: str | None,
    log_rec: dict[str, Any],
) -> None:
    repaired_path = affected_file_path / f"syntax_repaired_{idx}.py"
    failed_path = affected_file_path / f"syntax_failed_repaired_{idx}.py"

    try:
        os.remove(str(repaired_path))
    except FileNotFoundError:
        pass

    with open(str(failed_path), "w", encoding="utf-8") as f:
        f.write(final_code)

    log_rec.update(
        {
            "compile_error_word_after": error_word,
            "path_out": str(failed_path),
            "compile_error_message_after": clean_error_message(error_message) if error_message else None,
        }
    )
