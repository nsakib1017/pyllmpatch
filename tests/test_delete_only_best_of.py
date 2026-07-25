"""Tests for `utils.delete_only_compilation.delete_only_best_of` -- the
"best-of-both" delete-only fallback helper.

Context (see pipeline/runner.py's syntactic-repair delete-only fallback):
after the LLM exhausts its retries, the fallback used to run
`delete_lines_until_compilable_with_oracle` on only the LLM's last
(possibly mangled) output. That frequently can't recover (LLM-injected
`<mask_N>` tokens, unbalanced parens are not something line-deletion can
fix), even though running the SAME deletion search on the pristine
original decompiled source often reaches a compiling file. `delete_only_best_of`
runs the deletion search independently over BOTH sources and keeps
whichever result is better, per the rule documented on the function.

The `compile_check` shim below matches `compile_new_pyc`'s contract:
`compile_check(src, out_py_path, out_pyc_path, version) -> {"is_compiled":
bool, "error_description": str | None}`, writing the source unconditionally
and a stand-in ``.pyc`` only on success -- exactly like the real
`pipeline.runner.compile_new_pyc`, just backed by the host `compile()`
builtin instead of `compile_version` so the test needs no interpreter
version plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.logging_utils import extract_line_number
from utils.delete_only_compilation import delete_only_best_of
from utils.file_helpers import get_error_word_message_from_content


def _host_compile_check(src, out_py_path, out_pyc_path, version=None):
    with open(out_py_path, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        compile(src, out_py_path, "exec")
    except SyntaxError as exc:
        return {"is_compiled": False, "error_description": f"{type(exc).__name__}: {exc}"}
    with open(out_pyc_path, "wb") as f:
        f.write(b"FAKEPYC")
    return {"is_compiled": True, "error_description": None}


def _run(tmp_path, labeled_sources, *, suffix=""):
    out_py = str(tmp_path / f"out{suffix}.py")
    out_pyc = str(tmp_path / f"out{suffix}.pyc")
    err_txt = str(tmp_path / f"err{suffix}.txt")
    result = delete_only_best_of(
        labeled_sources,
        compile_check=_host_compile_check,
        extract_line_number=extract_line_number,
        get_error_word_message_from_content=get_error_word_message_from_content,
        out_py_path=out_py,
        out_pyc_path=out_pyc,
        version=None,
        base_window=0,
        max_iters=200,
        max_deleted_ratio=0.95,
        min_remaining_lines=1,
        err_txt_path=err_txt,
    )
    return result, out_py, out_pyc


def test_llm_unrecoverable_original_recoverable_picks_original(tmp_path):
    """LLM output is corrupted beyond what deletion can fix (mask token,
    unbalanced construct); the pristine original deletes down to something
    that compiles. The winner must be "original"."""
    original = "def f():\n    x = 1\n    return x\n"
    llm_broken = "def f(:\n    <mask_25> = broken(\n"

    (fixed_code, del_log, last_res, winner), out_py, out_pyc = _run(
        tmp_path, [("original", original), ("llm", llm_broken)]
    )

    assert winner == "original"
    assert last_res["is_compiled"] is True
    assert fixed_code.strip() != ""
    # canonical output paths hold the winner's fixed code.
    assert Path(out_py).read_text(encoding="utf-8") == fixed_code
    assert Path(out_pyc).read_bytes() == b"FAKEPYC"


def test_both_compile_more_content_preserved_wins(tmp_path):
    """Both sources are recoverable by deletion, but the LLM source has more
    surviving valid code around its one broken trailing line, so it keeps
    more content after deletion. The winner must be "llm"."""
    original = "def f():\n    return 1\nBROKEN_LINE(\n"
    llm_more_content = (
        "def f():\n"
        "    return 1\n"
        "\n"
        "def g():\n"
        "    return 2\n"
        "\n"
        "def h():\n"
        "    return 3\n"
        "BROKEN_LINE(\n"
    )

    (fixed_code, del_log, last_res, winner), out_py, out_pyc = _run(
        tmp_path, [("original", original), ("llm", llm_more_content)]
    )

    assert winner == "llm"
    assert last_res["is_compiled"] is True
    assert len(fixed_code.splitlines()) > len("def f():\n    return 1\n".splitlines())
    assert Path(out_py).read_text(encoding="utf-8") == fixed_code


def test_tie_prefers_first_candidate_in_input_order(tmp_path):
    """Documented deterministic tie-break: when both candidates compile and
    preserve the exact same amount of content, the FIRST entry in
    `labeled_sources` wins -- so callers passing
    `[("original", ...), ("llm", ...)]` prefer "original" on a tie."""
    tie_original = "x = 1\nBROKEN(\n"
    tie_llm = "y = 1\nBROKEN(\n"

    (fixed_code, del_log, last_res, winner), out_py, out_pyc = _run(
        tmp_path, [("original", tie_original), ("llm", tie_llm)]
    )

    assert winner == "original"
    assert last_res["is_compiled"] is True

    # Reversing input order flips the tie-break winner too -- confirms the
    # rule is genuinely "first in input order", not label-specific.
    (fixed_code2, del_log2, last_res2, winner2), _, _ = _run(
        tmp_path, [("llm", tie_llm), ("original", tie_original)], suffix="_rev"
    )
    assert winner2 == "llm"
    assert last_res2["is_compiled"] is True


def test_no_usable_source_returns_empty_result(tmp_path):
    """Both candidate sources are empty/falsy -- nothing to run deletion on."""
    result, out_py, out_pyc = _run(tmp_path, [("original", ""), ("llm", "")])

    assert result == ("", [], {"is_compiled": False}, None)
    # canonical output paths must not be touched when there's nothing to run.
    assert not Path(out_py).exists()
    assert not Path(out_pyc).exists()


def test_neither_compiles_picks_the_one_that_got_furthest(tmp_path):
    """Neither source reaches a compiling state within the iteration/ratio
    budget (both bottom out at an unterminated `call(` that deletion can
    never close). The winner is whichever recorded MORE deletion attempts
    -- a proxy for "got furthest" toward compiling: the original has 6
    throwaway args to chip away at one at a time (6 deletion attempts) vs
    the LLM source's 2 (2 deletion attempts), so "original" wins."""
    stubborn_original = "call(\n" + "".join(f"    arg{i},\n" for i in range(6))
    stubborn_llm = "call(\n" + "".join(f"    arg{i},\n" for i in range(2))

    (fixed_code, del_log, last_res, winner), out_py, out_pyc = _run(
        tmp_path,
        [("original", stubborn_original), ("llm", stubborn_llm)],
        suffix="_stubborn",
    )

    assert last_res["is_compiled"] is False
    assert winner == "original"
    # Sanity: output paths hold the winner's (non-compiling) fixed code.
    assert Path(out_py).read_text(encoding="utf-8") == fixed_code
