from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def append_log(path: Path, record: dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def clean_error_message(error_message: str | None):
    if not error_message:
        return None
    cleaned_message = error_message.lower().strip()
    cleaned_message = re.sub(r"\(.*?\)", "", cleaned_message).strip()
    return cleaned_message


def extract_line_number(error_msg: str):
    m = re.search(r"line\s+(\d+)", error_msg)
    return int(m.group(1)) if m else None


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
