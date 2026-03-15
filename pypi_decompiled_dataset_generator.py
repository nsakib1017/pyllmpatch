#!/usr/bin/env python3

import os
import csv
import re
import gc
import time
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from utils.generate_bytecode import *

load_dotenv()

BASE_PYTHON_FILES = Path(os.getenv("BASE_DIR_PYTHON_FILES"))
PYTHON_VERSIONS = {PythonVersion((3, x)) for x in range(10, 14)}
OUTPUT_CSV_BASE = Path(os.getenv("PROJECT_ROOT_DIR")) / "dataset" / "pycdc" / "pypi_decompiled_syntax_errors.csv"

MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_DECOMPILED_BYTES = 5 * 1024 * 1024

DECOMPILERS = [
    # "pylingual",
    "pycdc"
]

CSV_FIELDS = [
    "file_hash",
    "file",
    "error_message",
    "error_description",
    "error",
]


def choose_source_py(hash_dir: Path):
    candidates = [
        p for p in hash_dir.iterdir()
        if p.is_file()
        and p.suffix == ".py"
        and not p.name.startswith("decompiled")
    ]

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def iter_hash_dirs(root: Path):
    """
    Iterate over hash directories directly from the filesystem.
    Each subdirectory name is treated as the file hash.
    """
    for p in root.iterdir():
        if p.is_dir():
            yield {
                "file_hash": p.name,
                "hash_dir": p,
            }


def run_pylingual(pyc_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pylingual",
        "-q",
        "-o", str(out_dir),
        str(pyc_file),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    expected_file = out_dir / f"decompiled_{pyc_file.name.replace('.pyc', '.py')}"
    stderr_file = None
    return result, expected_file, stderr_file


def run_pycdc(pyc_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / f"decompiled_{pyc_file.name.replace('.pyc', '.py')}"
    error_file = out_dir / f"{pyc_file.stem}_pycdc.stderr.txt"

    with output_file.open("w", encoding="utf-8") as out_f, error_file.open("w", encoding="utf-8") as err_f:
        result = subprocess.run(
            ["pycdc", str(pyc_file)],
            stdout=out_f,
            stderr=err_f,
            check=False,
        )

    return result, output_file, error_file


def run_decompiler(decompiler_name: str, pyc_file: Path, out_dir: Path):
    if decompiler_name == "pylingual":
        return run_pylingual(pyc_file, out_dir)
    if decompiler_name == "pycdc":
        return run_pycdc(pyc_file, out_dir)
    raise ValueError(f"Unsupported decompiler: {decompiler_name}")


def find_decompiled_file(out_dir: Path, pyc_file: Path):
    expected_name = f"decompiled_{pyc_file.name.replace('.pyc', '.py')}"
    expected_path = out_dir / expected_name

    if expected_path.exists():
        return expected_path

    candidates = list(out_dir.rglob("decompiled_*.py"))
    if not candidates:
        return None

    pyc_stem = pyc_file.stem

    exact_stem_matches = [p for p in candidates if p.stem == f"decompiled_{pyc_stem}"]
    if exact_stem_matches:
        exact_stem_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return exact_stem_matches[0]

    loose_matches = [p for p in candidates if pyc_stem in p.stem]
    if loose_matches:
        loose_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return loose_matches[0]

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def get_error_word_message_from_text(text):
    file_line_re = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in (.+))?')
    error_line_re = re.compile(r'^\s*(\w+(?:Error|Exception))(?:\s*:\s*(.*))?$')
    sorry_line_re = re.compile(r'^\s*Sorry:\s*(\w+(?:Error|Exception))\s*:\s*(.*?)\s*\(([^,]+),\s*line\s*(\d+)\)\s*$')

    error_word = None
    message = None

    for line in text.splitlines():
        m_file = file_line_re.match(line)
        if m_file:
            continue

        m_err = error_line_re.match(line)
        if m_err:
            error_word = m_err.group(1)
            message = (m_err.group(2) or "").strip()
            break

        m_sorry = sorry_line_re.match(line)
        if m_sorry:
            error_word = m_sorry.group(1)
            message = (m_sorry.group(2) or "").strip()
            break

    return error_word, message


def syntax_check(file_path: Path, file_hash: str, source_file_name: str, version: PythonVersion):
    pyc_file = file_path.parent / "__pycache__" / f"{file_path.stem}.cpython-{version.major}{version.minor}.pyc"
    pyc_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        compile_version(file_path, pyc_file, version)
        return None

    except CompileError as e:
        err_text = str(e).strip()
        error_word, message = get_error_word_message_from_text(err_text)

        return {
            "file_hash": file_hash,
            "file": source_file_name,
            "error_message": message if message else err_text,
            "error_description": err_text,
            "error": error_word if error_word else err_text,
        }


def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_file(path: Path):
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass


def build_output_csv_path(base_path: Path, decompiler_name: str, version: PythonVersion) -> Path:
    return base_path.with_name(
        f"{base_path.stem}_{decompiler_name}_{version.as_str()}{base_path.suffix}"
    )


def main():
    root = BASE_PYTHON_FILES

    print(f"Root directory: {root}")
    print("Reading hash directories from filesystem")
    print(f"Max source bytes: {MAX_SOURCE_BYTES}")
    print(f"Max decompiled bytes: {MAX_DECOMPILED_BYTES}")
    print(f"Decompilers: {DECOMPILERS}")
    print()

    versions_sorted = sorted(PYTHON_VERSIONS, key=lambda v: v.as_tuple())

    for decompiler_name in DECOMPILERS:
        print("#" * 70)
        print(f"Starting decompiler: {decompiler_name}")
        print("#" * 70)
        print()

        for version in versions_sorted:
            print("=" * 70)
            print(f"Starting Python {version.as_str()} with {decompiler_name}")
            print("=" * 70)

            rows = []
            processed = 0
            skipped = 0
            errors = 0
            crashes = 0

            row_iter = iter_hash_dirs(root)

            for idx, item in enumerate(row_iter, start=1):
                # if idx > 5:
                #     break

                file_hash = item.get("file_hash", "<unknown>")

                pyc_file = None
                decompiled_file = None

                try:
                    hash_dir = item["hash_dir"]

                    print(f"[{decompiler_name} | {version.as_str()}] {file_hash}")

                    if not hash_dir.exists():
                        print("  -> skip (directory missing)")
                        skipped += 1
                        continue

                    source_py = choose_source_py(hash_dir)
                    if source_py is None:
                        print("  -> skip (source file not found)")
                        skipped += 1
                        continue

                    print(f"  -> source file: {source_py.name}")

                    source_size = source_py.stat().st_size
                    if source_size > MAX_SOURCE_BYTES:
                        print(f"  -> skip (source too large: {source_size} bytes)")
                        skipped += 1
                        continue

                    print("  -> step: compile source")
                    pyc_file = hash_dir / "__pycache__" / f"{source_py.stem}.cpython-{version.major}{version.minor}.pyc"
                    pyc_file.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        compile_version(source_py, pyc_file, version)
                        print(f"  -> compiled source to {pyc_file.name}")
                    except CompileError as e:
                        print(f"  -> skip (source compile failed: {e})")
                        skipped += 1
                        continue

                    decompiled_out_dir = hash_dir / f"decompiled_output_{decompiler_name}"

                    print(f"  -> step: run {decompiler_name}")
                    res, expected_decompiled_file, stderr_file = run_decompiler(
                        decompiler_name, pyc_file, decompiled_out_dir
                    )

                    if res.returncode != 0:
                        print(f"  -> skip ({decompiler_name} failed: exit code {res.returncode})")
                        if stderr_file is not None and stderr_file.exists():
                            print(f"  -> stderr saved to: {stderr_file}")
                        skipped += 1
                        continue

                    print(f"  -> {decompiler_name} finished, output dir: {decompiled_out_dir}")

                    print("  -> step: locate decompiled file")
                    decompiled_file = None
                    for attempt in range(3):
                        if expected_decompiled_file is not None and expected_decompiled_file.exists():
                            decompiled_file = expected_decompiled_file
                            break

                        decompiled_file = find_decompiled_file(decompiled_out_dir, pyc_file)
                        if decompiled_file is not None:
                            break

                        time.sleep(0.5)

                    if decompiled_file is None:
                        existing = list(decompiled_out_dir.rglob("*")) if decompiled_out_dir.exists() else []
                        print(f"  -> skip (decompiled file missing, found {len(existing)} item(s) in output dir)")
                        for p in existing[:10]:
                            print(f"     - {p}")
                        skipped += 1
                        continue

                    print(f"  -> decompiled file: {decompiled_file.name}")

                    decompiled_size = decompiled_file.stat().st_size
                    if decompiled_size > MAX_DECOMPILED_BYTES:
                        print(f"  -> skip (decompiled file too large: {decompiled_size} bytes)")
                        skipped += 1
                        continue

                    print("  -> step: compile decompiled file")
                    err = syntax_check(decompiled_file, file_hash, source_py.name, version)
                    if err:
                        rows.append(err)
                        errors += 1
                        print("  -> syntax error detected")
                    else:
                        print("  -> syntax OK")

                    processed += 1
                    print()

                except Exception as e:
                    crashes += 1
                    print(f"  -> ERROR processing {file_hash}: {e}")
                    print()

                gc.collect()

            version_output_csv = build_output_csv_path(OUTPUT_CSV_BASE, decompiler_name, version)
            save_csv(rows, version_output_csv)

            print("-" * 70)
            print(f"Decompiler: {decompiler_name}")
            print(f"Version {version.as_str()} finished")
            print(f"Processed: {processed}")
            print(f"Skipped: {skipped}")
            print(f"Syntax errors: {errors}")
            print(f"Unexpected crashes: {crashes}")
            print(f"Saved: {version_output_csv}")
            print("-" * 70)
            print()


if __name__ == "__main__":
    main()