#!/usr/bin/env python3

import os
import csv
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from utils.generate_bytecode import *

load_dotenv()

BASE_PYTHON_FILES = Path(os.getenv("BASE_DIR_PYTHON_FILES"))
PYTHON_VERSIONS = {PythonVersion((3, x)) for x in range(10, 15)}
HASH_FILES_CSV = Path(os.getenv("PROJECT_ROOT_DIR")) / "dataset" / os.getenv("BASE_DATASET_NAME")
OUTPUT_CSV = Path(os.getenv("PROJECT_ROOT_DIR")) / "dataset" / "decompiled_syntax_errors_pylingual.csv"

CSV_FIELDS = [
    "file_hash",
    "file",
    "error_message",
    "error_description",
    "error",
]


def get_source_name_from_decompiled_name(decompiled_name: str) -> str:
    name = decompiled_name

    if name.startswith("decompiled_"):
        name = name[len("decompiled_"):]

    name = re.sub(r"\.cpython-\d+\.py$", ".py", name)

    return name


def iter_hash_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"file_hash", "file"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain 'file_hash' and 'file' columns")

        for row in reader:
            file_hash = row["file_hash"].strip()
            file_name = row["file"].strip()

            if file_hash and file_name:
                yield {
                    "file_hash": file_hash,
                    "file": file_name,
                }


def choose_source_py(hash_dir: Path, decompiled_file_name_from_csv: str):
    source_name = get_source_name_from_decompiled_name(decompiled_file_name_from_csv)
    source_path = hash_dir / source_name

    if source_path.is_file():
        return source_path

    return None


def run_pylingual(pyc_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pylingual",
        "-q",
        "-o", str(out_dir),
        str(pyc_file),
    ]

    return subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def find_decompiled_file(out_dir: Path, pyc_file: Path):
    decompiled_name = f"decompiled_{pyc_file.name.replace('.pyc', '.py')}"
    candidate = out_dir / decompiled_name

    if candidate.exists():
        return candidate

    return None


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


def syntax_check(file_path: Path, file_hash: str, version: PythonVersion):
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
            "file": file_path.name,
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


def main():
    root = BASE_PYTHON_FILES

    print(f"Root directory: {root}")
    print(f"Reading CSV: {HASH_FILES_CSV}")
    print()

    versions_sorted = sorted(PYTHON_VERSIONS, key=lambda v: v.as_tuple())

    for version in versions_sorted:
        print("=" * 70)
        print(f"Starting Python {version.as_str()}")
        print("=" * 70)

        rows = []
        processed = 0
        skipped = 0
        errors = 0
        crashes = 0

        row_iter = iter_hash_rows(HASH_FILES_CSV)

        for idx, item in enumerate(row_iter, start=1):
            if idx > 5:
                break

            file_hash = item.get("file_hash", "<unknown>")

            try:
                csv_file_name = item["file"]
                hash_dir = root / file_hash

                print(f"[{version.as_str()}] {file_hash}")

                if not hash_dir.exists():
                    print("  -> skip (directory missing)")
                    skipped += 1
                    continue

                source_py = choose_source_py(hash_dir, csv_file_name)
                if source_py is None:
                    print(f"  -> skip (source file not found for {csv_file_name})")
                    skipped += 1
                    continue

                print(f"  -> source file: {source_py.name}")

                pyc_file = hash_dir / "__pycache__" / f"{source_py.stem}.cpython-{version.major}{version.minor}.pyc"
                pyc_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    compile_version(source_py, pyc_file, version)
                    print(f"  -> compiled source to {pyc_file.name}")
                except CompileError as e:
                    print(f"  -> skip (source compile failed: {e})")
                    skipped += 1
                    continue

                decompiled_out_dir = hash_dir / f"decompiled_output_py{version.major}{version.minor}"

                res = run_pylingual(pyc_file, decompiled_out_dir)
                if res.returncode != 0:
                    print(f"  -> skip (pylingual failed: exit code {res.returncode})")
                    skipped += 1
                    continue

                print(f"  -> pylingual finished, output dir: {decompiled_out_dir}")

                decompiled_file = find_decompiled_file(decompiled_out_dir, pyc_file)
                if decompiled_file is None:
                    print("  -> skip (decompiled file missing)")
                    skipped += 1
                    continue

                print(f"  -> decompiled file: {decompiled_file.name}")

                err = syntax_check(decompiled_file, file_hash, version)
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

        version_output_csv = OUTPUT_CSV.with_name(
            f"{OUTPUT_CSV.stem}_{version.as_str()}{OUTPUT_CSV.suffix}"
        )

        save_csv(rows, version_output_csv)

        print("-" * 70)
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