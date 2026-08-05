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
INPUT_DATASET_CSV = Path(os.getenv("PROJECT_ROOT_DIR")) / "dataset" / "pylingual" / os.getenv("BASE_DATASET_NAME")
OUTPUT_CSV = Path(os.getenv("PROJECT_ROOT_DIR")) / "dataset" / "pycdc" / "pylingual_dataset_summary_pycdc.csv"

MAX_DECOMPILED_BYTES = 5 * 1024 * 1024

CSV_FIELDS = [
    "file_hash",
    "file",
    "error_message",
    "error_description",
    "error",
    "bytecode_version"
]


def iter_input_rows(csv_path: Path):

    with csv_path.open(newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        required = {"file_hash", "error_type", "bytecode_version"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain 'file_hash', 'error_type', and 'bytecode_version'"
            )

        row_idx = 0

        while True:

            try:
                row = next(reader)

            except StopIteration:
                break

            except csv.Error as e:
                print(f"  -> skip row {row_idx} (CSV parsing error: {e})")
                row_idx += 1
                continue

            row_idx += 1

            try:

                if (row.get("error_type") or "").strip() != "syntactic_error":
                    continue

                file_hash = (row.get("file_hash") or "").strip()
                bytecode_version = (row.get("bytecode_version") or "").strip()

                if not file_hash or not bytecode_version:
                    continue

                m = re.match(r"(\d+)(?:\.(\d+))?", bytecode_version)

                if not m:
                    print(f"  -> skip row {row_idx} (invalid bytecode version)")
                    continue

                major = int(m.group(1))
                minor = int(m.group(2)) if m.group(2) else 0

                yield {
                    "file_hash": file_hash,
                    "version": PythonVersion((major, minor)),
                }

            except Exception as e:
                print(f"  -> skip row {row_idx} (row processing error: {e})")
                continue

def find_candidate_pyc_files(hash_dir: Path):
    candidates = []

    for p in hash_dir.rglob("*.pyc"):
        if "pycdc_decompilation_output" in p.parts:
            continue
        candidates.append(p)

    candidates.sort(key=lambda p: str(p))
    return candidates


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


def get_error_word_message_from_text(text):
    file_line_re = re.compile(r'^\s*File "([^"]+)", line (\d+)')
    error_line_re = re.compile(r'^\s*(\w+(?:Error|Exception))(?:\s*:\s*(.*))?$')

    error_word = None
    message = None

    for line in text.splitlines():
        if file_line_re.match(line):
            continue

        m = error_line_re.match(line)
        if m:
            error_word = m.group(1)
            message = (m.group(2) or "").strip()
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
            "bytecode_version": version.as_str(),
        }


def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():

    print(f"Root directory: {BASE_PYTHON_FILES}")
    print(f"Input CSV: {INPUT_DATASET_CSV}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Max decompiled bytes: {MAX_DECOMPILED_BYTES}")
    print()

    rows = []
    processed = 0
    skipped = 0
    errors = 0
    crashes = 0

    for idx, item in enumerate(iter_input_rows(INPUT_DATASET_CSV), start=1):

        # if idx > 100:
        #     break

        file_hash = item["file_hash"]
        version = item["version"]

        try:

            hash_dir = BASE_PYTHON_FILES / file_hash

            print(f"[{idx}] {file_hash} | Python {version.as_str()}")

            if not hash_dir.exists():
                print("  -> skip (directory missing)")
                skipped += 1
                continue

            print("  -> step: locate candidate pyc files")

            candidate_pycs = find_candidate_pyc_files(hash_dir)

            if not candidate_pycs:
                print("  -> skip (no .pyc files found)")
                skipped += 1
                continue

            pyc_file = candidate_pycs[0]

            print(f"  -> using pyc: {pyc_file}")

            out_dir = hash_dir / "pycdc_decompilation_output"

            print("  -> step: run pycdc")

            res, expected_decompiled_file, stderr_file = run_pycdc(pyc_file, out_dir)

            if res.returncode != 0:
                print(f"  -> skip (pycdc failed: exit code {res.returncode})")
                if stderr_file.exists():
                    print(f"  -> stderr saved to: {stderr_file}")
                skipped += 1
                continue

            print(f"  -> decompiled file: {expected_decompiled_file}")

            if not expected_decompiled_file.exists():
                print("  -> skip (decompiled file missing)")
                skipped += 1
                continue

            decompiled_size = expected_decompiled_file.stat().st_size

            if decompiled_size > MAX_DECOMPILED_BYTES:
                print(f"  -> skip (decompiled file too large: {decompiled_size} bytes)")
                skipped += 1
                continue

            print("  -> step: compile decompiled file")

            err = syntax_check(expected_decompiled_file, file_hash, version)

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

    save_csv(rows, OUTPUT_CSV)

    print("-" * 70)
    print("Finished")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Syntax errors: {errors}")
    print(f"Unexpected crashes: {crashes}")
    print(f"Saved: {OUTPUT_CSV}")
    print("-" * 70)


if __name__ == "__main__":
    main()