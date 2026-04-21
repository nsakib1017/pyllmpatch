#!/usr/bin/env python3

import os
import csv
import re
import gc
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from utils.generate_bytecode import *

load_dotenv()

BASE_PYTHON_FILES = Path(os.getenv("ROOT_FOR_FILES")) / Path(os.getenv("BASE_DIR_PYTHON_FILES_PYPI"))
PYTHON_VERSIONS = {PythonVersion((3, x)) for x in range(10, 11)}

OUTPUT_CSV_BASE = Path(os.getenv("PROJECT_ROOT_DIR")) / "dataset" / "pylingual" / "pypi_decompiled_syntax_errors_pylingual_3.10.csv"
OUTPUT_JSONL_BASE = Path(os.getenv("PROJECT_ROOT_DIR")) / "dataset" / "pylingual" / "pypi_decompiled_analysis_pylingual_3.10.jsonl"

MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_DECOMPILED_BYTES = 5 * 1024 * 1024

DECOMPILERS = [
    "pylingual",
    # "pycdc"
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

    stdout_file = out_dir / f"{pyc_file.stem}_pylingual.stdout.txt"
    stderr_file = out_dir / f"{pyc_file.stem}_pylingual.stderr.txt"
    meta_file = out_dir / f"{pyc_file.stem}_pylingual.meta.txt"

    cmd = [
        "pylingual",
        "-o", str(out_dir),
        str(pyc_file),
    ]

    with stdout_file.open("w", encoding="utf-8") as out_f, stderr_file.open("w", encoding="utf-8") as err_f:
        result = subprocess.run(
            cmd,
            stdout=out_f,
            stderr=err_f,
            text=True,
            check=False,
        )

    with meta_file.open("w", encoding="utf-8") as f:
        f.write(f"command: {' '.join(cmd)}\n")
        f.write(f"returncode: {result.returncode}\n")
        f.write(f"pyc_file: {pyc_file}\n")
        f.write(f"output_dir: {out_dir}\n")
        f.write(f"stdout_file: {stdout_file}\n")
        f.write(f"stderr_file: {stderr_file}\n")

    expected_file = out_dir / f"decompiled_{pyc_file.name.replace('.pyc', '.py')}"
    return result, expected_file, stdout_file, stderr_file


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
            text=True,
        )

    return result, output_file, output_file, error_file


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


def build_output_jsonl_path(base_path: Path, decompiler_name: str, version: PythonVersion) -> Path:
    return base_path.with_name(
        f"{base_path.stem}_{decompiler_name}_{version.as_str()}{base_path.suffix}"
    )


def append_jsonl(record: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_text_if_exists(path: Path):
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def parse_pylingual_stdout(stdout_text: str):
    """
    Parse Pylingual stdout to extract:
      - code object success rate
      - failed code objects from the equivalence results table
      - all table rows if needed for later analysis
    """
    result = {
        "code_object_success_rate": None,
        "failed_code_objects": [],
        "equivalence_rows": [],
    }

    if not stdout_text:
        return result

    rate_match = re.search(
        r'([0-9]+(?:\.[0-9]+)?)%\s+code object success rate',
        stdout_text,
        flags=re.IGNORECASE
    )
    if rate_match:
        try:
            result["code_object_success_rate"] = float(rate_match.group(1))
        except ValueError:
            pass

    # Parse unicode table rows like:
    # │ <module>.foo │ Failure │ Different bytecode │
    table_row_re = re.compile(
        r'^\s*│\s*(?P<code_object>.*?)\s*│\s*(?P<success>Success|Failure)\s*│\s*(?P<message>.*?)\s*│\s*$'
    )

    for line in stdout_text.splitlines():
        m = table_row_re.match(line)
        if not m:
            continue

        code_object = m.group("code_object").strip()
        success = m.group("success").strip()
        message = m.group("message").strip()

        row = {
            "code_object": code_object,
            "success": success,
            "message": message,
        }
        result["equivalence_rows"].append(row)

        if success.lower() == "failure":
            result["failed_code_objects"].append(row)

    return result


def classify_result(syntax_err, code_object_success_rate, decompiler_returncode):
    if syntax_err is not None:
        return "syntactic_error"

    if decompiler_returncode != 0:
        return "decompiler_failure"

    if code_object_success_rate is None:
        return "unknown"

    if code_object_success_rate < 100.0:
        return "semantic_error"

    return "success"


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
            syntax_errors = 0
            semantic_errors = 0
            successes = 0
            unknowns = 0
            crashes = 0

            row_items = list(iter_hash_dirs(root))
            version_output_csv = build_output_csv_path(OUTPUT_CSV_BASE, decompiler_name, version)
            version_output_jsonl = build_output_jsonl_path(OUTPUT_JSONL_BASE, decompiler_name, version)

            for idx, item in enumerate(row_items, start=1):
                file_hash = item.get("file_hash", "<unknown>")

                pyc_file = None
                decompiled_file = None
                stdout_file = None
                stderr_file = None

                record = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "decompiler": decompiler_name,
                    "bytecode_version": version.as_str(),
                    "file_hash": file_hash,
                    "file": None,
                    "status": "unknown",
                    "syntax_error": False,
                    "semantic_error": False,
                    "code_object_success_rate": None,
                    "failed_code_objects": [],
                    "equivalence_rows": [],
                    "error_message": None,
                    "error_description": None,
                    "error": None,
                    "decompiler_returncode": None,
                    "stdout_file": None,
                    "stderr_file": None,
                    "decompiled_file": None,
                    "source_file": None,
                    "skipped": False,
                    "skip_reason": None,
                    "exception": None,
                }

                try:
                    hash_dir = item["hash_dir"]

                    print(f"[{decompiler_name} | {version.as_str()}] {file_hash} ({idx}/{len(row_items)})")

                    if not hash_dir.exists():
                        print("  -> skip (directory missing)")
                        skipped += 1
                        record["skipped"] = True
                        record["skip_reason"] = "directory missing"
                        append_jsonl(record, version_output_jsonl)
                        continue

                    source_py = choose_source_py(hash_dir)
                    if source_py is None:
                        print("  -> skip (source file not found)")
                        skipped += 1
                        record["skipped"] = True
                        record["skip_reason"] = "source file not found"
                        append_jsonl(record, version_output_jsonl)
                        continue

                    record["file"] = source_py.name
                    record["source_file"] = str(source_py)

                    print(f"  -> source file: {source_py.name}")

                    source_size = source_py.stat().st_size
                    if source_size > MAX_SOURCE_BYTES:
                        print(f"  -> skip (source too large: {source_size} bytes)")
                        skipped += 1
                        record["skipped"] = True
                        record["skip_reason"] = f"source too large: {source_size}"
                        append_jsonl(record, version_output_jsonl)
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
                        record["skipped"] = True
                        record["skip_reason"] = "source compile failed"
                        record["exception"] = str(e)
                        append_jsonl(record, version_output_jsonl)
                        continue

                    decompiled_out_dir = hash_dir / f"decompiled_output_{decompiler_name}"

                    print(f"  -> step: run {decompiler_name}")
                    res, expected_decompiled_file, stdout_file, stderr_file = run_decompiler(
                        decompiler_name, pyc_file, decompiled_out_dir
                    )

                    record["decompiler_returncode"] = res.returncode
                    record["stdout_file"] = str(stdout_file) if stdout_file else None
                    record["stderr_file"] = str(stderr_file) if stderr_file else None

                    if res.returncode != 0:
                        print(f"  -> skip ({decompiler_name} failed: exit code {res.returncode})")
                        if stderr_file is not None and stderr_file.exists():
                            print(f"  -> stderr saved to: {stderr_file}")

                        skipped += 1
                        record["status"] = "decompiler_failure"
                        record["skipped"] = True
                        record["skip_reason"] = f"{decompiler_name} failed: exit code {res.returncode}"
                        append_jsonl(record, version_output_jsonl)
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
                        record["skipped"] = True
                        record["skip_reason"] = "decompiled file missing"
                        append_jsonl(record, version_output_jsonl)
                        continue

                    record["decompiled_file"] = str(decompiled_file)

                    print(f"  -> decompiled file: {decompiled_file.name}")

                    decompiled_size = decompiled_file.stat().st_size
                    if decompiled_size > MAX_DECOMPILED_BYTES:
                        print(f"  -> skip (decompiled file too large: {decompiled_size} bytes)")
                        skipped += 1
                        record["skipped"] = True
                        record["skip_reason"] = f"decompiled file too large: {decompiled_size}"
                        append_jsonl(record, version_output_jsonl)
                        continue

                    stdout_text = read_text_if_exists(stdout_file)
                    parsed_stdout = parse_pylingual_stdout(stdout_text)
                    record["code_object_success_rate"] = parsed_stdout["code_object_success_rate"]
                    record["failed_code_objects"] = parsed_stdout["failed_code_objects"]
                    record["equivalence_rows"] = parsed_stdout["equivalence_rows"]

                    print("  -> step: compile decompiled file")
                    err = syntax_check(decompiled_file, file_hash, source_py.name, version)

                    if err:
                        record["syntax_error"] = True
                        record["error_message"] = err.get("error_message")
                        record["error_description"] = err.get("error_description")
                        record["error"] = err.get("error")
                        print("  -> syntax error detected")
                    else:
                        print("  -> syntax OK")

                    status = classify_result(
                        syntax_err=err,
                        code_object_success_rate=record["code_object_success_rate"],
                        decompiler_returncode=res.returncode,
                    )
                    record["status"] = status
                    record["semantic_error"] = (status == "semantic_error")

                    if status == "syntactic_error":
                        syntax_errors += 1
                        rows.append(err)
                        print("  -> classified as syntactic_error")

                    elif status == "semantic_error":
                        semantic_errors += 1
                        print(f"  -> classified as semantic_error (success rate: {record['code_object_success_rate']})")

                    elif status == "success":
                        successes += 1
                        print("  -> classified as success")

                    else:
                        unknowns += 1
                        print("  -> classified as unknown")

                    append_jsonl(record, version_output_jsonl)

                    processed += 1
                    print()

                except Exception as e:
                    crashes += 1
                    record["status"] = "exception"
                    record["exception"] = str(e)
                    append_jsonl(record, version_output_jsonl)
                    print(f"  -> ERROR processing {file_hash}: {e}")
                    print()

                gc.collect()

            save_csv(rows, version_output_csv)

            print("-" * 70)
            print(f"Decompiler: {decompiler_name}")
            print(f"Version {version.as_str()} finished")
            print(f"Processed: {processed}")
            print(f"Skipped: {skipped}")
            print(f"Syntactic errors: {syntax_errors}")
            print(f"Semantic errors: {semantic_errors}")
            print(f"Successes: {successes}")
            print(f"Unknowns: {unknowns}")
            print(f"Unexpected crashes: {crashes}")
            print(f"Saved CSV: {version_output_csv}")
            print(f"Saved JSONL: {version_output_jsonl}")
            print("-" * 70)
            print()


if __name__ == "__main__":
    main()