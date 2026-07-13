"""Bridge the downloaded pyllm_final_dataset layout to the repair pipeline's expected layout.

The pipeline resolves a case via fetch_pyllmpatch_repair_paths(file_hash, source):
  <BASE_DIR>/<file_hash>/*.pyc                         -> GT pyc  (_first_top_level_pyc)
  <BASE_DIR>/<file_hash>/decompiler_output/indented_*.py   -> derived source (highest_indented_file)
  <BASE_DIR>/<file_hash>/decompiler_output/indented_*.pyc  -> derived pyc

The download instead is multi-version:
  pyllm_compiled/python-3.XX/<hash>/<name>.pyc                              (GT pyc)
  pyllm_decompilations/pyllm_3XX/*/decompilation_results/<hash>.py         (decompiled source)

So for every semantic (hash, version) we materialize <ADAPTER>/<hash>_<ver3>/ (ver3 = "311"..):
  gt.pyc                           -> symlink to the download GT pyc
  decompiler_output/indented_1.py  -> symlink to the download decompiled source
  decompiler_output/indented_1.pyc -> COMPILED from the decompiled source under that version

`file_hash` for the run becomes "<hash>_<ver3>" so one BASE_DIR + one CSV covers all versions;
the pipeline reads the real Python version from the pyc magic, so no version param is needed.

Resumable (skips a case whose indented_1.pyc already exists) and parallel. A decompiled source
that does not compile under its version is logged and skipped (not semantic-repairable). Emits a
run CSV. CPU-only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

VER3_TO_TUPLE = {"311": (3, 11), "312": (3, 12), "313": (3, 13), "314": (3, 14), "315": (3, 15)}


def _download_paths(dest: str, h: str, ver3: str):
    """(gt_pyc, decompiled_py) from the download, or (None, None) if either is missing."""
    gt = glob.glob(f"{dest}/pyllm_compiled/python-3.{ver3[1:]}/{h}/*.pyc")
    dec = glob.glob(f"{dest}/pyllm_decompilations/pyllm_{ver3}/*/decompilation_results/{h}.py")
    return (gt[0] if gt else None), (dec[0] if dec else None)


def _semantic_cases(stats_path: str, versions):
    """Yield (hash, ver3) for every semantic case (Different control flow / Different bytecode)."""
    d = json.load(open(stats_path))
    for h, vers in d.items():
        for v, info in vers.items():
            if v not in versions:
                continue
            cat = info.get("category")
            cat = cat if isinstance(cat, str) else (cat[0] if isinstance(cat, list) and cat else "?")
            if cat != "Different":
                continue
            errs = " ".join(info.get("errors") or [])
            if "Different control flow" in errs or "Different bytecode" in errs:
                yield h, v


def _materialize_one(args):
    dest, adapter, h, ver3 = args
    from utils.generate_bytecode import compile_version

    case_id = f"{h}_{ver3}"
    case_dir = Path(adapter) / case_id
    derived_pyc = case_dir / "decompiler_output" / "indented_1.pyc"
    if derived_pyc.exists():
        return (case_id, ver3, "cached")

    gt_pyc, dec_py = _download_paths(dest, h, ver3)
    if not gt_pyc or not dec_py:
        return (case_id, ver3, "missing_src")

    (case_dir / "decompiler_output").mkdir(parents=True, exist_ok=True)
    gt_link = case_dir / "gt.pyc"
    src_link = case_dir / "decompiler_output" / "indented_1.py"
    for link, target in ((gt_link, gt_pyc), (src_link, dec_py)):
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.abspath(target))

    try:
        compile_version(str(src_link), str(derived_pyc), VER3_TO_TUPLE[ver3])
    except Exception as e:  # decompiled source doesn't compile under this version
        return (case_id, ver3, f"compile_fail:{type(e).__name__}")
    if not derived_pyc.exists():
        return (case_id, ver3, "compile_no_output")
    return (case_id, ver3, "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/home/mxs220189/pylingual_collaboration/pyllm_final_dataset")
    ap.add_argument("--adapter", default="/home/mxs220189/pylingual_collaboration/pyllm_adapter")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--versions", default="311,312,313,314,315")
    ap.add_argument("--limit", type=int, default=0, help="cap cases (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    versions = set(args.versions.split(","))
    cases = list(_semantic_cases(f"{args.dest}/stats.json", versions))
    if args.limit:
        cases = cases[: args.limit]
    print(f"semantic cases to materialize: {len(cases)} (versions {sorted(versions)})", flush=True)
    Path(args.adapter).mkdir(parents=True, exist_ok=True)

    tasks = [(args.dest, args.adapter, h, v) for h, v in cases]
    from collections import Counter
    outcomes = Counter()
    ok_ids = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_materialize_one, t) for t in tasks]
        for f in as_completed(futs):
            case_id, ver3, status = f.result()
            key = status.split(":")[0]
            outcomes[key] += 1
            if status in ("ok", "cached"):
                ok_ids.append((case_id, ver3))
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)}  {dict(outcomes)}", flush=True)

    print(f"\nDONE {done} cases: {dict(outcomes)}", flush=True)
    csv_out = args.csv_out or f"{args.adapter}/semantic_run.csv"
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_hash", "error_type", "error", "error_message", "error_description",
                    "bytecode_version", "source"])
        for case_id, ver3 in sorted(ok_ids):
            w.writerow([case_id, "semantic_error", "", "", "", f"3.{ver3[1:]}", "pylingual"])
    print(f"run CSV ({len(ok_ids)} compilable cases) -> {csv_out}", flush=True)


if __name__ == "__main__":
    main()
