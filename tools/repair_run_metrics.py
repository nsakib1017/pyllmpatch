"""Perfect-decompilation metric dashboard for a semantic-repair run.

Reports THREE honest metrics so progress toward the >60% goal is measured correctly and
the decompiler-failure-bound residual stays explicit:

  * OBJECT-level perfect rate  — fraction of initially-failing code objects driven to Equal
                                 (where repair shows real progress).
  * FILE-level perfect rate    — fraction of files that become fully equivalent (the goal;
                                 gated by the file's weakest object).
  * WINNABLE-subset rate       — file-perfect rate among files whose every failing object is
                                 in principle repairable from GT bytecode (control_flow /
                                 missing_object files are decompiler-bound and excluded).

Usage:
  python tools/repair_run_metrics.py <run_output_dir> [--dataset CSV]
Reads every <run_output_dir>/**/result.json (each carries gt_pyc / derived_pyc / final_pyc).
"""
import argparse, csv, glob, json, sys
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "pylingual"))
sys.path.insert(0, str(R))
csv.field_size_limit(sys.maxsize)
from pylingual.equivalence_check import compare_pyc  # noqa: E402
from utils.semantic_operators.structural import _real_try_blocks  # noqa: E402


def _winnable_object(t) -> bool:
    """Repairable in principle from GT bytecode by editing derived source."""
    if t.message == "Different bytecode" and t.failed_offset is not None:
        return True
    if t.message == "Extra bytecode":
        return True
    if t.message == "Different control flow":
        try:
            return len(_real_try_blocks(t.bc_a)) == len(_real_try_blocks(t.bc_b)) + 1
        except Exception:
            return False
    return False


def _failing(gt: str, der: str):
    try:
        return [t for t in compare_pyc(Path(gt), Path(der)) if not t.success]
    except BaseException:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dataset", default=None, help="optional CSV mapping file_hash->bytecode_version")
    args = ap.parse_args()

    ver_of = {}
    if args.dataset and Path(args.dataset).exists():
        for row in csv.DictReader(open(args.dataset)):
            ver_of[row["file_hash"]] = row.get("bytecode_version", "?")

    per = defaultdict(lambda: {"files": 0, "file_perfect": 0, "winnable_files": 0,
                               "winnable_perfect": 0, "obj_fail0": 0, "obj_fixed": 0,
                               "cd0": 0, "cd1": 0})
    rows = sorted(glob.glob(str(Path(args.run_dir) / "**" / "result.json"), recursive=True))
    seen = set()
    for p in rows:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        gt, der, fin = d.get("gt_pyc"), d.get("derived_pyc"), d.get("final_pyc")
        fh = (gt or "").split("/")[-2]
        if not gt or fh in seen:
            continue
        seen.add(fh)
        ver = next((v for h, v in ver_of.items() if fh.startswith(h[:16]) or h.startswith(fh[:16])), "?")
        init_fail = _failing(gt, der)
        final_fail = _failing(gt, fin) if fin else init_fail
        if init_fail is None or final_fail is None:
            continue
        b = per[ver]
        b["files"] += 1
        n0, n1 = len(init_fail), len(final_fail)
        b["obj_fail0"] += n0
        b["obj_fixed"] += max(0, n0 - n1)
        ib = (d.get("initial_summary") or {}).get("combined_distance")
        fb = (d.get("final_summary") or {}).get("combined_distance")
        if isinstance(ib, int) and isinstance(fb, int):
            b["cd0"] += ib
            b["cd1"] += fb
        perfect = (n1 == 0) or ((d.get("pylingual_verification") or {}).get("all_equal") is True)
        if perfect:
            b["file_perfect"] += 1
        winnable = all(_winnable_object(t) for t in init_fail) if init_fail else False
        if winnable:
            b["winnable_files"] += 1
            if perfect:
                b["winnable_perfect"] += 1

    tot = defaultdict(int)
    for b in per.values():
        for k, v in b.items():
            tot[k] += v

    def pct(a, b):
        return f"{100*a/b:.0f}%" if b else "—"

    print(f"run: {args.run_dir}   files={tot['files']}")
    print(f"  OBJECT-level perfect: {tot['obj_fixed']}/{tot['obj_fail0']} objects fixed "
          f"({pct(tot['obj_fixed'], tot['obj_fail0'])})")
    print(f"  FILE-level perfect  : {tot['file_perfect']}/{tot['files']} ({pct(tot['file_perfect'], tot['files'])})")
    print(f"  WINNABLE-subset     : {tot['winnable_perfect']}/{tot['winnable_files']} perfect "
          f"({pct(tot['winnable_perfect'], tot['winnable_files'])})  "
          f"[winnable = {tot['winnable_files']}/{tot['files']} files, {pct(tot['winnable_files'], tot['files'])}]")
    print(f"  combined_distance   : {tot['cd0']} -> {tot['cd1']} ({pct(tot['cd0']-tot['cd1'], tot['cd0'])} cut)")
    if len(per) > 1:
        print("  per version [files | file-perfect | obj-fixed | winnable-perfect]:")
        for v in sorted(per):
            b = per[v]
            print(f"    {v}: {b['files']} | {b['file_perfect']} | {b['obj_fixed']}/{b['obj_fail0']} | "
                  f"{b['winnable_perfect']}/{b['winnable_files']}")


if __name__ == "__main__":
    main()
