import os
import shutil
from pathlib import Path

ROOT_DIR = Path("/home/diogenes/pylingual_colaboration/pypi_downloaded")
DRY_RUN = False 


def move_decompiled_files(root: Path, dry_run: bool = True):
    decompiled_files = [f for f in root.glob("decompiled_*.cpython-312.py") if f.is_file()]
    print(f"Found {len(decompiled_files)} decompiled file(s) in {root}")

    subdir_map = {}
    for py_file in root.rglob("*.py"):
        if py_file.parent == root:
            continue
        subdir_map[py_file.stem] = py_file.parent

    moved = 0
    not_found = []

    for decompiled in decompiled_files:
        name = decompiled.name
        if not name.startswith("decompiled_"):
            continue

        target_stem = name[len("decompiled_"):].split(".cpython-312.py")[0]

        if target_stem in subdir_map:
            target_dir = subdir_map[target_stem]
            dest = target_dir / decompiled.name
            action = "Would move" if dry_run else "→ Moving"
            print(f"{action} {decompiled.name} → {target_dir}")

            if not dry_run:
                shutil.move(str(decompiled), str(dest))
            moved += 1
        else:
            not_found.append(name)

    print(f"\n✅ {'(Dry run)' if dry_run else ''} Done. {'Would move' if dry_run else 'Moved'} {moved} file(s).")
    if not_found:
        print("⚠️ Could not find matching subdirectories for:")
        for nf in not_found:
            print("  -", nf)


if __name__ == "__main__":
    move_decompiled_files(ROOT_DIR, dry_run=DRY_RUN)
