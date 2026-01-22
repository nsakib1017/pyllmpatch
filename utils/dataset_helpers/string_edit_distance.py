import pandas as pd
from pathlib import Path
import re
import sys

from rapidfuzz.distance import Levenshtein


# -------------------------
# File utilities
# -------------------------

def read_text(path: Path) -> str:
    # Read as UTF-8; replace undecodable bytes so the comparison can proceed
    return path.read_text(encoding="utf-8", errors="replace")


def strip_comments_and_whitespace(text: str) -> str:
    """
    Text-only minifier:
      1) For each line, drop everything from the first '#' to end-of-line.
      2) Remove ALL whitespace (spaces, tabs, newlines, etc.).
    NOTE: This treats '#' inside strings as comment starts too (by design).
    """
    lines = []
    for line in text.splitlines():
        hash_idx = line.find('#')
        if hash_idx != -1:
            line = line[:hash_idx]
        lines.append(line)

    no_comments = "".join(lines)
    return re.sub(r"\s+", "", no_comments)


# -------------------------
# CSV helpers
# -------------------------

def read_csv_file(file_name: str) -> pd.DataFrame:
    if "__file__" in globals():
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path.cwd()

    csv_path = base_dir / file_name

    print(f"📄 CSV path: {csv_path.resolve()}")

    if not csv_path.exists():
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded CSV with {len(df)} rows × {len(df.columns)} columns.")
    return df


def save_csv_file(df: pd.DataFrame, file_name: Path):
    df.to_csv(file_name, index=False)
    print(f"\n💾 Results saved to: {file_name.resolve()}")


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":

    # Early cutoff: distances larger than this are "far enough"
    MAX_DIST = 5000

    results = []

    print("=" * 80)
    print("🧩 Starting decompiled file comparison")
    print("=" * 80)

    df = read_csv_file("../../dataset/decompiled_syntax_errors.csv")
    total_rows = len(df)
    print(f"Total records to process: {total_rows}\n")

    for idx, row in df.iterrows():
        file_hash = row["file_hash"]
        file_name = row["file"]
        raw_name = file_name.split(".")[0].replace("decompiled_", "")
        path1 = Path(
            f"../../results/experiment_outputs/20251224T045257Z/"
            f"938dc4c31bbc44c5aed8ac2b69cf2185/"
            f"{file_hash}/syntax_repaired_{file_name}"
        )
        path2 = Path(
            f"/home/mxs220189/pylingual_collaboration/"
            f"pypi_downloaded/{file_hash}/{file_name}"
        )
        path3 = Path(
            f"/home/mxs220189/pylingual_collaboration/"
            f"pypi_downloaded/{file_hash}/{raw_name}.py"
        )

        print("-" * 80)
        print(f"[{idx+1}/{total_rows}] 🔍 Hash: {file_hash}")

        d32 = None
        d31 = None
        status = "OK"

        missing = [str(p) for p in (path1, path2, path3) if not p.is_file()]
        if missing:
            print("❌ Missing files:")
            for m in missing:
                print(f"   - {m}")
            status = "missing"
        else:
            try:
                s1 = strip_comments_and_whitespace(read_text(path1))
                s2 = strip_comments_and_whitespace(read_text(path2))
                s3 = strip_comments_and_whitespace(read_text(path3))

                # Fast C++ Levenshtein with early cutoff
                d32 = Levenshtein.distance(s3, s2, score_cutoff=MAX_DIST)
                d31 = Levenshtein.distance(s3, s1, score_cutoff=MAX_DIST)

                print(f"✅ Distance (lookup vs decompiled): {d32}")
                print(f"✅ Distance (lookup vs repaired)  : {d31}")

            except Exception as e:
                print(f"⚠️ Error during processing: {e}")
                status = "error"

        results.append({
            "file_hash": file_hash,
            "decompiled_file_name": file_name,
            "raw_file_name": path3.name,
            "d_lookup_vs_decompiled": d32,
            "d_lookup_vs_repaired": d31,
            "status": status,
        })

    print("\n" + "=" * 80)
    print("🏁 Finished processing all comparisons.")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    save_csv_file(
        results_df,
        Path("../../dataset/decompiled_comparison_results_qwen_7b_enhanced.csv")
    )

    print("\n📊 Results preview:")
    print(results_df.head())
