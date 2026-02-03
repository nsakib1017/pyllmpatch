import pandas as pd
from pathlib import Path
import re
import sys

from rapidfuzz.distance import Levenshtein
from file_helpers import LIST_OF_LOG_FILES

from dotenv import load_dotenv
import os
load_dotenv()
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



def read_jsonl_file(file_name: str) -> pd.DataFrame:
    if "__file__" in globals():
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path.cwd()

    jsonl_path = base_dir / file_name

    print(f"📄 JSONL path: {jsonl_path.resolve()}")

    if not jsonl_path.exists():
        raise FileNotFoundError(f"❌ JSONL file not found: {jsonl_path}")

    df = pd.read_json(jsonl_path, lines=True)

    print(f"✅ Loaded JSONL with {len(df)} rows × {len(df.columns)} columns.")

    return df

def save_csv_file(df: pd.DataFrame, file_name: Path):
    df.to_csv(file_name, index=False)
    print(f"\n💾 Results saved to: {file_name.resolve()}")


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    MAX_DIST = 5000

    models = LIST_OF_LOG_FILES.keys()

    for model in models:
        for config, path in LIST_OF_LOG_FILES[model].items():
            
            if not Path(path).is_file():
                continue

            print("=" * 80)
            print("🧩 Starting decompiled file comparison")
            print("=" * 80)

            df = read_csv_file(f"{os.getenv('PROJECT_ROOT_DIR')}/dataset/decompiled_syntax_errors.csv")
            df_log = read_jsonl_file(path)
            total_rows = len(df)

            results = []

            for idx, row in df.iterrows():
                df_log_success = df_log[df_log["compiled_success"] == True]
                file_hash = row["file_hash"]
                file_name = row["file"]
                if file_hash not in df_log_success["file_hash"].values:
                    continue

                raw_name = file_name.split(".")[0].replace("decompiled_", "")

                path1 = Path(f"{Path(path).parent}/{file_hash}/syntax_repaired_{file_name}")
                path2 = Path(f"{os.getenv('BASE_DIR_PYTHON_FILES')}/{file_hash}/{file_name}")
                path3 = Path(f"{os.getenv('BASE_DIR_PYTHON_FILES')}/{file_hash}/{raw_name}.py")

                d32 = None
                d31 = None
                status = "OK"

                missing = [str(p) for p in (path1, path2, path3) if not p.is_file()]
                if missing:
                    status = "missing"
                else:
                    try:
                        s1 = strip_comments_and_whitespace(read_text(path1))
                        s2 = strip_comments_and_whitespace(read_text(path2))
                        s3 = strip_comments_and_whitespace(read_text(path3))

                        d32 = Levenshtein.distance(s3, s2, score_cutoff=MAX_DIST)
                        d31 = Levenshtein.distance(s3, s1, score_cutoff=MAX_DIST)

                    except Exception:
                        status = "error"

                # ✅ append ONCE per row
                results.append({
                    "file_hash": file_hash,
                    "decompiled_file_name": file_name,
                    "raw_file_name": path3.name,
                    "d_lookup_vs_decompiled": d32,
                    "d_lookup_vs_repaired": d31,
                    "status": status,
                })

            results_df = pd.DataFrame(results)
            out_path = Path(
                f"{os.getenv('PROJECT_ROOT_DIR')}/dataset/"
                f"decompiled_comparison_results_{model}_{config}.csv"
            )
            save_csv_file(results_df, out_path)

            print("=" * 80)
            print("🧩 Ending decompiled file comparison")
            print("=" * 80)
