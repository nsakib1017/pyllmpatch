import pandas as pd
from pathlib import Path
import re
import sys

from rapidfuzz.distance import Levenshtein

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


def save_csv_file(df: pd.DataFrame, file_name: Path):
    df.to_csv(file_name, index=False)
    print(f"\n💾 Results saved to: {file_name.resolve()}")


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":

    # Early cutoff: distances larger than this are "far enough"
    MAX_DIST = 5000
    LIST_OF_LOG_FILES = {
        "gemini": {
            "flash":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20251110T012915Z/daf95c71075048e1b3458c3c109344fd/run_log_daf95c71075048e1b3458c3c109344fd.jsonl",
            "flash-lite":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20251022T022513Z/e2648c12511c48558c29f4c5300aa6fe/run_log_e2648c12511c48558c29f4c5300aa6fe.jsonl",
            "pro":f"{os.getenv('PROJECT_ROOT_DIR')}/logs/run_log_9c729f3ab91c42f39b74e51fd102ebf2.jsonl", 
        },
        "qwen_7b": {
            "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T220209Z/d622ad5599af48dba5cf4d3435eb0545/run_log_d622ad5599af48dba5cf4d3435eb0545_with_config_0.jsonl",
            "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T220209Z/59ab85724c794122905a946c15baa405/run_log_59ab85724c794122905a946c15baa405_with_config_1.jsonl",
            "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T220209Z/b9c48dc9529b4cc4812337f5cd92f047/run_log_b9c48dc9529b4cc4812337f5cd92f047_with_config_2.jsonl",
        },
        "qwen_32b": {
            "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260129T025137Z/294cecf0e82049b58bc599cf48d0622b/run_log_294cecf0e82049b58bc599cf48d0622b_with_config_0.jsonl",
            "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260129T025137Z/3ef782858fa24527bb5a4755d31d169d/run_log_3ef782858fa24527bb5a4755d31d169d_with_config_1.jsonl",
            "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260129T025137Z/4ce329173d8e4307876d20bf4f7b5f33/run_log_4ce329173d8e4307876d20bf4f7b5f33_with_config_2.jsonl",
        },
        "deepseek-r1": {
            "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T212519Z/f52db77b7dc349a3bbc2615801c94643/run_log_f52db77b7dc349a3bbc2615801c94643_with_config_0.jsonl",
            "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T212519Z/f16e38fb574f41fb90efe88e5af39907/run_log_f16e38fb574f41fb90efe88e5af39907_with_config_1.jsonl",
            "config_2":None,
        },
        "granite": {
            "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T175015Z/b3f723500e5c47168ee6d851e8aee71f/run_log_b3f723500e5c47168ee6d851e8aee71f_with_config_0.jsonl",
            "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T175015Z/0a2b899e13b54024af5d59612a2e9802/run_log_0a2b899e13b54024af5d59612a2e9802_with_config_1.jsonl",
            "config_2":None,
        },
        "mistral": {
            "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T191624Z/3ea156e177624726b401d3c54a475539/run_log_3ea156e177624726b401d3c54a475539_with_config_0.jsonl",
            "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T191624Z/156b448efa6541378f9a1b5b46c4c6b3/run_log_156b448efa6541378f9a1b5b46c4c6b3_with_config_1.jsonl",
            "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260124T191624Z/b66bcdd98ecb466ab7734d9f65b1bd68/run_log_b66bcdd98ecb466ab7734d9f65b1bd68_with_config_2.jsonl",
        },
        "phi-4": {
            "config_0":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260125T155805Z/22aa93a4cc4f4d17ba63c2688b170f54/run_log_22aa93a4cc4f4d17ba63c2688b170f54_with_config_0.jsonl",
            "config_1":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260125T155805Z/67ac1a2443ab4ac8bfe7bba365801501/run_log_67ac1a2443ab4ac8bfe7bba365801501_with_config_1.jsonl",
            "config_2":f"{os.getenv('PROJECT_ROOT_DIR')}/results/experiment_outputs/20260125T155805Z/8cc795eb35404e8e8da0d6ec87eb7032/run_log_8cc795eb35404e8e8da0d6ec87eb7032_with_config_2.jsonl",
        },
    }

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
            f"{os.getenv('BASE_DIR_PYTHON_FILES')}/{file_hash}/{file_name}"
        )
        path3 = Path(
            f"{os.getenv('BASE_DIR_PYTHON_FILES')}/{file_hash}/{raw_name}.py"
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
