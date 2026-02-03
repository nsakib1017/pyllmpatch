from typing import List
import pandas as pd
from pathlib import Path
import re
import sys

from rapidfuzz.distance import Levenshtein
from file_helpers import LIST_OF_LOG_FILES

from dotenv import load_dotenv
import os
import torch
from sentence_transformers import SentenceTransformer, util


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



# ------------------ Embedding long texts with chunking ------------------ #
def chunk_by_tokens(text: str, tokenizer, max_tokens: int, stride: int) -> List[str]:
    """
    Split `text` into overlapping chunks by tokenizer tokens, then decode each
    chunk back to text for SentenceTransformer.encode.
    """
    # Tokenize once (no specials) then slice token ids.
    toks = tokenizer.encode(text, add_special_tokens=False)
    n = len(toks)
    if n == 0:
        return [""]  # keep shape consistent

    chunks = []
    i = 0
    while i < n:
        window = toks[i : i + max_tokens]
        chunk_text = tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        if i + max_tokens >= n:
            break
        i += max(max_tokens - stride, 1)
    return chunks



@torch.no_grad()
def embed_long_text(model: SentenceTransformer, text: str, max_tokens: int = 384, stride: int = 64) -> torch.Tensor:
    """
    Embed long text by averaging L2-normalized embeddings over token-based chunks.
    """
    tokenizer = model.tokenizer
    chunks = chunk_by_tokens(text, tokenizer, max_tokens=max_tokens, stride=stride)
    # Encode chunks -> normalized embeddings (SentenceTransformer returns normalized if normalize_embeddings=True is set)
    # We’ll explicitly normalize after.
    embs = model.encode(
        chunks,
        batch_size=8,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    # L2-normalize each chunk vector
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    # Average then normalize again
    mean_emb = embs.mean(dim=0, keepdim=True)
    mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
    return mean_emb.squeeze(0)  # shape: (d,)



def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

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
    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_model = sentence_model.to(device)

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

                        # Compute cosine similarities
                        e1 = embed_long_text(sentence_model, s1, max_tokens=512, stride=128)
                        e2 = embed_long_text(sentence_model, s2, max_tokens=512, stride=128)
                        e3 = embed_long_text(sentence_model, s3, max_tokens=512, stride=128)
                        
                        sim_31 = cosine_similarity(e3, e1)
                        sim_32 = cosine_similarity(e3, e2)

                        dist_32 = 1.0 - sim_32
                        dist_31 = 1.0 - sim_31
                    except Exception:
                        status = "error"

                results.append({
                    "file_hash": file_hash,
                    "decompiled_file_name": file_name,
                    "raw_file_name": path3.name,
                    "d_lookup_vs_decompiled": d32,
                    "d_lookup_vs_repaired": d31,
                    "d_lookup_vs_decompiled_cosine_similarity": sim_32,
                    "d_lookup_vs_repaired_cosine_similarity": sim_31,
                    "d_lookup_vs_decompiled_cosine_distance": dist_32,
                    "d_lookup_vs_repaired_cosine_distance": dist_31,
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
