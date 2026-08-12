"""
Phase 4 — Full corpus embedding.

Embeds all classical Arabic verses from the Ashaar dataset using
voyage-4-nano (Q8_0 GGUF) and stores them in a LanceDB table.

Why LanceDB over ChromaDB/FAISS:
  - We initially tried ChromaDB, but at 2.5 million × 2048-dim vectors,
    Chroma's Rust background compaction thread kept crashing under the GPU's
    fast write speed ("Failed to apply logs to the hnsw segment writer").
  - FAISS requires managing metadata manually (SQLite).
  - LanceDB handles millions of vectors effortlessly because it writes
    directly to a columnar format (Lance) on disk without a fragile 
    WAL-compaction loop, and has built-in metadata filtering.

This script is RESUMABLE: each verse gets a deterministic ID (SHA-256 of
text_index). If the DB already contains that ID, the verse is skipped.
Interrupt and restart safely at any time.

Usage:
    $env:HF_HOME = "D:\\hf-cache"
    uv run python embed_corpus.py              # full corpus
    uv run python embed_corpus.py --limit 1000 # timing test
"""

import sys
import os
import hashlib
import time
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow as pa

# Import the preprocessing pipeline we built in Phase 2
from preprocess import run_pipeline

# ── Config ────────────────────────────────────────────────────────────────────
LANCE_DIR     = "./lancedb"
CLASSICAL_ONLY = True    # drop عامي/شعبي/'-' rows (see preprocess.py for rationale)
N_CTX         = 512      # model context window — verses are 10-30 tokens; 512 is ample
import os
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

USE_OPENROUTER_EMBED = os.getenv("CLOUD_DEPLOYMENT", "false").lower() == "true"
TABLE_NAME    = "ashaar_baits_qwen3" if USE_OPENROUTER_EMBED else "ashaar_baits"
GGUF_REPO     = "jsonMartin/voyage-4-nano-gguf"


# ── CLI args (for timing test) ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Process only the first N chunks (for timing estimates)")
args = parser.parse_args()


# ── 1. Connect to LanceDB ─────────────────────────────────────────────────────
try:
    import lancedb
except ImportError:
    print("ERROR: lancedb not installed. Run: uv add lancedb pyarrow pandas")
    sys.exit(1)

schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 4096 if USE_OPENROUTER_EMBED else 2048)),
    pa.field("text_index", pa.string()),
    pa.field("text_display", pa.string()),
    pa.field("poem_title", pa.string()),
    pa.field("poet_name", pa.string()),
    pa.field("poet_era", pa.string()),
    pa.field("poem_meter", pa.string()),
    pa.field("poem_theme", pa.string()),
    pa.field("poem_language_type", pa.string()),
    pa.field("bait_index", pa.int32()),
    pa.field("is_orphan", pa.int32()),
    pa.field("poem_url", pa.string()),
    pa.field("poet_url", pa.string()),
])

db = lancedb.connect(LANCE_DIR)
try:
    tbl = db.open_table(TABLE_NAME)
    already_indexed = len(tbl)
except Exception:
    tbl = db.create_table(TABLE_NAME, schema=schema)
    already_indexed = 0

print(f"LanceDB '{TABLE_NAME}': {already_indexed:,} docs already indexed")


if USE_OPENROUTER_EMBED:
    # ── 2. Initialize OpenRouter API ───────────────────────────────────────────────
    api_key = os.getenv("opentouter_api", "").strip().strip("'\"")
    if not api_key:
        print("ERROR: opentouter_api environment variable not found. Cannot embed without API key.")
        sys.exit(1)

    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=api_key,
    )

    def embed_batch(texts: list[str]) -> list[np.ndarray]:
        """
        Embed a batch of strings using OpenRouter API and return 4096-dim L2-normalised vectors.
        Includes an exponential backoff retry loop. If the whole batch fails (e.g. 403 Security Policy),
        falls back to embedding verse-by-verse, skipping problematic verses with a zero vector.
        """
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=texts,
                    model="qwen/qwen3-embedding-8b"
                )
                
                results = []
                for item in response.data:
                    vec = np.array(item.embedding, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    results.append(vec / norm if norm > 0 else vec)
                return results
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"\n[Warning] API error: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"\n[Error] Batch failed after {max_retries} attempts: {e}. Falling back to 1-by-1...")
                    
        # Fallback: embed one by one
        results = []
        for text in texts:
            try:
                # Add a tiny sleep to avoid spamming the API on 1-by-1 fallback
                time.sleep(0.1)
                resp = client.embeddings.create(input=[text], model="qwen/qwen3-embedding-8b")
                vec = np.array(resp.data[0].embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                results.append(vec / norm if norm > 0 else vec)
            except Exception as inner_e:
                print(f"\n[Warning] Skipping verse due to API error: {inner_e}")
                # Return a zero vector for the problematic verse so the pipeline survives
                results.append(np.zeros(4096, dtype=np.float32))
                
        return results
else:
    # ── 2. Download and load model files ──────────────────────────────────────────
    print("\nChecking model files (cached after first download)...")
    gguf_path   = hf_hub_download(repo_id=GGUF_REPO, filename="voyage-4-nano-q8_0.gguf")
    linear_path = hf_hub_download(repo_id=GGUF_REPO, filename="voyage-4-nano-linear.pt")
    print(f"  GGUF  : {gguf_path}")
    print(f"  linear: {linear_path}")

    # ── 3. Load linear projection (no torch needed) ───────────────────────────────
    import zipfile

    def load_linear_pt(path: str) -> np.ndarray:
        with zipfile.ZipFile(path, "r") as zf:
            data_file = next(n for n in zf.namelist() if n.endswith("data/0"))
            with zf.open(data_file) as f:
                raw = f.read()
        return np.frombuffer(raw, dtype=np.float16).reshape(2048, 1024).astype(np.float32)

    print("\nLoading linear projection layer...")
    linear_weight = load_linear_pt(linear_path)
    print(f"  Shape: {linear_weight.shape}  (maps 1024-dim GGUF output → 2048-dim original)")

    # ── 4. Load GGUF model ────────────────────────────────────────────────────────
    import os
    try:
        os.add_dll_directory(r"D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\lib")
    except Exception:
        pass

    from llama_cpp import Llama

    print("\nLoading voyage-4-nano GGUF (GPU)...")
    llm = Llama(
        model_path=gguf_path,
        embedding=True,
        pooling_type=1,   # mean pooling — required for this model architecture
        n_ctx=N_CTX,
        n_batch=512,      # Increased from 1 to allow GPU to process tokens in parallel
        n_gpu_layers=-1,  # Offload all layers to GPU
        verbose=True,
    )
    print("  Model ready.")

    # ── 5. Embed batch ────────────────────────────────────────────────────────────
    def embed_batch(texts: list[str]) -> list[np.ndarray]:
        """
        Embed a batch of strings and return 2048-dim L2-normalised vectors.
        Batching + n_batch=512 is critical to actually saturate the GPU.
        """
        try:
            llm._ctx.kv_cache_clear()
        except AttributeError:
            llm.reset()

        raw_outputs = llm.embed(texts)
        
        # llama_cpp.embed might return a flat list if given 1 text, or list of lists
        if raw_outputs and not isinstance(raw_outputs[0], (list, tuple)):
            raw_outputs = [raw_outputs]

        results = []
        for raw in raw_outputs:
            vec = np.array(raw, dtype=np.float32)           # shape: (1024,)
            proj = vec @ linear_weight.T                    # shape: (2048,)
            norm = np.linalg.norm(proj)
            results.append(proj / norm if norm > 0 else proj) # L2-normalised
        return results
# ── 6. Deterministic chunk ID ─────────────────────────────────────────────────
def chunk_id(text_index: str) -> str:
    """First 32 hex chars of SHA-256(text_index)."""
    return hashlib.sha256(text_index.encode("utf-8")).hexdigest()[:32]


# ── 7. Preprocess corpus ──────────────────────────────────────────────────────
print(f"\nRunning preprocessing pipeline (classical_only={CLASSICAL_ONLY})...")
chunks = run_pipeline(classical_only=CLASSICAL_ONLY, limit=args.limit)
total = len(chunks)

# Deduplicate by ID in case the pipeline produces any duplicate texts
seen = {}
for c in chunks:
    cid = chunk_id(c["text_index"])
    seen[cid] = c

# If we are doing a limited test, make sure the eval_set verses are prioritized!
import json
try:
    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    golden_ids = set(item["expected_id"] for item in eval_set)
    # Reorder chunks: golden IDs first, then the rest
    golden_chunks = [c for cid, c in seen.items() if cid in golden_ids]
    other_chunks = [c for cid, c in seen.items() if cid not in golden_ids]
    chunks = golden_chunks + other_chunks
    print(f"Prioritized {len(golden_chunks)} golden verses for evaluation.")
except Exception as e:
    print("Could not prioritize golden set:", e)
    chunks = list(seen.values())

print(f"After dedup: {len(chunks):,} unique chunks (removed {total - len(chunks):,} duplicates)")

# Skip already-indexed IDs (resumability)
print("Loading existing IDs to skip previously indexed verses...")
if already_indexed > 0:
    import pandas as pd
    # LanceDB returns Arrow tables. Convert to Pandas to get list of strings quickly.
    existing = set(tbl.search().limit(10_000_000).select(["id"]).to_pandas()["id"].tolist())
else:
    existing = set()

pending  = [
    c for c in chunks 
    if chunk_id(c["text_index"]) not in existing 
    and len(c["text_index"].strip()) > 1
]
print(f"Already indexed: {len(existing):,} | To embed: {len(pending):,}")

if not pending:
    print("\nNothing to do — corpus is fully indexed.")
    print(f"Total in table: {len(tbl):,}")
    sys.exit(0)


# ── 8. Embed and store ────────────────────────────────────────────────────────
print(f"\nEmbedding {len(pending):,} verses...")
if args.limit:
    print(f"  [timing-test mode: --limit {args.limit}]")

try:
    from tqdm import tqdm
    bar = tqdm(total=len(pending), unit="verse", dynamic_ncols=True)
except ImportError:
    bar = None

t_start = time.time()
BATCH_SIZE = 100  # Safe batch size for OpenRouter API payload limits

for i in range(0, len(pending), BATCH_SIZE):
    batch_chunks = pending[i : i + BATCH_SIZE]
    batch_texts = [c["text_index"] for c in batch_chunks]
    
    # 1. Embed the whole batch on GPU
    batch_embs = embed_batch(batch_texts)
    
    # 2. Prepare LanceDB payload
    buf = []
    for chunk, emb in zip(batch_chunks, batch_embs):
        cid = chunk_id(chunk["text_index"])
        buf.append({
            "id": cid,
            "vector": emb.tolist(),
            "text_index": chunk["text_index"],
            "text_display": chunk["text_display"] or "",
            "poem_title": chunk["poem_title"] or "",
            "poet_name": chunk["poet_name"] or "",
            "poet_era": chunk["poet_era"] or "",
            "poem_meter": chunk["poem_meter"] or "",
            "poem_theme": chunk["poem_theme"] or "",
            "poem_language_type": chunk["poem_language_type"] or "",
            "bait_index": int(chunk["bait_index"]),
            "is_orphan": int(chunk["is_orphan"]),
            "poem_url": chunk["poem_url"] or "",
            "poet_url": chunk["poet_url"] or "",
        })

    # 3. Append to LanceDB
    tbl.add(buf)
    
    if bar:
        bar.update(len(batch_chunks))
    elif i > 0 and i % 1000 < BATCH_SIZE:
        elapsed = time.time() - t_start
        rate    = i / elapsed
        eta_s   = (len(pending) - i) / rate
        print(f"  [{i:>7,}/{len(pending):,}]  "
              f"{rate:.1f} v/s  |  ETA {int(eta_s//3600)}h {int((eta_s%3600)//60)}m")

if bar:
    bar.close()

elapsed = time.time() - t_start
rate    = len(pending) / elapsed

print(f"\n{'='*55}")
print(f"  Embedded : {len(pending):,} verses")
print(f"  Time     : {elapsed/60:.1f} min  ({rate:.2f} verses/sec)")
print(f"  Total in collection: {len(tbl):,}")

if args.limit:
    remaining = max(0, total - args.limit)
    eta_h = remaining / rate / 3600
    print(f"\n  [Timing estimate]")
    print(f"  Speed     : {rate:.2f} v/s")
    print(f"  Full corpus (~{total:,} verses) ETA: {eta_h:.1f} hours")
    print(f"  → {'Proceed overnight' if eta_h < 14 else 'Consider GPU acceleration (see PROGRESS.md)'}")
print(f"{'='*55}")

# Free Llama explicitly before Python interpreter tears down modules to prevent
# 'NoneType object is not callable' in _exit_wrappers.
del llm
