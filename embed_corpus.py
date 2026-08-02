"""
Phase 4 — Full corpus embedding.

Embeds all classical Arabic verses from the Ashaar dataset using
voyage-4-nano (Q8_0 GGUF) and stores them in a ChromaDB collection.

Why ChromaDB over FAISS or LanceDB:
  - Built-in WHERE-clause metadata filtering (poet, era, meter, theme)
    without needing a separate SQLite join layer.
  - Single pip install, no C++ compilation needed.
  - HNSW index handles 250k × 2048-dim vectors well inside its designed range.
  - FAISS is faster at search but you have to manage metadata storage yourself.
  - LanceDB would be the right call at 10x the scale (millions of docs).

This script is RESUMABLE: each verse gets a deterministic ID (SHA-256 of
text_index). If ChromaDB already contains that ID, the verse is skipped.
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

# Import the preprocessing pipeline we built in Phase 2
from preprocess import run_pipeline

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR    = "./chromadb"
COLLECTION    = "ashaar_baits"
CHROMA_BATCH  = 512      # how many docs to upsert to ChromaDB at once
CLASSICAL_ONLY = True    # drop عامي/شعبي/'-' rows (see preprocess.py for rationale)
N_CTX         = 512      # model context window — verses are 10-30 tokens; 512 is ample
GGUF_REPO     = "jsonMartin/voyage-4-nano-gguf"


# ── CLI args (for timing test) ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Process only the first N chunks (for timing estimates)")
args = parser.parse_args()


# ── 1. Connect to ChromaDB ────────────────────────────────────────────────────
# PersistentClient writes the HNSW index + SQLite metadata to CHROMA_DIR on disk.
# get_or_create_collection is idempotent — safe to call on every run.
try:
    import chromadb
except ImportError:
    print("ERROR: chromadb not installed. Run: uv add chromadb")
    sys.exit(1)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},  # cosine distance for embedding similarity
)
already_indexed = collection.count()
print(f"ChromaDB '{COLLECTION}': {already_indexed:,} docs already indexed")


# ── 2. Download and load model files ──────────────────────────────────────────
print("\nChecking model files (cached after first download)...")
gguf_path   = hf_hub_download(repo_id=GGUF_REPO, filename="voyage-4-nano-q8_0.gguf")
linear_path = hf_hub_download(repo_id=GGUF_REPO, filename="voyage-4-nano-linear.pt")
print(f"  GGUF  : {gguf_path}")
print(f"  linear: {linear_path}")


# ── 3. Load linear projection (no torch needed) ───────────────────────────────
# PyTorch .pt files since v1.6 are ZIP archives.
# The raw tensor data lives in archive/data/0 as little-endian binary.
# Shape is [2048, 1024], dtype is float16 (file size ≈ 4.2 MB = 2048×1024×2 bytes).
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
    pooling_type=2,   # mean pooling — required for this model architecture
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
    """
    First 32 hex chars of SHA-256(text_index).

    Why content-based rather than positional (e.g. row_i_bait_j)?
    If the dataset is re-downloaded or re-ordered, a positional ID
    would map to a different verse. A content hash is stable as long
    as the normalised text doesn't change — exactly the right invariant
    for a resumable indexing job.

    32 chars (128 bits) gives collision probability of 1 in 2^128 for
    a 250k-item collection — effectively zero.
    """
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
chunks = list(seen.values())
print(f"After dedup: {len(chunks):,} unique chunks (removed {total - len(chunks):,} duplicates)")

# Skip already-indexed IDs (resumability)
# We fetch just IDs from ChromaDB — no embeddings pulled, so it's fast.
existing = set(collection.get(include=[])["ids"])
pending  = [c for c in chunks if chunk_id(c["text_index"]) not in existing]
print(f"Already indexed: {len(existing):,} | To embed: {len(pending):,}")

if not pending:
    print("\nNothing to do — corpus is fully indexed.")
    print(f"Total in collection: {collection.count():,}")
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
BATCH_SIZE = 256  # process texts in chunks of 256 to saturate the GPU

for i in range(0, len(pending), BATCH_SIZE):
    batch_chunks = pending[i : i + BATCH_SIZE]
    batch_texts = [c["text_index"] for c in batch_chunks]
    
    # 1. Embed the whole batch on GPU
    batch_embs = embed_batch(batch_texts)
    
    # 2. Prepare ChromaDB payload
    buf_ids, buf_docs, buf_metas = [], [], []
    for chunk in batch_chunks:
        cid = chunk_id(chunk["text_index"])
        
        meta = {
            "text_display"       : chunk["text_display"]        or "",
            "poem_title"         : chunk["poem_title"]           or "",
            "poet_name"          : chunk["poet_name"]            or "",
            "poet_era"           : chunk["poet_era"]             or "",
            "poem_meter"         : chunk["poem_meter"]           or "",
            "poem_theme"         : chunk["poem_theme"]           or "",
            "poem_language_type" : chunk["poem_language_type"]   or "",
            "bait_index"         : int(chunk["bait_index"]),
            "is_orphan"          : int(chunk["is_orphan"]),
            "poem_url"           : chunk["poem_url"]             or "",
            "poet_url"           : chunk["poet_url"]             or "",
        }
        buf_ids.append(cid)
        buf_docs.append(chunk["text_index"])
        buf_metas.append(meta)

    # 3. Upsert to ChromaDB
    collection.add(
        ids=buf_ids,
        embeddings=[e.tolist() for e in batch_embs],
        documents=buf_docs,
        metadatas=buf_metas,
    )
    
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
print(f"  Total in collection: {collection.count():,}")

if args.limit:
    chunks_per_poem  = len(chunks) / args.limit       # avg baits per poem from test
    # The full Ashaar dataset has ~254k poems. After classical_only filter
    # (~3.4% dropped), estimated ~245k poems remain.
    full_poem_est    = 245_000
    full_verse_est   = int(full_poem_est * chunks_per_poem)
    eta_h            = full_verse_est / rate / 3600
    print(f"\n  [Timing estimate — full corpus]")
    print(f"  Speed              : {rate:.2f} v/s")
    print(f"  Baits/poem (sample): {chunks_per_poem:.1f}")
    print(f"  Est. full corpus   : ~{full_verse_est:,} verses")
    print(f"  ETA on CPU         : ~{eta_h:.1f} hours")
    if eta_h < 14:
        print(f"  → Feasible overnight on CPU. Run without --limit.")
    else:
        print(f"  → Consider GPU acceleration (run: nvidia-smi to check CUDA version).")
print(f"{'='*55}")
