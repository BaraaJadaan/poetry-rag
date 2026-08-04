"""
Phase 5 — Hybrid Search Retriever.

Provides a HybridRetriever class that connects to LanceDB and the local
GGUF model to execute Semantic, Keyword, and Hybrid searches.

Run this script directly to launch an interactive search CLI.
"""

import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

try:
    import lancedb
except ImportError:
    print("ERROR: lancedb not installed. Run: uv add lancedb pyarrow pandas")
    sys.exit(1)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
LANCE_DIR = "./lancedb"
TABLE_NAME = "ashaar_baits"
GGUF_REPO = "jsonMartin/voyage-4-nano-gguf"


class HybridRetriever:
    def __init__(self):
        print("Initializing HybridRetriever...")
        self.db = lancedb.connect(LANCE_DIR)
        
        try:
            self.tbl = self.db.open_table(TABLE_NAME)
        except Exception:
            print(f"ERROR: Could not open LanceDB table '{TABLE_NAME}'. Did you run embed_corpus.py?")
            sys.exit(1)
            
        print(f"Connected to LanceDB. Table size: {len(self.tbl):,} verses.")
        
        # Ensure FTS index exists
        self._ensure_fts_index()
        
        # Load embedding model
        self._load_model()
        
    def _ensure_fts_index(self):
        """Creates the Tantivy full-text search index if it doesn't exist."""
        # LanceDB doesn't have an easy "has_index" check, so we just try a dummy query.
        try:
            self.tbl.search("test", query_type="fts").limit(1).to_pandas()
        except Exception:
            print("FTS index not found. Building it now (this takes ~1 minute for 3.4M rows)...")
            # Using create_index config for newer LanceDB versions
            try:
                self.tbl.create_fts_index("text_index", replace=True)
            except AttributeError:
                # LanceDB >= 0.25
                from lancedb.index import FTS
                self.tbl.create_index("text_index", config=FTS())
            print("FTS index built successfully.")

    def _load_model(self):
        """Loads voyage-4-nano GGUF and the linear projection layer."""
        print("Loading local embedding model...")
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename="voyage-4-nano-q8_0.gguf")
        linear_path = hf_hub_download(repo_id=GGUF_REPO, filename="voyage-4-nano-linear.pt")
        
        # Load linear projection without PyTorch
        with zipfile.ZipFile(linear_path, "r") as zf:
            data_file = next(n for n in zf.namelist() if n.endswith("data/0"))
            with zf.open(data_file) as f:
                raw = f.read()
        self.linear_weight = np.frombuffer(raw, dtype=np.float16).reshape(2048, 1024).astype(np.float32)
        
        # Add Torch DLL path for Llama-cpp if ComfyUI is installed locally
        try:
            os.add_dll_directory(r"D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\lib")
        except Exception:
            pass
            
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=gguf_path,
            embedding=True,
            pooling_type=1, # Mean Pooling
            n_ctx=512,
            verbose=False,
        )
        print("Model loaded and ready.")

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single search query."""
        raw_outputs = self.llm.embed([text])
        if raw_outputs and not isinstance(raw_outputs[0], (list, tuple)):
            raw_outputs = [raw_outputs]
            
        vec = np.array(raw_outputs[0], dtype=np.float32)
        proj = vec @ self.linear_weight.T
        norm = np.linalg.norm(proj)
        return proj / norm if norm > 0 else proj

    def search_semantic(self, query: str, limit: int = 10, filter_sql: str = None) -> pd.DataFrame:
        """Finds verses with similar semantic meaning using vector similarity."""
        emb = self.embed_query(query)
        builder = self.tbl.search(emb.tolist())
        if filter_sql:
            builder = builder.where(filter_sql)
        return builder.limit(limit).to_pandas()

    def search_keyword(self, query: str, limit: int = 10, filter_sql: str = None) -> pd.DataFrame:
        """Finds exact keywords using BM25 / Full Text Search."""
        builder = self.tbl.search(query, query_type="fts")
        if filter_sql:
            builder = builder.where(filter_sql)
        return builder.limit(limit).to_pandas()

    def search_hybrid(self, query: str, limit: int = 10, filter_sql: str = None) -> pd.DataFrame:
        """
        Combines Semantic and Keyword search using Reciprocal Rank Fusion (RRF).
        This guarantees that results matching BOTH exactly and conceptually rise to the top.
        """
        # 1. Get top 100 from both strategies to ensure good fusion overlap
        df_sem = self.search_semantic(query, limit=100, filter_sql=filter_sql)
        df_key = self.search_keyword(query, limit=100, filter_sql=filter_sql)
        
        # 2. Assign RRF scores (score = 1 / (60 + rank))
        rrf_scores = {}
        
        for rank, row in enumerate(df_sem.itertuples()):
            vid = row.id
            if vid not in rrf_scores:
                rrf_scores[vid] = {"score": 0.0, "row": row}
            rrf_scores[vid]["score"] += 1.0 / (60 + rank)
            
        for rank, row in enumerate(df_key.itertuples()):
            vid = row.id
            if vid not in rrf_scores:
                rrf_scores[vid] = {"score": 0.0, "row": row}
            rrf_scores[vid]["score"] += 1.0 / (60 + rank)
            
        # 3. Sort by combined score and return top K
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        top_k = sorted_results[:limit]
        
        # Convert back to DataFrame
        if not top_k:
            return pd.DataFrame()
            
        # Reconstruct DataFrame from the namedtuples
        rows = [x["row"]._asdict() for x in top_k]
        # Remove the Index field from itertuples
        for r in rows:
            r.pop("Index", None)
            
        df = pd.DataFrame(rows)
        # Add our custom RRF score column for visibility
        df["rrf_score"] = [x["score"] for x in top_k]
        return df


# ── Interactive CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  POETRY RAG - HYBRID SEARCH CLI")
    print("="*50)
    
    retriever = HybridRetriever()
    
    print("\nReady! Enter a search query. Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nQuery > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break
                
            print("\nSearching...")
            t0 = time.time()
            
            # Use hybrid search
            results = retriever.search_hybrid(query, limit=5)
            
            t1 = time.time()
            
            if results.empty:
                print("No results found.")
                continue
                
            print(f"\nTop 5 Results (found in {(t1-t0)*1000:.0f} ms):\n")
            
            for i, row in results.iterrows():
                # Some rows might not have text_display if they were cleaned aggressively
                display = row.get("text_display") or row.get("text_index")
                poet = row.get("poet_name", "Unknown")
                era = row.get("poet_era", "Unknown Era")
                score = row.get("rrf_score", 0)
                
                print(f"{i+1}. {display}")
                print(f"   — {poet} ({era}) | RRF Score: {score:.4f}\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during search: {e}")
            
    print("\nExiting search CLI. Goodbye!")
