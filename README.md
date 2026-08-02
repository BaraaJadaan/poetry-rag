# Arabic Poetry RAG (Retrieval-Augmented Generation)

An end-to-end RAG system for classical Arabic poetry using local embeddings (`voyage-4-nano`), ChromaDB vector store, and Metadata filtering.

---

## 🚀 Quick Start & Installation

Since local vector database storage (`chromadb/`) and temporary build artifacts are ignored by Git (`.gitignore`), follow these steps to set up and run the project on a new machine:

### 1. Prerequisites
- Python >= 3.13
- [`uv`](https://github.com/astral-sh/uv) package manager (recommended) or standard `pip`

### 2. Clone the Repository
```bash
git clone https://github.com/BaraaJadaan/poetry-rag.git
cd poetry-rag
```

### 3. Install Dependencies
Using `uv`:
```bash
uv sync
```
*(Or with standard pip: `pip install -e .`)*

### 4. Build / Generate the Local Vector Database (`chromadb/`)
Run `embed_corpus.py` to preprocess the dataset, generate vector embeddings, and initialize the local ChromaDB database:

```bash
# Test run with a limited batch (e.g. 1000 verses)
uv run python embed_corpus.py --limit 1000

# Full corpus embedding (resumable if interrupted)
uv run python embed_corpus.py
```

This will automatically create the `chromadb/` directory locally with all index data and metadata intact.

---

## 📁 Repository Structure

- **`embed_corpus.py`**: Pipeline to preprocess poetry verses, compute embeddings via `voyage-4-nano`, and save to ChromaDB.
- **`preprocess.py`**: Data cleaning and tokenization routines.
- **`embed_sample.py`**: Small standalone script for testing vector embeddings.
- **`pyproject.toml`**: Dependency configuration.
- **`PROGRESS.md`**: Project implementation roadmap and phase tracker.
