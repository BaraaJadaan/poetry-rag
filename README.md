# Arabic Poetry RAG (Retrieval-Augmented Generation)

An end-to-end RAG system for classical Arabic poetry using embeddings, a LanceDB vector store, hybrid retrieval, and metadata filtering.

---

## 🚀 Quick Start & Installation

Since local vector database storage (`lancedb/`) and temporary build artifacts are ignored by Git (`.gitignore`), follow these steps to set up and run the project on a new machine:

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

### 4. Track Retrieval Experiments
MLflow is kept in a separate dependency group because it is needed for evaluation
experiments, not for the FastAPI serving image:

```bash
uv sync --group mlops
uv run python evaluate.py --mlflow --output evaluation_results.json
uv run mlflow ui --backend-store-uri ./mlruns
```

The GitHub Actions workflow runs a small deterministic retrieval fixture and builds the
Docker image on every push. It does not call the embedding API or rebuild the production
corpus.

### 5. Build / Generate the Local Vector Database (`lancedb/`)
Run `embed_corpus.py` to preprocess the dataset, generate vector embeddings, and initialize the local LanceDB database:

```bash
# Test run with a limited batch (e.g. 1000 verses)
uv run python embed_corpus.py --limit 1000

# Full corpus embedding (resumable if interrupted)
uv run python embed_corpus.py
```

This will automatically create the `lancedb/` directory locally with all vector data and metadata intact.

---

## 📁 Repository Structure

- **`embed_corpus.py`**: Pipeline to preprocess poetry verses, compute embeddings, and save to LanceDB.
- **`preprocess.py`**: Data cleaning and tokenization routines.
- **`embed_sample.py`**: Small standalone script for testing vector embeddings.
- **`pyproject.toml`**: Dependency configuration.
- **`PROGRESS.md`**: Project implementation roadmap and phase tracker.
