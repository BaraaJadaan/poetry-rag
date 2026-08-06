#!/bin/bash
set -e

# Download the OmniCoder model if it's missing
if [ ! -f "models/OmniCoder-Claude-uncensored-V2-Q4_K_M.gguf" ]; then
    echo "Downloading OmniCoder GGUF model..."
    uv run python download_model.py
else
    echo "Model already exists in models/. Skipping download."
fi

# Build the LanceDB vector store if it's missing
if [ ! -d "lancedb" ]; then
    echo "LanceDB vector store not found. Embedding the corpus..."
    echo "This may take a long time (hours/days) on a free CPU. Make sure this runs in the background!"
    # Make sure we don't crash due to huggingface token errors or similar
    uv run python embed_corpus.py || echo "Warning: embed_corpus.py encountered an error, but continuing..."
else
    echo "LanceDB vector store already exists. Skipping embedding."
fi

echo "Starting FastAPI server..."
exec uv run uvicorn app:app --host 0.0.0.0 --port 8000
