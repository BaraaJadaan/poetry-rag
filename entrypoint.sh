#!/bin/bash
set -e

# Download the Qwen 3.5 2B model if it's missing (only locally)
if [ "$CLOUD_DEPLOYMENT" = "true" ]; then
    echo "Cloud deployment detected. Skipping local model download."
elif [ ! -f "models/Qwen3.5-2B-MTP-Q4_K_M.gguf" ]; then
    echo "Downloading Qwen 3.5 2B MTP GGUF model..."
    uv run python download_model.py
else
    echo "Model already exists in models/. Skipping download."
fi

# Build the expected LanceDB table if it's missing.
# The mount point itself always exists when Docker attaches the named volume,
# so checking only for the directory can incorrectly skip a required build.
if [ "$CLOUD_DEPLOYMENT" = "true" ]; then
    LANCE_TABLE="ashaar_baits_qwen3"
else
    LANCE_TABLE="ashaar_baits"
fi

if ! uv run python - "$LANCE_TABLE" <<'PY'
import sys
import lancedb

table_name = sys.argv[1]
try:
    db = lancedb.connect("./lancedb")
    db.open_table(table_name)
except Exception:
    raise SystemExit(1)
PY
then
    echo "LanceDB table '$LANCE_TABLE' not found. Embedding the corpus..."
    echo "This may take a long time (hours/days) on a free CPU. Make sure this runs in the background!"
    # In cloud mode embed_corpus.py uses the OpenRouter embedding API.
    uv run python embed_corpus.py
else
    echo "LanceDB table '$LANCE_TABLE' already exists. Skipping embedding."
fi

echo "Starting FastAPI server..."
exec uv run uvicorn app:app --host 0.0.0.0 --port 8000
