"""
Phase 3 — Embedding sanity check using voyage-4-nano GGUF locally.

Downloads two files on first run (cached to D:/hf-cache afterward):
  voyage-4-nano-q8_0.gguf   ~372 MB
  voyage-4-nano-linear.pt     ~4.2 MB

The GGUF model outputs 1024-dim embeddings. The linear projection
maps them to the correct 2048-dim output space, matching the original
VoyageAI model exactly (0.9999 cosine similarity per the repo README).

Run:
  $env:HF_HOME = "D:\\hf-cache"
  uv run python embed_sample.py
"""

import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 0. Check HF_HOME is set so downloads go to D:\hf-cache ──────────────────
hf_home = os.environ.get("HF_HOME", "")
if not hf_home:
    print("TIP: Set $env:HF_HOME = 'D:\\hf-cache' to cache models to D: drive.")

# ── 1. Download model files ───────────────────────────────────────────────────
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: Run: uv add huggingface-hub")
    sys.exit(1)


def load_linear_pt_as_numpy(path: str) -> np.ndarray:
    """
    Load a simple PyTorch .pt weight tensor without requiring torch.

    PyTorch >= 1.6 serialises tensors as ZIP archives. The raw tensor
    data lives at archive/data/0 as little-endian binary. We know the
    shape is [2048, 1024] and the dtype is float16 (file ~4.2 MB =
    2048 * 1024 * 2 bytes). We load it directly and upcast to float32
    for the numpy matrix multiply.
    """
    import zipfile
    SHAPE = (2048, 1024)
    DTYPE = np.float16  # confirmed by file size

    with zipfile.ZipFile(path, "r") as zf:
        # The raw data file inside the zip
        names = zf.namelist()
        data_file = next(n for n in names if n.endswith("data/0"))
        with zf.open(data_file) as f:
            raw = f.read()

    arr = np.frombuffer(raw, dtype=DTYPE).reshape(SHAPE).astype(np.float32)
    return arr

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: Run: uv add llama-cpp-python")
    sys.exit(1)

REPO = "jsonMartin/voyage-4-nano-gguf"

print("Downloading voyage-4-nano-q8_0.gguf (~372 MB, one-time)...")
gguf_path = hf_hub_download(repo_id=REPO, filename="voyage-4-nano-q8_0.gguf")
print(f"  -> {gguf_path}")

print("Downloading voyage-4-nano-linear.pt (~4.2 MB)...")
linear_path = hf_hub_download(repo_id=REPO, filename="voyage-4-nano-linear.pt")
print(f"  -> {linear_path}")

# ── 2. Load linear projection matrix ─────────────────────────────────────────
# The GGUF model has hidden_dim=1024. The original model outputs 2048-dim.
# The linear layer maps (batch, 1024) -> (batch, 2048).
print("\nLoading linear projection layer (no torch required)...")
linear_weight = load_linear_pt_as_numpy(linear_path)
print(f"  Projection shape: {linear_weight.shape}  (expected (2048, 1024))")

# ── 3. Load GGUF model ────────────────────────────────────────────────────────
# Critical flags for voyage-4-nano (bidirectional encoder):
#   pooling_type=2  -> mean pooling
#   rope_scaling_type not needed — but attention must be non-causal
# llama-cpp-python exposes these via Llama() kwargs.
print("\nLoading GGUF model into llama-cpp-python...")
llm = Llama(
    model_path=gguf_path,
    embedding=True,
    pooling_type=2,       # 2 = mean pooling (matches --pooling mean)
    n_ctx=512,            # sufficient for ~15-word Arabic verses
    n_batch=8,
    verbose=False,
)
print("  Model loaded.")

# ── 4. Embed function with projection + re-normalisation ──────────────────────
def apply_projection(vec_1024: np.ndarray, linear_weight: np.ndarray) -> np.ndarray:
    """
    Apply the linear projection and L2 re-normalise.
      vec_1024:     shape (1024,) — raw GGUF output
      linear_weight: shape (2048, 1024)
      returns:       shape (2048,) — L2-normalised, matches original model
    """
    vec_2048 = vec_1024 @ linear_weight.T          # (1024,) @ (1024, 2048) -> (2048,)
    norm = np.linalg.norm(vec_2048)
    return vec_2048 / norm if norm > 0 else vec_2048

# ── 5. Test verses (text_index form — normalised, no diacritics) ──────────────
test_verses = [
    # [0] Patience / grief  ──────────────────────────────────────────
    "اصبر لكل مصيبة وتجلد واعلم بان المرء غير مخلد",
    # [1] Patience / endurance ───────────────────────────────────────
    "انما الصبر عند الصدمة الاولى",
    # [2] Courage / sword ────────────────────────────────────────────
    "السيف اصدق انباء من الكتب في حده الحد بين الجد واللعب",
    # [3] War / battle — عنترة بن شداد (pre-Islamic, different era & style) ──
    # "ولقد شفى نفسي وأبرأ سقمها قيل الفوارس ويك عنتر أقدم"
    # Normalized (no diacritics):
    "ولقد شفى نفسي وابرا سقمها قيل الفوارس ويك عنتر اقدم",
    # [4] Love ───────────────────────────────────────────────────────
    "وما كنت ممن يدخل العشق قلبه ولكن من يبصر جفونك يعشق",
]

labels = ["[0] Patience/grief (المتنبي)", "[1] Patience/endurance (الرسول)",
          "[2] Courage/sword (المتنبي)", "[3] War/battle (عنترة)", "[4] Love"]

# ── 6. Embed each verse with explicit KV cache clear between calls ────────────
# The batch embed (llm.embed(list)) still processes items sequentially through
# the same internal context — the KV cache leaks across inputs just as in
# a manual loop. The fix: explicitly clear the cache before each embed.
#
# Diagnostic: print the first 5 raw values of each 1024-dim vector.
# If verses [2] and [3] show identical raw[:5], the model is genuinely
# collapsing them (model failure). If they differ, cache clear fixed it.
print(f"\nEmbedding {len(test_verses)} verses individually with KV cache clear...")
embeddings = []

for i, v in enumerate(test_verses):
    # Clear the KV cache before each embed call
    try:
        llm._ctx.kv_cache_clear()        # preferred: direct cache clear
        cache_clear_method = "kv_cache_clear"
    except AttributeError:
        llm.reset()                       # fallback: context reset
        cache_clear_method = "reset"

    raw = llm.embed(v)

    # Handle both list-of-lists and flat list returns
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]

    vec_1024 = np.array(raw, dtype=np.float32)
    vec_final = apply_projection(vec_1024, linear_weight)
    embeddings.append(vec_final)

    print(f"  [{i}] raw[:5] = {[f'{x:.4f}' for x in vec_1024[:5]]}")

# ── 7. Cosine similarity matrix ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Cosine Similarities")
print("=" * 60)

PAIRS = {
    (0, 1): "expect HIGH  (patience vs patience)",
    (2, 3): "expect HIGH  (war vs war)",
    (0, 2): "expect LOW   (patience vs war)",
    (0, 3): "expect LOW   (patience vs war)",
    (1, 2): "expect LOW   (patience vs war)",
    (1, 3): "expect LOW   (patience vs war)",
    (0, 4): "expect LOW   (patience vs love)",
    (1, 4): "expect LOW   (patience vs love)",
    (2, 4): "expect LOW   (war vs love)",
    (3, 4): "expect LOW   (war vs love)",
}

for i in range(len(test_verses)):
    for j in range(i + 1, len(test_verses)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        note = PAIRS.get((i, j), "")
        print(f"  [{i}] & [{j}]  {sim:+.4f}   {note}")

print("\nVerse index:")
for i, (label, verse) in enumerate(zip(labels, test_verses)):
    print(f"  {label}: {verse[:45]}...")
