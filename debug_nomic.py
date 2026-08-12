import json
import numpy as np
import os
from huggingface_hub import hf_hub_download

print("Downloading Nomic Embed v1.5 GGUF...")
gguf_path = hf_hub_download(repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF", filename="nomic-embed-text-v1.5.Q8_0.gguf")

try:
    os.add_dll_directory(r"D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\lib")
except Exception:
    pass

from llama_cpp import Llama

print("Loading Nomic model locally...")
llm = Llama(
    model_path=gguf_path,
    embedding=True,
    pooling_type=1, # Nomic uses mean pooling by default typically
    n_ctx=512,
    verbose=False,
)

with open("eval_set.json", "r", encoding="utf-8") as f:
    eval_set = json.load(f)

print("\nTesting 5 queries from golden set to see if it understands Arabic...")
for i, item in enumerate(eval_set[:5]):
    query = item["query"]
    target = item["expected_text"]
    
    q_emb = np.array(llm.embed([query])[0], dtype=np.float32)
    t_emb = np.array(llm.embed([target])[0], dtype=np.float32)
    
    q_norm = q_emb / np.linalg.norm(q_emb)
    t_norm = t_emb / np.linalg.norm(t_emb)
    
    sim = np.dot(q_norm, t_norm)
    print(f"[{i+1}] Sim: {sim:.4f} | Query: {query}")
