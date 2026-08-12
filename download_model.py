import os
from huggingface_hub import hf_hub_download

MODEL_REPO = "unsloth/Qwen3.5-2B-MTP-GGUF"
MODEL_FILE = "Qwen3.5-2B-MTP-Q4_K_M.gguf"

if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists(os.path.join("models", MODEL_FILE)):
    print(f"Downloading {MODEL_FILE} from Hugging Face into models/ directory...")
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir="models",
        local_dir_use_symlinks=False
    )
    print("Download complete!")
else:
    print(f"Model {MODEL_FILE} already exists. Skipping download.")
