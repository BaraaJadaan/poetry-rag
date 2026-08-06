import os
from huggingface_hub import hf_hub_download

MODEL_REPO = "brxce/OmniCoder-Claude-uncensored-V2-GGUF"
MODEL_FILE = "OmniCoder-Claude-uncensored-V2-Q4_K_M.gguf"

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
