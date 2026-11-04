
from huggingface_hub import snapshot_download
from pathlib import Path

# Thư mục gốc chứa mô hình
BASE_DIR = Path("models")

# 1. Tải LLM
print("Dang tai mo hinh Qwen2.5-3B-Instruct...")

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    local_dir=BASE_DIR / "llm" / "Qwen2.5-3B-Instruct",
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.txt"
    ]
)

print("Da tai xong LLM!")

# 2. Tải Embedding
print("Dang tai mo hinh Embedding...")

snapshot_download(
    repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_dir=BASE_DIR / "embedding" / "paraphrase-multilingual-MiniLM-L12-v2",
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.model",
        "*.txt"
    ]
)

print("Da tai xong Embedding!")
print("Hoan tat tai ca hai mo hinh.")
