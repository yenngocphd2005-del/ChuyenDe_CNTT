"""
Cấu hình ứng dụng - đọc từ file .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Cấu hình chung cho ứng dụng."""

    # Database
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "cuchi_chatbot")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # Model paths
    LLM_MODEL_PATH: str = os.getenv(
        "LLM_MODEL_PATH",
        str(BASE_DIR / "models" / "llm" / "Qwen2.5-3B-Instruct"),
    )
    EMBEDDING_MODEL_PATH: str = os.getenv(
        "EMBEDDING_MODEL_PATH",
        str(BASE_DIR / "models" / "embedding" / "paraphrase-multilingual-MiniLM-L12-v2"),
    )

    # FAISS
    FAISS_INDEX_PATH: str = os.getenv(
        "FAISS_INDEX_PATH",
        str(BASE_DIR / "data" / "faiss_index"),
    )

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # RAG settings
    RAG_TOP_K: int = 3
    RAG_SCORE_THRESHOLD: float = 0.40  # Lọc tài liệu liên quan cao nhất

    # LLM settings - 95 tokens cân bằng hoàn hảo giữa độ đầy đủ thông tin và tốc độ (~60-70s)
    MAX_NEW_TOKENS: int = 95
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.85
    REPETITION_PENALTY: float = 1.15

    # Agent settings
    MAX_AGENT_STEPS: int = 3


settings = Settings()
