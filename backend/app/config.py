from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    MOCK_LLM: bool = True

    LLM_BASE_URL: str = "http://localhost:8001/v1"
    LLM_API_KEY: str = "local-key"
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

    UPLOAD_DIR: str = str(PROJECT_ROOT / "data" / "uploads")
    CHROMA_PATH: str = str(PROJECT_ROOT / "data" / "chroma")
    SQLITE_PATH: str = str(PROJECT_ROOT / "data" / "sqlite" / "paperops.db")

    TOP_K: int = 5
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150

    LLM_MAX_TOKENS: int = 800
    LLM_TEMPERATURE: float = 0.2
    MAX_CONTEXT_CHARS: int = 3500

    CHROMA_COLLECTION: str = "research_documents"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


    ENABLE_RERANKER: bool = False
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N: int = 20

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
    )



settings = Settings()