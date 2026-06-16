from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    MOCK_LLM: bool = True

    LLM_BASE_URL: str = "http://localhost:8001/v1"
    LLM_API_KEY: str = "local-key"
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

    UPLOAD_DIR: str = "./data/uploads"
    CHROMA_PATH: str = "./data/chroma"
    SQLITE_PATH: str = "./data/sqlite/paperops.db"

    TOP_K: int = 5

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
    )


settings = Settings()