from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MOCK_LLM: bool = True

    LLM_BASE_URL: str = "http://localhost:8001/v1"
    LLM_API_KEY: str = "local-key"
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

    UPLOAD_DIR: str = "./data/uploads"
    CHROMA_PATH: str = "./data/chroma"
    SQLITE_PATH: str = "./data/sqlite/paperops.db"

    TOP_K: int = 5

    class Config:
        env_file = ".env"


settings = Settings()