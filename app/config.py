from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # App
    APP_NAME: str = "RAG Document Assistant"
    DEBUG: bool = False

    # Auth
    SECRET_KEY: str = "change-this-to-a-random-32-byte-hex-string"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24h

    # LLM
    LLM_PROVIDER: Literal["gemini", "openai"] = "gemini"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Embeddings (local, no API key needed)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector store
    CHROMA_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "rag_docs"

    # Chunking
    CHUNK_SIZE: int = 800       # words per chunk
    CHUNK_OVERLAP: int = 150    # word overlap between chunks
    TOP_K: int = 5              # retrieved chunks per query

    # Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_MB: int = 50

    # DB
    DATABASE_URL: str = "sqlite:///./rag_app.db"

    class Config:
        env_file = ".env"


settings = Settings()
