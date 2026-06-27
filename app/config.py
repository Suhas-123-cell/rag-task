from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "RAG Document Assistant"

    # Auth — SECRET_KEY has no default; startup will fail if missing from .env
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24h

    # LLM
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.1-8b-instant"

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

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
