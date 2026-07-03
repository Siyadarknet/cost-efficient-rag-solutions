import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))

    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    TOP_K: int = int(os.getenv("TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))

    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")


settings = Settings()