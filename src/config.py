from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    fallback_llm_model: str = "mistral"

    # Retrieval
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_vector: int = 10
    top_k_bm25: int = 10
    top_k_rerank: int = 5
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    # Paths
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    chroma_dir: Path = Path("data/chroma_db")
    bm25_index_path: Path = Path("data/bm25_index.pkl")

    # Generation
    max_new_tokens: int = 1024
    temperature: float = 0.0

    class Config:
        env_file = ".env"


settings = Settings()
