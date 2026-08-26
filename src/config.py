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

    # ------------------------------------------------------------------
    # v2 — agentic layer
    # ------------------------------------------------------------------

    # SEC requires a descriptive User-Agent with a real contact address on
    # every EDGAR request. Requests without one get throttled or blocked.
    sec_user_agent: str = "SEC RAG Analyst khuship.study@gmail.com"

    # Companies the agentic layer knows about
    agent_tickers: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    # XBRL facts are cached on disk so repeated questions don't re-hit EDGAR
    xbrl_cache_dir: Path = Path("data/xbrl_cache")
    xbrl_cache_ttl_hours: int = 168  # 1 week; 10-K facts change rarely

    # Knowledge graph built from already-ingested chunks
    graph_path: Path = Path("data/graph/knowledge_graph.json")

    # Optional Neo4j export (the graph agent works without it)
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "yourpassword"

    # Verification
    verification_enabled: bool = True
    verification_max_evidence_chars: int = 6000

    class Config:
        env_file = ".env"


settings = Settings()
