"""Retrieval layer: interfaces, storage backends, and retrieval strategies."""
from src.retrieval.base import Document, Reranker, Retriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import OllamaEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_router import apply_financial_query_routing
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.vector_store import ChromaVectorStore

__all__ = [
    # Interfaces
    "Document",
    "Reranker",
    "Retriever",
    # Strategies
    "BM25Retriever",
    "HybridRetriever",
    "VectorRetriever",
    # Storage backends
    "BM25Store",
    "ChromaVectorStore",
    "OllamaEmbedder",
    # Query analysis
    "apply_financial_query_routing",
]
