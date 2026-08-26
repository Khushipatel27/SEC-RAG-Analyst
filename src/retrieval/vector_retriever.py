"""
Dense retrieval.

Embeds the query and asks the vector store for its nearest neighbours.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.retrieval.base import Document, Retriever
from src.retrieval.embedder import OllamaEmbedder
from src.retrieval.vector_store import ChromaVectorStore


class VectorRetriever(Retriever):
    """
    Semantic retrieval over ChromaDB.

    Strong on paraphrase, weak on exact tokens — a query for "NVDA" will not
    reliably surface the chunk containing that literal string, which is the gap
    :class:`BM25Retriever` exists to close.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: OllamaEmbedder,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        logger.info("VectorRetriever initialized")

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[Document]:
        """
        Embed *query* and return its *k* nearest chunks.

        Returns documents shaped ``{text, metadata, distance, score}``, where
        ``score`` is cosine similarity (1.0 = identical).
        """
        query_embedding = self._embedder.embed_text(query)
        results = self._vector_store.search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
        )
        logger.debug(f"VectorRetriever returned {len(results)} results (k={k})")
        return results
