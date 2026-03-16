"""
Cross-Encoder Reranker
Uses sentence-transformers CrossEncoder to rerank retrieved chunks.
"""
from __future__ import annotations

from typing import Any

from loguru import logger
from sentence_transformers import CrossEncoder

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Reranks a list of retrieved chunks using a cross-encoder model.

    The cross-encoder receives (query, passage) pairs and produces
    relevance scores that are more accurate than bi-encoder cosine
    similarity alone.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        logger.info(f"Loading CrossEncoder model: {model_name}")
        self._model = CrossEncoder(model_name)
        self._model_name = model_name
        logger.success(f"CrossEncoder loaded: {model_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank *chunks* against *query* and return the top *top_k*.

        Parameters
        ----------
        query : str
            The user's question.
        chunks : list[dict]
            Retrieved chunk dicts (each must have a "text" key).
        top_k : int
            Number of top results to return.

        Returns
        -------
        list[dict]
            Top-k chunks sorted by rerank_score descending.
            Each dict has an added "rerank_score" key (float).
        """
        if not chunks:
            logger.warning("Reranker received empty chunk list; returning []")
            return []

        if not query.strip():
            logger.warning("Reranker received empty query; returning first top_k chunks")
            return chunks[:top_k]

        # Build (query, passage) pairs
        pairs: list[tuple[str, str]] = [
            (query, chunk.get("text", "")) for chunk in chunks
        ]

        logger.info(
            f"CrossEncoder reranking {len(chunks)} chunks for query: '{query[:80]}'"
        )

        # Score all pairs in one forward pass
        scores: list[float] = self._model.predict(pairs).tolist()

        # Log original order vs reranked order
        original_order = list(range(len(chunks)))
        scored_chunks = [
            (score, idx, chunk)
            for idx, (score, chunk) in enumerate(zip(scores, chunks))
        ]
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        reranked_order = [idx for _, idx, _ in scored_chunks]
        logger.debug(
            f"Original order: {original_order[:10]} → "
            f"Reranked order: {reranked_order[:10]}"
        )

        # Build result list
        results: list[dict] = []
        for score, _, chunk in scored_chunks[:top_k]:
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(score)
            results.append(chunk_copy)

        logger.info(
            f"Reranking complete: {len(chunks)} → {len(results)} chunks "
            f"(top score: {results[0]['rerank_score']:.4f} if results else 'N/A')"
            if results
            else f"Reranking complete: {len(chunks)} → 0 chunks"
        )
        return results
