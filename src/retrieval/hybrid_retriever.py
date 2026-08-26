"""
Hybrid retrieval: dense + sparse, merged with Reciprocal Rank Fusion and
narrowed by a cross-encoder reranker.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.config import settings
from src.retrieval.base import Document, Reranker, Retriever
from src.retrieval.query_router import apply_financial_query_routing

# RRF constant (default 60 as per the original paper)
_RRF_K = 60


class HybridRetriever(Retriever):
    """
    Composes a dense and a sparse retriever, fuses their rankings with RRF, and
    optionally applies a second-pass reranker.

    Retrieval runs in two stages with different widths. Fusion collects a wide
    candidate set (``fusion_k``) because RRF only rewards documents that both
    strategies rank highly, and that signal needs depth to be meaningful. The
    reranker then narrows that set to the *k* documents actually sent to the
    LLM. Passing ``reranker=None`` skips the second stage and returns the fused
    top-k directly, which is how the evaluator isolates the reranker's
    contribution.

    Both collaborators are typed as the :class:`Retriever` interface rather than
    as concrete classes, so a different dense backend or a filtered variant can
    be substituted without editing this class.
    """

    def __init__(
        self,
        vector_retriever: Retriever,
        bm25_retriever: Retriever,
        reranker: Optional[Reranker] = None,
        fusion_k: Optional[int] = None,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._fusion_k = fusion_k if fusion_k is not None else settings.top_k_vector
        logger.info(
            f"HybridRetriever initialized | fusion_k={self._fusion_k} | "
            f"reranker={'yes' if reranker else 'no'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[Document]:
        """
        Return the *k* best documents for *query*.

        With a reranker configured this is: dense + sparse → RRF over
        ``fusion_k`` candidates → rerank → top *k*.
        """
        documents, _ = self.retrieve_with_trace(query, k=k, filters=filters)
        return documents

    def retrieve_with_trace(
        self,
        query: str,
        k: int = 10,
        filters: Optional[dict] = None,
    ) -> tuple[list[Document], dict]:
        """
        Like :meth:`retrieve`, but also returns what happened on the way.

        Exists so callers can report how many candidates were considered before
        reranking without reaching into the retriever's internals or the
        retriever holding per-query state on itself.

        Returns
        -------
        tuple[list[Document], dict]
            The documents, and a trace dict ``{num_fused, reranked}``.
        """
        logger.info(f"HybridRetriever.retrieve | query='{query[:80]}' | k={k}")

        if self._reranker is None:
            fused = self.fuse(query, k=k, filters=filters)
            return fused, {"num_fused": len(fused), "reranked": False}

        fused = self.fuse(query, k=self._fusion_k, filters=filters)
        reranked = self._reranker.rerank(query=query, chunks=fused, top_k=k)
        logger.info(f"Reranked {len(fused)} fused candidates → {len(reranked)}")
        return reranked, {"num_fused": len(fused), "reranked": True}

    def fuse(
        self,
        query: str,
        k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[Document]:
        """
        Run both retrievers and merge their rankings with RRF, without reranking.

        Returns
        -------
        list[Document]
            Top-k unique chunks sorted by RRF score (descending). Each dict
            contains all original chunk fields plus ``hybrid_score``.
        """
        vector_results = self._vector_retriever.retrieve(
            query=query,
            k=settings.top_k_vector,
            filters=filters,
        )
        bm25_results = self._bm25_retriever.retrieve(
            query=query,
            k=settings.top_k_bm25,
            filters=filters,
        )

        fused = self._rrf_fusion(vector_results, bm25_results, k_final=k)

        logger.info(
            f"Hybrid search: {len(vector_results)} vector + "
            f"{len(bm25_results)} BM25 → {len(fused)} fused results"
        )
        return fused

    @staticmethod
    def apply_financial_query_routing(query: str) -> dict:
        """
        Derive metadata filters from *query*.

        Delegates to :mod:`src.retrieval.query_router`; kept here so callers
        holding a retriever can route without a second import.
        """
        return apply_financial_query_routing(query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        vector_results: list[Document],
        bm25_results: list[Document],
        k_final: int,
    ) -> list[Document]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        score(d) = Σ  1 / (rank_i(d) + k)   for each list i
        """
        # Map chunk text → cumulative RRF score + payload
        rrf_scores: dict[str, float] = {}
        chunk_payloads: dict[str, Document] = {}

        # --- Vector results ---
        for rank, item in enumerate(vector_results, start=1):
            # Use text as deduplication key (chunk_id may not be in vector results)
            text_key = item.get("text", "")
            if not text_key:
                continue
            rrf_scores[text_key] = rrf_scores.get(text_key, 0.0) + 1.0 / (rank + _RRF_K)
            if text_key not in chunk_payloads:
                # Flatten metadata into the top-level dict for consistency
                payload = dict(item.get("metadata", {}))
                payload["text"] = text_key
                payload["vector_score"] = item.get("score", 0.0)
                chunk_payloads[text_key] = payload

        # --- BM25 results ---
        for rank, item in enumerate(bm25_results, start=1):
            text_key = item.get("text", "")
            if not text_key:
                continue
            rrf_scores[text_key] = rrf_scores.get(text_key, 0.0) + 1.0 / (rank + _RRF_K)
            if text_key not in chunk_payloads:
                payload = dict(item)
                chunk_payloads[text_key] = payload
            else:
                chunk_payloads[text_key]["bm25_score"] = item.get("bm25_score", 0.0)

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)

        results: list[Document] = []
        for key in sorted_keys[:k_final]:
            payload = chunk_payloads[key]
            payload["hybrid_score"] = rrf_scores[key]
            results.append(payload)

        return results
