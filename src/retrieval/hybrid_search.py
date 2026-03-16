"""
Hybrid Search (Vector + BM25) with Reciprocal Rank Fusion (RRF)
and financial query routing.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger

from src.config import settings
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import OllamaEmbedder
from src.retrieval.vector_store import ChromaVectorStore

# ---------------------------------------------------------------------------
# Constants for query routing
# ---------------------------------------------------------------------------

_NUMERICAL_KEYWORDS = frozenset(
    [
        "revenue", "income", "profit", "loss", "earnings", "eps",
        "margin", "assets", "liabilities", "cash", "debt", "sales",
        "expenses", "cost", "ebitda", "capex", "dividend", "shares",
        "operating", "gross", "net", "total", "r&d", "research",
        "billion", "million", "percent", "%", "growth", "decline",
        "quarter", "annual", "fiscal", "ytd", "qoq", "yoy",
    ]
)

_RISK_STRATEGY_KEYWORDS = frozenset(
    [
        "risk", "strategy", "outlook", "competition", "market",
        "regulatory", "litigation", "legal", "future", "plan",
        "guidance", "challenge", "opportunity", "management",
        "discussion", "analysis", "mda", "md&a", "overview",
        "segment", "business", "operations",
    ]
)

_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

_TICKER_PATTERN = re.compile(
    r"\b(AAPL|MSFT|GOOGL|GOOG|AMZN|NVDA|Apple|Microsoft|Alphabet|Amazon|NVIDIA)\b",
    re.IGNORECASE,
)

_TICKER_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
    "goog": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
}

# RRF constant (default 60 as per the original paper)
_RRF_K = 60


class HybridSearcher:
    """
    Combines dense vector search (ChromaDB) and sparse keyword search (BM25)
    using Reciprocal Rank Fusion (RRF) for result merging.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_store: BM25Store,
        embedder: OllamaEmbedder,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._embedder = embedder
        logger.info("HybridSearcher initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k_final: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform hybrid retrieval and return the top *k_final* results.

        Parameters
        ----------
        query : str
            Natural language query.
        k_final : int
            Number of final results to return.
        filters : dict, optional
            Metadata filters forwarded to ChromaDB.
            Supported keys: ticker, year, block_type, section.

        Returns
        -------
        list[dict]
            Top-k unique chunks sorted by RRF score (descending).
            Each dict contains all original chunk fields plus "hybrid_score".
        """
        logger.info(f"HybridSearcher.search | query='{query[:80]}' | k_final={k_final}")

        # 1. Dense vector search
        query_embedding = self._embedder.embed_text(query)
        vector_results = self._vector_store.search(
            query_embedding=query_embedding,
            k=settings.top_k_vector,
            filters=filters,
        )

        # 2. Sparse BM25 search
        bm25_results = self._bm25_store.search(query=query, k=settings.top_k_bm25)

        # 3. RRF fusion
        fused = self._rrf_fusion(vector_results, bm25_results, k_final=k_final)

        logger.info(
            f"Hybrid search: {len(vector_results)} vector + "
            f"{len(bm25_results)} BM25 → {len(fused)} fused results"
        )
        return fused

    def apply_financial_query_routing(self, query: str) -> dict:
        """
        Analyse the query and return a filters/hints dict.

        Returns
        -------
        dict with optional keys:
            ticker       – exact ticker string
            year         – four-digit year string
            block_type   – "table" or "text"
            section      – section hint string (for downstream use)
            boost_tables – bool (hint for callers to weight tables higher)
        """
        query_lower = query.lower()
        query_tokens = set(re.findall(r"\w+", query_lower))
        result: dict[str, Any] = {}

        # --- Year detection ---
        year_matches = _YEAR_PATTERN.findall(query)
        if year_matches:
            # Use the most recent year if multiple
            result["year"] = max(year_matches)

        # --- Ticker/company detection ---
        ticker_matches = _TICKER_PATTERN.findall(query)
        if ticker_matches:
            raw = ticker_matches[0].lower()
            ticker = _TICKER_MAP.get(raw, raw.upper())
            result["ticker"] = ticker

        # --- Numerical / table query? ---
        numerical_hits = query_tokens & _NUMERICAL_KEYWORDS
        if numerical_hits:
            result["boost_tables"] = True
            result["block_type"] = "table"
            logger.debug(f"Query routing: numerical keywords found → {numerical_hits}")

        # --- Risk / strategy / qualitative query? ---
        risk_hits = query_tokens & _RISK_STRATEGY_KEYWORDS
        if risk_hits and not numerical_hits:
            result["block_type"] = "text"
            result["section"] = "MD&A"
            logger.debug(f"Query routing: risk/strategy keywords found → {risk_hits}")

        logger.info(f"Query routing result: {result}")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        k_final: int,
    ) -> list[dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        score(d) = Σ  1 / (rank_i(d) + k)   for each list i
        """
        # Map chunk text → cumulative RRF score + payload
        rrf_scores: dict[str, float] = {}
        chunk_payloads: dict[str, dict] = {}

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
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        results: list[dict] = []
        for key in sorted_keys[:k_final]:
            payload = chunk_payloads[key]
            payload["hybrid_score"] = rrf_scores[key]
            results.append(payload)

        return results
