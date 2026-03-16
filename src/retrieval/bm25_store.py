"""
BM25 Sparse Retrieval Store
Uses rank-bm25 (BM25Okapi) with joblib persistence.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src.config import settings


def _tokenize(text: str) -> list[str]:
    """Lowercase + whitespace split tokenization (mirrors query tokenization)."""
    return text.lower().split()


class BM25Store:
    """
    Wraps BM25Okapi for sparse keyword retrieval over chunk dicts.

    Usage
    -----
    store = BM25Store()
    store.build_index(chunks)          # or store.load_index()
    results = store.search("revenue", k=10)
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[dict] = []
        self._index_path: Path = Path(settings.bm25_index_path)
        logger.info(f"BM25Store initialized | index_path={self._index_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build a BM25Okapi index from *chunks* and persist it to disk.

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts (each must have a "text" key).
        """
        logger.info(f"Building BM25 index over {len(chunks)} chunks...")

        self._chunks = list(chunks)
        tokenized_corpus: list[list[str]] = [
            _tokenize(chunk.get("text", "")) for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

        # Persist
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "chunks": self._chunks,
        }
        joblib.dump(payload, self._index_path)
        logger.success(
            f"BM25 index built and saved to {self._index_path} "
            f"({len(self._chunks)} documents)"
        )

    def load_index(self) -> bool:
        """
        Load a previously saved BM25 index from disk.

        Returns
        -------
        bool
            True if the index was loaded successfully, False otherwise.
        """
        if not self._index_path.exists():
            logger.warning(f"BM25 index not found at {self._index_path}")
            return False

        try:
            payload = joblib.load(self._index_path)
            self._bm25 = payload["bm25"]
            self._chunks = payload["chunks"]
            logger.success(
                f"BM25 index loaded from {self._index_path} "
                f"({len(self._chunks)} documents)"
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to load BM25 index: {exc}")
            return False

    def search(self, query: str, k: int = 10) -> list[dict]:
        """
        Search the BM25 index for the top-k chunks matching *query*.

        Parameters
        ----------
        query : str
            The search query.
        k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Top-k chunk dicts, each with an added "bm25_score" key (float, 0-1).
        """
        if self._bm25 is None:
            logger.error("BM25 index not loaded. Call build_index() or load_index() first.")
            return []

        if not query.strip():
            logger.warning("Empty query passed to BM25Store.search(); returning []")
            return []

        tokenized_query = _tokenize(query)
        raw_scores: np.ndarray = self._bm25.get_scores(tokenized_query)

        # Normalize scores to [0, 1]
        max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0
        normalized: np.ndarray = raw_scores / max_score

        # Get top-k indices (descending)
        effective_k = min(k, len(self._chunks))
        top_indices = np.argsort(normalized)[::-1][:effective_k]

        results: list[dict] = []
        for idx in top_indices:
            score = float(normalized[idx])
            if score <= 0.0:
                continue
            chunk_copy = dict(self._chunks[idx])
            chunk_copy["bm25_score"] = score
            results.append(chunk_copy)

        logger.debug(
            f"BM25 search for '{query[:60]}' returned {len(results)} results "
            f"(max_score={max_score:.4f})"
        )
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_documents(self) -> int:
        return len(self._chunks)

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None
