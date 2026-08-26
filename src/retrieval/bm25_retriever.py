"""
Sparse retrieval.

Thin adapter putting the BM25 index behind the :class:`Retriever` interface.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.retrieval.base import Document, Retriever
from src.retrieval.bm25_store import BM25Store


class BM25Retriever(Retriever):
    """
    Keyword retrieval over a BM25Okapi index.

    Catches what dense retrieval misses: tickers, dollar amounts, and exact
    line-item names that embeddings blur together.
    """

    def __init__(self, bm25_store: BM25Store) -> None:
        self._bm25_store = bm25_store
        logger.info("BM25Retriever initialized")

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[Document]:
        """
        Return the *k* highest-scoring chunks for *query*.

        Documents are flat chunk dicts with an added ``bm25_score`` in [0, 1].

        *filters* is accepted for interface conformance but ignored: the BM25
        index stores no queryable metadata. Dropping the constraint is the safe
        failure here — a sparse hit that the dense side also finds still gets
        filtered upstream, whereas returning nothing would silently remove the
        lexical half of hybrid search whenever a filter is present.
        """
        if filters:
            logger.debug(
                f"BM25Retriever ignoring unsupported filters: {sorted(filters)}"
            )

        results = self._bm25_store.search(query=query, k=k)
        logger.debug(f"BM25Retriever returned {len(results)} results (k={k})")
        return results
