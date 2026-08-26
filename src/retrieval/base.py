"""
Retrieval interfaces.

Every retrieval strategy in the system — dense, sparse, or a composition of
both — implements :class:`Retriever`. Callers depend on this interface rather
than on ChromaDB, rank-bm25, or any other concrete backend, which is what lets
the v2 agentic layer swap or wrap a strategy without touching the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

# A retrieved chunk. Kept as a plain dict rather than a dataclass because the
# payload is heterogeneous by design: keys vary with the strategy that produced
# it (``vector_score``, ``bm25_score``, ``hybrid_score``, ``rerank_score``) and
# the whole dict is forwarded to the generation layer and serialised to JSON by
# the API. The alias documents the contract without constraining the payload.
Document = dict[str, Any]


@runtime_checkable
class Reranker(Protocol):
    """
    Second-pass precision filter applied to an already-retrieved candidate set.

    Declared as a Protocol so :class:`HybridRetriever` depends on the behaviour
    rather than on ``CrossEncoderReranker`` specifically — tests inject a
    lightweight stand-in, and swapping the model is a constructor change.
    """

    def rerank(
        self, query: str, chunks: list[Document], top_k: int = 5
    ) -> list[Document]:
        ...


class Retriever(ABC):
    """
    Abstract base for anything that turns a query into ranked documents.

    Implementations must not mutate the documents held by their underlying
    store; callers are free to add scoring keys to whatever is returned.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[Document]:
        """
        Return the *k* most relevant documents for *query*.

        Parameters
        ----------
        query : str
            Natural language query.
        k : int
            Maximum number of documents to return. Implementations may return
            fewer if the corpus is smaller or scores fall below a threshold.
        filters : dict, optional
            Metadata constraints (``ticker``, ``year``, ``block_type``,
            ``section``). Implementations that cannot filter — sparse retrieval,
            for instance — ignore this rather than failing, so that a filter
            never silently empties a result set it does not understand.

        Returns
        -------
        list[Document]
            Ranked best-first.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Short identifier used in logs and evaluation reports."""
        return type(self).__name__
