"""
Unit tests for the Retriever interface and its three implementations.

Covers the interface contract, each strategy in isolation, and the composition
and staging logic in HybridRetriever.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.retrieval.base import Reranker, Retriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, _RRF_K
from src.retrieval.vector_retriever import VectorRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_retriever(vector_store, embedder) -> VectorRetriever:
    return VectorRetriever(vector_store=vector_store, embedder=embedder)


@pytest.fixture
def bm25_retriever(bm25_store) -> BM25Retriever:
    return BM25Retriever(bm25_store=bm25_store)


@pytest.fixture
def hybrid(vector_retriever, bm25_retriever, reranker) -> HybridRetriever:
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )


@pytest.fixture
def hybrid_no_rerank(vector_retriever, bm25_retriever) -> HybridRetriever:
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=None,
    )


def _texts(docs: list[dict]) -> list[str]:
    return [d.get("text", "") for d in docs]


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


def test_retriever_is_abstract():
    """Retriever cannot be instantiated without implementing retrieve()."""
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated():
    """A subclass that forgets retrieve() fails loudly at construction."""

    class Incomplete(Retriever):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "fixture_name", ["vector_retriever", "bm25_retriever", "hybrid"]
)
def test_all_implementations_satisfy_the_interface(fixture_name, request):
    """Every concrete retriever is a Retriever and exposes a name."""
    retriever = request.getfixturevalue(fixture_name)
    assert isinstance(retriever, Retriever)
    assert retriever.name == type(retriever).__name__
    assert callable(retriever.retrieve)


def test_reranker_protocol_is_satisfied_by_the_test_double(reranker):
    """The Reranker protocol matches structurally, with no explicit subclassing."""
    assert isinstance(reranker, Reranker)


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------


def test_vector_retriever_returns_ranked_documents(vector_retriever):
    results = vector_retriever.retrieve("Apple total net sales fiscal 2023", k=5)

    assert 0 < len(results) <= 5
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "results must be ranked best-first"


def test_vector_retriever_respects_k(vector_retriever):
    assert len(vector_retriever.retrieve("revenue", k=3)) <= 3
    assert len(vector_retriever.retrieve("revenue", k=1)) <= 1


def test_vector_retriever_applies_metadata_filters(vector_retriever):
    """A ticker filter must exclude every other issuer."""
    results = vector_retriever.retrieve("revenue", k=10, filters={"ticker": "MSFT"})

    assert results, "filter should not empty the result set for a common term"
    assert all(r["metadata"]["ticker"] == "MSFT" for r in results)


def test_vector_retriever_combines_filters_with_and(vector_retriever):
    results = vector_retriever.retrieve(
        "net sales", k=10, filters={"ticker": "AMZN", "year": "2022"}
    )
    assert results
    for r in results:
        assert r["metadata"]["ticker"] == "AMZN"
        assert r["metadata"]["year"] == "2022"


def test_vector_retriever_embeds_the_query_once(vector_store, embedder, mocker=None):
    """The query is embedded exactly once per retrieve() call."""
    calls: list[str] = []

    class CountingEmbedder:
        def embed_text(self, text: str) -> list[float]:
            calls.append(text)
            return embedder.embed_text(text)

    retriever = VectorRetriever(vector_store=vector_store, embedder=CountingEmbedder())
    retriever.retrieve("Microsoft cloud revenue", k=5)

    assert calls == ["Microsoft cloud revenue"]


def test_vector_retriever_returns_empty_for_unknown_vocabulary(vector_retriever):
    """A query sharing no terms with the corpus yields nothing, not an error."""
    assert vector_retriever.retrieve("zzzz qqqq wwww", k=5) == []


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


def test_bm25_retriever_returns_scored_documents(bm25_retriever):
    results = bm25_retriever.retrieve("Data Center revenue accelerated computing", k=5)

    assert results
    assert all("bm25_score" in r for r in results)

    scores = [r["bm25_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0), "top hit is normalised to 1.0"


def test_bm25_retriever_finds_exact_tokens_vector_search_would_blur(bm25_retriever):
    """Lexical retrieval is here to catch literal strings like 'TSMC'."""
    results = bm25_retriever.retrieve("TSMC foundries", k=3)
    assert results
    assert "TSMC" in results[0]["text"]


def test_bm25_retriever_respects_k(bm25_retriever):
    assert len(bm25_retriever.retrieve("revenue", k=2)) <= 2


def test_bm25_retriever_handles_empty_query(bm25_retriever):
    assert bm25_retriever.retrieve("   ", k=5) == []


def test_bm25_retriever_ignores_filters_rather_than_failing(bm25_retriever):
    """
    The sparse index holds no queryable metadata. Passing filters must degrade
    to an unfiltered search instead of raising or silently returning nothing.
    """
    unfiltered = bm25_retriever.retrieve("revenue growth", k=5)
    filtered = bm25_retriever.retrieve(
        "revenue growth", k=5, filters={"ticker": "MSFT"}
    )
    assert _texts(filtered) == _texts(unfiltered)


def test_bm25_retriever_does_not_mutate_the_stored_chunks(bm25_store, documents):
    """Adding bm25_score must not leak back into the index's own chunk dicts."""
    BM25Retriever(bm25_store=bm25_store).retrieve("revenue", k=5)
    assert all("bm25_score" not in d for d in documents)


# ---------------------------------------------------------------------------
# HybridRetriever — fusion
# ---------------------------------------------------------------------------


def test_hybrid_fuse_returns_k_unique_documents(hybrid):
    fused = hybrid.fuse("Apple revenue and research and development", k=6)

    assert len(fused) == 6
    assert len(set(_texts(fused))) == 6, "fusion must deduplicate"
    assert all("hybrid_score" in d for d in fused)

    scores = [d["hybrid_score"] for d in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_score_matches_the_published_formula(hybrid):
    """A document ranked 1st by both strategies scores exactly 2/(1+k)."""
    shared = {"text": "shared passage", "metadata": {"ticker": "AAPL"}, "score": 0.9}
    vector_results = [shared, {"text": "vector only", "metadata": {}, "score": 0.5}]
    bm25_results = [
        {"text": "shared passage", "bm25_score": 1.0},
        {"text": "bm25 only", "bm25_score": 0.4},
    ]

    fused = hybrid._rrf_fusion(vector_results, bm25_results, k_final=3)

    assert fused[0]["text"] == "shared passage"
    assert fused[0]["hybrid_score"] == pytest.approx(2.0 / (1 + _RRF_K))


def test_rrf_fusion_skips_documents_with_no_text(hybrid):
    """Empty text cannot be a dedup key, so such entries are dropped."""
    fused = hybrid._rrf_fusion(
        [{"text": "", "metadata": {}, "score": 0.9}, {"text": "real", "metadata": {}}],
        [{"text": "", "bm25_score": 1.0}],
        k_final=5,
    )
    assert _texts(fused) == ["real"]


def test_hybrid_surfaces_documents_neither_strategy_ranks_first(hybrid):
    """
    Fusion's value is the union: the result set contains documents contributed
    by each side, not just the dense side's ranking.
    """
    query = "supply chain risk from manufacturing partners and foundries"
    fused = hybrid.fuse(query, k=10)

    vector_only = hybrid._vector_retriever.retrieve(query, k=settings.top_k_vector)
    bm25_only = hybrid._bm25_retriever.retrieve(query, k=settings.top_k_bm25)

    fused_texts = set(_texts(fused))
    assert fused_texts & set(_texts(vector_only))
    assert fused_texts & set(_texts(bm25_only))


# ---------------------------------------------------------------------------
# HybridRetriever — staging and reranking
# ---------------------------------------------------------------------------


def test_retrieve_reranks_a_wider_candidate_set_than_it_returns(hybrid):
    """
    The two-stage width is the whole point: fuse fusion_k candidates, return k.
    """
    documents, trace = hybrid.retrieve_with_trace("Apple revenue", k=3)

    assert len(documents) == 3
    assert trace["reranked"] is True
    assert trace["num_fused"] > 3
    assert trace["num_fused"] <= settings.top_k_vector
    assert all("rerank_score" in d for d in documents)


def test_retrieve_without_a_reranker_returns_fused_top_k(hybrid_no_rerank):
    documents, trace = hybrid_no_rerank.retrieve_with_trace("Apple revenue", k=4)

    assert len(documents) == 4
    assert trace["reranked"] is False
    assert trace["num_fused"] == 4
    assert all("rerank_score" not in d for d in documents)
    assert all("hybrid_score" in d for d in documents)


def test_retrieve_matches_retrieve_with_trace(hybrid):
    """The convenience method must not diverge from the traced one."""
    plain = hybrid.retrieve("Microsoft cloud revenue", k=5)
    traced, _ = hybrid.retrieve_with_trace("Microsoft cloud revenue", k=5)
    assert _texts(plain) == _texts(traced)


def test_fusion_k_is_configurable(vector_retriever, bm25_retriever, reranker):
    narrow = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        fusion_k=3,
    )
    _, trace = narrow.retrieve_with_trace("revenue", k=2)
    assert trace["num_fused"] == 3


def test_fusion_k_defaults_to_the_configured_vector_depth(hybrid):
    assert hybrid._fusion_k == settings.top_k_vector


def test_hybrid_propagates_filters_to_the_dense_side(hybrid):
    """A ticker filter must survive fusion and reranking."""
    # Query terms are matched literally by the fixture embedder, so this uses
    # "revenues" — the wording Alphabet's filings actually use.
    documents = hybrid.retrieve("total revenues", k=5, filters={"ticker": "GOOGL"})

    assert documents
    # Sparse retrieval cannot filter, so assert the dense contribution is
    # constrained: every document carrying a ticker is the requested one.
    dense_hits = [d for d in documents if d.get("vector_score") is not None]
    assert dense_hits
    assert all(d.get("ticker") == "GOOGL" for d in dense_hits)


def test_hybrid_handles_a_query_matching_nothing(hybrid):
    assert hybrid.retrieve("zzzz qqqq wwww", k=5) == []


def test_query_routing_is_reachable_from_the_retriever(hybrid):
    routing = hybrid.apply_financial_query_routing(
        "What was Apple's total revenue in 2023?"
    )
    assert routing["ticker"] == "AAPL"
    assert routing["year"] == "2023"
    assert routing["block_type"] == "table"
