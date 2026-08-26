"""
Retrieval accuracy regression tests.

Measures recall for each strategy over the labelled fixture set in conftest.py
and asserts floors, so a change that quietly degrades ranking fails CI.

Measured at the time of writing (10 labelled queries, 32-document corpus):

    strategy          Recall@1   Recall@5
    vector_only          0.750      1.000
    bm25_only            0.350      1.000
    hybrid               0.750      1.000
    hybrid_rerank        0.850      1.000

Recall@5 is saturated, so it is asserted at 1.0 as a strict regression guard.
Recall@1 is the cutoff that actually separates the strategies and is where the
reranker's contribution shows up.

Scope note: these numbers characterise the retrieval *pipeline* (fusion,
staging, reranking) over a 20-document fixture corpus with a bag-of-words
embedder. They are not a measurement of the production nomic-embed-text +
cross-encoder stack on real 10-K filings, and are not comparable to the figures
in data/eval/. Run scripts/run_v1_v2_comparison.py for that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever
from tests.conftest import LABELLED_QUERIES

K = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_id(document: dict) -> str:
    """Chunk id, wherever the strategy happened to put it."""
    metadata = document.get("metadata") or {}
    return document.get("chunk_id") or metadata.get("chunk_id", "")


def recall_at_k(retrieved: list[dict], relevant: set[str], k: int = K) -> float:
    """
    Fraction of the relevant documents that appear in the top *k*.

    This is real recall — |retrieved ∩ relevant| / |relevant| — unlike
    RAGEvaluator.compute_retrieval_metrics, which can only ever return 0.0
    or 1.0.
    """
    if not relevant:
        return 0.0
    found = {_chunk_id(d) for d in retrieved[:k]} & relevant
    return len(found) / len(relevant)


def mean_recall(retriever, k: int = K) -> float:
    scores = [
        recall_at_k(retriever.retrieve(item["query"], k=k), item["relevant"], k)
        for item in LABELLED_QUERIES
    ]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strategies(vector_store, embedder, bm25_store, reranker) -> dict:
    vector = VectorRetriever(vector_store=vector_store, embedder=embedder)
    sparse = BM25Retriever(bm25_store=bm25_store)
    return {
        "vector_only": vector,
        "bm25_only": sparse,
        "hybrid": HybridRetriever(vector, sparse, reranker=None),
        "hybrid_rerank": HybridRetriever(vector, sparse, reranker=reranker),
    }


# ---------------------------------------------------------------------------
# Recall@5 — every strategy finds the answer within five results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy", ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]
)
def test_recall_at_5_is_perfect(strategies, strategy):
    """
    Every strategy places the relevant document in its top 5 for every query.

    The floor is 1.0 rather than something slacker precisely because the metric
    saturates here: any drop at all is a regression worth failing on.
    """
    score = mean_recall(strategies[strategy], k=5)
    assert score == pytest.approx(1.0), (
        f"{strategy} Recall@5 regressed to {score:.3f}"
    )


# ---------------------------------------------------------------------------
# Recall@1 — where the strategies actually differ
# ---------------------------------------------------------------------------
#
# Recall@5 cannot rank the strategies against each other because all four score
# 1.0. Recall@1 is the discriminating cutoff on this corpus: it asks whether the
# best answer was placed *first*, which is what the reranker exists to fix.


@pytest.mark.parametrize(
    "strategy,floor",
    [
        ("vector_only", 0.70),
        ("bm25_only", 0.30),
        ("hybrid", 0.70),
        ("hybrid_rerank", 0.80),
    ],
)
def test_recall_at_1_meets_floor(strategies, strategy, floor):
    score = mean_recall(strategies[strategy], k=1)
    assert score >= floor, (
        f"{strategy} Recall@1 regressed to {score:.3f}, below floor {floor:.2f}"
    )


def test_reranking_improves_top_1_accuracy(strategies):
    """
    The cross-encoder stage has to earn its latency cost: reranking the fused
    candidates must put the right document first more often than fusion alone.
    """
    fused_only = mean_recall(strategies["hybrid"], k=1)
    reranked = mean_recall(strategies["hybrid_rerank"], k=1)

    assert reranked > fused_only, (
        f"reranking did not improve Recall@1 ({reranked:.3f} vs {fused_only:.3f})"
    )


def test_dense_retrieval_beats_sparse_at_rank_1(strategies):
    """
    Sparse retrieval ranks by term frequency, so near-miss distractors that
    repeat a query term outrank the correct year. This is the specific weakness
    fusion exists to cover, and the margin is wide enough not to be noise.
    """
    assert mean_recall(strategies["vector_only"], k=1) > mean_recall(
        strategies["bm25_only"], k=1
    )


def test_hybrid_is_never_worse_than_either_half(strategies):
    """
    Fusion must not lose ground to the strategies it combines. Asserted as >=:
    on a corpus this small hybrid legitimately ties dense retrieval rather than
    beating it, and claiming otherwise would be overfitting to the fixture.
    """
    scores = {name: mean_recall(r, k=1) for name, r in strategies.items()}

    assert scores["hybrid"] >= scores["vector_only"], scores
    assert scores["hybrid"] >= scores["bm25_only"], scores
    assert scores["hybrid_rerank"] >= scores["hybrid"], scores


def test_every_labelled_query_retrieves_something(strategies):
    """No query in the set may come back empty from the full pipeline."""
    retriever = strategies["hybrid_rerank"]
    empty = [
        item["query"]
        for item in LABELLED_QUERIES
        if not retriever.retrieve(item["query"], k=K)
    ]
    assert not empty, f"queries returned nothing: {empty}"


@pytest.mark.parametrize("item", LABELLED_QUERIES, ids=lambda i: i["relevant"])
def test_each_query_individually(strategies, item):
    """
    Per-query recall, so a failure names the query that broke rather than only
    moving an aggregate.
    """
    retrieved = strategies["hybrid_rerank"].retrieve(item["query"], k=K)
    score = recall_at_k(retrieved, item["relevant"])
    assert score > 0.0, (
        f"no relevant document in top-{K} for {item['query']!r}; "
        f"got {[_chunk_id(d) for d in retrieved]}"
    )


# ---------------------------------------------------------------------------
# Recall metric itself
# ---------------------------------------------------------------------------


def test_recall_at_k_computes_partial_credit():
    """Two relevant documents, one retrieved → 0.5, not 1.0."""
    retrieved = [{"chunk_id": "a"}, {"chunk_id": "x"}, {"chunk_id": "y"}]
    assert recall_at_k(retrieved, {"a", "b"}) == pytest.approx(0.5)


def test_recall_at_k_respects_the_cutoff():
    """A relevant document ranked 6th does not count towards Recall@5."""
    retrieved = [{"chunk_id": f"pad{i}"} for i in range(5)] + [{"chunk_id": "a"}]
    assert recall_at_k(retrieved, {"a"}, k=5) == 0.0
    assert recall_at_k(retrieved, {"a"}, k=6) == 1.0


def test_recall_at_k_reads_ids_from_vector_metadata():
    """Dense results nest chunk_id under metadata; recall must still find it."""
    retrieved = [{"metadata": {"chunk_id": "a"}}]
    assert recall_at_k(retrieved, {"a"}) == pytest.approx(1.0)


def test_recall_at_k_handles_no_ground_truth():
    assert recall_at_k([{"chunk_id": "a"}], set()) == 0.0
