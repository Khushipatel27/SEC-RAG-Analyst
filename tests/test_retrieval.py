"""
Tests for chunker, hybrid search, query routing, and RRF fusion.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import FinancialChunker
from src.retrieval.hybrid_search import HybridSearcher, _RRF_K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(block_type: str = "text", text: str = "sample text", **kwargs) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "ticker": kwargs.get("ticker", "AAPL"),
        "year": kwargs.get("year", "2023"),
        "page_num": kwargs.get("page_num", 1),
        "section": kwargs.get("section", "ITEM 1"),
        "block_type": block_type,
        "contains_numbers": kwargs.get("contains_numbers", False),
        "source_file": "AAPL_2023_10K.pdf",
        "char_count": len(text),
    }


def _make_parsed_doc(
    text_blocks: list[dict] | None = None,
    table_blocks: list[dict] | None = None,
) -> dict:
    return {
        "file_path": "data/raw/AAPL_2023_10K.pdf",
        "ticker": "AAPL",
        "year": "2023",
        "num_pages": 3,
        "text_blocks": text_blocks or [],
        "table_blocks": table_blocks or [],
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunker():
    return FinancialChunker()


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search.return_value = [
        {
            "text": f"Vector result {i}",
            "metadata": {"ticker": "AAPL", "year": "2023", "section": "ITEM 1",
                         "page_num": i, "block_type": "text"},
            "distance": 0.1 * i,
            "score": 1.0 - 0.1 * i,
        }
        for i in range(1, 8)
    ]
    return store


@pytest.fixture
def mock_bm25_store():
    store = MagicMock()
    store.search.return_value = [
        {
            "chunk_id": str(uuid.uuid4()),
            "text": f"BM25 result {i}",
            "ticker": "AAPL",
            "year": "2023",
            "section": "ITEM 7",
            "page_num": i + 10,
            "block_type": "text",
            "contains_numbers": False,
            "source_file": "AAPL_2023_10K.pdf",
            "char_count": 20,
            "bm25_score": 1.0 / i,
        }
        for i in range(1, 8)
    ]
    return store


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 768
    return embedder


@pytest.fixture
def hybrid_searcher(mock_vector_store, mock_bm25_store, mock_embedder):
    return HybridSearcher(
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store,
        embedder=mock_embedder,
    )


# ---------------------------------------------------------------------------
# Test 1: Chunker never splits tables
# ---------------------------------------------------------------------------


def test_chunker_never_splits_tables(chunker):
    """A single table block should always produce exactly one chunk."""
    # Table with text much longer than chunk_size
    long_table_text = " | ".join([f"Row {i} Col1 {i*100} Col2 {i*200}" for i in range(200)])

    table_block = {
        "text": long_table_text,
        "dataframe": {"columns": ["Row", "Col1", "Col2"], "data": []},
        "page_num": 5,
        "block_type": "table",
        "section": "ITEM 8",
        "contains_numbers": True,
    }

    parsed_doc = _make_parsed_doc(table_blocks=[table_block])
    chunks = chunker.chunk_document(parsed_doc)

    table_chunks = [c for c in chunks if c["block_type"] == "table"]
    assert len(table_chunks) == 1, (
        f"Expected exactly 1 table chunk, got {len(table_chunks)}"
    )
    assert table_chunks[0]["text"] == long_table_text


# ---------------------------------------------------------------------------
# Test 2: Chunker respects chunk_size for text blocks
# ---------------------------------------------------------------------------


def test_chunker_respects_chunk_size(chunker):
    """Text chunks should not exceed chunk_size + chunk_overlap + buffer."""
    from src.config import settings

    # 5 paragraphs, each well above chunk_size
    long_para = "This is a sentence about Apple financial results. " * 30
    text_blocks = [
        {
            "text": long_para,
            "page_num": i,
            "block_type": "text",
            "section": "ITEM 7",
        }
        for i in range(1, 6)
    ]

    parsed_doc = _make_parsed_doc(text_blocks=text_blocks)
    chunks = chunker.chunk_document(parsed_doc)

    text_chunks = [c for c in chunks if c["block_type"] == "text"]
    assert len(text_chunks) > 0

    # Allow chunk_size + chunk_overlap + small buffer (50 chars) for edge cases
    max_allowed = settings.chunk_size + settings.chunk_overlap + 50
    oversized = [c for c in text_chunks if c["char_count"] > max_allowed]
    assert len(oversized) == 0, (
        f"{len(oversized)} chunks exceed max allowed size of {max_allowed}. "
        f"Largest: {max(c['char_count'] for c in text_chunks)} chars"
    )


# ---------------------------------------------------------------------------
# Test 3: Hybrid search returns k results
# ---------------------------------------------------------------------------


def test_hybrid_search_returns_k_results(hybrid_searcher):
    """HybridSearcher.search should return exactly k_final unique results."""
    k = 5
    results = hybrid_searcher.search("What is Apple's revenue?", k_final=k)

    assert len(results) == k, f"Expected {k} results, got {len(results)}"
    assert all("text" in r for r in results), "All results should have 'text'"
    assert all("hybrid_score" in r for r in results), "All results should have 'hybrid_score'"

    # Scores should be descending
    scores = [r["hybrid_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by hybrid_score"


# ---------------------------------------------------------------------------
# Test 4: Query routing detects numerical queries → table filter
# ---------------------------------------------------------------------------


def test_query_routing_detects_numerical_query(hybrid_searcher):
    """Queries with revenue/income keywords should trigger table block_type filter."""
    numerical_queries = [
        "What was Apple's total revenue in 2023?",
        "Show me net income for AAPL fiscal year 2022",
        "What is the operating income margin?",
    ]

    for query in numerical_queries:
        routing = hybrid_searcher.apply_financial_query_routing(query)
        assert routing.get("boost_tables") is True or routing.get("block_type") == "table", (
            f"Expected table routing for query: '{query}', got: {routing}"
        )


# ---------------------------------------------------------------------------
# Test 5: RRF fusion combines scores correctly
# ---------------------------------------------------------------------------


def test_rrf_fusion_combines_scores_correctly(hybrid_searcher):
    """
    Manually verify RRF formula: score(d) = 1/(rank + 60)
    A document appearing at rank 1 in both lists should score highest.
    """
    shared_text = "Apple total revenue 394 billion 2023"
    only_vector_text = "Only in vector results"
    only_bm25_text = "Only in BM25 results"

    vector_results = [
        {"text": shared_text, "metadata": {"ticker": "AAPL"}, "score": 0.9, "distance": 0.1},
        {"text": only_vector_text, "metadata": {"ticker": "AAPL"}, "score": 0.8, "distance": 0.2},
    ]

    bm25_results = [
        {
            "text": shared_text,
            "ticker": "AAPL", "year": "2023", "section": "ITEM 7",
            "page_num": 1, "block_type": "text", "contains_numbers": True,
            "source_file": "AAPL.pdf", "char_count": len(shared_text),
            "bm25_score": 1.0,
        },
        {
            "text": only_bm25_text,
            "ticker": "AAPL", "year": "2023", "section": "ITEM 8",
            "page_num": 2, "block_type": "text", "contains_numbers": False,
            "source_file": "AAPL.pdf", "char_count": len(only_bm25_text),
            "bm25_score": 0.8,
        },
    ]

    fused = hybrid_searcher._rrf_fusion(vector_results, bm25_results, k_final=3)

    # The shared_text chunk appears at rank 1 in BOTH lists
    # Expected RRF score = 1/(1+60) + 1/(1+60) = 2/61 ≈ 0.03279
    assert len(fused) > 0, "RRF fusion returned empty list"

    # First result should be the shared chunk (highest score)
    top_text = fused[0]["text"]
    assert top_text == shared_text, (
        f"Expected shared chunk at top (rank 1 in both lists), got: '{top_text}'"
    )

    expected_score = 2.0 / (1 + _RRF_K)
    actual_score = fused[0]["hybrid_score"]
    assert abs(actual_score - expected_score) < 1e-6, (
        f"RRF score mismatch: expected {expected_score:.6f}, got {actual_score:.6f}"
    )
