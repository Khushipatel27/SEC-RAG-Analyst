"""
Tests for FinancialRAGChain (prompt selection, context formatting, generation).
All ollama calls are mocked.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.chain import FinancialRAGChain
from src.generation.prompts import (
    COMPARISON_PROMPT,
    FINANCIAL_QA_PROMPT,
    SUMMARY_PROMPT,
    TABLE_EXTRACTION_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_chunk(text: str = "Apple revenue 394 billion", **kwargs) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "ticker": kwargs.get("ticker", "AAPL"),
        "year": kwargs.get("year", "2023"),
        "page_num": kwargs.get("page_num", 5),
        "section": kwargs.get("section", "ITEM 7A"),
        "block_type": "text",
        "contains_numbers": True,
        "source_file": "AAPL_2023_10K.pdf",
        "char_count": len(text),
        "rerank_score": kwargs.get("rerank_score", 0.95),
    }


def _make_table_chunk(text: str = "Year | Revenue\n2023 | 394,328", **kwargs) -> dict:
    chunk = _make_text_chunk(text, **kwargs)
    chunk["block_type"] = "table"
    return chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chain():
    return FinancialRAGChain()


@pytest.fixture
def sample_text_chunks():
    return [_make_text_chunk() for _ in range(3)]


@pytest.fixture
def sample_table_chunks():
    return [_make_table_chunk() for _ in range(4)]


# ---------------------------------------------------------------------------
# Test 1: Prompt selection logic
# ---------------------------------------------------------------------------


def test_prompt_selection_logic(chain, sample_text_chunks, sample_table_chunks):
    """Verify that compare/vs queries route to COMPARISON_PROMPT."""
    # Comparison queries
    compare_queries = [
        "Compare Apple and Microsoft revenue",
        "What is the difference between AAPL and MSFT margins?",
        "Apple vs Google R&D spending",
        "Contrast NVDA and AMZN operating income",
    ]
    for q in compare_queries:
        selected = chain.select_prompt(q, sample_text_chunks)
        assert selected is COMPARISON_PROMPT, (
            f"Expected COMPARISON_PROMPT for query: '{q}', got different template"
        )

    # Summary queries
    summary_queries = [
        "Give me a summary of Apple's 2023 performance",
        "Provide an overview of Microsoft's business",
        "Summarize NVIDIA's financial results",
    ]
    for q in summary_queries:
        selected = chain.select_prompt(q, sample_text_chunks)
        assert selected is SUMMARY_PROMPT, (
            f"Expected SUMMARY_PROMPT for query: '{q}'"
        )

    # Table-majority chunks
    table_query = "What are the revenue figures?"
    selected = chain.select_prompt(table_query, sample_table_chunks)
    assert selected is TABLE_EXTRACTION_PROMPT, (
        "Expected TABLE_EXTRACTION_PROMPT when majority of chunks are tables"
    )

    # Default
    default_query = "What are Apple's key risk factors?"
    selected = chain.select_prompt(default_query, sample_text_chunks)
    assert selected is FINANCIAL_QA_PROMPT, (
        f"Expected FINANCIAL_QA_PROMPT for default query: '{default_query}'"
    )


# ---------------------------------------------------------------------------
# Test 2: Context formatting includes citations
# ---------------------------------------------------------------------------


def test_context_formatting_includes_citations(chain):
    """format_context must include ticker, year, section, page_num, block_type."""
    chunks = [
        _make_text_chunk(
            text="Apple total net sales were 394 billion",
            ticker="AAPL",
            year="2023",
            section="ITEM 8",
            page_num=42,
        ),
        _make_table_chunk(
            text="Year | Revenue\n2023 | 394,328",
            ticker="MSFT",
            year="2022",
            section="ITEM 7",
            page_num=15,
        ),
    ]

    context = chain.format_context(chunks)

    # Must include all citation fields for each chunk
    assert "AAPL" in context, "Context should include ticker AAPL"
    assert "2023" in context, "Context should include year 2023"
    assert "ITEM 8" in context, "Context should include section ITEM 8"
    assert "42" in context, "Context should include page 42"
    assert "text" in context, "Context should include block_type text"

    assert "MSFT" in context, "Context should include ticker MSFT"
    assert "2022" in context, "Context should include year 2022"
    assert "ITEM 7" in context, "Context should include section ITEM 7"
    assert "15" in context, "Context should include page 15"
    assert "table" in context, "Context should include block_type table"

    # Check citation header format
    assert "[Source:" in context, "Context should contain [Source: ...] citation header"
    assert "10-K" in context, "Citation header should reference 10-K"
    assert "---" in context, "Chunk separator '---' should be present"

    # Check chunk text is present
    assert "Apple total net sales" in context
    assert "Year | Revenue" in context


# ---------------------------------------------------------------------------
# Test 3: generate() returns required fields
# ---------------------------------------------------------------------------


def test_answer_dict_has_required_fields(chain, sample_text_chunks):
    """generate() must return a dict with answer, sources, latency_ms, model."""
    mock_response = {
        "message": {
            "content": "Apple's total revenue in FY2023 was $394.3 billion, "
                       "representing a 3% increase year-over-year."
        }
    }

    with patch("ollama.chat", return_value=mock_response) as mock_chat:
        result = chain.generate(
            query="What was Apple's revenue in 2023?",
            chunks=sample_text_chunks,
        )

    # Verify required fields
    required_fields = [
        "answer",
        "sources",
        "prompt_used",
        "num_chunks",
        "latency_ms",
        "model",
        "context_length",
    ]
    for field in required_fields:
        assert field in result, f"Missing required field: '{field}'"

    # Verify field types
    assert isinstance(result["answer"], str), "answer should be a string"
    assert len(result["answer"]) > 0, "answer should not be empty"
    assert isinstance(result["sources"], list), "sources should be a list"
    assert isinstance(result["latency_ms"], float), "latency_ms should be a float"
    assert result["latency_ms"] >= 0, "latency_ms should be non-negative"
    assert isinstance(result["model"], str), "model should be a string"
    assert isinstance(result["num_chunks"], int), "num_chunks should be an int"
    assert result["num_chunks"] == len(sample_text_chunks)
    assert isinstance(result["context_length"], int), "context_length should be an int"
    assert result["context_length"] > 0

    # Verify sources structure
    for src in result["sources"]:
        assert "ticker" in src
        assert "year" in src
        assert "section" in src
        assert "page_num" in src
        assert "block_type" in src
        assert "relevance_score" in src
        assert "text_preview" in src

    # Verify ollama.chat was called
    mock_chat.assert_called_once()
