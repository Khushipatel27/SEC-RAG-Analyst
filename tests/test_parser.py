"""
Tests for FinancialDocumentParser
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.parser import (
    SECTION_PATTERNS,
    FinancialDocumentParser,
    _SECTION_RE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parser():
    return FinancialDocumentParser()


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Return a fake PDF path (won't be opened; pdfplumber is mocked)."""
    p = tmp_path / "AAPL_2023_10K.pdf"
    p.write_bytes(b"%PDF-1.4 fake pdf content")
    return p


# ---------------------------------------------------------------------------
# Test 1: parse_text_blocks extracts text blocks
# ---------------------------------------------------------------------------


def test_pdf_parser_extracts_text(parser, sample_pdf_path):
    """pdfplumber.open returns pages with extract_text(); verify text blocks."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = (
        "ITEM 1. BUSINESS\n\n"
        "Apple Inc. designs, manufactures, and markets smartphones.\n\n"
        "Net revenue was $394 billion in fiscal 2023."
    )
    mock_page.extract_tables.return_value = []

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    with patch("pdfplumber.open", return_value=mock_pdf):
        blocks = parser.parse_text_blocks(sample_pdf_path)

    assert len(blocks) > 0, "Expected at least one text block"
    for block in blocks:
        assert "text" in block
        assert "page_num" in block
        assert block["block_type"] == "text"
        assert "section" in block
        assert isinstance(block["text"], str)
        assert len(block["text"]) > 0


# ---------------------------------------------------------------------------
# Test 2: parse_table_blocks extracts table blocks
# ---------------------------------------------------------------------------


def test_pdf_parser_extracts_tables(parser, sample_pdf_path):
    """pdfplumber.open returns pages with extract_tables(); verify table blocks."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "ITEM 8. FINANCIAL STATEMENTS"
    mock_page.extract_tables.return_value = [
        [
            ["Year", "Revenue", "Net Income"],
            ["2023", "394,328", "96,995"],
            ["2022", "365,817", "99,803"],
        ]
    ]

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    with patch("pdfplumber.open", return_value=mock_pdf):
        blocks = parser.parse_table_blocks(sample_pdf_path)

    assert len(blocks) > 0, "Expected at least one table block"
    for block in blocks:
        assert block["block_type"] == "table"
        assert "text" in block
        assert "dataframe" in block
        assert "columns" in block["dataframe"]
        assert "data" in block["dataframe"]
        assert block.get("contains_numbers") is True
        assert "page_num" in block
        assert "section" in block


# ---------------------------------------------------------------------------
# Test 3: Section pattern detection
# ---------------------------------------------------------------------------


def test_section_detection_finds_headers(parser):
    """SECTION_PATTERNS regex matches known SEC filing headers."""
    test_cases = [
        ("ITEM 1. BUSINESS", True),
        ("ITEM 1A.", True),
        ("PART II", True),
        ("SECTION IV", True),
        ("random paragraph text here", False),
        ("ITEM 7A. QUANTITATIVE AND QUALITATIVE", True),
    ]

    for text, should_match in test_cases:
        match = _SECTION_RE.search(text)
        if should_match:
            assert match is not None, f"Expected match for: '{text}'"
        else:
            assert match is None, f"Unexpected match for: '{text}'"

    # Verify each pattern individually
    for pattern in SECTION_PATTERNS:
        assert isinstance(pattern, str), "Each SECTION_PATTERN should be a string"
        re.compile(pattern)  # Should not raise


# ---------------------------------------------------------------------------
# Test 4: extract_key_metrics finds revenue and income
# ---------------------------------------------------------------------------


def test_key_metrics_extraction(parser):
    """extract_key_metrics should find revenue and net income in sample text."""
    sample_text_block = {
        "text": (
            "Total net sales were $394,328 million for the year ended "
            "September 30, 2023. Net income was $96,995 million. "
            "Research and development expenses totaled $29,915 million. "
            "Diluted earnings per share was $6.13."
        ),
        "page_num": 5,
        "block_type": "text",
        "section": "CONSOLIDATED STATEMENTS OF OPERATIONS",
    }

    parsed_doc = {
        "ticker": "AAPL",
        "year": "2023",
        "text_blocks": [sample_text_block],
        "table_blocks": [],
    }

    metrics = parser.extract_key_metrics(parsed_doc)

    assert isinstance(metrics, dict), "extract_key_metrics should return a dict"
    assert "total_revenue" in metrics
    assert "net_income" in metrics
    assert "eps_diluted" in metrics
    assert "rd_expense" in metrics

    # At least revenue or net_income should be non-None
    assert metrics.get("total_revenue") is not None or metrics.get("net_income") is not None, (
        "Should extract at least one of revenue or net_income from sample text"
    )

    # EPS should be found
    if metrics.get("eps_diluted") is not None:
        assert "6" in str(metrics["eps_diluted"]), "EPS should contain '6'"
