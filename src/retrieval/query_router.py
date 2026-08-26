"""
Financial query routing.

Analyses a question and derives metadata filters from it — ticker, fiscal year,
and whether the answer is likely to live in a table or in narrative prose. This
is pure query analysis with no retrieval dependency, which is why it sits beside
the retrievers rather than inside one.
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

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


def apply_financial_query_routing(query: str) -> dict:
    """
    Analyse *query* and return a filters/hints dict.

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
