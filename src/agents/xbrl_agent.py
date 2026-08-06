"""
XBRL Agent
Retrieves exact, SEC-tagged financial figures straight from EDGAR's structured
XBRL data — no retrieval, no LLM, no chance of a hallucinated number.

Every figure carries the accession number of the filing it was tagged in, so a
downstream verification step can check the citation is real.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.config import settings

# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------
# Companies do not all tag the same concept for the same line item — Alphabet
# reports revenue as us-gaap:Revenues while Apple uses the longer
# RevenueFromContractWithCustomer... tag. Each metric therefore maps to an
# ordered fallback chain; the first concept that resolves wins.
METRIC_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
    ],
    "cost_of_revenue": [
        "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:CostOfRevenue",
    ],
    "gross_profit": ["us-gaap:GrossProfit"],
    "research_and_development": ["us-gaap:ResearchAndDevelopmentExpense"],
    "sga_expense": [
        "us-gaap:SellingGeneralAndAdministrativeExpense",
        "us-gaap:GeneralAndAdministrativeExpense",
    ],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "net_income": ["us-gaap:NetIncomeLoss"],
    "income_tax_expense": ["us-gaap:IncomeTaxExpenseBenefit"],
    "eps_diluted": ["us-gaap:EarningsPerShareDiluted"],
    "eps_basic": ["us-gaap:EarningsPerShareBasic"],
    "total_assets": ["us-gaap:Assets"],
    "total_liabilities": ["us-gaap:Liabilities"],
    "stockholders_equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash_and_equivalents": ["us-gaap:CashAndCashEquivalentsAtCarryingValue"],
    "inventory": ["us-gaap:InventoryNet"],
    "long_term_debt": [
        "us-gaap:LongTermDebtNoncurrent",
        "us-gaap:LongTermDebt",
    ],
    "operating_cash_flow": [
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
        "us-gaap:PaymentsToAcquireProductiveAssets",
    ],
    "share_repurchases": [
        "us-gaap:PaymentsForRepurchaseOfCommonStock",
    ],
    "dividends_paid": [
        "us-gaap:PaymentsOfDividendsCommonStock",
        "us-gaap:PaymentsOfDividends",
    ],
}

# Free-text phrases → canonical metric key. Longest match wins, so multi-word
# aliases are checked before single words.
METRIC_ALIASES: dict[str, str] = {
    "revenue": "revenue",
    "revenues": "revenue",
    "net sales": "revenue",
    "total revenue": "revenue",
    "sales": "revenue",
    "top line": "revenue",
    "cost of revenue": "cost_of_revenue",
    "cost of sales": "cost_of_revenue",
    "cogs": "cost_of_revenue",
    "gross profit": "gross_profit",
    "gross margin": "gross_profit",
    "r&d": "research_and_development",
    "rd": "research_and_development",
    "research and development": "research_and_development",
    "research & development": "research_and_development",
    "research spend": "research_and_development",
    "sg&a": "sga_expense",
    "selling general and administrative": "sga_expense",
    "operating income": "operating_income",
    "operating profit": "operating_income",
    "net income": "net_income",
    "net profit": "net_income",
    "profit": "net_income",
    "earnings": "net_income",
    "bottom line": "net_income",
    "income tax": "income_tax_expense",
    "tax expense": "income_tax_expense",
    "diluted eps": "eps_diluted",
    "eps": "eps_diluted",
    "earnings per share": "eps_diluted",
    "basic eps": "eps_basic",
    "total assets": "total_assets",
    "assets": "total_assets",
    "total liabilities": "total_liabilities",
    "liabilities": "total_liabilities",
    "shareholders equity": "stockholders_equity",
    "stockholders equity": "stockholders_equity",
    "equity": "stockholders_equity",
    "cash": "cash_and_equivalents",
    "cash and equivalents": "cash_and_equivalents",
    "inventory": "inventory",
    "inventories": "inventory",
    "long term debt": "long_term_debt",
    "long-term debt": "long_term_debt",
    "debt": "long_term_debt",
    "operating cash flow": "operating_cash_flow",
    "cash from operations": "operating_cash_flow",
    "capex": "capex",
    "capital expenditure": "capex",
    "capital expenditures": "capex",
    "buybacks": "share_repurchases",
    "share repurchases": "share_repurchases",
    "stock buybacks": "share_repurchases",
    "dividends": "dividends_paid",
    "dividends paid": "dividends_paid",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class XBRLFact:
    """A single exact figure pulled from SEC XBRL data."""

    ticker: str
    metric: str
    fiscal_year: int
    value: float
    unit: str
    concept: str
    label: str
    accession: str
    form_type: str
    filing_date: str
    period_start: Optional[str]
    period_end: Optional[str]
    statement_type: Optional[str]

    @property
    def citation(self) -> str:
        """Human-readable citation pointing at the exact SEC filing."""
        return (
            f"{self.ticker} FY{self.fiscal_year} {self.form_type} "
            f"(accession {self.accession}, filed {self.filing_date}) "
            f"[XBRL: {self.concept}]"
        )

    @property
    def value_formatted(self) -> str:
        """Format the value the way a financial analyst would write it."""
        if self.unit == "USD/shares":
            return f"${self.value:,.2f}"
        if self.unit != "USD":
            return f"{self.value:,.2f} {self.unit}"

        # Sign goes outside the currency symbol: -$2.72B, not $-2.72B
        # (Amazon's FY2022 net income is negative, so this path is exercised.)
        sign = "-" if self.value < 0 else ""
        abs_value = abs(self.value)
        if abs_value >= 1e9:
            return f"{sign}${abs_value / 1e9:,.2f}B"
        if abs_value >= 1e6:
            return f"{sign}${abs_value / 1e6:,.2f}M"
        return f"{sign}${abs_value:,.0f}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation
        d["value_formatted"] = self.value_formatted
        return d


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class XBRLAgent:
    """
    Looks up exact financial figures from SEC XBRL company facts.

    Unlike the retrieval pipeline, this agent never guesses: a metric is either
    tagged in the filing (returned with its accession number) or it is not
    (returned as None). That property is what makes the numbers trustworthy.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = Path(cache_dir or settings.xbrl_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # EntityFacts objects are expensive to build — keep them per process
        self._facts_cache: dict[str, Any] = {}
        # Disk-backed fact cache, keyed by ticker
        self._disk_cache: dict[str, dict] = {}

        self._identity_set = False
        logger.info(f"XBRLAgent initialized | cache_dir={self._cache_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_metric(text: str) -> Optional[str]:
        """
        Map a free-text metric phrase to a canonical metric key.

        Longest alias wins so that "operating income" is not shadowed by the
        shorter "income"-style aliases.

        Parameters
        ----------
        text : str

        Returns
        -------
        str | None  – canonical metric key, or None if nothing matched.
        """
        lowered = text.lower()
        best: Optional[str] = None
        best_len = 0

        for alias, canonical in METRIC_ALIASES.items():
            if alias in lowered and len(alias) > best_len:
                best = canonical
                best_len = len(alias)

        return best

    @staticmethod
    def resolve_metrics(text: str) -> list[str]:
        """
        Map free text to *every* metric it names, in order of appearance.

        Ratio questions ("R&D as a percentage of revenue") name two metrics, and
        resolve_metric() would only ever surface the longer alias. Matches are
        claimed longest-first so that "cost of revenue" consumes its span before
        the shorter "revenue" alias can match inside it.
        """
        lowered = text.lower()
        claimed: list[tuple[int, int]] = []
        found: list[tuple[int, str]] = []

        for alias in sorted(METRIC_ALIASES, key=len, reverse=True):
            canonical = METRIC_ALIASES[alias]
            start = lowered.find(alias)
            while start != -1:
                end = start + len(alias)
                overlaps = any(s < end and start < e for s, e in claimed)
                if not overlaps:
                    claimed.append((start, end))
                    found.append((start, canonical))
                start = lowered.find(alias, start + 1)

        # De-duplicate while preserving the order the metrics appear in the text
        ordered: list[str] = []
        for _, canonical in sorted(found):
            if canonical not in ordered:
                ordered.append(canonical)
        return ordered

    @staticmethod
    def supported_metrics() -> list[str]:
        """Return every canonical metric this agent can look up."""
        return sorted(METRIC_CONCEPTS.keys())

    def get_metric(
        self,
        ticker: str,
        metric: str,
        fiscal_year: int,
    ) -> Optional[XBRLFact]:
        """
        Fetch one exact annual (FY) figure for *ticker*.

        Parameters
        ----------
        ticker : str
            e.g. "AAPL". Case-insensitive.
        metric : str
            Canonical key (see supported_metrics()) or a free-text phrase such
            as "R&D", which is resolved via resolve_metric().
        fiscal_year : int
            The company's fiscal year, not the calendar year of filing.

        Returns
        -------
        XBRLFact | None
            None when the concept is not tagged for that company/year.
        """
        ticker = ticker.upper().strip()
        canonical = metric if metric in METRIC_CONCEPTS else self.resolve_metric(metric)

        if canonical is None:
            logger.warning(f"Unknown metric '{metric}' — no XBRL concept mapping")
            return None

        # --- Disk cache hit? ---
        cached = self._read_cache(ticker, canonical, fiscal_year)
        if cached is not None:
            logger.debug(f"XBRL cache hit: {ticker} {canonical} FY{fiscal_year}")
            return XBRLFact(**cached)

        # --- Resolve against EDGAR ---
        facts = self._get_entity_facts(ticker)
        if facts is None:
            return None

        concepts = METRIC_CONCEPTS[canonical]
        fact = None
        for concept in concepts:
            # edgartools warns loudly for every concept that does not resolve;
            # that is expected while walking the fallback chain.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fact = facts.get_annual_fact(concept, fiscal_year=fiscal_year)
                except Exception as exc:  # noqa: BLE001 - edgartools raises broadly
                    logger.debug(f"Concept '{concept}' lookup failed: {exc}")
                    fact = None
            if fact is not None:
                break

        if fact is None:
            logger.warning(
                f"No XBRL fact for {ticker} {canonical} FY{fiscal_year} "
                f"(tried {len(concepts)} concept(s))"
            )
            return None

        result = XBRLFact(
            ticker=ticker,
            metric=canonical,
            fiscal_year=int(getattr(fact, "fiscal_year", fiscal_year) or fiscal_year),
            value=float(fact.numeric_value),
            unit=str(getattr(fact, "unit", "USD") or "USD"),
            concept=str(fact.concept),
            label=str(getattr(fact, "label", canonical) or canonical),
            accession=str(getattr(fact, "accession", "") or ""),
            form_type=str(getattr(fact, "form_type", "") or ""),
            filing_date=str(getattr(fact, "filing_date", "") or ""),
            period_start=str(fact.period_start) if getattr(fact, "period_start", None) else None,
            period_end=str(fact.period_end) if getattr(fact, "period_end", None) else None,
            statement_type=str(getattr(fact, "statement_type", "") or "") or None,
        )

        self._write_cache(ticker, canonical, fiscal_year, result)
        logger.success(
            f"XBRL {ticker} {canonical} FY{fiscal_year} = "
            f"{result.value_formatted} ({result.accession})"
        )
        return result

    def get_metrics(
        self,
        tickers: list[str],
        metrics: list[str],
        fiscal_years: list[int],
    ) -> list[XBRLFact]:
        """
        Fetch the cross-product of tickers × metrics × years.

        Missing facts are skipped rather than raising, so a partially-tagged
        request still returns everything that could be resolved.
        """
        results: list[XBRLFact] = []
        for ticker in tickers:
            for metric in metrics:
                for year in fiscal_years:
                    fact = self.get_metric(ticker, metric, year)
                    if fact is not None:
                        results.append(fact)
        return results

    def answer(
        self,
        tickers: list[str],
        metrics: list[str],
        fiscal_years: list[int],
    ) -> dict:
        """
        Agent entry point used by the supervisor.

        Returns
        -------
        dict
            {facts, evidence, summary, missing, agent}
        """
        start = time.perf_counter()
        facts = self.get_metrics(tickers, metrics, fiscal_years)

        missing = [
            {"ticker": t, "metric": m, "fiscal_year": y}
            for t in tickers
            for m in metrics
            for y in fiscal_years
            if not any(
                f.ticker == t.upper()
                and f.fiscal_year == y
                and f.metric == (m if m in METRIC_CONCEPTS else self.resolve_metric(m))
                for f in facts
            )
        ]

        lines = [
            f"{f.ticker} FY{f.fiscal_year} {f.label}: {f.value_formatted} "
            f"— {f.citation}"
            for f in facts
        ]

        return {
            "agent": "xbrl",
            "facts": [f.to_dict() for f in facts],
            "evidence": "\n".join(lines),
            "summary": (
                f"Retrieved {len(facts)} exact XBRL fact(s) from SEC filings."
                if facts
                else "No matching XBRL facts were tagged for this request."
            ),
            "missing": missing,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_entity_facts(self, ticker: str) -> Optional[Any]:
        """Load (and memoise) the EntityFacts object for *ticker*."""
        if ticker in self._facts_cache:
            return self._facts_cache[ticker]

        try:
            from edgar import Company, set_identity
        except ImportError:
            logger.error(
                "edgartools is not installed — run: pip install edgartools"
            )
            return None

        if not self._identity_set:
            # SEC blocks requests without a contact address in the User-Agent.
            set_identity(settings.sec_user_agent)
            self._identity_set = True

        try:
            facts = Company(ticker).get_facts()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load SEC facts for {ticker}: {exc}")
            return None

        if facts is None:
            logger.error(f"SEC returned no facts for {ticker}")
            return None

        self._facts_cache[ticker] = facts
        return facts

    def _cache_path(self, ticker: str) -> Path:
        return self._cache_dir / f"{ticker}.json"

    def _load_ticker_cache(self, ticker: str) -> dict:
        if ticker in self._disk_cache:
            return self._disk_cache[ticker]

        path = self._cache_path(ticker)
        data: dict = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Corrupt XBRL cache for {ticker}, ignoring: {exc}")
                data = {}

        self._disk_cache[ticker] = data
        return data

    def _read_cache(
        self, ticker: str, metric: str, fiscal_year: int
    ) -> Optional[dict]:
        cache = self._load_ticker_cache(ticker)
        entry = cache.get(f"{metric}:{fiscal_year}")
        if entry is None:
            return None

        age_hours = (time.time() - entry.get("_cached_at", 0)) / 3600
        if age_hours > settings.xbrl_cache_ttl_hours:
            return None

        fact = dict(entry)
        fact.pop("_cached_at", None)
        return fact

    def _write_cache(
        self, ticker: str, metric: str, fiscal_year: int, fact: XBRLFact
    ) -> None:
        cache = self._load_ticker_cache(ticker)
        entry = asdict(fact)
        entry["_cached_at"] = time.time()
        cache[f"{metric}:{fiscal_year}"] = entry

        try:
            with open(self._cache_path(ticker), "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to write XBRL cache for {ticker}: {exc}")
