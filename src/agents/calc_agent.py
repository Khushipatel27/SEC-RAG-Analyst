"""
Calculation Agent
Performs arithmetic over exact XBRL facts in plain Python.

The LLM is deliberately kept out of this step. Language models are unreliable
at multi-step arithmetic, and a wrong growth rate in a financial answer is
indistinguishable from a right one to the reader. Every calculation here is
deterministic and carries the citations of the inputs it consumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.agents.xbrl_agent import XBRLFact


@dataclass
class Calculation:
    """A single derived figure, with a fully traceable derivation."""

    name: str
    value: float
    unit: str  # "%", "x", "USD", ...
    formula: str  # the arithmetic, with real numbers substituted in
    inputs: list[str] = field(default_factory=list)  # citations of source facts
    explanation: str = ""

    # Structured provenance, so downstream grouping never has to parse `name`
    kind: str = ""  # "growth" | "cagr" | "ratio" | "difference" | "absolute_change"
    ticker: Optional[str] = None
    metric: Optional[str] = None

    # A warning that must travel with the number wherever it is shown. A growth
    # rate computed across a sign change is arithmetically valid and practically
    # misleading, so the caveat belongs in the evidence, not just the metadata.
    caveat: str = ""

    @property
    def value_formatted(self) -> str:
        # Changes carry a sign (+8% is different from -8%); levels such as a
        # margin do not (a 19.57% R&D margin is not "+19.57%").
        signed = self.kind in ("growth", "cagr", "difference")

        if self.unit == "percentage points":
            return f"{self.value:+.2f} pp"
        if self.unit == "%":
            return f"{self.value:+.2f}%" if signed else f"{self.value:.2f}%"
        if self.unit == "x":
            return f"{self.value:.3f}x"
        if self.unit == "USD":
            # Sign goes outside the currency symbol: -$11.04B, not $-11.04B
            sign = "-" if self.value < 0 else ""
            abs_value = abs(self.value)
            if abs_value >= 1e9:
                return f"{sign}${abs_value / 1e9:,.2f}B"
            if abs_value >= 1e6:
                return f"{sign}${abs_value / 1e6:,.2f}M"
            return f"{sign}${abs_value:,.0f}"
        return f"{self.value:,.4f}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "unit": self.unit,
            "value_formatted": self.value_formatted,
            "formula": self.formula,
            "inputs": self.inputs,
            "explanation": self.explanation,
            "kind": self.kind,
            "ticker": self.ticker,
            "metric": self.metric,
            "caveat": self.caveat,
        }


class CalculationAgent:
    """
    Computes growth rates, margins, and ratios from XBRLFact inputs.

    Every public method returns a Calculation carrying the formula with real
    numbers substituted, so the arithmetic can be checked by hand.
    """

    def __init__(self) -> None:
        logger.info("CalculationAgent initialized")

    # ------------------------------------------------------------------
    # Primitive operations
    # ------------------------------------------------------------------

    def growth_rate(self, start: XBRLFact, end: XBRLFact) -> Optional[Calculation]:
        """
        Percentage change from *start* to *end*.

        Returns None when the starting value is zero (undefined growth) or the
        two facts measure different things.
        """
        if start.metric != end.metric:
            logger.warning(
                f"Refusing to compute growth across different metrics: "
                f"{start.metric} → {end.metric}"
            )
            return None
        if start.value == 0:
            logger.warning(f"Undefined growth: {start.ticker} {start.metric} base is 0")
            return None

        # A sign change makes percentage growth meaningless (e.g. a loss turning
        # into a profit) — report it, but flag the interpretation.
        sign_change = (start.value < 0) != (end.value < 0)
        pct = (end.value - start.value) / abs(start.value) * 100

        explanation = (
            f"{start.ticker} {start.label} moved from {start.value_formatted} "
            f"(FY{start.fiscal_year}) to {end.value_formatted} (FY{end.fiscal_year})."
        )
        caveat = ""
        if sign_change:
            caveat = (
                f"the value changed sign between periods "
                f"({start.value_formatted} → {end.value_formatted}), so this "
                f"percentage is not a meaningful growth rate — report the "
                f"absolute change instead"
            )
            explanation += f" Note: {caveat}."

        return Calculation(
            name=f"{start.ticker} {start.metric} growth FY{start.fiscal_year}→FY{end.fiscal_year}",
            value=pct,
            unit="%",
            formula=(
                f"({end.value:,.0f} - {start.value:,.0f}) / |{start.value:,.0f}| "
                f"x 100 = {pct:+.2f}%"
            ),
            inputs=[start.citation, end.citation],
            explanation=explanation,
            kind="growth",
            ticker=start.ticker,
            metric=start.metric,
            caveat=caveat,
        )

    def absolute_change(
        self, start: XBRLFact, end: XBRLFact
    ) -> Optional[Calculation]:
        """
        Absolute change between two facts, in currency units.

        Emitted alongside every growth rate. Without it the synthesis model
        tends to subtract the two figures itself to phrase the answer — and it
        gets that subtraction wrong often enough to matter.
        """
        if start.metric != end.metric:
            return None

        delta = end.value - start.value
        return Calculation(
            name=f"{start.ticker} {start.metric} change FY{start.fiscal_year}→FY{end.fiscal_year}",
            value=delta,
            unit="USD",
            formula=f"{end.value:,.0f} - {start.value:,.0f} = {delta:,.0f}",
            inputs=[start.citation, end.citation],
            explanation=(
                f"{start.ticker} {start.label} changed by "
                f"{'+' if delta >= 0 else ''}{delta:,.0f} {start.unit} between "
                f"FY{start.fiscal_year} and FY{end.fiscal_year}."
            ),
            kind="absolute_change",
            ticker=start.ticker,
            metric=start.metric,
        )

    def cagr(self, start: XBRLFact, end: XBRLFact) -> Optional[Calculation]:
        """Compound annual growth rate between two facts."""
        years = end.fiscal_year - start.fiscal_year
        if years <= 0 or start.value <= 0 or end.value <= 0:
            logger.warning("CAGR requires positive values and a positive year span")
            return None

        rate = ((end.value / start.value) ** (1 / years) - 1) * 100
        return Calculation(
            name=f"{start.ticker} {start.metric} CAGR FY{start.fiscal_year}→FY{end.fiscal_year}",
            value=rate,
            unit="%",
            formula=(
                f"({end.value:,.0f} / {start.value:,.0f})^(1/{years}) - 1 "
                f"= {rate:+.2f}%"
            ),
            inputs=[start.citation, end.citation],
            explanation=(
                f"{start.ticker} {start.label} compounded at {rate:+.2f}% per year "
                f"over {years} year(s)."
            ),
            kind="cagr",
            ticker=start.ticker,
            metric=start.metric,
        )

    def ratio(
        self,
        numerator: XBRLFact,
        denominator: XBRLFact,
        name: Optional[str] = None,
        as_percent: bool = True,
    ) -> Optional[Calculation]:
        """
        Ratio of two facts, e.g. R&D / revenue.

        Both facts should belong to the same company and fiscal year; a mismatch
        is logged but permitted, because cross-company ratios are sometimes the
        point of the question.
        """
        if denominator.value == 0:
            logger.warning("Refusing to divide by a zero denominator")
            return None

        if numerator.ticker != denominator.ticker:
            logger.info(
                f"Cross-company ratio: {numerator.ticker} / {denominator.ticker}"
            )

        raw = numerator.value / denominator.value
        value = raw * 100 if as_percent else raw
        unit = "%" if as_percent else "x"

        label = name or f"{numerator.metric} / {denominator.metric}"
        return Calculation(
            name=f"{numerator.ticker} {label} FY{numerator.fiscal_year}",
            value=value,
            unit=unit,
            formula=(
                f"{numerator.value:,.0f} / {denominator.value:,.0f} = "
                f"{value:,.2f}{'%' if as_percent else 'x'}"
            ),
            inputs=[numerator.citation, denominator.citation],
            explanation=(
                f"{numerator.ticker} FY{numerator.fiscal_year}: {numerator.label} "
                f"was {value:,.2f}{'%' if as_percent else 'x'} of {denominator.label}."
            ),
            kind="ratio",
            ticker=numerator.ticker,
            metric=numerator.metric,
        )

    def margin(self, fact: XBRLFact, revenue: XBRLFact) -> Optional[Calculation]:
        """Convenience wrapper: express *fact* as a percentage of revenue."""
        return self.ratio(
            fact, revenue, name=f"{fact.metric} margin", as_percent=True
        )

    def difference(self, a: Calculation, b: Calculation) -> Optional[Calculation]:
        """
        Difference between two calculations — the "how much faster" step.

        Used to answer questions like "how much faster did NVIDIA's R&D grow
        than Microsoft's", where the answer is a gap between two growth rates.
        """
        if a.unit != b.unit:
            logger.warning(f"Cannot difference {a.unit} against {b.unit}")
            return None

        gap = a.value - b.value
        return Calculation(
            name=f"{a.name} vs {b.name}",
            value=gap,
            unit="percentage points" if a.unit == "%" else a.unit,
            formula=f"{a.value:+.2f} - {b.value:+.2f} = {gap:+.2f}",
            inputs=list(dict.fromkeys(a.inputs + b.inputs)),
            explanation=(
                f"{a.name} ({a.value_formatted}) exceeded {b.name} "
                f"({b.value_formatted}) by {abs(gap):.2f} "
                f"{'percentage points' if a.unit == '%' else a.unit}."
                if gap >= 0
                else
                f"{a.name} ({a.value_formatted}) trailed {b.name} "
                f"({b.value_formatted}) by {abs(gap):.2f} "
                f"{'percentage points' if a.unit == '%' else a.unit}."
            ),
            kind="difference",
            metric=a.metric,
        )

    # ------------------------------------------------------------------
    # Agent entry point
    # ------------------------------------------------------------------

    def answer(self, facts: list[XBRLFact], operation: str = "auto") -> dict:
        """
        Derive every calculation the supplied facts support.

        Parameters
        ----------
        facts : list[XBRLFact]
            Typically the output of XBRLAgent.get_metrics().
        operation : str
            "growth", "margin", "compare_growth", or "auto" (do whatever the
            shape of the fact set allows).

        Returns
        -------
        dict  – {agent, calculations, evidence, summary, latency_ms}
        """
        start_time = time.perf_counter()
        calcs: list[Calculation] = []

        # Index facts by (ticker, metric) → {year: fact}
        grouped: dict[tuple[str, str], dict[int, XBRLFact]] = {}
        for f in facts:
            grouped.setdefault((f.ticker, f.metric), {})[f.fiscal_year] = f

        # --- Growth rates: any metric observed in 2+ years ---
        if operation in ("auto", "growth", "compare_growth"):
            for (ticker, metric), by_year in grouped.items():
                years = sorted(by_year)
                if len(years) < 2:
                    continue
                growth = self.growth_rate(by_year[years[0]], by_year[years[-1]])
                if growth is not None:
                    calcs.append(growth)
                delta = self.absolute_change(by_year[years[0]], by_year[years[-1]])
                if delta is not None:
                    calcs.append(delta)

        # --- Margins: any metric alongside revenue for the same ticker/year ---
        if operation in ("auto", "margin", "ratio"):
            for (ticker, metric), by_year in grouped.items():
                if metric == "revenue":
                    continue
                revenue_by_year = grouped.get((ticker, "revenue"))
                if not revenue_by_year:
                    continue
                for year, fact in by_year.items():
                    rev = revenue_by_year.get(year)
                    if rev is not None:
                        m = self.margin(fact, rev)
                        if m is not None:
                            calcs.append(m)

        # --- Cross-company comparison: difference between growth rates ---
        # Only meaningful when exactly two companies were asked about, and only
        # between growth rates of the same metric.
        if operation in ("auto", "compare_growth"):
            by_metric: dict[str, list[Calculation]] = {}
            for c in calcs:
                if c.kind == "growth" and c.metric:
                    by_metric.setdefault(c.metric, []).append(c)

            comparisons: list[Calculation] = []
            for metric, group in by_metric.items():
                if len({c.ticker for c in group}) != 2 or len(group) != 2:
                    continue
                # Always subtract the smaller from the larger. Otherwise the sign
                # depends on the order the tickers happened to be parsed in, and
                # a "how much faster" question gets an answer phrased as a
                # negative gap — which the synthesis model tends to mis-state.
                ordered = sorted(group, key=lambda c: c.value, reverse=True)
                diff = self.difference(ordered[0], ordered[1])
                if diff is not None:
                    comparisons.append(diff)
            calcs.extend(comparisons)

        lines = []
        for c in calcs:
            line = f"{c.name}: {c.value_formatted}  [{c.formula}]"
            if c.caveat:
                line += f"\n  ⚠ CAVEAT — {c.caveat}."
            lines.append(line)

        return {
            "agent": "calculation",
            "calculations": [c.to_dict() for c in calcs],
            "evidence": "\n".join(lines),
            "summary": (
                f"Computed {len(calcs)} derived figure(s) from exact XBRL inputs."
                if calcs
                else "No calculations were possible from the supplied facts."
            ),
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
