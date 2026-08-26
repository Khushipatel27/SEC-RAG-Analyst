"""
Tests for the v2 agentic layer.

Network (SEC EDGAR) and LLM (ollama) calls are mocked throughout, so the suite
runs offline and deterministically. The arithmetic assertions use figures
hand-verified against the real filings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.calc_agent import CalculationAgent
from src.agents.graph_agent import GraphAgent
from src.agents.supervisor import analyse_question
from src.agents.verification_agent import VerificationAgent
from src.agents.xbrl_agent import XBRLAgent, XBRLFact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact(
    ticker: str = "NVDA",
    metric: str = "research_and_development",
    year: int = 2023,
    value: float = 7_339_000_000.0,
) -> XBRLFact:
    return XBRLFact(
        ticker=ticker,
        metric=metric,
        fiscal_year=year,
        value=value,
        unit="USD",
        concept="us-gaap:ResearchAndDevelopmentExpense",
        label="Research and Development Expense",
        accession="0001045810-23-000017",
        form_type="10-K",
        filing_date="2023-02-24",
        period_start="2022-01-31",
        period_end="2023-01-29",
        statement_type="IncomeStatement",
    )


# ---------------------------------------------------------------------------
# XBRL agent — metric resolution
# ---------------------------------------------------------------------------


class TestMetricResolution:
    def test_resolves_common_aliases(self):
        assert XBRLAgent.resolve_metric("What was Apple's revenue?") == "revenue"
        assert XBRLAgent.resolve_metric("R&D spending") == "research_and_development"
        assert XBRLAgent.resolve_metric("net income for 2023") == "net_income"

    def test_longest_alias_wins(self):
        # "operating income" must not be shadowed by a shorter alias
        assert XBRLAgent.resolve_metric("operating income") == "operating_income"
        assert XBRLAgent.resolve_metric("cost of revenue") == "cost_of_revenue"

    def test_unknown_metric_returns_none(self):
        assert XBRLAgent.resolve_metric("what colour is the logo") is None

    def test_resolve_metrics_finds_both_sides_of_a_ratio(self):
        metrics = XBRLAgent.resolve_metrics(
            "What was NVIDIA's R&D as a percentage of revenue?"
        )
        assert "research_and_development" in metrics
        assert "revenue" in metrics

    def test_resolve_metrics_does_not_match_revenue_inside_cost_of_revenue(self):
        metrics = XBRLAgent.resolve_metrics("What was the cost of revenue?")
        assert metrics == ["cost_of_revenue"]


class TestXBRLFactFormatting:
    def test_citation_includes_accession(self):
        fact = _fact()
        assert "0001045810-23-000017" in fact.citation
        assert "10-K" in fact.citation

    def test_value_formatted_scales_to_billions(self):
        assert _fact(value=7_339_000_000.0).value_formatted == "$7.34B"
        assert _fact(value=5_268_000.0).value_formatted == "$5.27M"

    def test_negative_values_put_the_sign_outside_the_symbol(self):
        # Amazon's FY2022 net income was a loss
        assert _fact(value=-2_722_000_000.0).value_formatted == "-$2.72B"

    def test_per_share_units_are_not_scaled(self):
        fact = _fact(value=6.13)
        fact.unit = "USD/shares"
        assert fact.value_formatted == "$6.13"


# ---------------------------------------------------------------------------
# Calculation agent — verified against real filing figures
# ---------------------------------------------------------------------------


class TestCalculationAgent:
    def setup_method(self):
        self.calc = CalculationAgent()

    def test_growth_rate_matches_hand_calculation(self):
        # NVDA R&D: $5,268M (FY22) → $7,339M (FY23) = +39.31%
        start = _fact(year=2022, value=5_268_000_000.0)
        end = _fact(year=2023, value=7_339_000_000.0)

        result = self.calc.growth_rate(start, end)

        assert result is not None
        assert result.value == pytest.approx(39.3128, abs=0.001)
        assert result.unit == "%"
        assert result.kind == "growth"
        # Both source citations must be carried through
        assert len(result.inputs) == 2

    def test_growth_rate_refuses_mismatched_metrics(self):
        start = _fact(metric="revenue", year=2022)
        end = _fact(metric="net_income", year=2023)
        assert self.calc.growth_rate(start, end) is None

    def test_growth_rate_refuses_zero_base(self):
        start = _fact(year=2022, value=0.0)
        end = _fact(year=2023, value=100.0)
        assert self.calc.growth_rate(start, end) is None

    def test_absolute_change_is_exact(self):
        """AAPL revenue: $394,328M → $383,285M = -$11,043M."""
        start = _fact(metric="revenue", year=2022, value=394_328_000_000.0)
        end = _fact(metric="revenue", year=2023, value=383_285_000_000.0)

        result = self.calc.absolute_change(start, end)

        assert result is not None
        assert result.value == pytest.approx(-11_043_000_000.0)
        assert result.value_formatted == "-$11.04B"

    def test_growth_answer_also_emits_absolute_change(self):
        """The delta must be computed, not left for the LLM to subtract."""
        facts = [
            _fact(metric="revenue", year=2022, value=394_328_000_000.0),
            _fact(metric="revenue", year=2023, value=383_285_000_000.0),
        ]
        kinds = {c["kind"] for c in self.calc.answer(facts, operation="growth")["calculations"]}
        assert "absolute_change" in kinds
        assert "growth" in kinds

    def test_sign_change_growth_carries_a_caveat_into_the_evidence(self):
        """
        Amazon's net income went from a $2.72B loss to a $30.43B profit. The
        resulting +1217% "growth" is arithmetically valid and misleading, so the
        warning must reach the evidence the synthesis model reads — not just the
        metadata.
        """
        start = _fact(ticker="AMZN", metric="net_income", year=2022, value=-2_722_000_000.0)
        end = _fact(ticker="AMZN", metric="net_income", year=2023, value=30_425_000_000.0)

        result = self.calc.growth_rate(start, end)
        assert result is not None
        assert result.caveat

        evidence = self.calc.answer([start, end], operation="growth")["evidence"]
        assert "CAVEAT" in evidence

    def test_same_sign_growth_has_no_caveat(self):
        start = _fact(year=2022, value=5_268_000_000.0)
        end = _fact(year=2023, value=7_339_000_000.0)
        assert self.calc.growth_rate(start, end).caveat == ""

    def test_ratio_matches_hand_calculation(self):
        # NVDA FY23: R&D 7,339 / revenue 26,974 = 27.21%
        rd = _fact(value=7_339_000_000.0)
        revenue = _fact(metric="revenue", value=26_974_000_000.0)

        result = self.calc.ratio(rd, revenue, as_percent=True)

        assert result is not None
        assert result.value == pytest.approx(27.2077, abs=0.001)

    def test_ratio_refuses_zero_denominator(self):
        rd = _fact(value=7_339_000_000.0)
        revenue = _fact(metric="revenue", value=0.0)
        assert self.calc.ratio(rd, revenue) is None

    def test_margin_is_not_signed_but_growth_is(self):
        rd = _fact(value=7_339_000_000.0)
        revenue = _fact(metric="revenue", value=26_974_000_000.0)
        margin = self.calc.ratio(rd, revenue, as_percent=True)
        growth = self.calc.growth_rate(
            _fact(year=2022, value=5_268_000_000.0), _fact(year=2023)
        )

        assert not margin.value_formatted.startswith("+")
        assert growth.value_formatted.startswith("+")

    def test_difference_reports_percentage_points(self):
        nvda = self.calc.growth_rate(
            _fact(year=2022, value=5_268_000_000.0),
            _fact(year=2023, value=7_339_000_000.0),
        )
        msft = self.calc.growth_rate(
            _fact(ticker="MSFT", year=2022, value=24_512_000_000.0),
            _fact(ticker="MSFT", year=2023, value=27_195_000_000.0),
        )

        gap = self.calc.difference(nvda, msft)

        assert gap is not None
        assert gap.value == pytest.approx(28.3672, abs=0.001)
        assert gap.unit == "percentage points"

    def test_answer_produces_growth_and_comparison(self):
        facts = [
            _fact(ticker="NVDA", year=2022, value=5_268_000_000.0),
            _fact(ticker="NVDA", year=2023, value=7_339_000_000.0),
            _fact(ticker="MSFT", year=2022, value=24_512_000_000.0),
            _fact(ticker="MSFT", year=2023, value=27_195_000_000.0),
        ]

        result = self.calc.answer(facts, operation="compare_growth")
        kinds = {c["kind"] for c in result["calculations"]}

        assert "growth" in kinds
        assert "difference" in kinds

    def test_comparison_sign_is_order_independent(self):
        """The gap must not flip sign based on ticker parse order."""
        nvda = [
            _fact(ticker="NVDA", year=2022, value=5_268_000_000.0),
            _fact(ticker="NVDA", year=2023, value=7_339_000_000.0),
        ]
        msft = [
            _fact(ticker="MSFT", year=2022, value=24_512_000_000.0),
            _fact(ticker="MSFT", year=2023, value=27_195_000_000.0),
        ]

        forward = self.calc.answer(nvda + msft, operation="compare_growth")
        reverse = self.calc.answer(msft + nvda, operation="compare_growth")

        def gap(result):
            return next(
                c["value"] for c in result["calculations"] if c["kind"] == "difference"
            )

        assert gap(forward) == pytest.approx(gap(reverse), abs=1e-6)
        assert gap(forward) > 0


# ---------------------------------------------------------------------------
# Verification agent
# ---------------------------------------------------------------------------


class TestVerificationAgent:
    def setup_method(self):
        self.verifier = VerificationAgent()
        self.evidence = (
            "AAPL FY2023 Revenue: $383.29B — AAPL FY2023 10-K "
            "(accession 0000320193-23-000106)\n"
            "NVDA research_and_development growth FY2022->FY2023: +39.31%"
        )

    def test_grounded_answer_passes(self):
        score, unsupported, grounded, total = self.verifier.check_numeric_grounding(
            "Apple's FY2023 revenue was $383.29B.", self.evidence
        )
        assert score == 1.0
        assert unsupported == []

    def test_hallucinated_figure_is_caught(self):
        score, unsupported, _, _ = self.verifier.check_numeric_grounding(
            "Apple's FY2023 revenue was $412.50B.", self.evidence
        )
        assert score == 0.0
        assert unsupported

    def test_scaled_notation_matches_raw_number(self):
        """$383.29B in the answer must match 383,285,000,000 in the evidence."""
        score, _, _, _ = self.verifier.check_numeric_grounding(
            "Revenue was $383.29B.", "Revenue was 383,285,000,000 dollars."
        )
        assert score == 1.0

    def test_percentages_are_not_matched_against_absolute_values(self):
        score, unsupported, _, _ = self.verifier.check_numeric_grounding(
            "R&D grew 383.29%.", self.evidence
        )
        assert score == 0.0
        assert unsupported

    def test_years_are_not_treated_as_claims(self):
        _, _, _, total = self.verifier.check_numeric_grounding(
            "In 2023 Apple filed its 10-K.", self.evidence
        )
        assert total == 0

    def test_parses_negative_currency_with_sign_outside_symbol(self):
        """The calculation agent emits -$11.04B; the verifier must read it."""
        score, unsupported, _, total = self.verifier.check_numeric_grounding(
            "Revenue fell by -$11.04B.",
            "AAPL revenue change FY2022→FY2023: -$11.04B",
        )
        assert total == 1
        assert score == 1.0
        assert unsupported == []

    def test_verify_without_evidence_is_unsupported(self):
        result = self.verifier.verify("q", "Revenue was $1B.", "", use_llm=False)
        assert result.verdict == "unsupported"
        assert result.confidence == 0.0

    def test_numeric_failure_cannot_be_overridden_by_llm(self):
        """A hallucinated figure stays unsupported even if the judge says SUPPORTED."""
        with patch.object(
            VerificationAgent, "check_entailment", return_value=("SUPPORTED", "ok")
        ):
            result = self.verifier.verify(
                "q", "Revenue was $999.99B.", self.evidence, use_llm=True
            )
        assert result.verdict != "grounded"

    def test_partial_grounding_is_reported(self):
        result = self.verifier.verify(
            "q",
            "Revenue was $383.29B and net income was $150.00B.",
            self.evidence,
            use_llm=False,
        )
        assert result.verdict == "partially_grounded"


# ---------------------------------------------------------------------------
# Supervisor routing
# ---------------------------------------------------------------------------


class TestQuestionAnalysis:
    def test_detects_single_ticker_and_metric(self):
        a = analyse_question("What was Apple's revenue in 2023?")
        assert a["tickers"] == ["AAPL"]
        assert "revenue" in a["metrics"]
        assert a["years"] == [2023]

    def test_detects_two_companies_for_comparison(self):
        a = analyse_question("How much faster did NVIDIA's R&D grow vs Microsoft's?")
        assert set(a["tickers"]) == {"NVDA", "MSFT"}
        assert a["wants_comparison"] is True
        assert a["wants_growth"] is True

    def test_detects_relationship_question(self):
        a = analyse_question("Which companies share supply chain risk exposure?")
        assert a["wants_relationship"] is True
        assert a["risk"] == "supply_chain"

    def test_detects_narrative_question(self):
        a = analyse_question("What did Amazon say about their AI strategy?")
        assert a["tickers"] == ["AMZN"]
        assert a["wants_narrative"] is True

    def test_parses_fiscal_year_shorthand(self):
        a = analyse_question("What was NVIDIA's revenue in FY23?")
        assert 2023 in a["years"]

    def test_alphabet_and_google_map_to_same_ticker(self):
        assert analyse_question("Alphabet revenue")["tickers"] == ["GOOGL"]
        assert analyse_question("Google revenue")["tickers"] == ["GOOGL"]


# ---------------------------------------------------------------------------
# Graph agent
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_file(tmp_path: Path) -> Path:
    graph = {
        "nodes": [
            {"id": "Company:Apple", "name": "Apple", "type": "Company"},
            {"id": "Company:NVIDIA", "name": "NVIDIA", "type": "Company"},
            {"id": "RiskFactor:supply_chain", "name": "supply_chain", "type": "RiskFactor"},
            {
                "id": "RiskFactor:customer_concentration",
                "name": "customer_concentration",
                "type": "RiskFactor",
            },
        ],
        "edges": [
            {
                "source": "Apple",
                "type": "DISCLOSES_RISK",
                "target": "supply_chain",
                "ticker": "AAPL",
                "year": "2023",
                "weight": 40,
                "evidence": [
                    {
                        "chunk_id": "c1",
                        "page_num": 7,
                        "section": "PART I",
                        "ticker": "AAPL",
                        "year": "2023",
                        "text_preview": "custom components...",
                    }
                ],
            },
            {
                "source": "NVIDIA",
                "type": "DISCLOSES_RISK",
                "target": "supply_chain",
                "ticker": "NVDA",
                "year": "2023",
                "weight": 55,
                "evidence": [],
            },
            {
                "source": "NVIDIA",
                "type": "DISCLOSES_RISK",
                "target": "customer_concentration",
                "ticker": "NVDA",
                "year": "2023",
                "weight": 12,
                "evidence": [],
            },
        ],
        "stats": {"num_nodes": 4, "num_edges": 3},
    }
    path = tmp_path / "knowledge_graph.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    return path


class TestGraphAgent:
    def test_missing_graph_degrades_gracefully(self, tmp_path):
        agent = GraphAgent(graph_path=tmp_path / "nope.json")
        assert agent.is_available is False
        result = agent.answer("Which companies share supply chain risk?")
        assert result["query_type"] == "unavailable"

    def test_shared_risk_ranks_by_weight(self, graph_file):
        agent = GraphAgent(graph_path=graph_file)
        result = agent.shared_risk("supply_chain", year="2023")

        assert result["num_companies"] == 2
        assert result["companies"][0]["company"] == "NVIDIA"  # weight 55 > 40

    def test_common_risks_computes_intersection_and_difference(self, graph_file):
        agent = GraphAgent(graph_path=graph_file)
        result = agent.common_risks(["AAPL", "NVDA"], year="2023")

        shared = {r["risk"] for r in result["shared_risks"]}
        assert shared == {"supply_chain"}
        assert result["unique_risks"]["NVIDIA"] == ["customer_concentration"]

    def test_direct_answer_names_every_company(self, graph_file):
        """The deterministic renderer must not drop set members."""
        agent = GraphAgent(graph_path=graph_file)
        result = agent.answer("Which companies share supply chain risk exposure?")

        assert "NVIDIA" in result["direct_answer"]
        assert "Apple" in result["direct_answer"]

    def test_uniqueness_question_compares_against_all_companies(self, graph_file):
        """
        "Which risks does NVIDIA disclose that the others do not" names one
        company but requires a set difference across the whole corpus.
        """
        agent = GraphAgent(graph_path=graph_file)
        result = agent.answer(
            "Which risks does NVIDIA disclose that the other companies do not?",
            tickers=["NVDA"],
        )

        assert result["query_type"] == "common_risks"
        assert "customer_concentration" in result["direct_answer"].replace(" ", "_")

    def test_company_lists_read_as_english(self):
        assert GraphAgent._join(["Apple"]) == "Apple"
        assert GraphAgent._join(["Apple", "NVIDIA"]) == "Apple and NVIDIA"
        assert (
            GraphAgent._join(["Apple", "Microsoft", "NVIDIA"])
            == "Apple, Microsoft and NVIDIA"
        )

    def test_evidence_states_counts_explicitly(self, graph_file):
        """
        The deterministic answer says "2 companies"; the verifier can only trace
        a figure it can find, so the count must appear in the evidence too.
        """
        agent = GraphAgent(graph_path=graph_file)
        result = agent.answer("Which companies share supply chain risk exposure?")

        assert "2" in result["evidence"]
        assert "2" in result["direct_answer"]

    def test_resolves_risk_aliases(self):
        assert GraphAgent.resolve_risk("supply chain problems") == "supply_chain"
        assert GraphAgent.resolve_risk("cyber attacks") == "cybersecurity"
        assert GraphAgent.resolve_risk("what colour is it") is None
