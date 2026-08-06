"""
Supervisor
LangGraph orchestrator that routes a question to the specialists it needs,
collects their evidence, drafts an answer, and verifies it before returning.

Routing is rule-based, not LLM-based. A 3B model asked to pick tools returns
plausible-looking choices that are wrong often enough to matter, and a
misrouted question fails in a way that is hard to debug. The signals that
actually determine routing — a ticker, a metric name, a year, a comparison
word — are all extractable deterministically, so they are.

Graph shape
-----------
    route ─┬─→ xbrl → calc ─┐
           ├─→ graph ───────┤
           └─→ narrative ───┴─→ synthesize → verify → END
"""
from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Any, Optional, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.agents.calc_agent import CalculationAgent
from src.agents.graph_agent import GraphAgent
from src.agents.narrative_agent import NarrativeAgent
from src.agents.verification_agent import VerificationAgent
from src.agents.xbrl_agent import XBRLAgent, XBRLFact
from src.config import settings

# ---------------------------------------------------------------------------
# Question analysis
# ---------------------------------------------------------------------------

TICKER_PATTERNS: dict[str, list[str]] = {
    "AAPL": ["aapl", "apple"],
    "MSFT": ["msft", "microsoft"],
    "GOOGL": ["googl", "goog", "google", "alphabet"],
    "AMZN": ["amzn", "amazon"],
    "NVDA": ["nvda", "nvidia"],
}

_COMPARISON_WORDS = (
    "compare", "comparison", "versus", " vs", "faster", "slower", "more than",
    "less than", "difference", "differ", "outpace", "relative to", "against",
    "higher", "lower", "gap between",
)
_GROWTH_WORDS = (
    "grow", "growth", "increase", "decrease", "decline", "change", "rose",
    "fell", "yoy", "year over year", "year-over-year", "cagr", "trend",
)
_RATIO_WORDS = ("margin", "as a percentage", "as a % ", "ratio", "per dollar")
_RELATIONSHIP_WORDS = (
    "which companies", "which company", "who else", "share", "shared", "common",
    "both", "all five", "across companies", "overlap", "exposure", "exposed",
    "relationship", "connected", "in common",
)
_NARRATIVE_WORDS = (
    "say about", "said about", "describe", "discuss", "strategy", "explain",
    "why", "how does", "commentary", "outlook", "management", "approach",
    "what did", "opinion", "plan",
)

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
# "FY23" / "fiscal 2023"
_FY_RE = re.compile(r"\bfy\s?(\d{2,4})\b", re.IGNORECASE)


class QuestionAnalysis(TypedDict, total=False):
    tickers: list[str]
    metrics: list[str]
    years: list[int]
    risk: Optional[str]
    wants_comparison: bool
    wants_growth: bool
    wants_ratio: bool
    wants_relationship: bool
    wants_narrative: bool


def analyse_question(question: str) -> QuestionAnalysis:
    """Extract every routing signal the question carries."""
    lowered = question.lower()

    tickers = [
        ticker
        for ticker, aliases in TICKER_PATTERNS.items()
        if any(alias in lowered for alias in aliases)
    ]

    years = sorted({int(y) for y in _YEAR_RE.findall(question)})
    for match in _FY_RE.findall(question):
        year = int(match)
        if year < 100:
            year += 2000
        if year not in years:
            years.append(year)
    years = sorted(set(years))

    metrics = XBRLAgent.resolve_metrics(question)
    risk = GraphAgent.resolve_risk(question)

    return QuestionAnalysis(
        tickers=tickers,
        metrics=metrics,
        years=years,
        risk=risk,
        wants_comparison=any(w in lowered for w in _COMPARISON_WORDS),
        wants_growth=any(w in lowered for w in _GROWTH_WORDS),
        wants_ratio=any(w in lowered for w in _RATIO_WORDS),
        wants_relationship=any(w in lowered for w in _RELATIONSHIP_WORDS),
        wants_narrative=any(w in lowered for w in _NARRATIVE_WORDS),
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """State threaded through the LangGraph run."""

    question: str
    ticker_filter: Optional[str]
    year_filter: Optional[str]

    analysis: dict
    route: list[str]

    # Each specialist writes its own key, so parallel branches never collide.
    xbrl_result: dict
    calc_result: dict
    graph_result: dict
    narrative_result: dict

    draft_answer: str
    verification: dict
    final_answer: str

    # Appended to from multiple branches, hence the reducer.
    trace: Annotated[list[dict], operator.add]


SYNTHESIS_PROMPT = """You are a financial analyst answering a question about SEC \
10-K filings.

You have been given EVIDENCE gathered by specialist tools. Exact figures come \
from SEC XBRL data and calculations were performed in code — treat both as \
authoritative and reproduce them exactly as written.

Rules:
- Begin with ONE sentence that directly answers the question. Do not open by
  listing the evidence.
{numeric_rules}- Use ONLY the evidence below. Never introduce a figure that is not in it.
- Quote figures exactly as they appear. Do not round or recompute them.
- If a figure carries a CAVEAT, you must state that caveat alongside it.
- If the evidence lists several companies or entities, include EVERY one of
  them. Do not shorten the list.
- If the evidence does not answer part of the question, say so plainly.
- Be concise and direct. No preamble.

QUESTION:
{question}

EVIDENCE:
{evidence}

ANSWER:"""

# Only supplied when the evidence actually contains computed figures. Asking for
# "both the starting and ending values" on a question that has no numbers invites
# the model to invent them — which is exactly what it did on
# "which risks does NVIDIA disclose that the others do not?", fabricating a
# "$30 million trade secrets risk" that appears nowhere in any filing.
_NUMERIC_RULES = """- Then give the supporting figures. When describing a change over time, state
  BOTH the starting and ending values as well as the change itself. When
  comparing two companies, state the figure for BOTH companies.
- Cite the company and fiscal year for each figure you state.
"""

# Supplied instead when the evidence is purely qualitative.
_NON_NUMERIC_RULES = """- This question is not about figures. Do NOT introduce any dollar amounts,
  percentages, or numeric comparisons — answer in words, naming the companies
  and themes in the evidence.
"""


class SupervisorAgent:
    """
    Routes questions to specialists, then synthesises and verifies the answer.

    Parameters
    ----------
    pipeline : SECRAGPipeline | None
        The existing v1 pipeline, used for narrative questions. When None, the
        narrative specialist is unavailable and the supervisor routes around it.
    """

    def __init__(self, pipeline: Any = None) -> None:
        logger.info("Initialising SupervisorAgent...")

        self.xbrl = XBRLAgent()
        self.calc = CalculationAgent()
        self.graph = GraphAgent()
        self.verifier = VerificationAgent()
        self.narrative = NarrativeAgent(pipeline) if pipeline is not None else None

        self._model = settings.llm_model
        self._graph = self._build_graph()

        logger.success(
            f"SupervisorAgent ready | narrative={'yes' if self.narrative else 'no'} "
            f"| graph={'yes' if self.graph.is_available else 'no'}"
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("route", self._node_route)
        builder.add_node("xbrl", self._node_xbrl)
        builder.add_node("calc", self._node_calc)
        builder.add_node("graph", self._node_graph)
        builder.add_node("narrative", self._node_narrative)
        builder.add_node("synthesize", self._node_synthesize)
        builder.add_node("verify", self._node_verify)

        builder.add_edge(START, "route")

        # Fan out to whichever specialists the router selected.
        builder.add_conditional_edges(
            "route",
            self._select_branches,
            ["xbrl", "graph", "narrative", "synthesize"],
        )

        # Calculation always follows XBRL — it has nothing to compute otherwise.
        builder.add_edge("xbrl", "calc")
        builder.add_edge("calc", "synthesize")
        builder.add_edge("graph", "synthesize")
        builder.add_edge("narrative", "synthesize")

        builder.add_edge("synthesize", "verify")
        builder.add_edge("verify", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _node_route(self, state: AgentState) -> dict:
        """Decide which specialists this question needs."""
        question = state["question"]
        analysis = analyse_question(question)

        route: list[str] = []
        metrics = analysis.get("metrics") or []
        tickers = analysis.get("tickers") or []
        years = analysis.get("years") or []

        # --- Exact figures / arithmetic ---
        # A metric plus a company is enough to want the XBRL number; the
        # calculation node decides for itself whether anything is computable.
        if metrics and tickers:
            route.append("xbrl")

        # --- Relationship traversal ---
        # The graph answers set-shaped questions: who shares a risk, what two
        # companies have in common, what a company's disclosed risk profile is.
        # A risk *theme* appearing in the question is not on its own enough —
        # "what did Amazon say about AI" mentions a theme but wants prose.
        lowered = question.lower()
        asks_risk_profile = "risk" in lowered
        asks_segments = ("segment" in lowered or "product" in lowered) and tickers

        if self.graph.is_available and (
            analysis.get("wants_relationship")
            or (analysis.get("risk") and len(tickers) >= 2)
            or (analysis.get("risk") and not tickers)
            or asks_risk_profile
            or asks_segments
        ):
            route.append("graph")

        # --- Narrative retrieval ---
        # Used when the question is prose-shaped, when nothing structured was
        # detected at all, or as a safety net alongside a metric lookup that may
        # come back empty (Amazon, for instance, does not tag R&D in XBRL).
        if self.narrative is not None and (
            analysis.get("wants_narrative")
            or not route
            or (not metrics and not analysis.get("risk"))
        ):
            route.append("narrative")

        if not route:
            # Nothing matched and no narrative pipeline — go straight to
            # synthesis so the caller gets an honest "no evidence" answer.
            route.append("synthesize")

        logger.info(
            f"Route: {route} | tickers={tickers} metrics={metrics} "
            f"years={years} risk={analysis.get('risk')}"
        )

        return {
            "analysis": dict(analysis),
            "route": route,
            "trace": [
                {
                    "step": "route",
                    "selected": route,
                    "signals": dict(analysis),
                }
            ],
        }

    def _select_branches(self, state: AgentState) -> list[str]:
        """Conditional-edge selector: fan out to every routed specialist."""
        return state.get("route") or ["synthesize"]

    def _node_xbrl(self, state: AgentState) -> dict:
        analysis = state.get("analysis", {})
        tickers = analysis.get("tickers") or []
        metrics = analysis.get("metrics") or []
        years = analysis.get("years") or []

        # A growth or comparison question with only one year named needs the
        # prior year too, or there is nothing to compare against.
        if years and (analysis.get("wants_growth") or analysis.get("wants_comparison")):
            if len(years) == 1:
                years = sorted({years[0] - 1, years[0]})

        # No year mentioned: default to the two years in the corpus.
        if not years:
            years = [2022, 2023]

        result = self.xbrl.answer(tickers=tickers, metrics=metrics, fiscal_years=years)
        return {
            "xbrl_result": result,
            "trace": [
                {
                    "step": "xbrl",
                    "facts_found": len(result.get("facts", [])),
                    "missing": len(result.get("missing", [])),
                    "latency_ms": result.get("latency_ms"),
                }
            ],
        }

    def _node_calc(self, state: AgentState) -> dict:
        xbrl_result = state.get("xbrl_result") or {}
        fact_dicts = xbrl_result.get("facts", [])

        if not fact_dicts:
            return {
                "calc_result": {},
                "trace": [{"step": "calc", "skipped": "no XBRL facts to compute on"}],
            }

        # Rehydrate into XBRLFact so the calculator works with typed inputs.
        facts: list[XBRLFact] = []
        for d in fact_dicts:
            payload = {
                k: v
                for k, v in d.items()
                if k not in ("citation", "value_formatted")
            }
            try:
                facts.append(XBRLFact(**payload))
            except TypeError as exc:
                logger.warning(f"Could not rehydrate XBRL fact: {exc}")

        analysis = state.get("analysis", {})
        if analysis.get("wants_comparison"):
            operation = "compare_growth"
        elif analysis.get("wants_growth"):
            operation = "growth"
        elif analysis.get("wants_ratio"):
            operation = "ratio"
        else:
            operation = "auto"

        result = self.calc.answer(facts, operation=operation)
        return {
            "calc_result": result,
            "trace": [
                {
                    "step": "calc",
                    "operation": operation,
                    "calculations": len(result.get("calculations", [])),
                    "latency_ms": result.get("latency_ms"),
                }
            ],
        }

    def _node_graph(self, state: AgentState) -> dict:
        analysis = state.get("analysis", {})
        years = analysis.get("years") or []
        result = self.graph.answer(
            question=state["question"],
            tickers=analysis.get("tickers") or [],
            year=str(years[-1]) if years else None,
        )
        return {
            "graph_result": result,
            "trace": [
                {
                    "step": "graph",
                    "query_type": result.get("query_type"),
                    "latency_ms": result.get("latency_ms"),
                }
            ],
        }

    def _node_narrative(self, state: AgentState) -> dict:
        analysis = state.get("analysis", {})
        tickers = analysis.get("tickers") or []
        years = analysis.get("years") or []

        result = self.narrative.answer(
            question=state["question"],
            ticker=state.get("ticker_filter") or (tickers[0] if len(tickers) == 1 else None),
            year=state.get("year_filter") or (str(years[-1]) if len(years) == 1 else None),
        )
        return {
            "narrative_result": result,
            "trace": [
                {
                    "step": "narrative",
                    "sources": len(result.get("sources", [])),
                    "latency_ms": result.get("latency_ms"),
                }
            ],
        }

    def _node_synthesize(self, state: AgentState) -> dict:
        """Draft the answer from whatever evidence the specialists returned."""
        evidence = self._collect_evidence(state)

        if not evidence.strip():
            return {
                "draft_answer": (
                    "I could not find evidence to answer this question in the "
                    "indexed filings or SEC XBRL data."
                ),
                "trace": [{"step": "synthesize", "evidence_chars": 0}],
            }

        # --- Short-circuits: skip the LLM when one specialist already holds a
        # --- complete answer. Re-narrating it can only lose information.
        contributors = {
            name
            for name, key in (
                ("xbrl", "xbrl_result"),
                ("calc", "calc_result"),
                ("graph", "graph_result"),
                ("narrative", "narrative_result"),
            )
            if (state.get(key) or {}).get("evidence")
        }

        graph_result = state.get("graph_result") or {}
        if contributors == {"graph"} and graph_result.get("direct_answer"):
            return {
                "draft_answer": graph_result["direct_answer"],
                "trace": [
                    {
                        "step": "synthesize",
                        "mode": "deterministic (graph traversal)",
                        "evidence_chars": len(evidence),
                    }
                ],
            }

        narrative_result = state.get("narrative_result") or {}
        if contributors == {"narrative"} and narrative_result.get("answer"):
            # Exactly the v1 answer — this question was a retrieval problem all
            # along, and the v1 pipeline already generated from the full context.
            return {
                "draft_answer": narrative_result["answer"],
                "trace": [
                    {
                        "step": "synthesize",
                        "mode": "passthrough (v1 retrieval pipeline)",
                        "evidence_chars": len(evidence),
                    }
                ],
            }

        # Tailor the rules to the shape of the evidence. A question answered by
        # graph traversal has no figures to report, and telling the model to
        # report figures anyway is an invitation to invent them.
        has_figures = bool(
            (state.get("xbrl_result") or {}).get("evidence")
            or (state.get("calc_result") or {}).get("evidence")
        )
        prompt = SYNTHESIS_PROMPT.format(
            question=state["question"],
            evidence=evidence,
            numeric_rules=_NUMERIC_RULES if has_figures else _NON_NUMERIC_RULES,
        )

        start = time.perf_counter()
        try:
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={
                    "num_predict": settings.max_new_tokens,
                    "temperature": settings.temperature,
                },
            )
            draft = response["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Synthesis failed: {exc}")
            # Falling back to the raw evidence is better than returning nothing:
            # the figures are already exact and cited.
            draft = f"[Synthesis model unavailable: {exc}]\n\nEvidence collected:\n{evidence}"

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return {
            "draft_answer": draft,
            "trace": [
                {
                    "step": "synthesize",
                    "evidence_chars": len(evidence),
                    "latency_ms": elapsed,
                }
            ],
        }

    def _node_verify(self, state: AgentState) -> dict:
        draft = state.get("draft_answer", "")
        evidence = self._collect_evidence(state)

        if not settings.verification_enabled:
            return {
                "final_answer": draft,
                "verification": {"verdict": "skipped", "confidence": 0.0},
                "trace": [{"step": "verify", "skipped": "verification disabled"}],
            }

        result = self.verifier.verify(
            question=state["question"], answer=draft, evidence=evidence
        )

        # An answer that failed grounding is still returned — suppressing it
        # would hide the failure. It is returned with the warning attached.
        final = draft
        if result.verdict == "unsupported" and result.unsupported_numbers:
            final = (
                f"{draft}\n\n"
                f"⚠️ Verification warning: the following figures could not be "
                f"traced to the retrieved evidence: "
                f"{', '.join(result.unsupported_numbers[:6])}."
            )

        return {
            "final_answer": final,
            "verification": result.to_dict(),
            "trace": [
                {
                    "step": "verify",
                    "verdict": result.verdict,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Evidence assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_evidence(state: AgentState) -> str:
        """Concatenate every specialist's evidence, most authoritative first."""
        blocks: list[str] = []

        # Calculations come first deliberately. When a question asks "how much
        # faster", the derived figure *is* the answer — leading with the raw
        # inputs invites the model to restate them instead of answering.
        calc = state.get("calc_result") or {}
        if calc.get("evidence"):
            blocks.append(
                "=== DERIVED ANSWERS (computed in code — authoritative) ===\n"
                + calc["evidence"]
            )

        xbrl = state.get("xbrl_result") or {}
        if xbrl.get("evidence"):
            blocks.append(
                "=== SUPPORTING FIGURES (SEC XBRL — authoritative) ===\n"
                + xbrl["evidence"]
            )

        graph = state.get("graph_result") or {}
        if graph.get("evidence"):
            blocks.append(
                "=== RELATIONSHIPS (knowledge graph over filings) ===\n"
                + graph["evidence"]
            )

        narrative = state.get("narrative_result") or {}
        if narrative.get("evidence"):
            blocks.append(
                "=== RETRIEVED PASSAGES (10-K text) ===\n" + narrative["evidence"]
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _collect_sources(state: AgentState) -> list[dict]:
        """Build a unified citation list across every specialist."""
        sources: list[dict] = []

        for fact in (state.get("xbrl_result") or {}).get("facts", []):
            sources.append(
                {
                    "type": "xbrl",
                    "ticker": fact.get("ticker"),
                    "year": fact.get("fiscal_year"),
                    "concept": fact.get("concept"),
                    "accession": fact.get("accession"),
                    "form_type": fact.get("form_type"),
                    "value": fact.get("value_formatted"),
                    "citation": fact.get("citation"),
                }
            )

        for citation in (state.get("graph_result") or {}).get("citations", []):
            sources.append({"type": "graph", **citation})

        for source in (state.get("narrative_result") or {}).get("sources", []):
            sources.append({"type": "retrieval", **source})

        return sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        ticker_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
    ) -> dict:
        """
        Answer *question* through the agentic pipeline.

        Returns
        -------
        dict
            {answer, sources, agents_used, routing, verification, trace,
             latency_ms, model, agent_results}
        """
        start = time.perf_counter()
        logger.info(f"Supervisor.ask: '{question[:80]}'")

        initial: AgentState = {
            "question": question,
            "ticker_filter": ticker_filter,
            "year_filter": year_filter,
            "trace": [],
        }

        final_state = self._graph.invoke(initial)

        # A specialist counts as "used" only if it actually contributed evidence.
        # The calculation node runs after every XBRL lookup but often has nothing
        # to compute, and reporting it as used would overstate the routing.
        agents_used = [
            name
            for name, key in (
                ("xbrl", "xbrl_result"),
                ("calculation", "calc_result"),
                ("graph", "graph_result"),
                ("narrative", "narrative_result"),
            )
            if (final_state.get(key) or {}).get("evidence")
        ]

        elapsed = round((time.perf_counter() - start) * 1000, 2)

        return {
            "answer": final_state.get("final_answer", ""),
            "sources": self._collect_sources(final_state),
            "agents_used": agents_used,
            "routing": final_state.get("analysis", {}),
            "verification": final_state.get("verification", {}),
            "trace": final_state.get("trace", []),
            "latency_ms": elapsed,
            "model": self._model,
            "agent_results": {
                "xbrl": final_state.get("xbrl_result", {}),
                "calculation": final_state.get("calc_result", {}),
                "graph": final_state.get("graph_result", {}),
                "narrative": final_state.get("narrative_result", {}),
            },
        }
