"""
Graph Agent
Answers relationship questions by traversing the knowledge graph rather than
by text similarity.

"Which companies share supply chain risk exposure?" is not a retrieval problem.
No single chunk contains the answer — it only exists in the intersection of
what five separate filings disclose. Embedding search cannot compute an
intersection; a graph traversal can.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
from loguru import logger

from src.config import settings
from src.ingestion.graph_builder import RISK_TAXONOMY, TICKER_TO_COMPANY

# Free-text phrase → canonical risk theme, for question routing.
RISK_ALIASES: dict[str, str] = {
    "supply chain": "supply_chain",
    "supplier": "supply_chain",
    "sourcing": "supply_chain",
    "semiconductor": "semiconductor_supply",
    "chip": "semiconductor_supply",
    "foundry": "semiconductor_supply",
    "cyber": "cybersecurity",
    "cybersecurity": "cybersecurity",
    "security breach": "cybersecurity",
    "hacking": "cybersecurity",
    "privacy": "privacy_regulation",
    "gdpr": "privacy_regulation",
    "data protection": "privacy_regulation",
    "antitrust": "regulatory_antitrust",
    "regulation": "regulatory_antitrust",
    "regulatory": "regulatory_antitrust",
    "monopoly": "regulatory_antitrust",
    "intellectual property": "intellectual_property",
    "patent": "intellectual_property",
    "currency": "foreign_exchange",
    "foreign exchange": "foreign_exchange",
    "exchange rate": "foreign_exchange",
    "geopolitical": "geopolitical",
    "tariff": "geopolitical",
    "trade tension": "geopolitical",
    "export control": "geopolitical",
    "talent": "talent_retention",
    "personnel": "talent_retention",
    "hiring": "talent_retention",
    "competition": "competition",
    "competitive": "competition",
    "climate": "climate_environment",
    "environmental": "climate_environment",
    "litigation": "litigation",
    "lawsuit": "litigation",
    "legal proceeding": "litigation",
    "tax": "tax",
    "customer concentration": "customer_concentration",
    "product defect": "product_defect",
    "warranty": "product_defect",
    "recall": "product_defect",
    "artificial intelligence": "artificial_intelligence",
    "ai ": "artificial_intelligence",
    "machine learning": "artificial_intelligence",
    "pandemic": "pandemic_health",
    "covid": "pandemic_health",
    "inflation": "macroeconomic",
    "recession": "macroeconomic",
    "macroeconomic": "macroeconomic",
    "interest rate": "macroeconomic",
    "data center": "data_center_capacity",
}


class GraphAgent:
    """
    Queries the citation-backed knowledge graph built by KnowledgeGraphBuilder.

    Loads the graph JSON into a networkx MultiDiGraph. Neo4j is supported as an
    optional export target for visual browsing, but is never required — keeping
    the query path in-process means the agent has no runtime dependency on a
    database being up.
    """

    def __init__(self, graph_path: Optional[Path] = None) -> None:
        self._graph_path = Path(graph_path or settings.graph_path)
        self._graph: Optional[nx.MultiDiGraph] = None
        self._raw: dict = {"nodes": [], "edges": [], "stats": {}}
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._graph_path.exists():
            logger.warning(
                f"No knowledge graph at {self._graph_path} — run: "
                f"python -m src.ingestion.graph_builder"
            )
            self._graph = nx.MultiDiGraph()
            return

        try:
            self._raw = json.loads(self._graph_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to read knowledge graph: {exc}")
            self._graph = nx.MultiDiGraph()
            return

        g = nx.MultiDiGraph()
        for node in self._raw.get("nodes", []):
            g.add_node(node["name"], **node)
        for edge in self._raw.get("edges", []):
            g.add_edge(
                edge["source"],
                edge["target"],
                key=f"{edge['type']}:{edge['ticker']}:{edge['year']}",
                **edge,
            )

        self._graph = g
        logger.info(
            f"GraphAgent loaded {g.number_of_nodes()} nodes / "
            f"{g.number_of_edges()} edges from {self._graph_path}"
        )

    @property
    def is_available(self) -> bool:
        return self._graph is not None and self._graph.number_of_nodes() > 0

    @property
    def stats(self) -> dict:
        return self._raw.get("stats", {})

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_risk(text: str) -> Optional[str]:
        """Map a free-text phrase to a canonical risk theme (longest match wins)."""
        lowered = text.lower()
        best: Optional[str] = None
        best_len = 0
        for alias, canonical in RISK_ALIASES.items():
            if alias in lowered and len(alias) > best_len:
                best = canonical
                best_len = len(alias)
        # A question may also name the theme directly ("customer_concentration")
        if best is None:
            for theme in RISK_TAXONOMY:
                if theme in lowered:
                    return theme
        return best

    def _edges(self, edge_type: Optional[str] = None, year: Optional[str] = None):
        """Iterate edge attribute dicts, optionally filtered."""
        if self._graph is None:
            return
        for _, _, data in self._graph.edges(data=True):
            if edge_type and data.get("type") != edge_type:
                continue
            if year and str(data.get("year")) != str(year):
                continue
            yield data

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def shared_risk(
        self, risk: str, year: Optional[str] = None, min_companies: int = 2
    ) -> dict:
        """
        Which companies disclose *risk*, ranked by how heavily they discuss it.

        Weight is the number of distinct chunks in that filing matching the
        theme — a rough but honest proxy for emphasis.
        """
        rows = [
            e
            for e in self._edges("DISCLOSES_RISK", year)
            if e.get("target") == risk
        ]
        by_company: dict[str, dict] = {}
        for e in rows:
            entry = by_company.setdefault(
                e["source"],
                {"company": e["source"], "weight": 0, "years": [], "evidence": []},
            )
            entry["weight"] += e.get("weight", 0)
            entry["years"].append(e.get("year"))
            entry["evidence"].extend(e.get("evidence", [])[:2])

        companies = sorted(by_company.values(), key=lambda r: -r["weight"])
        for c in companies:
            c["years"] = sorted(set(c["years"]))
            c["evidence"] = c["evidence"][:3]

        return {
            "risk": risk,
            "companies": companies,
            "shared": len(companies) >= min_companies,
            "num_companies": len(companies),
        }

    def risk_profile(self, ticker: str, year: Optional[str] = None) -> dict:
        """Every risk theme a company discloses, ranked by emphasis."""
        company = TICKER_TO_COMPANY.get(ticker.upper(), ticker)
        rows = [
            e
            for e in self._edges("DISCLOSES_RISK", year)
            if e.get("source") == company
        ]
        by_risk: dict[str, dict] = {}
        for e in rows:
            entry = by_risk.setdefault(
                e["target"], {"risk": e["target"], "weight": 0, "evidence": []}
            )
            entry["weight"] += e.get("weight", 0)
            entry["evidence"].extend(e.get("evidence", [])[:1])

        return {
            "company": company,
            "ticker": ticker.upper(),
            "risks": sorted(by_risk.values(), key=lambda r: -r["weight"]),
        }

    def common_risks(
        self, tickers: list[str], year: Optional[str] = None
    ) -> dict:
        """
        Risk themes disclosed by *every* named company — a set intersection the
        retrieval pipeline structurally cannot compute.
        """
        companies = [TICKER_TO_COMPANY.get(t.upper(), t.upper()) for t in tickers]
        per_company: dict[str, dict[str, int]] = defaultdict(dict)

        for e in self._edges("DISCLOSES_RISK", year):
            if e["source"] in companies:
                per_company[e["source"]][e["target"]] = (
                    per_company[e["source"]].get(e["target"], 0) + e.get("weight", 0)
                )

        present = [c for c in companies if c in per_company]
        if not present:
            return {"companies": companies, "shared_risks": [], "unique_risks": {}}

        shared = set(per_company[present[0]])
        for c in present[1:]:
            shared &= set(per_company[c])

        shared_rows = [
            {
                "risk": risk,
                "total_weight": sum(per_company[c].get(risk, 0) for c in present),
                "by_company": {c: per_company[c].get(risk, 0) for c in present},
            }
            for risk in shared
        ]
        shared_rows.sort(key=lambda r: -r["total_weight"])

        # Risks unique to one company are often the more interesting signal:
        # everyone discloses litigation risk, only NVIDIA discloses customer
        # concentration.
        unique: dict[str, list[str]] = {}
        for company in present:
            others: set[str] = set()
            for other in present:
                if other != company:
                    others |= set(per_company[other])

            only = sorted(set(per_company[company]) - others)
            if only:
                unique[company] = only

        return {
            "companies": present,
            "shared_risks": shared_rows,
            "unique_risks": unique,
        }

    def geographic_exposure(
        self, geography: Optional[str] = None, ticker: Optional[str] = None,
        year: Optional[str] = None,
    ) -> dict:
        """Which companies name which geographies in their filings."""
        rows = list(self._edges("EXPOSED_TO", year))
        if geography:
            rows = [e for e in rows if e["target"].lower() == geography.lower()]
        if ticker:
            company = TICKER_TO_COMPANY.get(ticker.upper(), ticker.upper())
            rows = [e for e in rows if e["source"] == company]

        grouped: dict[str, dict] = {}
        for e in rows:
            key = f"{e['source']}→{e['target']}"
            entry = grouped.setdefault(
                key,
                {
                    "company": e["source"],
                    "geography": e["target"],
                    "weight": 0,
                    "evidence": [],
                },
            )
            entry["weight"] += e.get("weight", 0)
            entry["evidence"].extend(e.get("evidence", [])[:1])

        return {
            "geography": geography,
            "exposures": sorted(grouped.values(), key=lambda r: -r["weight"]),
        }

    def segments(self, ticker: str, year: Optional[str] = None) -> dict:
        """Business segments and products named in a company's filing."""
        company = TICKER_TO_COMPANY.get(ticker.upper(), ticker.upper())
        rows = [
            e for e in self._edges("OPERATES_SEGMENT", year) if e["source"] == company
        ]
        grouped: dict[str, dict] = {}
        for e in rows:
            entry = grouped.setdefault(
                e["target"], {"segment": e["target"], "weight": 0, "evidence": []}
            )
            entry["weight"] += e.get("weight", 0)
            entry["evidence"].extend(e.get("evidence", [])[:1])

        return {
            "company": company,
            "segments": sorted(grouped.values(), key=lambda r: -r["weight"]),
        }

    # ------------------------------------------------------------------
    # Agent entry point
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        tickers: Optional[list[str]] = None,
        year: Optional[str] = None,
    ) -> dict:
        """
        Route a relationship question to the right traversal.

        Returns
        -------
        dict  – {agent, query_type, result, evidence, summary, citations}
        """
        start = time.perf_counter()
        if not self.is_available:
            return {
                "agent": "graph",
                "query_type": "unavailable",
                "result": {},
                "evidence": "",
                "summary": (
                    "The knowledge graph has not been built. Run: "
                    "python -m src.ingestion.graph_builder"
                ),
                "citations": [],
                "latency_ms": 0.0,
            }

        tickers = [t.upper() for t in (tickers or [])]
        lowered = question.lower()
        risk = self.resolve_risk(question)

        # "which risks does NVIDIA disclose that the others do not" names one
        # company but is a question about the whole corpus — the answer is a set
        # difference, so every company has to be in the comparison.
        asks_uniqueness = any(
            phrase in lowered
            for phrase in (
                "do not", "don't", "does not", "unique", "only ", "others",
                "other companies", "no one else", "nobody else", "distinct",
            )
        )

        # --- Pick the traversal ---
        if len(tickers) == 1 and asks_uniqueness:
            query_type = "common_risks"
            all_tickers = list(TICKER_TO_COMPANY.keys())
            result = self.common_risks(all_tickers, year)
            evidence, citations = self._format_common_risks(result)

        elif len(tickers) >= 2:
            query_type = "common_risks"
            result = self.common_risks(tickers, year)
            evidence, citations = self._format_common_risks(result)

        elif risk and (
            "which compan" in lowered
            or "share" in lowered
            or "shared" in lowered
            or "both" in lowered
            or "common" in lowered
            or not tickers
        ):
            query_type = "shared_risk"
            result = self.shared_risk(risk, year)
            evidence, citations = self._format_shared_risk(result)

        elif tickers and ("segment" in lowered or "product" in lowered or "business" in lowered):
            query_type = "segments"
            result = self.segments(tickers[0], year)
            evidence, citations = self._format_segments(result)

        elif tickers:
            query_type = "risk_profile"
            result = self.risk_profile(tickers[0], year)
            evidence, citations = self._format_risk_profile(result)

        else:
            query_type = "geographic_exposure"
            result = self.geographic_exposure(year=year)
            evidence, citations = self._format_exposure(result)

        logger.info(f"GraphAgent query_type={query_type} tickers={tickers} risk={risk}")

        return {
            "agent": "graph",
            "query_type": query_type,
            "result": result,
            "evidence": evidence,
            "direct_answer": self._direct_answer(query_type, result),
            "summary": f"Traversed the knowledge graph ({query_type}).",
            "citations": citations,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    # ------------------------------------------------------------------
    # Deterministic prose
    # ------------------------------------------------------------------

    @staticmethod
    def _join(names: list[str]) -> str:
        """Join names as English prose: 'A', 'A and B', 'A, B and C'."""
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return f"{', '.join(names[:-1])} and {names[-1]}"

    def _direct_answer(self, query_type: str, result: dict) -> str:
        """
        Render the traversal result as prose, without an LLM.

        A graph result is a complete set — every company that discloses a risk,
        every risk two companies share. Handing that set to a small model to
        restate reliably loses members of it, which turns an exact answer into
        an approximate one. The set is written out directly instead.
        """
        if query_type == "shared_risk":
            rows = result.get("companies", [])
            if not rows:
                return f"No company in the corpus discloses '{result.get('risk')}'."
            risk = str(result.get("risk", "")).replace("_", " ")
            parts = [
                f"{r['company']} ({r['weight']} passages across "
                f"FY{'/FY'.join(r['years'])})"
                for r in rows
            ]
            return (
                f"{len(rows)} of the companies in the corpus disclose {risk} risk, "
                f"ranked by how much filing text they devote to it: "
                + "; ".join(parts)
                + "."
            )

        if query_type == "common_risks":
            companies = result.get("companies", [])
            shared = result.get("shared_risks", [])
            unique = result.get("unique_risks", {})
            named = self._join(companies)
            quantifier = "both" if len(companies) == 2 else f"all {len(companies)}"

            if not shared:
                return f"{named} share no risk themes in common."
            lines = [
                f"{named} {quantifier} disclose {len(shared)} risk themes: "
                + ", ".join(r["risk"].replace("_", " ") for r in shared)
                + "."
            ]
            for company, risks in unique.items():
                lines.append(
                    f"Disclosed only by {company}: "
                    + ", ".join(r.replace("_", " ") for r in risks)
                    + "."
                )
            return " ".join(lines)

        if query_type == "risk_profile":
            rows = result.get("risks", [])
            if not rows:
                return f"No risk disclosures found for {result.get('company')}."
            top = ", ".join(
                f"{r['risk'].replace('_', ' ')} ({r['weight']})" for r in rows[:8]
            )
            return (
                f"{result.get('company')} discloses {len(rows)} risk themes, "
                f"ranked by volume of filing text: {top}."
            )

        if query_type == "segments":
            rows = result.get("segments", [])
            if not rows:
                return f"No segments identified for {result.get('company')}."
            return (
                f"{result.get('company')} names {len(rows)} segments/products: "
                + ", ".join(f"{r['segment']} ({r['weight']} mentions)" for r in rows)
                + "."
            )

        if query_type == "geographic_exposure":
            rows = result.get("exposures", [])
            if not rows:
                return "No geographic exposure recorded."
            return "Geographic exposure named across filings: " + "; ".join(
                f"{r['company']} → {r['geography']} ({r['weight']})" for r in rows[:15]
            ) + "."

        return ""

    # ------------------------------------------------------------------
    # Evidence formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _cite(evidence: list[dict]) -> list[dict]:
        return [
            {
                "ticker": e.get("ticker"),
                "year": e.get("year"),
                "section": e.get("section"),
                "page_num": e.get("page_num"),
                "chunk_id": e.get("chunk_id"),
                "text_preview": e.get("text_preview", ""),
            }
            for e in evidence
        ]

    # Counts are stated explicitly in the evidence, not left implicit in the
    # number of lines. The traversal computes them deterministically, so they are
    # authoritative — but the verifier can only trace a figure it can actually
    # find, and would otherwise flag "5 companies" as an unsupported number.

    def _format_shared_risk(self, result: dict) -> tuple[str, list[dict]]:
        lines = [
            f"Risk theme: {result['risk']}",
            f"Companies disclosing it: {result['num_companies']}",
        ]
        citations: list[dict] = []
        for row in result["companies"]:
            lines.append(
                f"  {row['company']}: disclosed in FY{'/FY'.join(row['years'])}, "
                f"{row['weight']} matching passage(s)"
            )
            for ev in row["evidence"][:1]:
                lines.append(
                    f"      └─ {ev.get('ticker')} FY{ev.get('year')} "
                    f"p.{ev.get('page_num')} [{ev.get('section')}]: "
                    f"\"{ev.get('text_preview', '')[:160]}...\""
                )
            citations.extend(self._cite(row["evidence"][:2]))
        return "\n".join(lines), citations

    def _format_common_risks(self, result: dict) -> tuple[str, list[dict]]:
        lines = [f"Companies compared: {', '.join(result['companies'])}"]
        lines.append(
            f"Number of risk themes disclosed by all "
            f"{len(result['companies'])} companies: {len(result['shared_risks'])}"
        )
        for row in result["shared_risks"][:12]:
            by_co = ", ".join(
                f"{c} {w}" for c, w in sorted(row["by_company"].items(), key=lambda kv: -kv[1])
            )
            lines.append(f"  {row['risk']} (passages — {by_co})")
        if result.get("unique_risks"):
            lines.append("Risks unique to a single company:")
            for company, risks in result["unique_risks"].items():
                lines.append(f"  {company}: {', '.join(risks)}")
        return "\n".join(lines), []

    def _format_risk_profile(self, result: dict) -> tuple[str, list[dict]]:
        lines = [
            f"Risk profile for {result['company']} ({result['ticker']}):",
            f"Number of risk themes disclosed: {len(result['risks'])}",
        ]
        citations: list[dict] = []
        for row in result["risks"][:12]:
            lines.append(f"  {row['risk']}: {row['weight']} matching passage(s)")
            citations.extend(self._cite(row["evidence"][:1]))
        return "\n".join(lines), citations

    def _format_segments(self, result: dict) -> tuple[str, list[dict]]:
        lines = [
            f"Segments and products named by {result['company']}:",
            f"Number of segments identified: {len(result['segments'])}",
        ]
        citations: list[dict] = []
        for row in result["segments"]:
            lines.append(f"  {row['segment']}: {row['weight']} mention(s)")
            citations.extend(self._cite(row["evidence"][:1]))
        return "\n".join(lines), citations

    def _format_exposure(self, result: dict) -> tuple[str, list[dict]]:
        lines = ["Geographic exposure named across filings:"]
        citations: list[dict] = []
        for row in result["exposures"][:20]:
            lines.append(
                f"  {row['company']} → {row['geography']}: {row['weight']} mention(s)"
            )
            citations.extend(self._cite(row["evidence"][:1]))
        return "\n".join(lines), citations
