"""
Knowledge Graph Builder
Builds an entity/relationship graph over the already-ingested 10-K chunks.

Design note
-----------
An LLM-driven extractor (Graphiti and friends) was the obvious choice here, but
this project runs entirely on a local 3B model. Asking llama3.2 to emit
structured triples over 15k chunks of dense legal prose produces a graph whose
errors are invisible until you query it — and a silently wrong graph is worse
than no graph.

So extraction is deterministic: a curated taxonomy of risk themes, peer
companies, geographies, and business segments, matched against chunk text with
word-boundary patterns. It is less impressive on paper and considerably more
trustworthy. Every edge carries the chunk IDs, pages, and sections it was
derived from, so any relationship the graph asserts can be traced back to the
filing text that produced it.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger

from src.config import settings

# ---------------------------------------------------------------------------
# Extraction taxonomies
# ---------------------------------------------------------------------------

# Risk themes: canonical name → surface forms that indicate the theme.
RISK_TAXONOMY: dict[str, list[str]] = {
    "supply_chain": [
        "supply chain", "single source", "sole source", "component shortage",
        "supplier", "outsourcing partner", "manufacturing partner",
        "contract manufacturer", "logistics",
    ],
    "semiconductor_supply": [
        "semiconductor", "foundry", "wafer", "chip shortage", "fabrication",
        "integrated circuit",
    ],
    "cybersecurity": [
        "cybersecurity", "cyberattack", "cyber attack", "data breach", "malware",
        "ransomware", "information security", "security incident",
        "unauthorized access",
    ],
    "privacy_regulation": [
        "privacy", "gdpr", "data protection", "personal data", "personal information",
    ],
    "regulatory_antitrust": [
        "antitrust", "competition law", "regulatory scrutiny", "european commission",
        "department of justice", "federal trade commission", "monopol",
    ],
    "intellectual_property": [
        "intellectual property", "patent", "infringement", "trade secret",
        "copyright",
    ],
    "foreign_exchange": [
        "foreign exchange", "currency fluctuation", "exchange rate",
        "foreign currency",
    ],
    "geopolitical": [
        "geopolitical", "trade tension", "tariff", "export control", "sanction",
        "trade restriction",
    ],
    "talent_retention": [
        "key personnel", "attract and retain", "highly skilled", "qualified personnel",
    ],
    "competition": [
        "intense competition", "highly competitive", "competitive pressure",
        "new entrants",
    ],
    "climate_environment": [
        "climate change", "greenhouse gas", "environmental regulation",
        "carbon", "renewable energy",
    ],
    "litigation": [
        "legal proceedings", "lawsuit", "litigation", "class action", "claims against",
    ],
    "tax": [
        "tax rate", "tax legislation", "tax authorities", "deferred tax",
    ],
    "customer_concentration": [
        "concentration of credit risk", "significant customer",
        "limited number of customers", "customer concentration",
    ],
    "product_defect": [
        "product defect", "warranty", "product recall", "quality problem",
    ],
    "artificial_intelligence": [
        "artificial intelligence", "machine learning", "generative ai",
        "large language model",
    ],
    "pandemic_health": [
        "covid-19", "pandemic", "public health",
    ],
    "macroeconomic": [
        "inflation", "recession", "economic conditions", "interest rate",
        "consumer spending",
    ],
    "data_center_capacity": [
        "data center", "capacity constraints", "infrastructure investment",
    ],
}

# Peer companies mentioned inside filings. Keys are canonical names; values are
# the surface forms to match.
COMPANY_ALIASES: dict[str, list[str]] = {
    "Apple": ["apple inc", "apple's", "apple "],
    "Microsoft": ["microsoft"],
    "Alphabet": ["alphabet", "google"],
    "Amazon": ["amazon"],
    "NVIDIA": ["nvidia"],
    "Meta": ["meta platforms", "facebook"],
    "Intel": ["intel corporation", "intel "],
    "AMD": ["advanced micro devices", "amd "],
    "Samsung": ["samsung"],
    "TSMC": ["taiwan semiconductor", "tsmc"],
    "Qualcomm": ["qualcomm"],
    "Broadcom": ["broadcom"],
    "IBM": ["international business machines", "ibm "],
    "Oracle": ["oracle"],
    "Salesforce": ["salesforce"],
    "Netflix": ["netflix"],
    "Tesla": ["tesla"],
    "Sony": ["sony"],
    "Cisco": ["cisco"],
    "Adobe": ["adobe"],
    "Walmart": ["walmart"],
    "Alibaba": ["alibaba"],
}

# Geographies that matter for exposure analysis.
GEOGRAPHIES: list[str] = [
    "China", "Taiwan", "Japan", "India", "Korea", "Singapore", "Vietnam",
    "Europe", "United Kingdom", "Ireland", "Germany", "France", "Israel",
    "Russia", "Ukraine", "Mexico", "Brazil", "Canada", "Australia",
]

# Business segments / products, keyed by the filer they belong to.
SEGMENTS: dict[str, list[str]] = {
    "AAPL": ["iPhone", "iPad", "Mac", "Wearables", "Services", "App Store"],
    "MSFT": [
        "Azure", "Office", "Windows", "LinkedIn", "Xbox", "Dynamics",
        "Surface", "Intelligent Cloud", "Productivity and Business Processes",
    ],
    "GOOGL": [
        "Google Cloud", "YouTube", "Android", "Google Search", "Google Play",
        "Other Bets", "Waymo",
    ],
    "AMZN": [
        "AWS", "Amazon Web Services", "Prime", "Alexa", "Kindle",
        "Advertising services", "Fulfillment",
    ],
    "NVDA": [
        "Data Center", "GeForce", "Gaming", "Professional Visualization",
        "Automotive", "CUDA", "Omniverse", "DGX",
    ],
}

# Chunks shorter than this are usually headings or table fragments — too little
# context to justify asserting a relationship.
MIN_CHUNK_CHARS = 150
# Cap evidence stored per edge so the graph file stays a sane size.
MAX_EVIDENCE_PER_EDGE = 5


def _compile(terms: Iterable[str]) -> re.Pattern:
    """Compile surface forms into one word-boundary-aware alternation."""
    escaped = [re.escape(t.strip()) for t in terms if t.strip()]
    # \b works for terms starting/ending with word chars; terms with trailing
    # spaces (e.g. "intel ") rely on the space itself as the boundary.
    return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")", re.IGNORECASE)


_RISK_PATTERNS = {name: _compile(terms) for name, terms in RISK_TAXONOMY.items()}
_COMPANY_PATTERNS = {name: _compile(terms) for name, terms in COMPANY_ALIASES.items()}

# Geographies and segments are matched one canonical term at a time. Matching a
# combined alternation and using the matched text would key nodes off whatever
# casing happened to appear in the filing, so "Data Center" and "data center"
# would become two different nodes.
_GEO_PATTERNS = {geo: _compile([geo]) for geo in GEOGRAPHIES}
_SEGMENT_PATTERNS = {
    ticker: {name: _compile([name]) for name in names}
    for ticker, names in SEGMENTS.items()
}

# Ticker → canonical company name, so a filer and a mention resolve to one node.
TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
}


class KnowledgeGraphBuilder:
    """
    Builds a citation-backed knowledge graph from processed chunk JSON files.

    Node types
    ----------
    Company, RiskFactor, Geography, Segment

    Edge types
    ----------
    DISCLOSES_RISK   Company → RiskFactor
    EXPOSED_TO       Company → Geography
    OPERATES_SEGMENT Company → Segment
    MENTIONS         Company → Company
    """

    def __init__(self, processed_dir: Optional[Path] = None) -> None:
        self._processed_dir = Path(processed_dir or settings.processed_dir)
        logger.info(f"KnowledgeGraphBuilder | source={self._processed_dir}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, output_path: Optional[Path] = None) -> dict:
        """
        Walk every processed chunk file and emit the graph.

        Returns
        -------
        dict  – {nodes, edges, stats} (also written to *output_path*)
        """
        start = time.perf_counter()
        chunk_files = sorted(self._processed_dir.glob("*_chunks.json"))
        if not chunk_files:
            logger.error(f"No *_chunks.json found in {self._processed_dir}")
            return {"nodes": [], "edges": [], "stats": {}}

        nodes: dict[str, dict] = {}
        # (source, type, target, ticker, year) → edge record
        edges: dict[tuple, dict] = {}
        chunks_scanned = 0

        for path in chunk_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Skipping unreadable {path.name}: {exc}")
                continue

            ticker = str(payload.get("ticker", "")).upper()
            year = str(payload.get("year", ""))
            company = TICKER_TO_COMPANY.get(ticker, ticker)
            if not company:
                logger.warning(f"Skipping {path.name}: no ticker")
                continue

            self._add_node(nodes, company, "Company", ticker=ticker)

            for chunk in payload.get("chunks", []):
                text = chunk.get("text", "") or ""
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                chunks_scanned += 1
                self._extract_from_chunk(
                    nodes, edges, chunk, text, company, ticker, year
                )

            logger.info(f"Processed {path.name} ({ticker} {year})")

        graph = {
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "stats": self._compute_stats(nodes, edges, chunks_scanned, chunk_files),
        }
        graph["stats"]["build_time_seconds"] = round(time.perf_counter() - start, 2)

        out = Path(output_path or settings.graph_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(graph, indent=2), encoding="utf-8")

        logger.success(
            f"Knowledge graph built: {len(graph['nodes'])} nodes, "
            f"{len(graph['edges'])} edges from {chunks_scanned} chunks → {out}"
        )
        return graph

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_from_chunk(
        self,
        nodes: dict[str, dict],
        edges: dict[tuple, dict],
        chunk: dict,
        text: str,
        company: str,
        ticker: str,
        year: str,
    ) -> None:
        """Pull every entity/relationship this chunk supports."""
        citation = {
            "chunk_id": chunk.get("chunk_id", ""),
            "page_num": chunk.get("page_num"),
            "section": chunk.get("section", ""),
            "ticker": ticker,
            "year": year,
            "text_preview": text[:240].replace("\n", " "),
        }

        # --- Risk themes ---
        for risk, pattern in _RISK_PATTERNS.items():
            if pattern.search(text):
                self._add_node(nodes, risk, "RiskFactor")
                self._add_edge(
                    edges, company, "DISCLOSES_RISK", risk, ticker, year, citation
                )

        # --- Geographic exposure ---
        for geo, pattern in _GEO_PATTERNS.items():
            if pattern.search(text):
                self._add_node(nodes, geo, "Geography")
                self._add_edge(
                    edges, company, "EXPOSED_TO", geo, ticker, year, citation
                )

        # --- Business segments (only the filer's own segment vocabulary) ---
        for seg, pattern in _SEGMENT_PATTERNS.get(ticker, {}).items():
            if pattern.search(text):
                self._add_node(nodes, seg, "Segment", owner=ticker)
                self._add_edge(
                    edges, company, "OPERATES_SEGMENT", seg, ticker, year, citation
                )

        # --- Peer company mentions ---
        for peer, pattern in _COMPANY_PATTERNS.items():
            if peer == company:
                continue
            if pattern.search(text):
                self._add_node(nodes, peer, "Company")
                self._add_edge(
                    edges, company, "MENTIONS", peer, ticker, year, citation
                )

    # ------------------------------------------------------------------
    # Graph primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _add_node(
        nodes: dict[str, dict], name: str, node_type: str, **attrs: Any
    ) -> None:
        node_id = f"{node_type}:{name}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "name": name,
                "type": node_type,
                **attrs,
            }
        else:
            nodes[node_id].update({k: v for k, v in attrs.items() if v is not None})

    @staticmethod
    def _add_edge(
        edges: dict[tuple, dict],
        source: str,
        edge_type: str,
        target: str,
        ticker: str,
        year: str,
        citation: dict,
    ) -> None:
        # An edge is scoped to a company-year: "Apple disclosed supply chain risk
        # in FY2023" is a different assertion from the same risk in FY2022.
        key = (source, edge_type, target, ticker, year)
        edge = edges.get(key)
        if edge is None:
            edge = {
                "source": source,
                "type": edge_type,
                "target": target,
                "ticker": ticker,
                "year": year,
                "weight": 0,
                "evidence": [],
            }
            edges[key] = edge

        edge["weight"] += 1
        if len(edge["evidence"]) < MAX_EVIDENCE_PER_EDGE:
            edge["evidence"].append(citation)

    @staticmethod
    def _compute_stats(
        nodes: dict[str, dict],
        edges: dict[tuple, dict],
        chunks_scanned: int,
        chunk_files: list[Path],
    ) -> dict:
        node_types: dict[str, int] = defaultdict(int)
        for n in nodes.values():
            node_types[n["type"]] += 1

        edge_types: dict[str, int] = defaultdict(int)
        for e in edges.values():
            edge_types[e["type"]] += 1

        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "chunks_scanned": chunks_scanned,
            "documents": len(chunk_files),
        }

    # ------------------------------------------------------------------
    # Optional Neo4j export
    # ------------------------------------------------------------------

    def export_to_neo4j(self, graph: Optional[dict] = None) -> bool:
        """
        Mirror the graph into Neo4j for browsing in the Neo4j UI.

        Entirely optional — the graph agent queries the JSON file directly, so
        the system works with no database running. Returns False if the driver
        is missing or the connection fails.
        """
        if graph is None:
            path = Path(settings.graph_path)
            if not path.exists():
                logger.error(f"No graph at {path}; run build() first")
                return False
            graph = json.loads(path.read_text(encoding="utf-8"))

        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.error("neo4j driver not installed — run: pip install neo4j")
            return False

        try:
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")

                for node in graph["nodes"]:
                    session.run(
                        f"CREATE (n:{node['type']} {{id: $id, name: $name}})",
                        id=node["id"],
                        name=node["name"],
                    )

                for edge in graph["edges"]:
                    session.run(
                        f"""
                        MATCH (a {{name: $source}}), (b {{name: $target}})
                        CREATE (a)-[r:{edge['type']} {{
                            ticker: $ticker, year: $year, weight: $weight,
                            evidence: $evidence
                        }}]->(b)
                        """,
                        source=edge["source"],
                        target=edge["target"],
                        ticker=edge["ticker"],
                        year=edge["year"],
                        weight=edge["weight"],
                        evidence=json.dumps(edge["evidence"]),
                    )
            driver.close()
            logger.success(
                f"Exported {len(graph['nodes'])} nodes / {len(graph['edges'])} "
                f"edges to Neo4j at {settings.neo4j_uri}"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Neo4j export failed: {exc}")
            return False


def main() -> None:
    """CLI entry point: python -m src.ingestion.graph_builder"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build()
    stats = graph.get("stats", {})
    print("\n" + "=" * 60)
    print("  KNOWLEDGE GRAPH BUILD")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key:<22}: {value}")
    print("=" * 60 + "\n")

    if settings.neo4j_enabled:
        builder.export_to_neo4j(graph)


if __name__ == "__main__":
    main()
