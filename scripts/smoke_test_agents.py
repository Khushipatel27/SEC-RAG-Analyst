"""
smoke_test_agents.py
────────────────────
Checks each v2 specialist in isolation, then the supervisor end to end.

Unlike the unit tests (which mock everything and run offline), this exercises
the real dependencies — SEC EDGAR, the knowledge graph file, Ollama — and tells
you which one is broken when something does not work.

Run:  python scripts/smoke_test_agents.py
      python scripts/smoke_test_agents.py --full   (also runs the LLM steps)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()  # keep the output readable — this script prints its own status

PASS, FAIL, SKIP = "  [PASS]", "  [FAIL]", "  [SKIP]"
results: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"{PASS if condition else FAIL} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition))
    return condition


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


full = "--full" in sys.argv

# ---------------------------------------------------------------------------
section("1. XBRL agent — needs internet (SEC EDGAR)")
# ---------------------------------------------------------------------------
from src.agents.xbrl_agent import XBRLAgent  # noqa: E402

xbrl = XBRLAgent()

check(
    "resolve_metric('R&D spending')",
    XBRLAgent.resolve_metric("R&D spending") == "research_and_development",
)
check(
    "resolve_metrics finds both sides of a ratio",
    set(XBRLAgent.resolve_metrics("R&D as a percentage of revenue"))
    == {"research_and_development", "revenue"},
)

t = time.perf_counter()
fact = xbrl.get_metric("AAPL", "revenue", 2023)
elapsed = time.perf_counter() - t

if check("AAPL FY2023 revenue fetched from SEC", fact is not None,
         f"{elapsed:.1f}s"):
    # Apple's FY2023 revenue is 383,285,000,000 — a known, checkable figure.
    check(
        "value matches the filed figure ($383.29B)",
        abs(fact.value - 383_285_000_000) < 1_000_000,
        fact.value_formatted,
    )
    check(
        "citation carries a real accession number",
        fact.accession.startswith("0000320193-23"),
        fact.accession,
    )

missing = xbrl.get_metric("AMZN", "research_and_development", 2023)
check(
    "unavailable metric returns None rather than guessing",
    missing is None,
    "Amazon does not tag R&D in XBRL — expected",
)

# ---------------------------------------------------------------------------
section("2. Calculation agent — pure Python, no dependencies")
# ---------------------------------------------------------------------------
from src.agents.calc_agent import CalculationAgent  # noqa: E402

calc = CalculationAgent()
nvda_22 = xbrl.get_metric("NVDA", "research_and_development", 2022)
nvda_23 = xbrl.get_metric("NVDA", "research_and_development", 2023)

if nvda_22 and nvda_23:
    growth = calc.growth_rate(nvda_22, nvda_23)
    check(
        "NVDA R&D growth FY22→FY23 == +39.31%",
        growth is not None and abs(growth.value - 39.3128) < 0.01,
        growth.value_formatted if growth else "none",
    )
    check(
        "calculation carries both input citations",
        growth is not None and len(growth.inputs) == 2,
    )
else:
    print(f"{SKIP} calculation checks — XBRL unavailable")

# ---------------------------------------------------------------------------
section("3. Graph agent — needs data/graph/knowledge_graph.json")
# ---------------------------------------------------------------------------
from src.agents.graph_agent import GraphAgent  # noqa: E402

graph = GraphAgent()

if check(
    "knowledge graph loaded",
    graph.is_available,
    f"{graph.stats.get('num_nodes', 0)} nodes / {graph.stats.get('num_edges', 0)} edges"
    if graph.is_available
    else "run: python -m src.ingestion.graph_builder",
):
    shared = graph.shared_risk("supply_chain")
    check(
        "all 5 companies found for supply_chain risk",
        shared["num_companies"] == 5,
        f"{shared['num_companies']} companies",
    )

    common = graph.common_risks(["NVDA", "MSFT"], year="2023")
    check(
        "NVDA/MSFT intersection computed",
        len(common["shared_risks"]) > 0,
        f"{len(common['shared_risks'])} shared themes",
    )
    check(
        "NVDA unique risks identified",
        "customer_concentration" in common["unique_risks"].get("NVIDIA", []),
    )

    answer = graph.answer("Which companies share supply chain risk exposure?")
    check(
        "deterministic answer names every company",
        all(
            name in answer["direct_answer"]
            for name in ("NVIDIA", "Apple", "Amazon", "Microsoft", "Alphabet")
        ),
    )

# ---------------------------------------------------------------------------
section("4. Verification agent — deterministic layer (no LLM)")
# ---------------------------------------------------------------------------
from src.agents.verification_agent import VerificationAgent  # noqa: E402

verifier = VerificationAgent()
evidence = "AAPL FY2023 Revenue: $383.29B (accession 0000320193-23-000106)"

score, _, _, _ = verifier.check_numeric_grounding(
    "Apple's FY2023 revenue was $383.29B.", evidence
)
check("grounded figure passes", score == 1.0)

score, unsupported, _, _ = verifier.check_numeric_grounding(
    "Apple's FY2023 revenue was $412.50B.", evidence
)
check("hallucinated figure is caught", score == 0.0 and bool(unsupported),
      f"flagged {unsupported}")

score, _, _, _ = verifier.check_numeric_grounding(
    "Revenue was $383.29B.", "Revenue was 383,285,000,000 dollars."
)
check("scaled notation matches raw number", score == 1.0)

_, _, _, total = verifier.check_numeric_grounding(
    "In 2023 Apple filed its 10-K.", evidence
)
check("form types and years are not treated as claims", total == 0)

# ---------------------------------------------------------------------------
section("5. Supervisor routing — no LLM needed")
# ---------------------------------------------------------------------------
from src.agents.supervisor import analyse_question  # noqa: E402

cases = [
    ("What was Apple's revenue in 2023?", "AAPL", "revenue"),
    ("How much faster did NVIDIA's R&D grow vs Microsoft's?", "NVDA", "research_and_development"),
]
for question, ticker, metric in cases:
    a = analyse_question(question)
    check(
        f"routes {question[:40]!r}",
        ticker in a["tickers"] and metric in a["metrics"],
        f"tickers={a['tickers']} metrics={a['metrics']}",
    )

a = analyse_question("Which companies share supply chain risk exposure?")
check("relationship question detected", a["wants_relationship"] and a["risk"] == "supply_chain")

# ---------------------------------------------------------------------------
if full:
    section("6. Supervisor end to end — needs Ollama running")
    from src.agents.supervisor import SupervisorAgent  # noqa: E402

    sup = SupervisorAgent(pipeline=None)  # no v1 pipeline: XBRL/graph paths only

    for question, expected_agent in [
        ("What was Apple's revenue in 2023?", "xbrl"),
        ("How much faster did NVIDIA's R&D grow vs Microsoft's?", "calculation"),
        ("Which companies share supply chain risk exposure?", "graph"),
    ]:
        t = time.perf_counter()
        r = sup.ask(question)
        check(
            f"{question[:45]!r} → {expected_agent}",
            expected_agent in r["agents_used"],
            f"{r['agents_used']} · {r['verification'].get('verdict')} · "
            f"{time.perf_counter() - t:.1f}s",
        )
        print(f"         {r['answer'][:160].replace(chr(10), ' ')}")
else:
    print(f"\n{SKIP} end-to-end supervisor checks — re-run with --full "
          f"(needs Ollama running)")

# ---------------------------------------------------------------------------
passed = sum(1 for _, ok in results if ok)
total_checks = len(results)
print(f"\n{'=' * 70}")
print(f"  {passed}/{total_checks} checks passed")
print(f"{'=' * 70}\n")
sys.exit(0 if passed == total_checks else 1)
