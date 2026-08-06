"""
build_agentic_eval.py
─────────────────────
Generates data/eval/agentic_questions.json — the v2 evaluation set.

The v1 suite is 20 questions that a single retrieval pass can answer. This set
deliberately targets what that architecture cannot do:

  multi_hop_numeric   requires two exact figures and arithmetic between them
  cross_company       requires figures from two different filings
  ratio               requires a derived quantity, not a stated one
  relationship        requires a set operation across all five companies
  qualitative         plain narrative, included as a control — v2 should not
                      beat v1 here, and if it does the comparison is suspect

Reference answers for the numeric questions are generated from SEC XBRL facts
rather than written by hand, so the ground truth is the filing itself. Note the
consequence honestly: on numeric questions v2 shares a data source with the
reference answer, and is expected to win. The relationship and qualitative
questions carry no such advantage, which is why they are in the set.

Run:  python scripts/build_agentic_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.calc_agent import CalculationAgent  # noqa: E402
from src.agents.xbrl_agent import XBRLAgent  # noqa: E402

OUT_PATH = ROOT / "data" / "eval" / "agentic_questions.json"

xbrl = XBRLAgent()
calc = CalculationAgent()


def fact(ticker: str, metric: str, year: int):
    f = xbrl.get_metric(ticker, metric, year)
    if f is None:
        raise SystemExit(f"Missing XBRL fact: {ticker} {metric} FY{year}")
    return f


def growth_q(ticker: str, metric: str, label: str, y0: int, y1: int, difficulty: str) -> dict:
    a, b = fact(ticker, metric, y0), fact(ticker, metric, y1)
    g = calc.growth_rate(a, b)
    return {
        "question": f"How much did {label} change for {ticker} between FY{y0} and FY{y1}?",
        "reference_answer": (
            f"{ticker} {label} went from {a.value_formatted} in FY{y0} to "
            f"{b.value_formatted} in FY{y1}, a change of {g.value:+.2f}%."
        ),
        "ticker": ticker,
        "year": str(y1),
        "question_type": "multi_hop_numeric",
        "difficulty": difficulty,
        "expected_agents": ["xbrl", "calculation"],
    }


def ratio_q(ticker: str, metric: str, label: str, year: int, difficulty: str) -> dict:
    num, den = fact(ticker, metric, year), fact(ticker, "revenue", year)
    r = calc.ratio(num, den, as_percent=True)
    return {
        "question": f"What was {ticker}'s {label} as a percentage of revenue in FY{year}?",
        "reference_answer": (
            f"{ticker} spent {num.value_formatted} on {label} against revenue of "
            f"{den.value_formatted} in FY{year}, or {r.value:.2f}% of revenue."
        ),
        "ticker": ticker,
        "year": str(year),
        "question_type": "ratio",
        "difficulty": difficulty,
        "expected_agents": ["xbrl", "calculation"],
    }


def cross_company_q(
    t0: str, t1: str, metric: str, label: str, y0: int, y1: int
) -> dict:
    a0, a1 = fact(t0, metric, y0), fact(t0, metric, y1)
    b0, b1 = fact(t1, metric, y0), fact(t1, metric, y1)
    g0, g1 = calc.growth_rate(a0, a1), calc.growth_rate(b0, b1)
    faster, slower = (g0, g1) if g0.value >= g1.value else (g1, g0)
    gap = abs(g0.value - g1.value)
    return {
        "question": f"How much faster did {t0}'s {label} grow than {t1}'s between FY{y0} and FY{y1}?",
        "reference_answer": (
            f"{t0} {label} grew {g0.value:+.2f}% (from {a0.value_formatted} to "
            f"{a1.value_formatted}) while {t1} grew {g1.value:+.2f}% (from "
            f"{b0.value_formatted} to {b1.value_formatted}). "
            f"{faster.ticker} grew faster by {gap:.2f} percentage points."
        ),
        "ticker": None,
        "year": str(y1),
        "question_type": "cross_company",
        "difficulty": "hard",
        "expected_agents": ["xbrl", "calculation"],
    }


questions: list[dict] = []

# --- multi-hop numeric: two figures + arithmetic ---
questions.append(growth_q("AAPL", "revenue", "revenue", 2022, 2023, "medium"))
questions.append(growth_q("NVDA", "research_and_development", "R&D expense", 2022, 2023, "medium"))
questions.append(growth_q("GOOGL", "net_income", "net income", 2022, 2023, "medium"))
questions.append(growth_q("AMZN", "operating_cash_flow", "operating cash flow", 2022, 2023, "hard"))
questions.append(growth_q("MSFT", "revenue", "revenue", 2022, 2023, "medium"))

# --- ratios: derived, never stated directly in the filing ---
questions.append(ratio_q("NVDA", "research_and_development", "R&D spend", 2023, "hard"))
questions.append(ratio_q("AAPL", "research_and_development", "R&D spend", 2023, "hard"))
questions.append(ratio_q("MSFT", "research_and_development", "R&D spend", 2023, "hard"))
questions.append(ratio_q("GOOGL", "net_income", "net income", 2023, "medium"))

# --- cross-company: figures from two separate filings ---
questions.append(
    cross_company_q("NVDA", "MSFT", "research_and_development", "R&D spend", 2022, 2023)
)
questions.append(
    cross_company_q("GOOGL", "AAPL", "revenue", "revenue", 2022, 2023)
)
questions.append(
    cross_company_q("AMZN", "AAPL", "net_income", "net income", 2022, 2023)
)

# --- relationship: set operations the retrieval pipeline cannot compute ---
questions.extend(
    [
        {
            "question": "Which companies share supply chain risk exposure?",
            "reference_answer": (
                "All five companies in the corpus (Apple, Microsoft, Alphabet, "
                "Amazon and NVIDIA) disclose supply chain risk in their 10-K "
                "filings. NVIDIA and Apple devote the most filing text to it."
            ),
            "ticker": None,
            "year": "2023",
            "question_type": "relationship",
            "difficulty": "hard",
            "expected_agents": ["graph"],
        },
        {
            "question": "What risk themes do NVIDIA and Microsoft both disclose?",
            "reference_answer": (
                "NVIDIA and Microsoft both disclose supply chain, privacy "
                "regulation, tax, litigation, foreign exchange, cybersecurity, "
                "macroeconomic, intellectual property, geopolitical, artificial "
                "intelligence, pandemic and climate risk. Customer "
                "concentration, semiconductor supply and data centre capacity "
                "are disclosed by NVIDIA only."
            ),
            "ticker": None,
            "year": "2023",
            "question_type": "relationship",
            "difficulty": "hard",
            "expected_agents": ["graph"],
        },
        {
            "question": "Which risks does NVIDIA disclose that the other companies do not?",
            "reference_answer": (
                "NVIDIA uniquely discloses customer concentration risk and "
                "semiconductor supply risk relative to the other filers in the "
                "corpus, reflecting its fabless manufacturing model and "
                "reliance on a small number of large customers."
            ),
            "ticker": "NVDA",
            "year": "2023",
            "question_type": "relationship",
            "difficulty": "hard",
            "expected_agents": ["graph"],
        },
        {
            "question": "Which geographies does Apple name as exposure in its filings?",
            "reference_answer": (
                "Apple names China, Taiwan, Japan, India, Korea, Singapore, "
                "Vietnam, Europe, Ireland and Australia in its 10-K filings."
            ),
            "ticker": "AAPL",
            "year": "2023",
            "question_type": "relationship",
            "difficulty": "medium",
            "expected_agents": ["graph"],
        },
        # --- qualitative controls: v1's home ground ---
        {
            "question": "How does Apple describe its approach to customer privacy?",
            "reference_answer": (
                "Apple describes privacy as a fundamental human right and "
                "states that it designs its products to minimise data "
                "collection, keeping data on device where possible, and notes "
                "that privacy regulation such as GDPR creates compliance "
                "obligations and risk."
            ),
            "ticker": "AAPL",
            "year": "2023",
            "question_type": "qualitative",
            "difficulty": "medium",
            "expected_agents": ["narrative"],
        },
        {
            "question": "What does Microsoft say about competition in cloud services?",
            "reference_answer": (
                "Microsoft describes the cloud market as intensely competitive, "
                "naming competitors in infrastructure and platform services, "
                "and states that competition is based on price, performance, "
                "security and the breadth of services offered."
            ),
            "ticker": "MSFT",
            "year": "2023",
            "question_type": "qualitative",
            "difficulty": "medium",
            "expected_agents": ["narrative"],
        },
    ]
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(questions, indent=2), encoding="utf-8")

by_type: dict[str, int] = {}
for q in questions:
    by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1

print(f"\nWrote {len(questions)} questions → {OUT_PATH}")
for qtype, count in sorted(by_type.items()):
    print(f"  {qtype:<20} {count}")
print("\nSample reference answers:")
for q in questions[:3]:
    print(f"  Q: {q['question']}")
    print(f"     {q['reference_answer']}")
