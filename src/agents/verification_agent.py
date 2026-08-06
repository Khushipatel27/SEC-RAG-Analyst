"""
Verification Agent
Checks that a drafted answer is actually supported by the evidence the
specialists collected.

Two layers, in this order:

1. A deterministic numeric grounding check. Every figure asserted in the answer
   must trace back to a number in the evidence. This needs no LLM, cannot itself
   hallucinate, and catches the failure mode that matters most in financial QA —
   a number that appeared from nowhere.

2. An LLM entailment check, for claims that are not numeric. This is the weaker
   of the two (it is a 3B model judging its own family's output), so its verdict
   is advisory: it can lower confidence, and it can never override a numeric
   failure.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import ollama
from loguru import logger

from src.config import settings

# Matches 383,285 / $383.29B / 39.31% / 1.2 million
_NUMBER_RE = re.compile(
    r"""
    (?P<sign>[-+]?)
    \$?\s*
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<suffix>%|percent|percentage\s+points|pp\b|billion|bn\b|million|mm?\b|thousand|[BMKT]\b)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SCALE = {
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
    "t": 1e12,
}

# Years, section numbers, and small counts are not financial claims — checking
# them produces noise, not signal.
_IGNORE_EXACT = {1934, 1933, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026}

# Structural identifiers that contain digits but assert nothing numeric. These
# are removed before parsing, otherwise "10-K" is read as the figure 10 and the
# answer is reported as containing an unsupported number.
_NON_CLAIM_PATTERNS = [
    re.compile(r"\b\d{10}-\d{2}-\d{6}\b"),          # accession: 0000320193-23-000106
    re.compile(r"\b10-[KQ]\b", re.IGNORECASE),      # form types
    re.compile(r"\b8-K\b", re.IGNORECASE),
    re.compile(r"\b20-F\b", re.IGNORECASE),
    re.compile(r"\bS-\d+\b", re.IGNORECASE),
    re.compile(r"\bitem\s+\d+[A-Z]?\b", re.IGNORECASE),   # Item 1A, Item 7
    re.compile(r"\bpart\s+[IVX]+\b", re.IGNORECASE),
    re.compile(r"\bp\.\s?\d+\b", re.IGNORECASE),          # page references
    re.compile(r"\bFY\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\bus-gaap:\S+", re.IGNORECASE),          # XBRL concept names
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                 # ISO dates
]

# Relative tolerance when matching an answer figure to an evidence figure.
# Generous enough to absorb rounding ($383.29B vs 383,285,000,000).
_REL_TOL = 0.02


@dataclass
class VerificationResult:
    """Outcome of grounding a draft answer against collected evidence."""

    verdict: str  # "grounded" | "partially_grounded" | "unsupported"
    confidence: float  # 0.0 – 1.0
    numeric_grounding: float  # fraction of answer figures found in evidence
    total_numbers: int = 0
    grounded_numbers: int = 0
    unsupported_numbers: list[str] = field(default_factory=list)
    llm_verdict: Optional[str] = None
    llm_reasoning: str = ""
    checks: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "numeric_grounding": round(self.numeric_grounding, 3),
            "total_numbers": self.total_numbers,
            "grounded_numbers": self.grounded_numbers,
            "unsupported_numbers": self.unsupported_numbers,
            "llm_verdict": self.llm_verdict,
            "llm_reasoning": self.llm_reasoning,
            "checks": self.checks,
            "latency_ms": self.latency_ms,
        }


VERIFICATION_PROMPT = """You are a financial answer auditor. Your only job is to \
decide whether the ANSWER is fully supported by the EVIDENCE.

Rules:
- Judge only what the EVIDENCE supports. Do not use outside knowledge.
- If the ANSWER states a fact not present in the EVIDENCE, it is NOT supported.
- If the ANSWER is fully supported, reply SUPPORTED.
- If parts are supported and parts are not, reply PARTIAL.
- If the ANSWER contradicts the EVIDENCE or has no basis in it, reply UNSUPPORTED.

EVIDENCE:
{evidence}

ANSWER:
{answer}

Reply with exactly two lines:
VERDICT: <SUPPORTED|PARTIAL|UNSUPPORTED>
REASON: <one sentence>"""


class VerificationAgent:
    """Grounds a draft answer against the evidence that produced it."""

    def __init__(self) -> None:
        self._model = settings.llm_model
        self._max_evidence = settings.verification_max_evidence_chars
        logger.info(f"VerificationAgent initialized | model={self._model}")

    # ------------------------------------------------------------------
    # Numeric grounding (deterministic)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_numbers(text: str) -> list[tuple[str, float, bool]]:
        """
        Extract numbers as (surface_form, normalised_value, is_percent).

        Percentages are kept separate from absolute values so that "39.31%"
        is never matched against "$39.31B".
        """
        for pattern in _NON_CLAIM_PATTERNS:
            text = pattern.sub(" ", text)

        out: list[tuple[str, float, bool]] = []
        for m in _NUMBER_RE.finditer(text):
            raw = m.group("num").replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue

            suffix = (m.group("suffix") or "").lower().strip()
            is_percent = suffix in ("%", "percent", "percentage points", "pp")

            if not is_percent and suffix:
                value *= _SCALE.get(suffix.rstrip("."), 1.0)

            if m.group("sign") == "-":
                value = -value

            out.append((m.group(0).strip(), value, is_percent))
        return out

    def check_numeric_grounding(self, answer: str, evidence: str) -> tuple[float, list[str], int, int]:
        """
        Fraction of figures in *answer* that also appear in *evidence*.

        Returns
        -------
        (score, unsupported_surface_forms, grounded_count, total_count)
        """
        answer_numbers = self._parse_numbers(answer)
        evidence_numbers = self._parse_numbers(evidence)

        # Compare like with like
        ev_abs = [v for _, v, is_pct in evidence_numbers if not is_pct]
        ev_pct = [v for _, v, is_pct in evidence_numbers if is_pct]

        checkable: list[tuple[str, float, bool]] = [
            (surface, value, is_pct)
            for surface, value, is_pct in answer_numbers
            # Skip bare years and trivial counts
            if not (not is_pct and value in _IGNORE_EXACT)
        ]

        if not checkable:
            return 1.0, [], 0, 0

        unsupported: list[str] = []
        grounded = 0

        for surface, value, is_pct in checkable:
            pool = ev_pct if is_pct else ev_abs
            # Percentages may be stated either way round in the evidence
            # (a -6.66% gap described as 6.66), so compare on magnitude too.
            match = any(
                abs(value - candidate) <= _REL_TOL * max(abs(value), abs(candidate), 1e-9)
                or abs(abs(value) - abs(candidate))
                <= _REL_TOL * max(abs(value), abs(candidate), 1e-9)
                for candidate in pool
            )
            if match:
                grounded += 1
            else:
                unsupported.append(surface)

        score = grounded / len(checkable)
        return score, unsupported, grounded, len(checkable)

    # ------------------------------------------------------------------
    # LLM entailment (advisory)
    # ------------------------------------------------------------------

    def check_entailment(self, answer: str, evidence: str) -> tuple[Optional[str], str]:
        """Ask the local model whether the answer follows from the evidence."""
        prompt = VERIFICATION_PROMPT.format(
            evidence=evidence[: self._max_evidence],
            answer=answer,
        )
        try:
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"num_predict": 200, "temperature": 0.0},
            )
            content = response["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Verification LLM call failed: {exc}")
            return None, f"Verification model unavailable: {exc}"

        verdict_match = re.search(
            r"VERDICT:\s*(SUPPORTED|PARTIAL|UNSUPPORTED)", content, re.IGNORECASE
        )
        reason_match = re.search(r"REASON:\s*(.+)", content, re.IGNORECASE)

        verdict = verdict_match.group(1).upper() if verdict_match else None
        reason = reason_match.group(1).strip() if reason_match else content[:200]

        if verdict is None:
            logger.warning(f"Could not parse verification verdict from: {content[:120]}")

        return verdict, reason

    # ------------------------------------------------------------------
    # Agent entry point
    # ------------------------------------------------------------------

    def verify(
        self,
        question: str,
        answer: str,
        evidence: str,
        use_llm: bool = True,
    ) -> VerificationResult:
        """
        Verify *answer* against *evidence*.

        The numeric check is authoritative: an answer with an ungrounded figure
        cannot be reported as grounded regardless of what the LLM judge says.
        """
        start = time.perf_counter()
        checks: list[str] = []

        if not evidence.strip():
            return VerificationResult(
                verdict="unsupported",
                confidence=0.0,
                numeric_grounding=0.0,
                checks=["No evidence was collected for this answer."],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # --- Layer 1: numeric grounding ---
        score, unsupported, grounded, total = self.check_numeric_grounding(
            answer, evidence
        )
        if total == 0:
            checks.append("No numeric claims to verify.")
        else:
            checks.append(
                f"Numeric grounding: {grounded}/{total} figures traced to evidence."
            )
        if unsupported:
            checks.append(f"Figures not found in evidence: {', '.join(unsupported[:6])}")

        # --- Layer 2: LLM entailment ---
        llm_verdict: Optional[str] = None
        llm_reasoning = ""
        if use_llm:
            llm_verdict, llm_reasoning = self.check_entailment(answer, evidence)
            if llm_verdict:
                checks.append(f"LLM entailment check: {llm_verdict}.")
            else:
                checks.append("LLM entailment check was inconclusive.")

        # --- Combine ---
        verdict, confidence = self._combine(score, total, llm_verdict)

        result = VerificationResult(
            verdict=verdict,
            confidence=confidence,
            numeric_grounding=score,
            total_numbers=total,
            grounded_numbers=grounded,
            unsupported_numbers=unsupported,
            llm_verdict=llm_verdict,
            llm_reasoning=llm_reasoning,
            checks=checks,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

        logger.info(
            f"Verification: {verdict} (confidence {confidence:.2f}, "
            f"numeric {grounded}/{total}, llm={llm_verdict})"
        )
        return result

    @staticmethod
    def _combine(
        numeric_score: float, total_numbers: int, llm_verdict: Optional[str]
    ) -> tuple[str, float]:
        """
        Fuse the two signals into one verdict.

        The numeric check dominates because it is the one that cannot be wrong
        about its own domain. The LLM verdict only adjusts confidence.
        """
        if total_numbers > 0:
            if numeric_score >= 0.999:
                verdict, confidence = "grounded", 0.9
            elif numeric_score >= 0.5:
                verdict, confidence = "partially_grounded", 0.6
            else:
                verdict, confidence = "unsupported", 0.25
        else:
            # Purely narrative answer — the LLM judge is all we have
            verdict, confidence = "grounded", 0.6

        if llm_verdict == "SUPPORTED":
            confidence = min(1.0, confidence + 0.1)
        elif llm_verdict == "PARTIAL":
            confidence = max(0.0, confidence - 0.1)
            if verdict == "grounded":
                verdict = "partially_grounded"
        elif llm_verdict == "UNSUPPORTED":
            confidence = max(0.0, confidence - 0.25)
            if verdict == "grounded":
                verdict = "partially_grounded"

        return verdict, confidence
