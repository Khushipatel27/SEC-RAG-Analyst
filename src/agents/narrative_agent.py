"""
Narrative Agent
Thin wrapper that exposes the existing v1 RAG pipeline as one tool among many.

The v1 pipeline is unchanged and still the right instrument for questions whose
answers live in prose — strategy discussion, risk narrative, management
commentary. v2 does not replace it; it stops treating it as the only option.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from loguru import logger


class NarrativeAgent:
    """Adapts SECRAGPipeline.ask() to the agent result contract."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        logger.info("NarrativeAgent initialized (wrapping SECRAGPipeline)")

    def answer(
        self,
        question: str,
        ticker: Optional[str] = None,
        year: Optional[str] = None,
    ) -> dict:
        """
        Run hybrid retrieval + rerank + generation, unchanged from v1.

        Returns
        -------
        dict  – {agent, answer, sources, evidence, summary, latency_ms}
        """
        start = time.perf_counter()

        try:
            response = self._pipeline.ask(
                question=question,
                ticker_filter=ticker,
                year_filter=year,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Narrative agent failed: {exc}")
            return {
                "agent": "narrative",
                "answer": "",
                "sources": [],
                "evidence": "",
                "summary": f"Retrieval pipeline failed: {exc}",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        sources = response.get("sources", [])
        answer = response.get("answer", "")

        # Two kinds of evidence, and they serve different consumers.
        #
        # The retrieved passages are what verification must check against —
        # grounding an answer in itself proves nothing. But the pipeline only
        # exposes 200-character previews of them, which is too thin for the
        # supervisor to synthesise from. So the v1 answer (which *was* generated
        # from the full reranked context) is carried alongside them.
        passage_parts = [
            f"[{s.get('ticker')} FY{s.get('year')} 10-K | {s.get('section')} | "
            f"p.{s.get('page_num')}]\n{s.get('text_preview', '')}"
            for s in sources
        ]

        evidence_blocks = []
        if answer:
            evidence_blocks.append(
                "Summary produced by the retrieval pipeline from the full "
                f"reranked context:\n{answer}"
            )
        if passage_parts:
            evidence_blocks.append(
                "Source passages (previews):\n" + "\n\n".join(passage_parts)
            )

        return {
            "agent": "narrative",
            "answer": answer,
            "sources": sources,
            "evidence": "\n\n".join(evidence_blocks),
            "summary": f"Retrieved {len(sources)} passage(s) via hybrid search + reranking.",
            "filters_applied": response.get("filters_applied"),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
