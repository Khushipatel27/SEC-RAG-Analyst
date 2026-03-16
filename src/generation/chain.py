"""
Financial RAG Generation Chain
Orchestrates prompt selection, context formatting, and LLM generation
via a locally running Ollama instance.
"""
from __future__ import annotations

import time
from typing import Any, Generator

import ollama
from loguru import logger

from src.config import settings
from src.generation.prompts import (
    COMPARISON_PROMPT,
    FINANCIAL_QA_PROMPT,
    SUMMARY_PROMPT,
    TABLE_EXTRACTION_PROMPT,
)

_COMPARE_KEYWORDS = frozenset(["compare", "vs", "versus", "difference", "differ",
                               "contrast", "between", "relative to"])
_SUMMARY_KEYWORDS = frozenset(["summary", "summarize", "summarise", "overview",
                               "overview of", "brief", "highlight"])


class FinancialRAGChain:
    """
    End-to-end generation chain for financial Q&A.

    Given a user query and a list of retrieved + reranked chunk dicts,
    this class selects the appropriate prompt template, formats the
    context, calls the Ollama LLM, and returns a structured response dict.
    """

    def __init__(self) -> None:
        self._model: str = settings.llm_model
        self._fallback_model: str = settings.fallback_llm_model
        self._max_tokens: int = settings.max_new_tokens
        self._temperature: float = settings.temperature
        logger.info(
            f"FinancialRAGChain initialized | model={self._model} | "
            f"fallback={self._fallback_model} | max_tokens={self._max_tokens}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_prompt(self, query: str, chunks: list[dict]) -> str:
        """
        Choose the most appropriate prompt template for the query.

        Logic
        -----
        1. "compare" / "vs" / "difference" → COMPARISON_PROMPT
        2. "summary" / "overview"           → SUMMARY_PROMPT
        3. Majority of chunks are tables    → TABLE_EXTRACTION_PROMPT
        4. Default                          → FINANCIAL_QA_PROMPT

        Returns
        -------
        str  – the selected prompt template string.
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        # Rule 1: comparison query
        if query_tokens & _COMPARE_KEYWORDS:
            logger.debug("Prompt selected: COMPARISON_PROMPT")
            return COMPARISON_PROMPT

        # Rule 2: summary/overview query
        if query_tokens & _SUMMARY_KEYWORDS:
            logger.debug("Prompt selected: SUMMARY_PROMPT")
            return SUMMARY_PROMPT

        # Rule 3: mostly table chunks
        if chunks:
            table_count = sum(
                1 for c in chunks if c.get("block_type") == "table"
                or c.get("metadata", {}).get("block_type") == "table"
            )
            if table_count > len(chunks) / 2:
                logger.debug("Prompt selected: TABLE_EXTRACTION_PROMPT")
                return TABLE_EXTRACTION_PROMPT

        # Default
        logger.debug("Prompt selected: FINANCIAL_QA_PROMPT")
        return FINANCIAL_QA_PROMPT

    def format_context(self, chunks: list[dict]) -> str:
        """
        Build a formatted context string from retrieved chunks.

        Each chunk is prefixed with a citation header:
          [Source: {ticker} {year} 10-K | Section: {section} | Page: {page_num} | Type: {block_type}]

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts (fields may be top-level or nested under "metadata").

        Returns
        -------
        str
        """
        parts: list[str] = []

        for i, chunk in enumerate(chunks, start=1):
            # Support both flat chunk dicts and {text, metadata, ...} dicts
            meta: dict = chunk.get("metadata") or {}
            ticker = chunk.get("ticker") or meta.get("ticker", "?")
            year = chunk.get("year") or meta.get("year", "?")
            section = chunk.get("section") or meta.get("section", "?")
            page_num = chunk.get("page_num") or meta.get("page_num", "?")
            block_type = chunk.get("block_type") or meta.get("block_type", "text")
            text = chunk.get("text", "")

            header = (
                f"[Source: {ticker} {year} 10-K | "
                f"Section: {section} | "
                f"Page: {page_num} | "
                f"Type: {block_type}]"
            )
            parts.append(f"{header}\n{text}\n---")

        return "\n\n".join(parts)

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate an answer for *query* given the retrieved *chunks*.

        Parameters
        ----------
        query : str
            The user's question.
        chunks : list[dict]
            Retrieved and (optionally) reranked chunk dicts.

        Returns
        -------
        dict
            {
                "answer": str,
                "sources": list[dict],
                "prompt_used": str,
                "num_chunks": int,
                "latency_ms": float,
                "model": str,
                "context_length": int,
            }
        """
        prompt_template = self.select_prompt(query, chunks)
        context = self.format_context(chunks)
        full_prompt = prompt_template.format(context=context, question=query)

        logger.info(
            f"Generating answer | model={self._model} | "
            f"chunks={len(chunks)} | context_len={len(context)}"
        )

        start = time.perf_counter()
        answer = self._call_ollama(full_prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000

        sources = self._build_sources(chunks)

        response = {
            "answer": answer,
            "sources": sources,
            "prompt_used": prompt_template[:60] + "...",
            "num_chunks": len(chunks),
            "latency_ms": round(elapsed_ms, 2),
            "model": self._model,
            "context_length": len(full_prompt),
        }

        logger.success(
            f"Generation complete in {elapsed_ms:.0f} ms | "
            f"answer_len={len(answer)} chars"
        )
        return response

    def generate_stream(
        self, query: str, chunks: list[dict]
    ) -> Generator[str, None, None]:
        """
        Streaming generation – yields tokens as they arrive from Ollama.

        Parameters
        ----------
        query : str
        chunks : list[dict]

        Yields
        ------
        str – individual token strings.
        """
        prompt_template = self.select_prompt(query, chunks)
        context = self.format_context(chunks)
        full_prompt = prompt_template.format(context=context, question=query)

        logger.info(
            f"Starting streaming generation | model={self._model} | "
            f"chunks={len(chunks)}"
        )

        try:
            stream = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": full_prompt}],
                stream=True,
                options={
                    "num_predict": self._max_tokens,
                    "temperature": self._temperature,
                },
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        except Exception as exc:
            logger.error(f"Streaming generation failed: {exc}")
            yield f"\n[Error during generation: {exc}]"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama chat endpoint, falling back to the fallback model on error."""
        for model in (self._model, self._fallback_model):
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    options={
                        "num_predict": self._max_tokens,
                        "temperature": self._temperature,
                    },
                )
                return response["message"]["content"]
            except Exception as exc:
                logger.warning(f"Model '{model}' failed: {exc}")
                if model == self._fallback_model:
                    logger.error("Both primary and fallback models failed.")
                    return f"[Generation failed: {exc}]"
        return "[Generation failed: unknown error]"

    @staticmethod
    def _build_sources(chunks: list[dict]) -> list[dict]:
        """Build a de-duplicated list of source citations from chunks."""
        seen: set[tuple] = set()
        sources: list[dict] = []

        for chunk in chunks:
            meta: dict = chunk.get("metadata") or {}
            ticker = chunk.get("ticker") or meta.get("ticker", "?")
            year = chunk.get("year") or meta.get("year", "?")
            section = chunk.get("section") or meta.get("section", "?")
            page_num = chunk.get("page_num") or meta.get("page_num", "?")
            block_type = chunk.get("block_type") or meta.get("block_type", "text")
            score = (
                chunk.get("rerank_score")
                or chunk.get("hybrid_score")
                or chunk.get("vector_score")
                or chunk.get("score")
                or 0.0
            )

            key = (ticker, year, section, page_num)
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "ticker": ticker,
                        "year": year,
                        "section": section,
                        "page_num": page_num,
                        "block_type": block_type,
                        "relevance_score": round(float(score), 4),
                        "text_preview": chunk.get("text", "")[:200],
                    }
                )

        return sources
