"""
Financial Document Chunker
Rules:
  1. NEVER split a table across chunks
  2. NEVER split a paragraph mid-sentence
  3. Keep section header with first chunk of that section
  4. Use RecursiveCharacterTextSplitter for text blocks
  5. Tables always become a single chunk regardless of size
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import settings


def _make_chunk(
    text: str,
    ticker: str,
    year: str,
    page_num: int,
    section: str,
    block_type: str,
    contains_numbers: bool,
    source_file: str,
) -> dict[str, Any]:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "ticker": ticker,
        "year": year,
        "page_num": page_num,
        "section": section,
        "block_type": block_type,
        "contains_numbers": contains_numbers,
        "source_file": source_file,
        "char_count": len(text),
    }


class FinancialChunker:
    """
    Converts a parsed document dict (from FinancialDocumentParser)
    into a flat list of chunk dicts ready for embedding and indexing.
    """

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, parsed_doc: dict) -> list[dict]:
        """
        Convert a parsed document into a list of chunk dicts.

        Parameters
        ----------
        parsed_doc : dict
            Output of FinancialDocumentParser.parse_document()

        Returns
        -------
        list[dict]  – each element is a chunk dict with all required fields.
        """
        ticker: str = parsed_doc.get("ticker", "UNKNOWN")
        year: str = parsed_doc.get("year", "0000")
        source_file: str = parsed_doc.get("file_path", "")

        chunks: list[dict] = []

        # --- Process text blocks ---
        text_blocks: list[dict] = parsed_doc.get("text_blocks", [])
        chunks.extend(
            self._chunk_text_blocks(text_blocks, ticker, year, source_file)
        )

        # --- Process table blocks (Rule 1 & 5: one chunk per table) ---
        table_blocks: list[dict] = parsed_doc.get("table_blocks", [])
        chunks.extend(
            self._chunk_table_blocks(table_blocks, ticker, year, source_file)
        )

        # --- Stats ---
        table_chunks = sum(1 for c in chunks if c["block_type"] == "table")
        text_chunks = sum(1 for c in chunks if c["block_type"] == "text")
        total_chars = sum(c["char_count"] for c in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0

        logger.info(
            f"[{ticker} {year}] Chunking complete: "
            f"{len(chunks)} total chunks | "
            f"{text_chunks} text | "
            f"{table_chunks} table | "
            f"avg {avg_chars:.0f} chars/chunk"
        )
        print(
            f"\n--- Chunking Stats [{ticker} {year}] ---\n"
            f"  Total chunks  : {len(chunks)}\n"
            f"  Text chunks   : {text_chunks}\n"
            f"  Table chunks  : {table_chunks}\n"
            f"  Avg chars     : {avg_chars:.0f}\n"
            f"  Total chars   : {total_chars}\n"
            f"------------------------------------\n"
        )

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_text_blocks(
        self,
        text_blocks: list[dict],
        ticker: str,
        year: str,
        source_file: str,
    ) -> list[dict]:
        """
        Split each text block with RecursiveCharacterTextSplitter.
        Rule 2: splitter uses sentence-aware separators.
        Rule 3: prepend the section header to the FIRST sub-chunk of each block.
        """
        chunks: list[dict] = []
        seen_sections: set[str] = set()

        for block in text_blocks:
            raw_text: str = block.get("text", "").strip()
            if not raw_text:
                continue

            page_num: int = block.get("page_num", 0)
            section: str = block.get("section", "Unknown")
            contains_numbers: bool = self._text_has_numbers(raw_text)

            sub_texts: list[str] = self._splitter.split_text(raw_text)
            if not sub_texts:
                continue

            for idx, sub_text in enumerate(sub_texts):
                # Rule 3: prepend section header if this is the first chunk
                # in this section and the header is not already present.
                if idx == 0 and section not in seen_sections:
                    if section not in sub_text:
                        sub_text = f"{section}\n\n{sub_text}"
                    seen_sections.add(section)

                chunks.append(
                    _make_chunk(
                        text=sub_text,
                        ticker=ticker,
                        year=year,
                        page_num=page_num,
                        section=section,
                        block_type="text",
                        contains_numbers=contains_numbers,
                        source_file=source_file,
                    )
                )

        return chunks

    def _chunk_table_blocks(
        self,
        table_blocks: list[dict],
        ticker: str,
        year: str,
        source_file: str,
    ) -> list[dict]:
        """
        Rule 1 & 5: each table becomes exactly ONE chunk.
        """
        chunks: list[dict] = []

        for block in table_blocks:
            raw_text: str = block.get("text", "").strip()
            if not raw_text:
                continue

            page_num: int = block.get("page_num", 0)
            section: str = block.get("section", "Unknown")
            contains_numbers: bool = block.get("contains_numbers", True)

            chunks.append(
                _make_chunk(
                    text=raw_text,
                    ticker=ticker,
                    year=year,
                    page_num=page_num,
                    section=section,
                    block_type="table",
                    contains_numbers=contains_numbers,
                    source_file=source_file,
                )
            )

        return chunks

    @staticmethod
    def _text_has_numbers(text: str) -> bool:
        import re
        return bool(re.search(r"\d", text))
