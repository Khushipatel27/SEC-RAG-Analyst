"""
Financial Document Parser
Primary: pdfplumber | Fallback: fitz (PyMuPDF)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Section detection patterns
# ---------------------------------------------------------------------------
SECTION_PATTERNS = [
    r"ITEM\s+\d+[A-Z]?\.",
    r"(?:PART|SECTION)\s+[IVX]+",
    r"^[A-Z][A-Z\s]{10,50}$",
]

_SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.MULTILINE)

# ---------------------------------------------------------------------------
# Metric extraction patterns  {label: [pattern, ...]}
# ---------------------------------------------------------------------------
_METRIC_PATTERNS: dict[str, list[str]] = {
    "total_revenue": [
        r"(?:total\s+)?net\s+(?:sales|revenue)[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
        r"total\s+revenue[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
        r"revenues?[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
    ],
    "net_income": [
        r"net\s+income[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
        r"net\s+earnings[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
    ],
    "eps_diluted": [
        r"diluted[^\n]*?(?:earnings|loss)\s+per\s+(?:common\s+)?share[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)",
        r"earnings\s+per\s+(?:diluted\s+)?share[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)",
    ],
    "total_assets": [
        r"total\s+assets[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
    ],
    "rd_expense": [
        r"research\s+(?:and|&)\s+development[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
        r"r\s*&\s*d[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
    ],
    "operating_income": [
        r"operating\s+income[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
        r"income\s+from\s+operations[^\n]*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
    ],
}


def _fix_spacing(text: str) -> str:
    """
    Repair spacing defects produced by naive HTML strippers.

    Applies rules in order:
    1. wordninja — split long all-lowercase merged tokens
       e.g. "isamultinational" → "is a multinational"
    2. camelCase boundary — "appleInc" → "apple Inc"
    3. Space after period / comma / colon / semicolon before a letter
    4. Space after closing parenthesis before a letter
    5. Strip long separator lines
    6. Collapse whitespace
    """
    # ── 1. wordninja: split merged all-lowercase tokens ──────────────────────
    try:
        import wordninja as _wn

        def _split_merged(m: "re.Match") -> str:
            word = m.group(0)
            parts = _wn.split(word)
            return " ".join(parts) if len(parts) > 1 else word

        text = re.sub(r"(?<![a-z])[a-z]{14,}(?![a-z])", _split_merged, text)
    except ImportError:
        pass

    # ── 2. camelCase boundary ─────────────────────────────────────────────────
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # ── 3. Space after punctuation before a letter ────────────────────────────
    text = re.sub(r"([.,:;])([A-Za-z])", r"\1 \2", text)

    # ── 4. Space after closing parenthesis ───────────────────────────────────
    text = re.sub(r"\)([A-Za-z])", r") \1", text)

    # ── 5. Strip long separator lines ────────────────────────────────────────
    text = re.sub(r"[=\-]{4,}", "\n", text)

    # ── 6. Collapse whitespace ────────────────────────────────────────────────
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _detect_section(text: str, previous_section: str = "Unknown") -> str:
    """Return the section header found in *text*, or *previous_section*."""
    match = _SECTION_RE.search(text)
    if match:
        return match.group(0).strip()
    return previous_section


def _ticker_year_from_filename(file_path: Path) -> tuple[str, str]:
    """
    Extract ticker and year from filenames like  AAPL_2023_10K.pdf
    Falls back to 'UNKNOWN' / '0000'.
    """
    stem = file_path.stem.upper()
    parts = stem.split("_")
    ticker = parts[0] if parts else "UNKNOWN"
    year = "0000"
    for part in parts[1:]:
        if re.fullmatch(r"\d{4}", part):
            year = part
            break
    return ticker, year


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class FinancialDocumentParser:
    """Parse SEC 10-K filings (PDF / HTM / TXT) into structured blocks."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_document(self, file_path: Path) -> dict[str, Any]:
        """
        Top-level entry point.

        Returns
        -------
        {
            "file_path": str,
            "ticker": str,
            "year": str,
            "num_pages": int,
            "text_blocks": list[dict],
            "table_blocks": list[dict],
            "metadata": dict,
        }
        """
        file_path = Path(file_path)
        logger.info(f"Parsing document: {file_path}")

        ticker, year = _ticker_year_from_filename(file_path)

        suffix = file_path.suffix.lower()
        if suffix in {".htm", ".html", ".txt"}:
            text_blocks, table_blocks, num_pages = self._parse_text_file(file_path)
        else:
            text_blocks = self.parse_text_blocks(file_path)
            table_blocks = self.parse_table_blocks(file_path)
            num_pages = max(
                (b.get("page_num", 0) for b in text_blocks + table_blocks),
                default=0,
            )

        parsed_doc: dict[str, Any] = {
            "file_path": str(file_path),
            "ticker": ticker,
            "year": year,
            "num_pages": num_pages,
            "text_blocks": text_blocks,
            "table_blocks": table_blocks,
            "metadata": {
                "ticker": ticker,
                "year": year,
                "source_file": file_path.name,
                "num_text_blocks": len(text_blocks),
                "num_table_blocks": len(table_blocks),
            },
        }

        key_metrics = self.extract_key_metrics(parsed_doc)
        parsed_doc["metadata"]["key_metrics"] = key_metrics

        logger.success(
            f"Parsed {file_path.name}: {len(text_blocks)} text blocks, "
            f"{len(table_blocks)} table blocks, {num_pages} pages"
        )
        return parsed_doc

    def parse_text_blocks(self, pdf_path: Path) -> list[dict]:
        """
        Extract text blocks from a PDF.

        Each block: {"text": str, "page_num": int, "block_type": "text", "section": str}
        """
        pdf_path = Path(pdf_path)
        try:
            return self._parse_text_with_pdfplumber(pdf_path)
        except Exception as primary_err:
            logger.warning(
                f"pdfplumber failed ({primary_err}); falling back to fitz for {pdf_path.name}"
            )
            try:
                return self._parse_text_with_fitz(pdf_path)
            except Exception as fallback_err:
                logger.error(f"fitz also failed ({fallback_err}) for {pdf_path.name}")
                return []

    def parse_table_blocks(self, pdf_path: Path) -> list[dict]:
        """
        Extract table blocks from a PDF.

        Each block:
        {
            "text": str,
            "dataframe": dict,   # {columns: [...], data: [[...]]}
            "page_num": int,
            "block_type": "table",
            "section": str,
            "contains_numbers": True,
        }
        """
        pdf_path = Path(pdf_path)
        try:
            return self._parse_tables_with_pdfplumber(pdf_path)
        except Exception as primary_err:
            logger.warning(
                f"pdfplumber table extraction failed ({primary_err}); "
                f"falling back to fitz for {pdf_path.name}"
            )
            try:
                return self._parse_tables_with_fitz(pdf_path)
            except Exception as fallback_err:
                logger.error(
                    f"fitz table extraction also failed ({fallback_err}) for {pdf_path.name}"
                )
                return []

    def extract_key_metrics(self, parsed_doc: dict) -> dict[str, str | None]:
        """
        Search all text + table blocks for known financial metrics.

        Returns
        -------
        {
            "total_revenue": "...",
            "net_income": "...",
            "eps_diluted": "...",
            "total_assets": "...",
            "rd_expense": "...",
            "operating_income": "...",
        }
        """
        # Combine all block text into one lower-cased corpus
        all_text_parts: list[str] = []
        for block in parsed_doc.get("text_blocks", []):
            all_text_parts.append(block.get("text", ""))
        for block in parsed_doc.get("table_blocks", []):
            all_text_parts.append(block.get("text", ""))
        corpus = "\n".join(all_text_parts).lower()

        results: dict[str, str | None] = {}
        for metric, patterns in _METRIC_PATTERNS.items():
            found: str | None = None
            for pattern in patterns:
                match = re.search(pattern, corpus, re.IGNORECASE)
                if match:
                    found = match.group(1).replace(",", "").strip()
                    break
            results[metric] = found

        logger.debug(
            f"Extracted key metrics for {parsed_doc.get('ticker','?')} "
            f"{parsed_doc.get('year','?')}: {results}"
        )
        return results

    # ------------------------------------------------------------------
    # Private: pdfplumber text
    # ------------------------------------------------------------------

    def _parse_text_with_pdfplumber(self, pdf_path: Path) -> list[dict]:
        import pdfplumber  # lazy import

        blocks: list[dict] = []
        current_section = "Unknown"

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                if not raw_text:
                    continue

                # Split into paragraphs (double newline) and single lines
                paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

                for para in paragraphs:
                    current_section = _detect_section(para, current_section)
                    blocks.append(
                        {
                            "text": para,
                            "page_num": page_num,
                            "block_type": "text",
                            "section": current_section,
                        }
                    )

        return blocks

    # ------------------------------------------------------------------
    # Private: pdfplumber tables
    # ------------------------------------------------------------------

    def _parse_tables_with_pdfplumber(self, pdf_path: Path) -> list[dict]:
        import pdfplumber  # lazy import

        blocks: list[dict] = []
        current_section = "Unknown"

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Update section from page text before extracting tables
                raw_text = page.extract_text() or ""
                current_section = _detect_section(raw_text, current_section)

                tables = page.extract_tables()
                if not tables:
                    continue

                for table in tables:
                    if not table:
                        continue

                    # Build a plain-text representation
                    rows_text: list[str] = []
                    for row in table:
                        if row:
                            rows_text.append(
                                " | ".join(str(cell) if cell is not None else "" for cell in row)
                            )
                    text_repr = "\n".join(rows_text)

                    # Build serialisable dataframe dict
                    if len(table) > 1:
                        columns = [str(c) if c is not None else "" for c in table[0]]
                        data = [
                            [str(cell) if cell is not None else "" for cell in row]
                            for row in table[1:]
                        ]
                    else:
                        columns = []
                        data = [
                            [str(cell) if cell is not None else "" for cell in row]
                            for row in table
                        ]

                    contains_numbers = bool(re.search(r"\d", text_repr))

                    blocks.append(
                        {
                            "text": text_repr,
                            "dataframe": {"columns": columns, "data": data},
                            "page_num": page_num,
                            "block_type": "table",
                            "section": current_section,
                            "contains_numbers": contains_numbers,
                        }
                    )

        return blocks

    # ------------------------------------------------------------------
    # Private: fitz (PyMuPDF) text fallback
    # ------------------------------------------------------------------

    def _parse_text_with_fitz(self, pdf_path: Path) -> list[dict]:
        import fitz  # PyMuPDF  # lazy import

        blocks: list[dict] = []
        current_section = "Unknown"

        doc = fitz.open(str(pdf_path))
        try:
            for page_num, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")
                if not raw_text:
                    continue

                paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    current_section = _detect_section(para, current_section)
                    blocks.append(
                        {
                            "text": para,
                            "page_num": page_num,
                            "block_type": "text",
                            "section": current_section,
                        }
                    )
        finally:
            doc.close()

        return blocks

    # ------------------------------------------------------------------
    # Private: fitz table fallback (extracts text around table-like areas)
    # ------------------------------------------------------------------

    def _parse_tables_with_fitz(self, pdf_path: Path) -> list[dict]:
        """
        PyMuPDF does not have a native table extractor in older versions.
        We look for text blocks whose content resembles tabular data
        (multiple pipe-separated or tab-separated values on consecutive lines).
        """
        import fitz  # lazy import

        blocks: list[dict] = []
        current_section = "Unknown"

        doc = fitz.open(str(pdf_path))
        try:
            for page_num, page in enumerate(doc, start=1):
                raw_text = page.get_text("text") or ""
                current_section = _detect_section(raw_text, current_section)

                # Heuristic: lines with 2+ consecutive numbers separated by spaces/tabs
                candidate_lines: list[str] = []
                for line in raw_text.splitlines():
                    num_count = len(re.findall(r"\b[\d,]+(?:\.\d+)?\b", line))
                    if num_count >= 2:
                        candidate_lines.append(line)

                if len(candidate_lines) >= 2:
                    text_repr = "\n".join(candidate_lines)
                    blocks.append(
                        {
                            "text": text_repr,
                            "dataframe": {"columns": [], "data": []},
                            "page_num": page_num,
                            "block_type": "table",
                            "section": current_section,
                            "contains_numbers": True,
                        }
                    )
        finally:
            doc.close()

        return blocks

    # ------------------------------------------------------------------
    # Private: plain-text / HTML file parsing
    # ------------------------------------------------------------------

    def _parse_text_file(
        self, file_path: Path
    ) -> tuple[list[dict], list[dict], int]:
        """Parse .htm, .html, .txt files."""
        suffix = file_path.suffix.lower()
        raw_text: str

        if suffix in {".htm", ".html"}:
            try:
                from bs4 import BeautifulSoup  # optional dependency

                html = file_path.read_text(encoding="utf-8", errors="replace")
                soup = BeautifulSoup(html, "html.parser")
                raw_text = soup.get_text(separator="\n")
            except ImportError:
                logger.warning("beautifulsoup4 not installed; stripping HTML tags with regex")
                html = file_path.read_text(encoding="utf-8", errors="replace")
                raw_text = re.sub(r"<[^>]+>", " ", html)
        else:
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            # Fix spacing defects from naive HTML stripping (no-space CamelCase)
            raw_text = _fix_spacing(raw_text)

        lines = raw_text.splitlines()
        num_pages = max(1, len(lines) // 60)  # approximate

        text_blocks: list[dict] = []
        current_section = "Unknown"

        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        page_num = 1
        for i, para in enumerate(paragraphs):
            # Simulate page breaks every ~30 paragraphs
            page_num = (i // 30) + 1
            current_section = _detect_section(para, current_section)
            text_blocks.append(
                {
                    "text": para,
                    "page_num": page_num,
                    "block_type": "text",
                    "section": current_section,
                }
            )

        return text_blocks, [], num_pages
