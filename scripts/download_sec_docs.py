"""
SEC 10-K Downloader
Downloads annual 10-K filings for target companies from the SEC EDGAR API.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Root of the project (two levels up from this script)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_LOG_PATH = ROOT_DIR / "data" / "download_log.json"

# ---------------------------------------------------------------------------
# Target companies
# ---------------------------------------------------------------------------
TARGET_COMPANIES = [
    {"ticker": "AAPL", "company": "Apple Inc", "cik": "0000320193"},
    {"ticker": "MSFT", "company": "Microsoft Corp", "cik": "0000789019"},
    {"ticker": "GOOGL", "company": "Alphabet Inc", "cik": "0001652044"},
    {"ticker": "AMZN", "company": "Amazon.com Inc", "cik": "0001018724"},
    {"ticker": "NVDA", "company": "NVIDIA Corp", "cik": "0001045810"},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SEC-RAG-Research/1.0; research@example.com)",
    "Accept-Encoding": "gzip, deflate",
    # NOTE: Do NOT set "Host" here — requests sets it automatically per URL.
    # A hardcoded "Host: data.sec.gov" breaks all requests to www.sec.gov.
}

SEC_BASE = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

# ---------------------------------------------------------------------------
# Core downloader
# ---------------------------------------------------------------------------


def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def download_10k(ticker: str, cik: str, year: int) -> Optional[Path]:
    """
    Download the 10-K filing for *ticker* / *cik* for fiscal *year*.

    Saves to  data/raw/{ticker}_{year}_10K.pdf  (or .htm if no PDF).

    Returns the saved Path, or None on failure.
    """
    session = _get_session()
    submissions_url = f"{SEC_BASE}/submissions/CIK{cik}.json"

    logger.info(f"Fetching submissions for {ticker} (CIK {cik}) from {submissions_url}")

    try:
        resp = session.get(submissions_url, timeout=30)
        resp.raise_for_status()
        submissions = resp.json()
    except Exception as exc:
        logger.error(f"Failed to fetch submissions for {ticker}: {exc}")
        return None

    time.sleep(1)  # polite delay

    # ------------------------------------------------------------------
    # Walk through all filings to find 10-K for the target fiscal year.
    # KEY: match on reportDate (the period the 10-K covers), NOT filingDate.
    # Example: Amazon's FY2022 10-K has reportDate=2022-12-31 but was
    # filed in February 2023 — matching on filingDate would give wrong year.
    # ------------------------------------------------------------------
    filings = submissions.get("filings", {})
    recent = filings.get("recent", {})

    forms = recent.get("form", [])
    report_dates = recent.get("reportDate", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])

    target_accession: Optional[str] = None
    target_primary: Optional[str] = None
    target_date: Optional[str] = None

    for form, report_date, filing_date, accession, primary_doc in zip(
        forms, report_dates, filing_dates, accession_numbers, primary_documents
    ):
        if form.strip().upper() not in {"10-K", "10-K/A"}:
            continue
        # Use reportDate to identify fiscal year (period the filing covers)
        rdate = report_date or filing_date or ""
        report_year = int(rdate.split("-")[0]) if rdate else 0
        if report_year == year:
            target_accession = accession
            target_primary = primary_doc
            target_date = filing_date
            logger.info(
                f"Found 10-K for {ticker} FY{year}: accession={accession}, "
                f"primary={primary_doc}, reportDate={report_date}, filingDate={filing_date}"
            )
            break

    if target_accession is None:
        logger.warning(f"No 10-K found for {ticker} FY{year}")
        return None

    # ------------------------------------------------------------------
    # Build document URL directly from submissions data.
    # We already have primaryDocument — no need for the index JSON step.
    # Accession number: 0001234567-23-012345  →  directory 000123456723012345
    # ------------------------------------------------------------------
    accession_clean = target_accession.replace("-", "")
    numeric_cik = cik.lstrip("0")
    base_url = f"{SEC_ARCHIVES}/{numeric_cik}/{accession_clean}"

    if not target_primary:
        logger.warning(f"No primary document listed for {ticker} FY{year}")
        return None

    primary_lower = target_primary.lower()
    pdf_url: Optional[str] = None
    htm_url: Optional[str] = None

    if primary_lower.endswith(".pdf"):
        pdf_url = f"{base_url}/{target_primary}"
    else:
        htm_url = f"{base_url}/{target_primary}"

    # ------------------------------------------------------------------
    # Download the file
    # ------------------------------------------------------------------
    if pdf_url:
        save_path = RAW_DIR / f"{ticker}_{year}_10K.pdf"
        download_url = pdf_url
        ext = ".pdf"
    elif htm_url:
        save_path = RAW_DIR / f"{ticker}_{year}_10K.htm"
        download_url = htm_url
        ext = ".htm"
    else:
        logger.warning(f"No downloadable document found for {ticker} FY{year}")
        return None

    logger.info(f"Downloading {download_url} → {save_path}")
    try:
        dl_resp = session.get(download_url, timeout=120, stream=True)
        dl_resp.raise_for_status()
        with open(save_path, "wb") as fout:
            for chunk in dl_resp.iter_content(chunk_size=8192):
                fout.write(chunk)
        logger.success(f"Saved {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        logger.error(f"Download failed for {ticker} FY{year}: {exc}")
        return None

    time.sleep(1)

    # If we got an HTM, convert it to text
    if ext == ".htm":
        txt_path = convert_htm_to_text(save_path)
        if txt_path:
            return txt_path

    return save_path


def download_all() -> None:
    """
    Download 2022 and 2023 10-Ks for all 5 target companies.
    Logs results to data/download_log.json.
    """
    years = [2022, 2023]
    tasks = [
        (company["ticker"], company["cik"], year)
        for company in TARGET_COMPANIES
        for year in years
    ]

    log_entries: list[dict] = []

    with tqdm(total=len(tasks), desc="Downloading SEC 10-Ks") as pbar:
        for ticker, cik, year in tasks:
            pbar.set_description(f"{ticker} {year}")
            saved_path = download_10k(ticker, cik, year)
            entry = {
                "ticker": ticker,
                "cik": cik,
                "year": year,
                "status": "success" if saved_path else "failed",
                "path": str(saved_path) if saved_path else None,
            }
            log_entries.append(entry)
            pbar.update(1)

    # Write download log
    DOWNLOAD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOWNLOAD_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)

    success_count = sum(1 for e in log_entries if e["status"] == "success")
    logger.success(
        f"Download complete: {success_count}/{len(tasks)} files downloaded. "
        f"Log saved to {DOWNLOAD_LOG_PATH}"
    )


def convert_htm_to_text(htm_path: Path) -> Optional[Path]:
    """
    Convert an HTML/HTM filing to readable plain-text.

    Uses BeautifulSoup when available (much better output).
    Falls back to a careful regex approach that preserves word spacing.
    """
    import re as _re

    htm_path = Path(htm_path)
    txt_path = htm_path.with_suffix(".txt")

    logger.info(f"Converting {htm_path.name} → {txt_path.name}")
    try:
        html_content = htm_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error(f"Failed to read {htm_path}: {exc}")
        return None

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script / style / hidden XBRL metadata entirely
        for tag in soup(["script", "style", "ix:hidden", "ix:header"]):
            tag.decompose()

        # Block-level tags → newline so paragraphs stay separate
        BLOCK = {"p","div","tr","li","h1","h2","h3","h4","h5","h6",
                 "table","thead","tbody","tfoot","section","article"}
        for tag in soup.find_all(True):
            if tag.name in BLOCK:
                tag.insert_before("\n")
                tag.insert_after("\n")

        # get_text with space separator keeps inline words apart
        text = soup.get_text(separator=" ")
        logger.info(f"Converted using BeautifulSoup")

    except ImportError:
        logger.warning("beautifulsoup4 not installed – using regex HTML stripper. "
                       "Run: pip install beautifulsoup4  for better text quality.")

        # Step 1 – remove <script> and <style> blocks entirely
        text = _re.sub(r"<(script|style)[^>]*>.*?</\1>", " ",
                       html_content, flags=_re.DOTALL | _re.IGNORECASE)

        # Step 2 – block-level tags → newline
        text = _re.sub(
            r"</?(?:p|div|tr|li|h[1-6]|table|thead|tbody|section|article)"
            r"(?:\s[^>]*)?>",
            "\n", text, flags=_re.IGNORECASE,
        )

        # Step 3 – <br> → newline,  <td>/<th> → tab separator
        text = _re.sub(r"<br\s*/?>", "\n", text, flags=_re.IGNORECASE)
        text = _re.sub(r"</?(?:td|th)(?:\s[^>]*)?>", " | ", text,
                       flags=_re.IGNORECASE)

        # Step 4 – remove ALL remaining tags, replace with a space
        text = _re.sub(r"<[^>]+>", " ", text)

        # Step 5 – decode HTML entities
        entities = {
            "&nbsp;": " ", "&#160;": " ", "&amp;": "&",
            "&lt;": "<", "&gt;": ">", "&quot;": '"',
            "&apos;": "'", "&ndash;": "–", "&mdash;": "—",
            "&lsquo;": "'", "&rsquo;": "'", "&ldquo;": """, "&rdquo;": """,
        }
        for ent, rep in entities.items():
            text = text.replace(ent, rep)
        # numeric entities  &#NNNN;  or  &#xHHHH;
        text = _re.sub(
            r"&#x([0-9a-fA-F]+);",
            lambda m: chr(int(m.group(1), 16)), text,
        )
        text = _re.sub(
            r"&#([0-9]+);",
            lambda m: chr(int(m.group(1))), text,
        )

    # ── Clean up whitespace ──────────────────────────────────────────────────
    # Collapse runs of spaces/tabs to a single space (but keep newlines)
    text = _re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ consecutive newlines → double newline (paragraph break)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading space on each line
    text = _re.sub(r"^ +", "", text, flags=_re.MULTILINE)
    # Strip leading/trailing whitespace overall
    text = text.strip()

    try:
        txt_path.write_text(text, encoding="utf-8")
        logger.success(f"Converted to text: {txt_path} "
                       f"({len(text):,} chars)")
        return txt_path
    except Exception as exc:
        logger.error(f"Failed to write {txt_path}: {exc}")
        return None


def reconvert_all() -> None:
    """
    Re-convert all existing .htm files in data/raw/ to .txt.
    Run this after  pip install beautifulsoup4  to get clean text.
    """
    htm_files = list(RAW_DIR.glob("*.htm"))
    if not htm_files:
        logger.warning(f"No .htm files found in {RAW_DIR}")
        return

    logger.info(f"Re-converting {len(htm_files)} HTM files…")
    ok = 0
    for htm in tqdm(htm_files, desc="Re-converting"):
        result = convert_htm_to_text(htm)
        if result:
            ok += 1
    logger.success(f"Re-converted {ok}/{len(htm_files)} files → {RAW_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        ROOT_DIR / "data" / "download.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )

    parser = argparse.ArgumentParser(description="SEC 10-K Downloader")
    parser.add_argument(
        "--reconvert", action="store_true",
        help="Re-convert existing .htm files to .txt (run after pip install beautifulsoup4)",
    )
    args = parser.parse_args()

    if args.reconvert:
        reconvert_all()
    else:
        download_all()
