"""
fix_and_reingest.py
───────────────────
One-shot script that:
  1. Checks / installs beautifulsoup4
  2. Re-converts every HTM file → clean TXT with proper word spacing
  3. Wipes the old garbled ChromaDB + BM25 index
  4. Re-ingests all TXT files through the full pipeline

Run once from the sec-rag-analyst directory:
    python scripts/fix_and_reingest.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "raw"
CHROMA    = ROOT / "data" / "chroma_db"
BM25      = ROOT / "data" / "bm25_index.pkl"
PROC_DIR  = ROOT / "data" / "processed"

# ── add project root to sys.path so src.* imports work ───────────────────────
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — ensure beautifulsoup4
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Checking beautifulsoup4")
print("="*60)
try:
    from bs4 import BeautifulSoup
    print("✓ beautifulsoup4 already installed")
except ImportError:
    print("Installing beautifulsoup4...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
    from bs4 import BeautifulSoup
    print("✓ beautifulsoup4 installed")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — re-convert HTM → TXT with BeautifulSoup
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Re-converting HTM files to clean TXT")
print("="*60)

import re

def htm_to_clean_text(htm_path: Path) -> str:
    """Convert an HTM filing to clean, properly-spaced plain text."""
    html = htm_path.read_text(encoding="utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags entirely
    for tag in soup(["script", "style", "ix:hidden", "ix:header",
                     "head", "meta", "link"]):
        tag.decompose()

    # Insert newlines around block-level elements
    BLOCK = {"p","div","tr","li","h1","h2","h3","h4","h5","h6",
             "table","thead","tbody","tfoot","section","article","br"}
    for tag in soup.find_all(True):
        if tag.name in BLOCK:
            tag.insert_before("\n")
            tag.insert_after("\n")

    # get_text with space separator — keeps inline words separated
    text = soup.get_text(separator=" ")

    # Clean up
    text = re.sub(r"[ \t]{2,}", " ", text)          # multiple spaces → one
    text = re.sub(r" \n", "\n", text)               # trailing space before newline
    text = re.sub(r"\n ", "\n", text)               # leading space after newline
    text = re.sub(r"\n{3,}", "\n\n", text)           # 3+ newlines → 2
    return text.strip()


htm_files = sorted(RAW_DIR.glob("*.htm"))
if not htm_files:
    print(f"No .htm files found in {RAW_DIR}")
    print("Run: python scripts/download_sec_docs.py  first")
    sys.exit(1)

converted = 0
for htm in htm_files:
    txt = htm.with_suffix(".txt")
    print(f"  Converting {htm.name} ...", end=" ", flush=True)
    try:
        text = htm_to_clean_text(htm)
        txt.write_text(text, encoding="utf-8")
        kb = len(text) // 1024
        print(f"✓  {kb:,} KB")
        converted += 1
    except Exception as e:
        print(f"✗  ERROR: {e}")

print(f"\n✓ Converted {converted}/{len(htm_files)} files")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — wipe old index
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Clearing old index")
print("="*60)

if CHROMA.exists():
    shutil.rmtree(CHROMA)
    print(f"✓ Deleted {CHROMA}")
else:
    print(f"  (chroma_db not found — nothing to delete)")

if BM25.exists():
    BM25.unlink()
    print(f"✓ Deleted {BM25}")
else:
    print(f"  (bm25_index.pkl not found — nothing to delete)")

if PROC_DIR.exists():
    shutil.rmtree(PROC_DIR)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Cleared {PROC_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — re-ingest all TXT files
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Re-ingesting all documents")
print("="*60)

txt_files = sorted(RAW_DIR.glob("*.txt"))
if not txt_files:
    print("No .txt files found — re-conversion may have failed.")
    sys.exit(1)

print(f"Found {len(txt_files)} files to ingest:\n")
for f in txt_files:
    print(f"  {f.name}")

print()

try:
    from src.pipeline import SECRAGPipeline
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

    pipeline = SECRAGPipeline()
    total_chunks = 0

    for txt_file in txt_files:
        # Extract ticker and year from filename: AAPL_2022_10K.txt
        parts = txt_file.stem.split("_")
        ticker = parts[0] if len(parts) >= 1 else "UNKNOWN"
        year   = parts[1] if len(parts) >= 2 else "0000"

        print(f"\n─── Ingesting {ticker} {year} ───")
        try:
            result = pipeline.ingest_document(
                file_path=txt_file,
                ticker=ticker,
                year=year,
            )
            chunks = result.get("num_chunks", 0)
            tables = result.get("table_chunks", 0)
            secs   = result.get("processing_time_seconds", 0)
            total_chunks += chunks
            print(f"  ✓ {chunks:,} chunks ({tables} tables) in {secs:.1f}s")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print("\n" + "="*60)
    print(f"✅  ALL DONE — {total_chunks:,} total chunks indexed")
    print("="*60)
    print("\nRestart the API and Streamlit, then refresh status.\n")

except Exception as e:
    print(f"\n✗ Pipeline error: {e}")
    print("Make sure the API is NOT running while ingesting.")
    raise
