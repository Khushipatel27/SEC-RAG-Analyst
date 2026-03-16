"""
SEC Financial RAG Analyst — Streamlit Dashboard
Dark finance theme | 4 tabs: Setup & Status | Ask the Filings | Company Profiles | RAG Evaluation
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEC RAG Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "http://localhost:8000"

COMPANIES = {
    "AAPL": {"name": "Apple Inc.",      "color": "#A8A9AD", "emoji": "🍎"},
    "MSFT": {"name": "Microsoft Corp.", "color": "#00A4EF", "emoji": "🪟"},
    "GOOGL":{"name": "Alphabet Inc.",   "color": "#34A853", "emoji": "🔍"},
    "AMZN": {"name": "Amazon.com",      "color": "#FF9900", "emoji": "📦"},
    "NVDA": {"name": "NVIDIA Corp.",    "color": "#76B900", "emoji": "🎮"},
}
TICKERS = list(COMPANIES.keys())
YEARS   = ["2022", "2023"]

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark finance theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global background ── */
.stApp { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background: #161b22; }

/* ── top header bar ── */
.rag-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 50%, #1a2332 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.rag-header h1 { margin: 0; font-size: 1.8rem; color: #58a6ff; }
.rag-header p  { margin: 0; color: #8b949e; font-size: 0.9rem; }

/* ── stat cards ── */
.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #58a6ff; }
.stat-card .value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.stat-card .label { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }

/* ── status pills ── */
.pill-ok  { background:#1a3a2a; color:#3fb950; border:1px solid #3fb950;
            padding:4px 12px; border-radius:20px; font-size:0.82rem; display:inline-block; }
.pill-err { background:#3a1a1a; color:#f85149; border:1px solid #f85149;
            padding:4px 12px; border-radius:20px; font-size:0.82rem; display:inline-block; }

/* ── step guide cards ── */
.step-card {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.step-card .step-num { color: #58a6ff; font-weight: 700; font-size: 0.85rem; }
.step-card .step-title { font-weight: 600; font-size: 1rem; margin: 2px 0; }
.step-card .step-desc { color: #8b949e; font-size: 0.85rem; }

/* ── question chip buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}

/* ── answer area: font sizes ── */
[data-testid="stMarkdownContainer"] h1 { font-size: 1.1rem  !important; color: #58a6ff; margin: 14px 0 6px 0 !important; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.0rem  !important; color: #58a6ff; margin: 12px 0 5px 0 !important; }
[data-testid="stMarkdownContainer"] h3 { font-size: 0.93rem !important; color: #79c0ff; margin: 10px 0 4px 0 !important; }
[data-testid="stMarkdownContainer"] p  { font-size: 0.87rem !important; line-height: 1.65 !important; margin: 3px 0 !important; }
[data-testid="stMarkdownContainer"] li { font-size: 0.87rem !important; line-height: 1.6  !important; }
[data-testid="stMarkdownContainer"] strong { color: #e6edf3 !important; }

/* ── markdown tables ── */
[data-testid="stMarkdownContainer"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 12px 0 16px 0 !important;
    font-size: 0.82rem !important;
    border-radius: 6px;
    overflow: hidden;
}
[data-testid="stMarkdownContainer"] thead tr {
    background: #1e3a5f !important;
}
[data-testid="stMarkdownContainer"] th {
    color: #79c0ff !important;
    padding: 8px 12px !important;
    border: 1px solid #30363d !important;
    text-align: left !important;
    font-weight: 600 !important;
    white-space: nowrap;
}
[data-testid="stMarkdownContainer"] td {
    padding: 6px 12px !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    vertical-align: top;
}
[data-testid="stMarkdownContainer"] tbody tr:nth-child(even) td { background: #161b22 !important; }
[data-testid="stMarkdownContainer"] tbody tr:hover td { background: #1c2a3a !important; }

/* ── chat answer box ── */
.answer-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 18px 22px;
    margin-top: 12px;
    line-height: 1.65;
    font-size: 0.88rem;
}

/* ── source chip ── */
.source-chip {
    background: #1c2a3a;
    border: 1px solid #1f6feb;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 10px;
    font-size: 0.82rem;
}

/* ── company badge ── */
.co-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
}

/* ── insight callout ── */
.insight-box {
    background: #1a2a1a;
    border: 1px solid #3fb950;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 16px;
}

/* ── tab label font ── */
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
if "qa_prefill"      not in st.session_state: st.session_state.qa_prefill      = ""
if "eval_result"     not in st.session_state: st.session_state.eval_result     = None
if "ingested_docs"   not in st.session_state: st.session_state.ingested_docs   = []

# Auto-fetch status once on first load so the header is always accurate
if "system_status" not in st.session_state:
    try:
        r = requests.get(f"{API_BASE}/status", timeout=3)
        st.session_state.system_status = r.json() if r.status_code == 200 else {}
    except Exception:
        st.session_state.system_status = {}

# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def api_get(endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return None


def api_post(endpoint: str, payload: dict, timeout: int = 300) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def get_status() -> dict:
    s = api_get("/status") or {}
    st.session_state.system_status = s
    return s


def _reconstruct_broken_table(text: str) -> str:
    """
    Merge single-cell-per-line table fragments into proper markdown table rows.

    The LLM sometimes emits every table cell on its own line:
        |Company
        |Revenue
        |---|
        |---|
        |AAPL
        |$365.3B

    This detects such blocks (lines where each line has only one pipe),
    infers the column count from the header lines before the first separator,
    and rebuilds: | Company | Revenue | ... |
    """
    import re as _re

    _SEP_RE = _re.compile(r"^\s*\|[\s\-|]+\|?\s*$")          # |---|---| lines
    _CELL_RE = _re.compile(r"^\s*\|[^|]*\|?\s*$")             # |single cell| lines

    lines = text.split("\n")
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect start of a broken-table block: line starts with | and has ≤ 1 inner pipe
        if _CELL_RE.match(line) and not _SEP_RE.match(line):
            block: list[str] = []
            j = i
            while j < len(lines) and (_CELL_RE.match(lines[j]) or _SEP_RE.match(lines[j])):
                block.append(lines[j].strip())
                j += 1

            # Only attempt reconstruction if most lines are single-cell (count ≤ 2 pipes)
            single = [l for l in block if not _SEP_RE.match(l) and l.count("|") <= 2]
            seps   = [l for l in block if _SEP_RE.match(l)]

            if len(single) >= 2 and seps:
                # Column count = number of non-sep lines before the first sep line
                n_cols = 0
                for l in block:
                    if _SEP_RE.match(l):
                        break
                    n_cols += 1

                if 2 <= n_cols <= 10:
                    # Extract all non-sep cell values
                    cells = [l.strip().strip("|").strip() for l in block if not _SEP_RE.match(l) and l.strip()]

                    out.append("")
                    # Header row
                    header = cells[:n_cols]
                    out.append("| " + " | ".join(header) + " |")
                    # Separator row
                    out.append("| " + " | ".join(["---"] * n_cols) + " |")
                    # Data rows
                    data = cells[n_cols:]
                    for k in range(0, len(data), n_cols):
                        row = data[k : k + n_cols]
                        while len(row) < n_cols:
                            row.append("")
                        out.append("| " + " | ".join(row) + " |")
                    out.append("")
                    i = j
                    continue

            # Can't reconstruct — pass block through unchanged
            out.extend(block)
            i = j
        else:
            out.append(line)
            i += 1

    return "\n".join(out)


def _fix_llm_spacing(text: str) -> str:
    """
    Full post-processing pass on LLM output → clean, rendered Streamlit markdown.

    Rules applied in order
    ──────────────────────
    0.  wordninja   — split merged all-lowercase tokens (≥14 chars)
    1.  ## headers  — force onto their own line, remove trailing dashes
    2.  Asterisks   — collapse *** / **** → **, fix * *text* * spaced bold
    3.  Tables A    — split ||  (inline row separator) → newline between rows
    4.  Tables B    — reconstruct single-cell-per-line blocks into full rows
    5.  Bullets     — fix "Label:****$X-*" → "- **Label:** $X"
                     remove orphan "- " empty lines
    6.  Numbers     — add space before billion/million/trillion/%
    7.  Dates       — "year2022" → "year 2022"
    8.  camelCase   — insert space at lowercase→uppercase boundary
    9.  Punctuation — space after . , ; ) when followed by a letter
    10. Whitespace  — collapse spaces, trim lines, collapse blank lines
    """
    import re as _re

    # ── 0a. Normalise Unicode pipe variants → ASCII | ─────────────────────────
    # llama3.2 sometimes emits ∣ (U+2223) or ｜ (U+FF5C) instead of |
    for _uc in ("\u2223", "\uff5c", "\u007c\u200b", "\u01c0", "\u2502"):
        text = text.replace(_uc, "|")

    # ── 0b. Collapse numbers split across lines inside table cells ────────────
    # Pattern: digit(s) \n unit-letter(s) \n ∣-or-| → merge onto one line
    # e.g.  "362.1\nB\n|"  →  "362.1B|"
    text = _re.sub(r"(\d[\d,.]*)\s*\n\s*([BMKTbmkt])\s*\n\s*\|", r"\1\2|", text)
    # Also handle:  "$\n362.1\nB"  →  "$362.1B"
    text = _re.sub(r"\$\s*\n\s*([\d,.]+)\s*\n\s*([BMKbmk])", r"$\1\2", text)
    # Merge a lone "B" / "M" line that follows a number line (no pipe)
    text = _re.sub(r"([\d,.]+)\n([BMKTbmkt])\n", r"\1\2\n", text)

    # ── 0. wordninja ──────────────────────────────────────────────────────────
    try:
        import wordninja as _wn
        def _split_tok(m: "_re.Match") -> str:
            parts = _wn.split(m.group(0))
            return " ".join(parts) if len(parts) > 1 else m.group(0)
        text = _re.sub(r"(?<![a-z])[a-z]{14,}(?![a-z])", _split_tok, text)
    except ImportError:
        pass

    # ── 1. Headers ────────────────────────────────────────────────────────────
    # Force blank line BEFORE any ## header embedded in a paragraph
    text = _re.sub(r"(?<!\n)(#{1,6})\s*([A-Za-z])", r"\n\n\1 \2", text)
    # Force blank line AFTER a header line
    text = _re.sub(r"(#{1,6} [^\n]+)\n(?!\n)", r"\1\n\n", text)
    # Remove trailing dash:  ## Section Name-  →  ## Section Name
    text = _re.sub(r"(#{1,6} [^\n]+?)-\s*$", r"\1", text, flags=_re.MULTILINE)

    # ── 2. Asterisks ─────────────────────────────────────────────────────────
    text = _re.sub(r"\*{3,}", "**", text)                      # *** / **** → **
    text = _re.sub(r"\* \*(.+?)\* \*", r"**\1**", text)        # * *text* * → **text**
    text = _re.sub(r"\*\* (.+?) \*\*", r"**\1**", text)        # ** text ** → **text**
    text = _re.sub(r"\*\*([^*\n]+)\*(?!\*)", r"**\1**", text)  # **text* → **text**

    # ── 3. Tables A: inline rows separated by || ──────────────────────────────
    text = _re.sub(r"\|\|", "|\n|", text)
    # Ensure a blank line before the first | of a table block
    text = _re.sub(r"([^\n|])\n(\|)", r"\1\n\n\2", text)

    # ── 4. Tables B: reconstruct single-cell-per-line blocks ──────────────────
    text = _reconstruct_broken_table(text)

    # ── 5. Bullets ────────────────────────────────────────────────────────────
    # "Label:****$365.3billion-*" → "- **Label:** $365.3billion"
    text = _re.sub(
        r"^([A-Z&][A-Za-z&\s/()]{1,40}):\*{2,4}([^\n*]+?)\*?-?\*?\s*$",
        r"- **\1:** \2",
        text, flags=_re.MULTILINE,
    )
    # Remove orphan empty bullet lines
    text = _re.sub(r"^-\s*$", "", text, flags=_re.MULTILINE)

    # ── 6. Numbers: space before unit ────────────────────────────────────────
    text = _re.sub(r"(\d)(billion|million|trillion|%)", r"\1 \2", text, flags=_re.IGNORECASE)

    # ── 7. Dates: year/quarter/month immediately followed by 4-digit year ────
    text = _re.sub(r"(year|quarter|month|fy)(\d{4})", r"\1 \2", text, flags=_re.IGNORECASE)

    # ── 8. camelCase ─────────────────────────────────────────────────────────
    text = _re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # ── 9. Punctuation spacing ────────────────────────────────────────────────
    text = _re.sub(r"\.([A-Za-z])", r". \1", text)
    text = _re.sub(r"([,;])([A-Za-z])", r"\1 \2", text)
    text = _re.sub(r"\)([A-Za-z])", r") \1", text)

    # ── 10. Whitespace ────────────────────────────────────────────────────────
    text = _re.sub(r"[ \t]{2,}", " ", text)
    text = _re.sub(r" +$", "", text, flags=_re.MULTILINE)
    text = _re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def stream_answer(question: str, ticker_f, year_f) -> tuple[str, float]:
    """Call SSE streaming endpoint; return (full_text, elapsed_seconds).

    Renders tokens using st.markdown() so that **bold**, ## headers,
    bullet points, and newlines display correctly — NOT inside a raw HTML div.
    """
    params: dict = {"question": question}
    if ticker_f: params["ticker_filter"] = ticker_f
    if year_f:   params["year_filter"]   = year_f

    answer_text = ""
    start = time.time()

    # Outer styled container — rendered once
    st.markdown(
        '<div style="background:#161b22; border:1px solid #30363d; '
        'border-left:4px solid #3fb950; border-radius:8px; padding:18px 22px; '
        'margin-top:8px;">',
        unsafe_allow_html=True,
    )
    placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        r = requests.get(f"{API_BASE}/ask/stream", params=params, stream=True, timeout=180)
        if r.status_code == 200:
            for line in r.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data:"):
                        token = decoded[5:].strip()
                        if token == "[DONE]":
                            break
                        if token.startswith("[ERROR"):
                            placeholder.error(token)
                            break
                        answer_text += token
                        # Render as proper markdown so **bold** / ## headers work
                        placeholder.markdown(answer_text + " ▌")
        elif r.status_code == 503:
            placeholder.error(f"Service unavailable: {r.text}")
    except Exception as exc:
        placeholder.error(f"Streaming error: {exc}")

    # Final render — apply full spacing fix so merged words become readable
    fixed = _fix_llm_spacing(answer_text)
    placeholder.markdown(fixed)
    return fixed, time.time() - start


def company_color(ticker: str) -> str:
    return COMPANIES.get(ticker, {}).get("color", "#58a6ff")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
status = st.session_state.system_status or {}
is_running   = status.get("ollama_running", False)
chunks_total = status.get("total_chunks", 0)
models_avail = status.get("models_available", [])
llm_ok       = any("llama"  in m.lower() for m in models_avail)
embed_ok     = any("nomic"  in m.lower() or "embed" in m.lower() for m in models_avail)

# Build warning string for header
missing = []
if not is_running:  missing.append("Start the API: uvicorn api.main:app --reload")
if not llm_ok:      missing.append("ollama pull llama3.2")
if not embed_ok:    missing.append("ollama pull nomic-embed-text")

api_pill  = f'<span class="pill-ok">● API Online</span>'  if is_running  else '<span class="pill-err">● API Offline — start uvicorn</span>'
llm_pill  = f'<span class="pill-ok">✓ llama3.2</span>'    if llm_ok      else '<span class="pill-err">✗ llama3.2 — run: ollama pull llama3.2</span>'
emb_pill  = f'<span class="pill-ok">✓ nomic-embed</span>' if embed_ok    else '<span class="pill-err">✗ nomic-embed-text — run: ollama pull nomic-embed-text</span>'

st.markdown(f"""
<div class="rag-header">
  <div style="font-size:2.4rem">📈</div>
  <div>
    <h1>SEC Financial RAG Analyst</h1>
    <p>Ask natural-language questions about Apple · Microsoft · Google · Amazon · NVIDIA 10-K filings</p>
  </div>
  <div style="margin-left:auto; text-align:right; line-height:1.9">
    {api_pill}<br>{llm_pill}<br>{emb_pill}<br>
    <span style="color:#8b949e; font-size:0.78rem">{chunks_total:,} chunks indexed</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Show setup banner if anything is missing
if missing:
    st.warning(
        "**Action needed before you can ask questions:**\n" +
        "\n".join(f"- `{m}`" for m in missing),
        icon="⚠️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🛠️  Setup & Status",
    "💬  Ask the Filings",
    "🏢  Company Profiles",
    "📊  RAG Evaluation",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Setup & Status
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    # ── Left: How-to guide ──────────────────────────────────────────────────
    with left:
        st.markdown("### 🚀 Getting Started — 4 Steps")
        st.markdown("""
<div class="step-card">
  <div class="step-num">STEP 1 · One-time setup</div>
  <div class="step-title">Start Ollama + pull models</div>
  <div class="step-desc">Run in a terminal:<br>
    <code>ollama serve</code><br>
    <code>ollama pull llama3.2</code><br>
    <code>ollama pull nomic-embed-text</code>
  </div>
</div>

<div class="step-card">
  <div class="step-num">STEP 2 · One-time setup</div>
  <div class="step-title">Download SEC 10-K filings</div>
  <div class="step-desc">Run in a terminal (takes ~3 min):<br>
    <code>python scripts/download_sec_docs.py</code><br>
    Downloads 10 filings → <code>data/raw/</code>
  </div>
</div>

<div class="step-card">
  <div class="step-num">STEP 3 · One-time setup</div>
  <div class="step-title">Ingest documents into the index</div>
  <div class="step-desc">Use the <b>Ingest Panel</b> on the right, or run:<br>
    <code>POST /ingest</code> for each ticker/year combination.<br>
    Each filing takes ~1–3 min to parse, chunk & embed.
  </div>
</div>

<div class="step-card" style="border-left-color:#3fb950">
  <div class="step-num" style="color:#3fb950">STEP 4 · Done!</div>
  <div class="step-title">Ask questions</div>
  <div class="step-desc">Go to the <b>💬 Ask the Filings</b> tab and start querying.<br>
    Hybrid search (vector + BM25) + cross-encoder reranking finds the best context.<br>
    llama3.2 generates a cited, grounded answer.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Right: Live status + ingest ─────────────────────────────────────────
    with right:
        st.markdown("### ⚡ System Status")
        refresh_col, _ = st.columns([1, 2])
        if refresh_col.button("🔄 Refresh Status", use_container_width=True):
            with st.spinner("Checking..."):
                status = get_status()

        status = st.session_state.system_status or {}
        models = status.get("models_available", [])
        llm_ok    = any("llama"   in m.lower() for m in models)
        embed_ok  = any("nomic"   in m.lower() or "embed" in m.lower() for m in models)
        chroma_ok = status.get("total_chunks", 0) > 0
        ollama_ok = status.get("ollama_running", False)

        c1, c2 = st.columns(2)
        def pill(ok, label):
            cls = "pill-ok" if ok else "pill-err"
            icon = "✓" if ok else "✗"
            return f'<span class="{cls}">{icon} {label}</span>'

        c1.markdown(pill(ollama_ok,  "Ollama running"),    unsafe_allow_html=True)
        c1.markdown("")
        c1.markdown(pill(llm_ok,    "llama3.2 loaded"),   unsafe_allow_html=True)
        c2.markdown(pill(embed_ok,  "nomic-embed-text"),  unsafe_allow_html=True)
        c2.markdown("")
        c2.markdown(pill(chroma_ok, "ChromaDB indexed"),  unsafe_allow_html=True)

        st.markdown("---")

        # Stats row
        companies_avail = status.get("companies_available", [])
        years_avail     = status.get("years_available", [])
        docs_indexed    = status.get("documents_indexed", 0)
        table_chunks    = status.get("table_chunks", 0)
        db_mb           = status.get("chroma_db_size_mb", 0)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.markdown(f'<div class="stat-card"><div class="value">{docs_indexed}</div><div class="label">Docs Indexed</div></div>', unsafe_allow_html=True)
        sc2.markdown(f'<div class="stat-card"><div class="value">{chunks_total:,}</div><div class="label">Total Chunks</div></div>', unsafe_allow_html=True)
        sc3.markdown(f'<div class="stat-card"><div class="value">{table_chunks:,}</div><div class="label">Table Chunks</div></div>', unsafe_allow_html=True)
        sc4.markdown(f'<div class="stat-card"><div class="value">{db_mb:.1f}</div><div class="label">DB Size (MB)</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📥 Ingest a Document")
        st.caption("Each filing must be downloaded first (Step 2), then ingested here.")

        ig1, ig2 = st.columns(2)
        ingest_ticker = ig1.selectbox("Ticker", TICKERS, key="ig_ticker")
        ingest_year   = ig2.selectbox("Year",   YEARS,   key="ig_year")

        default_path = f"data/raw/{ingest_ticker}_{ingest_year}_10K.txt"
        file_exists  = Path(default_path).exists()

        if file_exists:
            st.success(f"✓ File found: `{default_path}`")
        else:
            st.warning(f"File not found: `{default_path}` — run Step 2 first")

        if st.button(
            f"⚙️ Ingest {ingest_ticker} {ingest_year}",
            type="primary",
            use_container_width=True,
            key="ingest_btn",
            disabled=not file_exists,
        ):
            prog = st.progress(0, text=f"Parsing {ingest_ticker} {ingest_year}…")
            result = None
            with st.spinner("Parsing · Chunking · Embedding · Indexing…"):
                for pct, msg in [(20,"Parsing PDF/text…"),(50,"Chunking…"),(75,"Embedding…"),(90,"Indexing…")]:
                    time.sleep(0.3)
                    prog.progress(pct, text=msg)
                result = api_post("/ingest", {
                    "ticker":    ingest_ticker,
                    "year":      ingest_year,
                    "file_path": default_path,
                }, timeout=600)
            prog.progress(100, text="Done!")
            if result:
                st.success(
                    f"✅ Ingested **{ingest_ticker} {ingest_year}** — "
                    f"{result.get('num_chunks',0):,} chunks "
                    f"({result.get('table_chunks',0)} tables) in "
                    f"{result.get('processing_time_seconds',0):.1f}s"
                )
                get_status()
                st.rerun()

        # Indexed docs table
        docs_data = api_get("/documents")
        if docs_data and docs_data.get("documents"):
            st.markdown("---")
            st.markdown("### 📋 Indexed Documents")
            df_docs = pd.DataFrame(docs_data["documents"])
            if not df_docs.empty:
                display_cols = [c for c in
                    ["ticker","year","num_chunks","table_chunks","text_chunks","processing_time_seconds"]
                    if c in df_docs.columns]
                df_show = df_docs[display_cols].copy()
                df_show.columns = [c.replace("_"," ").title() for c in display_cols]
                st.dataframe(df_show, use_container_width=True, hide_index=True, height=220)

        # ── Re-index all documents ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔄 Re-index All Documents")
        st.caption(
            "Use this after fixing text extraction issues. "
            "Clears the existing ChromaDB index and re-ingests every downloaded filing."
        )
        with st.expander("⚠️ What this does (read before clicking)"):
            st.markdown("""
- Deletes all existing vectors from ChromaDB
- Re-reads every `data/raw/*.txt` file with the latest text-cleaning code
- Re-chunks, re-embeds, and re-indexes all documents
- Takes **10–20 minutes** — keep the API running

**When to use:** After running `python scripts/fix_and_reingest.py`
or if you see garbled / merged words in LLM answers.
""")

        if st.button("🔄 Re-index All Documents", type="secondary",
                     use_container_width=True, key="reingest_all"):
            import glob as _glob
            raw_files = sorted(_glob.glob("data/raw/*.txt"))
            if not raw_files:
                st.error("No .txt files found in `data/raw/`. Run `download_sec_docs.py` first.")
            else:
                total = len(raw_files)
                prog_ri = st.progress(0, text=f"Re-indexing 0/{total}…")
                errors_ri: list[str] = []
                all_chunks = 0
                for idx, fpath in enumerate(raw_files):
                    fname = Path(fpath).stem          # e.g. AAPL_2022_10K
                    parts = fname.split("_")
                    t = parts[0] if len(parts) >= 1 else "UNKNOWN"
                    y = parts[1] if len(parts) >= 2 else "0000"
                    prog_ri.progress(idx / total, text=f"Ingesting {t} {y}… ({idx+1}/{total})")
                    result_ri = api_post("/ingest", {
                        "ticker": t, "year": y, "file_path": fpath,
                    }, timeout=600)
                    if result_ri:
                        all_chunks += result_ri.get("num_chunks", 0)
                    else:
                        errors_ri.append(f"{t} {y}")
                prog_ri.progress(100, text="Done!")
                if errors_ri:
                    st.warning(f"Completed with errors on: {', '.join(errors_ri)}")
                else:
                    st.success(f"✅ Re-indexed {total} documents — {all_chunks:,} total chunks")
                get_status()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Ask the Filings
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Controls row ────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    with ctrl1:
        question_input = st.text_input(
            "Ask a question",
            value=st.session_state.qa_prefill,
            placeholder="e.g. What was Apple's revenue in 2023?",
            label_visibility="collapsed",
            key="qa_input",
        )
    with ctrl2:
        company_filter = st.selectbox("Company", ["All"] + TICKERS,
                                      label_visibility="visible", key="qa_co")
    with ctrl3:
        year_filter = st.selectbox("Year", ["All"] + YEARS,
                                   label_visibility="visible", key="qa_yr")
    with ctrl4:
        ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True, key="ask_btn")

    # ── Example question chips (categorised) ────────────────────────────────
    with st.expander("💡 Example questions — click to use", expanded=not st.session_state.chat_history):
        cat_cols = st.columns(3)

        EXAMPLES = {
            "💰 Financial Figures": [
                "What was Apple's total revenue in fiscal year 2023?",
                "What is Apple's diluted EPS for 2023?",
                "What was NVIDIA's net income in 2023?",
                "What were Microsoft's total assets in 2022?",
            ],
            "📊 Comparisons": [
                "Compare Microsoft and Google R&D spending in 2022",
                "How did Amazon's operating income change from 2022 to 2023?",
                "Which company had the highest revenue growth from 2022 to 2023?",
                "Compare Apple and Microsoft net income margins in 2023",
            ],
            "⚠️ Strategy & Risk": [
                "What are NVIDIA's key risk factors in 2023?",
                "What is Amazon's cloud strategy according to the 2023 10-K?",
                "What competition risks does Apple face?",
                "What does Microsoft say about AI in its 2023 10-K?",
            ],
        }

        for col, (category, questions) in zip(cat_cols, EXAMPLES.items()):
            col.markdown(f"**{category}**")
            for q in questions:
                if col.button(q, key=f"chip_{hash(q)}", use_container_width=True):
                    st.session_state.qa_prefill = q
                    st.rerun()

    st.markdown("---")

    # ── Answer area ─────────────────────────────────────────────────────────
    if ask_btn and question_input.strip():
        ticker_f = company_filter if company_filter != "All" else None
        year_f   = year_filter   if year_filter   != "All" else None

        # Show what we're doing
        context_parts = []
        if ticker_f:
            co = COMPANIES.get(ticker_f, {})
            context_parts.append(f"{co.get('emoji','')} **{ticker_f}**")
        if year_f:
            context_parts.append(f"📅 {year_f}")
        context_parts.append("🔀 Hybrid search + reranking")

        st.markdown(
            "**Context:** " + "  ·  ".join(context_parts) if context_parts
            else "**Context:** All companies · All years · Hybrid search"
        )

        with st.spinner("🔍 Searching filings…"):
            answer_text, elapsed = stream_answer(question_input, ticker_f, year_f)

        # Metadata badges
        meta_result = api_post("/ask", {
            "question":      question_input,
            "ticker_filter": ticker_f,
            "year_filter":   year_f,
            "stream":        False,
        }, timeout=120)

        if meta_result:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("⏱ Latency",   f"{meta_result.get('latency_ms',0):.0f} ms")
            b2.metric("🧠 Model",     meta_result.get("model","?"))
            b3.metric("📄 Chunks",    meta_result.get("num_chunks", 0))
            b4.metric("📝 Prompt",    meta_result.get("prompt_used","?")[:20])

            # Sources
            sources = meta_result.get("sources", [])
            if sources:
                with st.expander(f"📚 View {len(sources)} source chunk(s)"):
                    for i, src in enumerate(sources, 1):
                        ticker_s = src.get("ticker","?")
                        color_s  = company_color(ticker_s)
                        score_s  = src.get("relevance_score", 0) or src.get("rerank_score", 0)
                        type_s   = src.get("block_type","text")
                        type_icon= "📊" if type_s == "table" else "📄"
                        st.markdown(f"""
<div class="source-chip">
  <b>[{i}]</b>
  <span class="co-badge" style="background:{color_s}22; color:{color_s}; border:1px solid {color_s}55">{ticker_s}</span>
  {src.get('year','?')} ·
  {type_icon} {type_s.title()} ·
  Section: <i>{src.get('section','?')[:40]}</i> ·
  Page {src.get('page_num','?')} ·
  Score: <b>{score_s:.3f}</b>
</div>
""", unsafe_allow_html=True)
                        st.caption(src.get("text_preview", src.get("text",""))[:280] + "…")

        # Save to history
        st.session_state.chat_history.append({
            "question":   question_input,
            "answer":     answer_text,
            "ticker":     ticker_f,
            "year":       year_f,
            "latency_ms": meta_result.get("latency_ms",0) if meta_result else 0,
            "ts":         time.strftime("%H:%M"),
        })
        st.session_state.chat_history = st.session_state.chat_history[-6:]
        st.session_state.qa_prefill   = ""

    elif ask_btn:
        st.warning("Please enter a question above.")

    # ── Chat history ────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 🕑 Recent Questions")
        for item in reversed(st.session_state.chat_history):
            ticker_h = item.get("ticker","")
            co_info  = COMPANIES.get(ticker_h, {})
            badge    = f"{co_info.get('emoji','')} {ticker_h}" if ticker_h else "All"
            with st.expander(
                f"[{item.get('ts','')}] {item['question'][:70]}{'…' if len(item['question'])>70 else ''}",
                expanded=False,
            ):
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(item["answer"])  # render as markdown, not raw HTML
                c1h, c2h, c3h = st.columns(3)
                c1h.caption(f"Company: {badge}")
                c2h.caption(f"Year: {item.get('year','All')}")
                c3h.caption(f"Latency: {item.get('latency_ms',0):.0f} ms")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Company Profiles
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:

    # ── Selector row ────────────────────────────────────────────────────────
    co_col, yr_col, btn_col = st.columns([2, 1, 1])
    with co_col:
        profile_ticker = st.selectbox(
            "Company",
            TICKERS,
            format_func=lambda t: f"{COMPANIES[t]['emoji']} {t} — {COMPANIES[t]['name']}",
            key="prof_ticker",
        )
    with yr_col:
        profile_year = st.selectbox("Year", YEARS, key="prof_year")
    with btn_col:
        st.write("")
        load_btn = st.button("📊 Load Profile", type="primary",
                             use_container_width=True, key="load_profile")

    co_info  = COMPANIES[profile_ticker]
    co_color = co_info["color"]

    # ── Company hero banner ──────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{co_color}22 0%,#161b22 100%);
     border:1px solid {co_color}55; border-radius:12px;
     padding:20px 28px; margin:12px 0; display:flex; align-items:center; gap:16px">
  <span style="font-size:3rem; line-height:1">{co_info['emoji']}</span>
  <div>
    <div style="font-size:1.7rem; font-weight:800; color:{co_color}; letter-spacing:-0.5px">
      {co_info['name']}
    </div>
    <div style="color:#8b949e; font-size:0.95rem; margin-top:2px">
      <b style="color:#e6edf3">{profile_ticker}</b> · Fiscal Year {profile_year} · SEC 10-K Annual Report
    </div>
  </div>
  <div style="margin-left:auto; text-align:right">
    <div style="color:{co_color}; font-size:0.85rem; font-weight:600">TICKER</div>
    <div style="font-size:2rem; font-weight:800; color:#e6edf3">{profile_ticker}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Key metrics ─────────────────────────────────────────────────────────
    def fmt_val(v, prefix="$", suffix=""):
        """Auto-scale: show B for billions, M for millions."""
        if v is None: return "N/A"
        try:
            n = float(str(v).replace(",", "").replace("$", ""))
            if n >= 1_000_000: return f"{prefix}{n/1_000_000:.1f}T{suffix}"
            if n >= 1_000:     return f"{prefix}{n/1_000:.1f}B{suffix}"
            return f"{prefix}{n:.1f}M{suffix}"
        except:
            return str(v)

    if load_btn:
        with st.spinner(f"Loading {profile_ticker} {profile_year} metrics…"):
            metrics_data = api_get(f"/metrics/{profile_ticker}/{profile_year}")

        if metrics_data:
            km = metrics_data.get("key_metrics", {})
            st.markdown(f"#### 📊 {profile_ticker} FY{profile_year} — Key Financial Metrics")
            st.caption("Values extracted from the SEC 10-K filing via regex. Units auto-scaled.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("💵 Revenue",     fmt_val(km.get("total_revenue")))
            m2.metric("📈 Net Income",  fmt_val(km.get("net_income")))
            m3.metric("⚙️ Op. Income",  fmt_val(km.get("operating_income")))
            m4.metric("🔬 R&D Spend",   fmt_val(km.get("rd_expense")))
            # EPS is already a per-share dollar value — do NOT run through fmt_val
            eps_raw = km.get("eps_diluted")
            try:
                eps_disp = f"${float(str(eps_raw).replace(',','')):.2f}"
            except Exception:
                eps_disp = str(eps_raw) if eps_raw else "N/A"
            m5.metric("📌 EPS (dil.)", eps_disp)

            # Raw metrics detail table
            with st.expander("🔍 Raw extracted metrics"):
                st.json(km)
        else:
            st.warning(
                f"⚠️ No metrics found for **{profile_ticker} {profile_year}**. "
                "Make sure this filing has been ingested — go to the **🛠️ Setup & Status** tab."
            )

    st.markdown("---")

    # ── Financial trend charts ───────────────────────────────────────────────
    st.markdown("### 📉 Financial Trends — All Indexed Companies")
    st.caption("Charts populate automatically as you ingest more filings.")

    docs_resp  = api_get("/documents")
    chart_rows = []
    if docs_resp and docs_resp.get("documents"):
        for row in docs_resp["documents"]:
            km  = row.get("key_metrics") or {}
            rev = km.get("total_revenue")
            ni  = km.get("net_income")
            rd  = km.get("rd_expense")
            try:
                rev_f = float(str(rev).replace(",","")) if rev else None
                ni_f  = float(str(ni).replace(",",""))  if ni  else None
                rd_f  = float(str(rd).replace(",",""))  if rd  else None
                # Auto-scale to billions (> 1000 means millions → divide by 1000)
                chart_rows.append({
                    "Ticker":     row["ticker"],
                    "Year":       str(row["year"]),
                    "Revenue ($B)":    round(rev_f / 1000, 1) if rev_f and rev_f > 500 else rev_f,
                    "Net Income ($B)": round(ni_f  / 1000, 1) if ni_f  and abs(ni_f) > 500 else ni_f,
                    "R&D ($B)":        round(rd_f  / 1000, 1) if rd_f  and rd_f > 500 else rd_f,
                    "Revenue_raw":     rev_f,
                    "RD_raw":          rd_f,
                })
            except: pass

    if chart_rows:
        cdf       = pd.DataFrame(chart_rows)
        color_map = {t: company_color(t) for t in TICKERS}

        ch1, ch2 = st.columns(2)

        with ch1:
            rev_df = cdf.dropna(subset=["Revenue ($B)"])
            if not rev_df.empty:
                fig_rev = px.bar(
                    rev_df, x="Ticker", y="Revenue ($B)", color="Ticker",
                    facet_col="Year",
                    title="Annual Revenue by Company",
                    color_discrete_map=color_map,
                    template="plotly_dark",
                    text_auto=".1f",
                )
                fig_rev.update_traces(textposition="outside")
                fig_rev.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    showlegend=False, height=380,
                    yaxis_title="Revenue ($B)",
                )
                st.plotly_chart(fig_rev, use_container_width=True)

        with ch2:
            ni_df = cdf.dropna(subset=["Net Income ($B)"])
            if not ni_df.empty:
                fig_ni = px.bar(
                    ni_df, x="Ticker", y="Net Income ($B)", color="Ticker",
                    facet_col="Year",
                    title="Net Income by Company",
                    color_discrete_map=color_map,
                    template="plotly_dark",
                    text_auto=".1f",
                )
                fig_ni.update_traces(textposition="outside")
                fig_ni.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    showlegend=False, height=380,
                    yaxis_title="Net Income ($B)",
                )
                st.plotly_chart(fig_ni, use_container_width=True)

        rd_df = cdf.dropna(subset=["R&D ($B)", "Revenue_raw"]).copy()
        if not rd_df.empty and (rd_df["Revenue_raw"] > 0).all():
            rd_df["R&D % of Revenue"] = (
                rd_df["RD_raw"] / rd_df["Revenue_raw"] * 100
            ).round(1)
            fig_rd = px.bar(
                rd_df, x="Ticker", y="R&D % of Revenue", color="Ticker",
                facet_col="Year",
                title="R&D Investment as % of Revenue  (higher = more innovation spend)",
                color_discrete_map=color_map,
                template="plotly_dark",
                text_auto=".1f",
            )
            fig_rd.update_traces(textposition="outside")
            fig_rd.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                showlegend=False, height=360,
            )
            st.plotly_chart(fig_rd, use_container_width=True)

    else:
        st.markdown("""
<div style="background:#161b22; border:1px dashed #30363d; border-radius:10px;
     padding:32px; text-align:center; color:#8b949e;">
  <div style="font-size:2rem">📂</div>
  <div style="font-weight:600; margin:8px 0">No documents indexed yet</div>
  <div style="font-size:0.85rem">Go to <b>🛠️ Setup & Status</b> and ingest at least one filing to see charts here.</div>
</div>
""", unsafe_allow_html=True)

    # ── AI Executive Summary ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 AI Executive Summary")
    st.caption(
        f"Uses hybrid RAG search + llama3.2 to generate a structured summary of "
        f"**{profile_ticker} FY{profile_year}** from the actual 10-K filing text."
    )

    sum_col1, sum_col2 = st.columns([1, 2])
    with sum_col1:
        st.markdown(f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px;">
  <div style="font-weight:700; margin-bottom:8px">What this generates:</div>
  <ul style="color:#8b949e; font-size:0.85rem; margin:0; padding-left:18px">
    <li>Business overview</li>
    <li>Revenue &amp; profitability</li>
    <li>Key risks from Risk Factors</li>
    <li>Strategic initiatives from MD&amp;A</li>
    <li>All grounded in actual filing text</li>
  </ul>
</div>
""", unsafe_allow_html=True)
        st.write("")
        gen_btn = st.button(
            f"✨ Generate {profile_ticker} {profile_year} Summary",
            key="summary_btn", type="primary", use_container_width=True,
        )

    with sum_col2:
        if gen_btn:
            q = (
                f"You are a financial analyst. Write a clear, well-formatted executive "
                f"summary of {profile_ticker} ({co_info['name']}) for fiscal year {profile_year} "
                f"based on their SEC 10-K filing.\n\n"
                f"Use this exact structure with proper markdown:\n\n"
                f"## Company Overview\n"
                f"Write 2-3 sentences about the business.\n\n"
                f"## Key Financial Metrics\n"
                f"- **Revenue:** [value]\n"
                f"- **Net Income:** [value]\n"
                f"- **Operating Margin:** [value]\n"
                f"- **R&D Investment:** [value]\n\n"
                f"## Top Business Risks\n"
                f"- Risk 1\n- Risk 2\n- Risk 3\n\n"
                f"## Strategic Outlook\n"
                f"Write 2-3 sentences about strategy and future direction.\n\n"
                f"Use actual numbers from the filing. Write in plain English with proper spaces between all words."
            )

            # Stream the raw answer
            raw_parts: list[str] = []
            start_t = time.time()

            st.markdown(
                f'<div style="background:#161b22; border:1px solid #30363d; '
                f'border-left:4px solid {co_color}; border-radius:8px; padding:20px 24px;">',
                unsafe_allow_html=True,
            )
            stream_placeholder = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

            try:
                params_s: dict = {
                    "question": q,
                    "ticker_filter": profile_ticker,
                    "year_filter": profile_year,
                }
                r = requests.get(
                    f"{API_BASE}/ask/stream", params=params_s,
                    stream=True, timeout=180,
                )
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")
                            if decoded.startswith("data:"):
                                token = decoded[5:].strip()
                                if token == "[DONE]":
                                    break
                                if token.startswith("[ERROR"):
                                    stream_placeholder.error(token)
                                    break
                                raw_parts.append(token)
                                # Live preview — fix spacing on partial text
                                live = _fix_llm_spacing("".join(raw_parts))
                                stream_placeholder.markdown(live + " ▌")
            except Exception as exc:
                stream_placeholder.error(f"Streaming error: {exc}")

            # Final clean render with full post-processing
            final_text = _fix_llm_spacing("".join(raw_parts))
            stream_placeholder.markdown(final_text)
            elapsed = time.time() - start_t

            # Footer bar
            st.markdown(
                f'<div style="display:flex; gap:12px; margin-top:8px; font-size:0.78rem; color:#8b949e;">'
                f'<span>⏱ {elapsed:.1f}s</span>'
                f'<span>·</span>'
                f'<span>📄 {profile_ticker} {profile_year} 10-K</span>'
                f'<span>·</span>'
                f'<span>🧠 llama3.2</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:
            st.markdown(f"""
<div style="background:#161b22; border:1px dashed #30363d; border-radius:8px;
     padding:40px; text-align:center; color:#8b949e;">
  <span style="font-size:2rem">{co_info['emoji']}</span>
  <div style="font-weight:600; margin:8px 0">Click to generate an AI summary</div>
  <div style="font-size:0.82rem">
    Requires <b>{profile_ticker} {profile_year}</b> to be ingested in the Setup tab
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAG Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown("### 📊 RAG Evaluation Dashboard")
    st.caption(
        "This tab shows how well the retrieval pipeline performs across 20 hand-crafted "
        "financial Q&A pairs — the key differentiator of this project."
    )

    run_col, info_col = st.columns([1, 2])

    with run_col:
        st.markdown("#### Run Evaluation")
        st.caption("Runs all 20 questions through the full pipeline. Takes 5–10 min.")
        if st.button("▶ Run Fresh Evaluation", type="primary",
                     use_container_width=True, key="run_eval"):
            bar = st.progress(0, text="Starting…")
            with st.spinner("Running 20 evaluation questions…"):
                for p, msg in [
                    (10,"Retrieving chunks…"),
                    (40,"Generating answers…"),
                    (70,"Computing metrics…"),
                    (90,"Saving results…"),
                ]:
                    time.sleep(0.5)
                    bar.progress(p, text=msg)
                result = api_post("/evaluate", {}, timeout=900)
            bar.progress(100, text="✓ Complete")
            if result:
                st.session_state.eval_result = result
                st.success("Evaluation complete!")

        st.markdown("---")
        st.markdown("#### What's being measured?")
        st.markdown("""
| Metric | What it tests |
|---|---|
| **Keyword Overlap** | % reference keywords in answer |
| **Numerical Acc.** | Key numbers appear in answer |
| **Citation Rate** | Does answer cite sources? |
| **MRR** | Mean Reciprocal Rank of retrieval |
| **Recall@5** | Are relevant chunks in top 5? |
| **Latency** | End-to-end response time |
""")

    with info_col:
        # Try loading eval data
        eval_data = st.session_state.get("eval_result")
        if not eval_data:
            p = Path("data/eval/eval_results.json")
            if p.exists():
                eval_data = json.loads(p.read_text())
                st.session_state.eval_result = eval_data

        if eval_data:
            st.markdown("#### Answer Quality Metrics")
            em1, em2, em3, em4 = st.columns(4)
            kw = eval_data.get("avg_keyword_overlap", 0)
            na = eval_data.get("avg_numerical_accuracy", 0)
            cr = eval_data.get("citation_rate", 0)
            mr = eval_data.get("avg_mrr", 0)

            em1.metric("Keyword Overlap",  f"{kw*100:.1f}%")
            em2.metric("Numerical Acc.",   f"{na*100:.1f}%")
            em3.metric("Citation Rate",    f"{cr*100:.1f}%")
            em4.metric("Avg MRR",          f"{mr:.3f}")

            per_q = eval_data.get("per_question_results", [])
            if per_q:
                pq_df = pd.DataFrame(per_q)
                display_cols = [c for c in
                    ["question_id","question_type","difficulty","ticker","year",
                     "keyword_overlap","numerical_accuracy","latency_ms"]
                    if c in pq_df.columns]
                st.markdown("#### Per-Question Results")
                st.dataframe(
                    pq_df[display_cols].round(3),
                    use_container_width=True, hide_index=True, height=250,
                )

                if "question_type" in pq_df.columns and "keyword_overlap" in pq_df.columns:
                    type_perf = (
                        pq_df.groupby("question_type")["keyword_overlap"]
                        .mean().reset_index()
                        .rename(columns={"keyword_overlap":"Avg Keyword Overlap"})
                        .sort_values("Avg Keyword Overlap", ascending=False)
                    )
                    fig_type = px.bar(
                        type_perf, x="question_type", y="Avg Keyword Overlap",
                        color="question_type", title="Performance by Question Type",
                        template="plotly_dark",
                        labels={"question_type":"Question Type"},
                    )
                    fig_type.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_type, use_container_width=True)

                    best  = type_perf.iloc[0]["question_type"]
                    worst = type_perf.iloc[-1]["question_type"]
                    st.markdown(f"""
<div class="insight-box">
  💡 <b>Key Insight:</b> The system performs best on <b>{best}</b> questions
  and has the most room to improve on <b>{worst}</b> questions.
  Adding re-ranking typically improves Recall@5 by 15–25% over vector-only baselines.
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No evaluation results yet. Click **▶ Run Fresh Evaluation** to generate.")

    # ── Retrieval Method Comparison ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Retrieval Method Comparison")
    st.caption("Shows how hybrid search + reranking outperforms single-method retrieval")

    if st.button("▶ Run Retrieval Comparison", key="run_comparison"):
        with st.spinner("Comparing vector / BM25 / hybrid / hybrid+rerank… (~1–2 min)"):
            comp_result = api_post("/evaluate/compare", {}, timeout=300)
        if comp_result:
            st.success("Comparison complete! Results below.")
            st.rerun()
        else:
            st.error("Comparison failed — check the API logs.")

    comp_path = Path("data/eval/retrieval_comparison.json")
    if comp_path.exists():
        comp_data = json.loads(comp_path.read_text())
        comp_rows = [
            {"Method": m,
             "Recall@5":    d.get("avg_recall_at_k", 0),
             "Precision@5": d.get("avg_precision_at_k", 0),
             "MRR":         d.get("avg_mrr", 0)}
            for m, d in comp_data.items()
        ]
        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            c_left, c_right = st.columns([1, 2])

            c_left.markdown("**Results table**")
            c_left.dataframe(
                comp_df.round(4),
                use_container_width=True, hide_index=True,
            )

            fig_comp = go.Figure()
            metrics_comp = ["Recall@5", "Precision@5", "MRR"]
            colors_comp  = ["#58a6ff", "#3fb950", "#f78166"]
            for metric, color in zip(metrics_comp, colors_comp):
                fig_comp.add_trace(go.Bar(
                    name=metric,
                    x=comp_df["Method"],
                    y=comp_df[metric],
                    marker_color=color,
                ))
            fig_comp.update_layout(
                barmode="group",
                title="Retrieval Method Comparison",
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            c_right.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No comparison data yet. Click **▶ Run Retrieval Comparison** above.")
