# SEC Financial RAG Analyst — Complete Run Guide

## What You Need Running (3 separate terminals)

```
Terminal 1: Ollama (always running)
Terminal 2: FastAPI backend  →  http://localhost:8000
Terminal 3: Streamlit dashboard  →  http://localhost:8501
```

---

## FIRST-TIME SETUP (do once)

### 1. Activate your conda environment
```bash
conda activate rag_finance
cd sec-rag-analyst
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install beautifulsoup4 langchain-text-splitters
```

### 3. Pull the Ollama models (required — ~4 GB total)
```bash
ollama pull llama3.2           # ~2.0 GB  — the LLM that generates answers
ollama pull nomic-embed-text   # ~274 MB  — the embedding model for search
```
> These only need to be pulled once. Verify with: `ollama list`

### 4. Download SEC 10-K filings (one-time, ~3 min)
```bash
python scripts/download_sec_docs.py
```
Downloads 10 filings (AAPL, MSFT, GOOGL, AMZN, NVDA × 2022 & 2023) to `data/raw/`.

---

## EVERY TIME YOU USE THE APP

### Terminal 1 — Ollama (if not already running)
```bash
ollama serve
```
Leave this running in the background.

### Terminal 2 — Start the API
```bash
conda activate rag_finance
cd sec-rag-analyst
uvicorn api.main:app --reload
```
You should see: `Application startup complete.`
API docs available at: http://localhost:8000/docs

### Terminal 3 — Start the Dashboard
```bash
conda activate rag_finance
cd sec-rag-analyst
streamlit run app/streamlit_app.py
```
Opens automatically at: http://localhost:8501

---

## USING THE DASHBOARD (tab by tab)

### Tab 1 — Setup & Status
This is your control center and health check.

**Status pills (top right of header):**
| Pill | What it means | Fix if red |
|---|---|---|
| API Online | FastAPI is reachable | Run Terminal 2 |
| ✓ llama3.2 | LLM model is pulled | `ollama pull llama3.2` |
| ✓ nomic-embed | Embedding model is pulled | `ollama pull nomic-embed-text` |
| ChromaDB indexed | Documents are in the index | Ingest documents (see below) |

**Ingest a document:**
1. Select a ticker (e.g. AAPL) and year (e.g. 2023)
2. If the file was downloaded, you'll see a green "File found" message
3. Click **⚙️ Ingest AAPL 2023** — takes 1–3 minutes
4. Repeat for each company/year you want to query

> Tip: Ingest at least AAPL 2023 first to test the system end-to-end.

---

### Tab 2 — Ask the Filings
The main Q&A interface. **Requires at least one document ingested.**

**How to use:**
1. Type a question in the text box at the top — OR click one of the example chips
2. Optionally filter by Company and/or Year on the right
3. Click **🔍 Ask**
4. Watch the answer stream in real-time (like ChatGPT)
5. Click **"View X source chunks"** to see exactly which filing sections were used

**Good questions to start with:**
```
What was Apple's total revenue in fiscal year 2023?
What are NVIDIA's key risk factors in 2023?
How did Amazon's operating income change from 2022 to 2023?
What is Microsoft's cloud revenue strategy?
Compare Apple and Microsoft net income in 2023
```

**What the filters do:**
- **Company filter** — restricts search to only that company's filings
- **Year filter** — restricts to a specific fiscal year
- Leave both as "All" for cross-company/cross-year questions

---

### Tab 3 — Company Profiles
Visual financial dashboard for each indexed company.

**How to use:**
1. Select a Company and Year from the dropdowns
2. Click **📊 Load Profile** to see key metrics (Revenue, Net Income, R&D, EPS)
3. Charts below show trends across all indexed companies automatically
4. Click **✨ Generate Summary** for an AI-written executive summary

**Charts:**
- Revenue by company per year (bar chart)
- Net income trend over time (line chart)
- R&D as % of revenue comparison (shows who invests most in innovation)

---

### Tab 4 — RAG Evaluation
Measures how well the system actually works. **Requires all 10 documents ingested.**

**How to use:**
1. Click **▶ Run Fresh Evaluation** — takes 5–10 minutes
2. Reads 20 hand-crafted Q&A pairs from `data/eval/eval_questions.json`
3. Runs each through the full pipeline and scores the answers
4. Results saved to `data/eval/eval_results.json`

**Metrics explained:**
| Metric | What it means | Good score |
|---|---|---|
| Keyword Overlap | % of reference answer keywords found | > 0.5 |
| Numerical Accuracy | Key numbers appear in answer | > 0.6 |
| Citation Rate | Answers include source citations | > 0.8 |
| Avg MRR | Mean Reciprocal Rank of retrieval | > 0.5 |

---

## TROUBLESHOOTING

### "API Offline" in header
→ Start Terminal 2: `uvicorn api.main:app --reload`

### "✗ llama3.2 loaded"
→ Run: `ollama pull llama3.2`
→ Verify: `ollama list`

### "✗ nomic-embed-text"
→ Run: `ollama pull nomic-embed-text`
→ Verify: `ollama list`

### "model not found" error when asking questions
→ The embedding model isn't pulled. Run: `ollama pull nomic-embed-text`

### "ChromaDB indexed = 0" / no answers
→ No documents have been ingested. Go to Tab 1 and ingest filings.

### "File not found" in ingest panel
→ Download the filings first: `python scripts/download_sec_docs.py`

### 500 error when asking questions
→ Check the API terminal (Terminal 2) for the full error message

### Streamlit shows blank / errors on startup
→ Make sure the API is running first, then start Streamlit

---

## QUICK REFERENCE

```bash
# Full startup sequence (copy-paste into 3 terminals)

# Terminal 1
ollama serve

# Terminal 2
conda activate rag_finance
cd sec-rag-analyst
uvicorn api.main:app --reload

# Terminal 3
conda activate rag_finance
cd sec-rag-analyst
streamlit run app/streamlit_app.py
```

---

## FILE LOCATIONS

```
sec-rag-analyst/
├── data/raw/              ← downloaded SEC filings (.txt)
├── data/chroma_db/        ← vector database (auto-created on ingest)
├── data/bm25_index.pkl    ← keyword search index (auto-created)
├── data/eval/             ← evaluation questions & results
├── api/main.py            ← FastAPI backend (port 8000)
└── app/streamlit_app.py   ← Streamlit dashboard (port 8501)
```
