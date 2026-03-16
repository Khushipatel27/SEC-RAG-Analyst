<div align="center">

# 📊 SEC Financial RAG Analyst

### Ask plain-English questions about Apple, Microsoft, Google, Amazon & NVIDIA's annual filings — and get cited, grounded answers in real time.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 💡 What It Does

A **production-grade RAG system** that ingests real SEC 10-K filings and answers financial questions with cited, grounded responses — all running locally with **zero API cost**.

> *"Compare Apple and NVIDIA's profitability in 2022"*
> *"What are the top risks Microsoft disclosed in their 2023 10-K?"*
> *"What was Amazon's revenue growth between 2022 and 2023?"*

The system retrieves the most relevant passages from 10 real SEC filings, reranks them with a cross-encoder, and streams a structured answer with citations — token by token in the browser.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  Setup & Status │ Ask the Filings │ Company Profiles │ RAG  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE streaming
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend  :8000                      │
│  POST /ingest │ POST /ask │ GET /ask/stream │ GET /status   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    SECRAGPipeline                            │
│                                                             │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  Ingestion   │  │  Hybrid Search  │  │  Generation   │  │
│  │              │  │                 │  │               │  │
│  │ pdfplumber   │  │ ChromaDB vector │  │ Prompt Router │  │
│  │ BeautifulSoup│  │       +         │  │ llama3.2      │  │
│  │ 512-token    │  │ BM25Okapi sparse│  │ SSE streaming │  │
│  │ chunks       │  │       ↓         │  │               │  │
│  │ 64 overlap   │  │  RRF Fusion     │  │ Cross-Encoder │  │
│  └──────────────┘  │       ↓         │  │ Reranker      │  │
│                    │  Rerank top-5   │  └───────────────┘  │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
          │                              │
  ChromaDB (vectors)              BM25 index (.pkl)
  nomic-embed-text                rank-bm25
```

---

## 🚀 Why This Stands Out

Most RAG projects use vector search alone. This project implements the **full production retrieval stack**:

| Method | Strengths | Weakness |
|:---|:---|:---|
| Vector only | Semantic similarity | Misses exact numbers / names |
| BM25 only | Exact keyword matches | Misses paraphrased content |
| **Hybrid + RRF** | Semantic + lexical | Best of both worlds |
| **+ Cross-encoder rerank** | Precision re-scoring | Eliminates false positives |

> **Hybrid + reranking achieves ~52% higher Recall@5 vs vector-only baseline** — measured on the built-in 20-question evaluation suite.

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| 🤖 LLM | Ollama — llama3.2 (local, no API cost) |
| 🔀 Fallback LLM | Ollama — mistral |
| 🔢 Embeddings | nomic-embed-text via Ollama (768-dim) |
| 🗄️ Vector Store | ChromaDB — persistent, cosine similarity, HNSW index |
| 🔍 Sparse Search | BM25Okapi via rank-bm25 |
| ⚖️ Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| 📄 PDF Parsing | pdfplumber (primary), PyMuPDF fitz (fallback) |
| 🌐 HTML Parsing | BeautifulSoup4 — SEC EDGAR HTM filings |
| ⚡ API | FastAPI + SSE streaming (sse-starlette) |
| 🎨 Frontend | Streamlit — dark finance theme |
| ⚙️ Config | Pydantic-settings + .env |

---

## 📂 Dataset

10 real SEC 10-K annual filings downloaded directly from **SEC EDGAR**:

| Company | Ticker | FY2022 | FY2023 |
|:---|:---:|:---:|:---:|
| Apple Inc. | `AAPL` | ✅ | ✅ |
| Microsoft Corp. | `MSFT` | ✅ | ✅ |
| Alphabet Inc. | `GOOGL` | ✅ | ✅ |
| Amazon.com | `AMZN` | ✅ | ✅ |
| NVIDIA Corp. | `NVDA` | ✅ | ✅ |

Each filing is parsed, chunked into 512-token segments with 64-token overlap, embedded, and indexed in both ChromaDB and BM25.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- 8 GB RAM minimum (16 GB recommended)
- ~2 GB free disk space

### 1 — Clone & Install

```bash
git clone https://github.com/Khushipatel27/SEC-RAG-Analyst.git
cd SEC-RAG-Analyst

conda create -n rag_finance python=3.11
conda activate rag_finance

pip install -r requirements.txt
```

### 2 — Pull AI Models (one-time, ~4 GB)

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3 — Download SEC Filings (one-time)

```bash
python scripts/download_sec_docs.py
```

### 4 — Ingest & Index (one-time, ~15–30 min)

```bash
python scripts/fix_and_reingest.py
```

### 5 — Launch

```bash
# Terminal 1 — Backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** 🎉

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|:---|:---|
| 🛠️ **Setup & Status** | Live health checks for Ollama, models, and ChromaDB. Ingest individual filings. |
| 💬 **Ask the Filings** | Natural language Q&A with streaming answers, source citations, latency display, and chat history. |
| 🏢 **Company Profiles** | Revenue, net income, EPS, and R&D charts for each company across 2022–2023. |
| 📊 **RAG Evaluation** | Runs 20 benchmark questions. Shows MRR, Recall@5, Precision@5, citation rate, and retrieval method comparison. |

---

## 📡 API Reference

| Method | Endpoint | Description |
|:---:|:---|:---|
| `POST` | `/ingest` | Parse, chunk, embed, and index a filing |
| `POST` | `/ask` | Question answering (blocking) |
| `GET` | `/ask/stream` | Question answering (SSE token streaming) |
| `GET` | `/status` | System health + index statistics |
| `GET` | `/documents` | List all ingested documents |
| `GET` | `/metrics/{ticker}/{year}` | Key financial metrics for a filing |
| `POST` | `/evaluate` | Run full 20-question evaluation suite |
| `POST` | `/evaluate/compare` | Retrieval method comparison (no LLM required) |

> Interactive docs at **http://localhost:8000/docs**

---

## 📈 Evaluation Results

Measured on 20 hand-crafted financial Q&A pairs (5 companies × 2 years × question types):

| Method | Recall@5 | Precision@5 | MRR |
|:---|:---:|:---:|:---:|
| Vector Only | ~0.42 | ~0.38 | ~0.28 |
| BM25 Only | ~0.38 | ~0.34 | ~0.24 |
| Hybrid (RRF) | ~0.58 | ~0.51 | ~0.38 |
| **Hybrid + Rerank** ⭐ | **~0.64** | **~0.57** | **~0.44** |

---

## ⏱️ Performance

| Operation | Time |
|:---|:---|
| Ingestion per 10-K | 45–90 seconds |
| Query latency (end-to-end) | 3–8 seconds |
| First streaming token | 1–2 seconds |
| ChromaDB index size | ~150 MB |
| BM25 index size | ~50 MB |

*Tested on Intel i7, 16 GB RAM, CPU only (no GPU).*

---

## 💬 Example Queries

```
What was Apple's total revenue and net income in fiscal year 2023?

Compare the operating margins of Apple, NVIDIA, and Microsoft in 2022.

What are the top 3 business risks Amazon disclosed in their 2023 10-K?

How did NVIDIA's revenue change between 2022 and 2023 and what drove the growth?

Give me an executive summary of Microsoft's 2023 annual report.
```

---

## ⚙️ Configuration

All settings can be overridden via `.env`:

```env
LLM_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_VECTOR=10
TOP_K_BM25=10
TOP_K_RERANK=5
VECTOR_WEIGHT=0.6
BM25_WEIGHT=0.4
TEMPERATURE=0.0
MAX_NEW_TOKENS=1024
```

---

## 🗂️ Project Structure

```
sec-rag-analyst/
├── api/
│   └── main.py                  # FastAPI — 8 endpoints, CORS, SSE streaming
├── app/
│   └── streamlit_app.py         # Streamlit dashboard — 4 tabs, dark theme
├── src/
│   ├── config.py                # Pydantic settings (all tuneable via .env)
│   ├── pipeline.py              # SECRAGPipeline — ingest, ask, stream, status
│   ├── ingestion/
│   │   ├── parser.py            # PDF/HTM/TXT parsing, text cleaning
│   │   └── chunker.py           # Smart chunker — preserves table blocks
│   ├── retrieval/
│   │   ├── embedder.py          # Ollama embedding client with retry logic
│   │   ├── vector_store.py      # ChromaDB wrapper — filtered search
│   │   ├── bm25_store.py        # BM25 store — build, search, persist
│   │   └── hybrid_search.py     # RRF fusion + financial query routing
│   ├── generation/
│   │   ├── chain.py             # RAG chain — prompt routing + streaming
│   │   ├── prompts.py           # 4 prompt templates
│   │   └── reranker.py          # Cross-encoder reranker
│   └── evaluation/
│       └── evaluator.py         # 20-question eval — MRR, Recall@5
├── scripts/
│   ├── download_sec_docs.py     # SEC EDGAR downloader
│   └── fix_and_reingest.py      # HTM→TXT conversion + re-index
├── tests/
│   ├── test_parser.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## 🔮 Future Improvements

- [ ] Multi-hop reasoning across filings (agent-based)
- [ ] Time-series financial metric extraction
- [ ] Fine-tuned domain-specific embeddings
- [ ] Cloud deployment (Pinecone + managed LLM)
- [ ] Earnings call transcript support
- [ ] XBRL structured data parsing

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

*Built to demonstrate production-grade RAG system design with real financial data.*

⭐ **Star this repo if you found it useful!**

</div>
