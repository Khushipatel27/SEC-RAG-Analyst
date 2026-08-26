"""
FastAPI application for the SEC Financial RAG system.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import ollama as _ollama
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.config import settings
from src.pipeline import SECRAGPipeline

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SEC Financial RAG API",
    description=(
        "RAG system for SEC 10-K filings analysis.\n\n"
        "**/ask** — v1: hybrid retrieval → rerank → generate.\n\n"
        "**/ask/agentic** — v2: a supervisor routes the question to XBRL, "
        "calculation, graph, and retrieval specialists, then verifies the "
        "answer is grounded before returning it."
    ),
    version="2.0.0",
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"in {elapsed_ms:.1f} ms"
    )
    response.headers["X-Process-Time-Ms"] = str(round(elapsed_ms, 2))
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ---------------------------------------------------------------------------
# Pipeline singleton (initialised at startup)
# ---------------------------------------------------------------------------
pipeline: Optional[SECRAGPipeline] = None
supervisor: Optional["SupervisorAgent"] = None


@app.on_event("startup")
async def startup_event():
    global pipeline, supervisor
    logger.info("Starting up SEC RAG API – initialising pipeline...")
    try:
        pipeline = SECRAGPipeline()
        logger.success("Pipeline initialised successfully")
    except Exception as exc:
        logger.error(f"Pipeline initialisation failed: {exc}")
        pipeline = None

    # The agentic layer is additive: if it fails to start, /ask still works.
    try:
        from src.agents.supervisor import SupervisorAgent

        supervisor = SupervisorAgent(pipeline=pipeline)
        logger.success("Agentic supervisor initialised successfully")
    except Exception as exc:
        logger.error(f"Supervisor initialisation failed: {exc}")
        supervisor = None


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    ticker: str
    year: str
    file_path: str


class AskRequest(BaseModel):
    question: str
    ticker_filter: Optional[str] = None
    year_filter: Optional[str] = None
    stream: bool = False


class IngestResponse(BaseModel):
    ticker: str
    year: str
    num_chunks: int
    table_chunks: int
    text_chunks: int
    key_metrics: dict
    processing_time_seconds: float


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    prompt_used: str
    num_chunks: int
    latency_ms: float
    model: str
    context_length: int
    filters_applied: Optional[dict] = None
    retrieved_before_rerank: Optional[int] = None


class AgenticAskRequest(BaseModel):
    question: str
    ticker_filter: Optional[str] = None
    year_filter: Optional[str] = None
    include_agent_results: bool = False


class AgenticAskResponse(BaseModel):
    answer: str
    sources: list[dict]
    agents_used: list[str]
    routing: dict
    verification: dict
    trace: list[dict]
    latency_ms: float
    model: str
    agent_results: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _require_pipeline() -> SECRAGPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return pipeline


def _require_supervisor():
    if supervisor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Agentic supervisor not initialised. Check that edgartools and "
                "langgraph are installed and that the knowledge graph has been "
                "built (python -m src.ingestion.graph_builder)."
            ),
        )
    return supervisor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(request: IngestRequest):
    """
    Ingest a single SEC 10-K filing into the RAG system.

    The file must already be present at *file_path* on the server filesystem.
    """
    p = _require_pipeline()
    fp = Path(request.file_path)
    if not fp.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {fp}")

    try:
        result = p.ingest_document(
            file_path=fp,
            ticker=request.ticker,
            year=request.year,
        )
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        ticker=request.ticker,
        year=request.year,
        **result,
    )


@app.post("/ask", response_model=AskResponse, tags=["QA"])
async def ask(request: AskRequest):
    """
    Answer a financial question using the RAG pipeline.

    Set *stream=true* to receive a server-sent events stream instead.
    """
    p = _require_pipeline()

    if request.stream:
        # Redirect to SSE endpoint data (for POST-based clients)
        # For true SSE use GET /ask/stream
        async def token_generator() -> AsyncGenerator[str, None]:
            for token in p.ask_stream(
                question=request.question,
                ticker_filter=request.ticker_filter,
                year_filter=request.year_filter,
            ):
                yield token

        return StreamingResponse(token_generator(), media_type="text/plain")

    try:
        result = p.ask(
            question=request.question,
            ticker_filter=request.ticker_filter,
            year_filter=request.year_filter,
        )
    except RuntimeError as exc:
        exc_str = str(exc)
        logger.error(f"Ask failed: {exc_str}")
        # Model-not-found → give a clear 503 with fix instructions
        if "not available in ollama" in exc_str.lower() or "ollama pull" in exc_str.lower():
            raise HTTPException(
                status_code=503,
                detail=exc_str,
            )
        raise HTTPException(status_code=500, detail=exc_str)
    except Exception as exc:
        logger.error(f"Ask failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(**result)


@app.post("/ask/agentic", response_model=AgenticAskResponse, tags=["QA"])
async def ask_agentic(request: AgenticAskRequest):
    """
    Answer a question through the v2 multi-agent pipeline.

    A supervisor inspects the question and dispatches to the specialists it
    needs — exact XBRL figures, in-code calculations, knowledge-graph traversal,
    and the v1 retrieval pipeline — then verifies the drafted answer is grounded
    in the evidence before returning it.

    The response reports which agents ran and why, so the routing decision is
    inspectable rather than implicit.
    """
    s = _require_supervisor()

    try:
        result = s.ask(
            question=request.question,
            ticker_filter=request.ticker_filter,
            year_filter=request.year_filter,
        )
    except Exception as exc:
        logger.error(f"Agentic ask failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if not request.include_agent_results:
        # Full specialist payloads are verbose; opt in when debugging routing.
        result = {**result, "agent_results": None}

    return AgenticAskResponse(**result)


@app.get("/agents/status", tags=["System"])
async def agents_status():
    """Report which specialists are available and what the graph contains."""
    if supervisor is None:
        return {"available": False, "reason": "Supervisor not initialised"}

    return {
        "available": True,
        "agents": {
            "xbrl": {
                "available": True,
                "supported_metrics": supervisor.xbrl.supported_metrics(),
                "tickers": settings.agent_tickers,
            },
            "calculation": {"available": True},
            "graph": {
                "available": supervisor.graph.is_available,
                "stats": supervisor.graph.stats,
            },
            "narrative": {"available": supervisor.narrative is not None},
            "verification": {"enabled": settings.verification_enabled},
        },
    }


@app.get("/ask/stream", tags=["QA"])
async def ask_stream(
    question: str = Query(..., description="Financial question"),
    ticker_filter: Optional[str] = Query(None, description="Filter by ticker (e.g. AAPL)"),
    year_filter: Optional[str] = Query(None, description="Filter by year (e.g. 2023)"),
):
    """
    Server-Sent Events streaming endpoint for financial Q&A.
    """
    p = _require_pipeline()

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            for token in p.ask_stream(
                question=question,
                ticker_filter=ticker_filter,
                year_filter=year_filter,
            ):
                yield {"data": token}
            yield {"data": "[DONE]"}
        except Exception as exc:
            logger.error(f"SSE stream error: {exc}")
            yield {"data": f"[ERROR: {exc}]"}

    return EventSourceResponse(event_generator())


@app.get("/status", tags=["System"])
async def status():
    """Return a full system status snapshot."""
    p = _require_pipeline()
    try:
        return p.get_system_status()
    except Exception as exc:
        logger.error(f"Status check failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents", tags=["System"])
async def list_documents():
    """List all ingested documents with metadata."""
    p = _require_pipeline()
    try:
        return {
            "documents": p._ingested_docs,
            "total": len(p._ingested_docs),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics/{ticker}/{year}", tags=["Data"])
async def get_metrics(ticker: str, year: str):
    """
    Retrieve extracted key financial metrics for a specific ticker and year.
    """
    p = _require_pipeline()
    ticker = ticker.upper()

    # Search in ingested docs log
    for doc in p._ingested_docs:
        if doc.get("ticker", "").upper() == ticker and str(doc.get("year", "")) == year:
            return {
                "ticker": ticker,
                "year": year,
                "key_metrics": doc.get("key_metrics", {}),
            }

    # Try loading from processed chunks file
    processed_path = Path("data/processed") / f"{ticker}_{year}_chunks.json"
    if processed_path.exists():
        import json

        with open(processed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "ticker": ticker,
            "year": year,
            "key_metrics": data.get("key_metrics", {}),
        }

    raise HTTPException(
        status_code=404,
        detail=f"No metrics found for {ticker} {year}. "
               f"Has this filing been ingested?",
    )


@app.post("/evaluate", tags=["Evaluation"])
async def run_evaluation():
    """
    Trigger a full RAG evaluation run.
    Runs 20 questions and returns aggregated metrics.
    """
    p = _require_pipeline()
    from src.evaluation.evaluator import RAGEvaluator

    evaluator = RAGEvaluator(pipeline=p)
    try:
        report = evaluator.run_evaluation(pipeline=p)
        return report
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate/compare", tags=["Evaluation"])
async def run_retrieval_comparison():
    """
    Run retrieval method comparison only (vector, BM25, hybrid, hybrid+rerank).
    Much faster than a full evaluation — no LLM calls needed.
    """
    p = _require_pipeline()
    from src.evaluation.evaluator import RAGEvaluator

    evaluator = RAGEvaluator(pipeline=p)
    try:
        result = evaluator.compare_retrieval_methods()
        return result
    except Exception as exc:
        logger.error(f"Retrieval comparison failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/models", tags=["System"])
async def list_models():
    """List all available Ollama models."""
    try:
        response = _ollama.list()
        models = [
            {
                "name": m.get("name", ""),
                "size": m.get("size", 0),
                "modified_at": str(m.get("modified_at", "")),
            }
            for m in response.get("models", [])
        ]
        return {"models": models, "total": len(models)}
    except Exception as exc:
        logger.warning(f"Failed to list Ollama models: {exc}")
        return {"models": [], "total": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Root health check
# ---------------------------------------------------------------------------


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "SEC Financial RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }
