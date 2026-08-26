"""
FastAPI endpoint tests — happy path and failure path for every route.

The TestClient is deliberately NOT used as a context manager: entering it would
run the startup event, which builds a real SECRAGPipeline (ChromaDB, a
cross-encoder download, and an Ollama connection). Making requests outside the
context manager skips lifespan entirely, so these tests stay offline and fast,
with the pipeline and supervisor globals patched per test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.main as api_main
from api.main import app


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


ANSWER_PAYLOAD = {
    "answer": "Apple's total net sales for fiscal 2023 were $383.3 billion [Source 1].",
    "sources": [{"text": "Total net sales ...", "ticker": "AAPL", "year": "2023"}],
    "prompt_used": "FINANCIAL_QA_PROMPT",
    "num_chunks": 5,
    "latency_ms": 1234.5,
    "model": "llama3.2",
    "context_length": 2048,
    "filters_applied": {"ticker": "AAPL", "year": "2023"},
    "retrieved_before_rerank": 10,
}

INGEST_RESULT = {
    "num_chunks": 120,
    "table_chunks": 30,
    "text_chunks": 90,
    "key_metrics": {"revenue": "$383.3B"},
    "processing_time_seconds": 61.2,
}

AGENTIC_PAYLOAD = {
    "answer": "NVIDIA's R&D grew 39% versus Microsoft's 14%.",
    "sources": [{"type": "xbrl", "concept": "us-gaap:ResearchAndDevelopmentExpense"}],
    "agents_used": ["xbrl", "calculation"],
    "routing": {"reason": "comparison of two tagged metrics"},
    "verification": {"verdict": "grounded", "confidence": 0.95},
    "trace": [{"agent": "xbrl", "ms": 120}],
    "latency_ms": 2200.0,
    "model": "llama3.2",
    "agent_results": {"xbrl": {"value": 8675000000}},
}


class FakePipeline:
    """Implements only what the API routes actually call."""

    def __init__(self) -> None:
        self._ingested_docs = [
            {
                "ticker": "AAPL",
                "year": "2023",
                "num_chunks": 120,
                "key_metrics": {"revenue": "$383.3B"},
            }
        ]

    def ask(self, question, ticker_filter=None, year_filter=None):
        return dict(ANSWER_PAYLOAD)

    def ask_stream(self, question, ticker_filter=None, year_filter=None):
        yield from ["Apple's ", "revenue ", "was ", "$383.3 billion."]

    def ingest_document(self, file_path, ticker=None, year=None):
        return dict(INGEST_RESULT)

    def get_system_status(self):
        return {
            "ollama_running": True,
            "models_available": ["llama3.2"],
            "documents_indexed": 1,
            "companies_available": ["AAPL"],
            "years_available": ["2023"],
            "total_chunks": 120,
            "table_chunks": 30,
            "chroma_db_size_mb": 12.5,
        }


class FakeXBRLAgent:
    @staticmethod
    def supported_metrics():
        return ["Revenues", "NetIncomeLoss"]


class FakeGraphAgent:
    is_available = True
    stats = {"nodes": 120, "edges": 340}


class FakeSupervisor:
    def __init__(self) -> None:
        self.xbrl = FakeXBRLAgent()
        self.graph = FakeGraphAgent()
        self.narrative = object()

    def ask(self, question, ticker_filter=None, year_filter=None):
        return dict(AGENTIC_PAYLOAD)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def pipeline(monkeypatch) -> FakePipeline:
    fake = FakePipeline()
    monkeypatch.setattr(api_main, "pipeline", fake)
    return fake


@pytest.fixture
def supervisor(monkeypatch) -> FakeSupervisor:
    fake = FakeSupervisor()
    monkeypatch.setattr(api_main, "supervisor", fake)
    return fake


@pytest.fixture
def no_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "pipeline", None)


@pytest.fixture
def no_supervisor(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "supervisor", None)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_root_reports_service_metadata(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["docs"] == "/docs"


def test_root_works_without_a_pipeline(client, no_pipeline):
    """The health check must not depend on the pipeline having initialised."""
    assert client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------


def test_ask_returns_an_answer_with_sources(client, pipeline):
    response = client.post("/ask", json={"question": "Apple revenue 2023?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ANSWER_PAYLOAD["answer"]
    assert body["sources"]
    assert body["retrieved_before_rerank"] == 10
    assert body["filters_applied"] == {"ticker": "AAPL", "year": "2023"}


def test_ask_passes_filters_through(client, pipeline, monkeypatch):
    seen = {}

    def capture(question, ticker_filter=None, year_filter=None):
        seen.update(
            question=question, ticker_filter=ticker_filter, year_filter=year_filter
        )
        return dict(ANSWER_PAYLOAD)

    monkeypatch.setattr(pipeline, "ask", capture)
    client.post(
        "/ask",
        json={"question": "revenue?", "ticker_filter": "MSFT", "year_filter": "2023"},
    )

    assert seen == {
        "question": "revenue?",
        "ticker_filter": "MSFT",
        "year_filter": "2023",
    }


def test_ask_streams_when_requested(client, pipeline):
    response = client.post("/ask", json={"question": "revenue?", "stream": True})
    assert response.status_code == 200
    assert response.text == "Apple's revenue was $383.3 billion."


def test_ask_rejects_a_missing_question(client, pipeline):
    """Pydantic validation error, before the pipeline is ever touched."""
    assert client.post("/ask", json={}).status_code == 422


def test_ask_returns_503_when_the_pipeline_is_down(client, no_pipeline):
    response = client.post("/ask", json={"question": "revenue?"})
    assert response.status_code == 503
    assert "not initialised" in response.json()["detail"]


def test_ask_returns_503_with_fix_instructions_for_a_missing_model(
    client, pipeline, monkeypatch
):
    """A missing Ollama model is a config problem, so it must not read as a 500."""

    def raise_model_missing(*args, **kwargs):
        raise RuntimeError(
            "Embedding model 'nomic-embed-text' is not available in Ollama. "
            "Fix: run  ollama pull nomic-embed-text"
        )

    monkeypatch.setattr(pipeline, "ask", raise_model_missing)
    response = client.post("/ask", json={"question": "revenue?"})

    assert response.status_code == 503
    assert "ollama pull" in response.json()["detail"]


def test_ask_returns_500_on_an_unexpected_failure(client, pipeline, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("chroma exploded")

    monkeypatch.setattr(pipeline, "ask", boom)
    response = client.post("/ask", json={"question": "revenue?"})

    assert response.status_code == 500
    assert "chroma exploded" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


def test_ingest_indexes_an_existing_file(client, pipeline, tmp_path):
    filing = tmp_path / "AAPL_2023_10K.pdf"
    filing.write_text("dummy filing")

    response = client.post(
        "/ingest",
        json={"ticker": "AAPL", "year": "2023", "file_path": str(filing)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["num_chunks"] == 120
    assert body["table_chunks"] == 30


def test_ingest_returns_404_for_a_missing_file(client, pipeline, tmp_path):
    response = client.post(
        "/ingest",
        json={
            "ticker": "AAPL",
            "year": "2023",
            "file_path": str(tmp_path / "nope.pdf"),
        },
    )
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]


def test_ingest_returns_500_when_parsing_fails(client, pipeline, tmp_path, monkeypatch):
    filing = tmp_path / "corrupt.pdf"
    filing.write_text("not really a pdf")

    def boom(*args, **kwargs):
        raise RuntimeError("pdfplumber could not open the file")

    monkeypatch.setattr(pipeline, "ingest_document", boom)
    response = client.post(
        "/ingest",
        json={"ticker": "AAPL", "year": "2023", "file_path": str(filing)},
    )

    assert response.status_code == 500
    assert "pdfplumber" in response.json()["detail"]


def test_ingest_returns_503_when_the_pipeline_is_down(client, no_pipeline, tmp_path):
    filing = tmp_path / "AAPL.pdf"
    filing.write_text("x")
    response = client.post(
        "/ingest",
        json={"ticker": "AAPL", "year": "2023", "file_path": str(filing)},
    )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /ask/agentic
# ---------------------------------------------------------------------------


def test_agentic_ask_reports_routing_and_verification(client, supervisor):
    response = client.post("/ask/agentic", json={"question": "R&D growth?"})

    assert response.status_code == 200
    body = response.json()
    assert body["agents_used"] == ["xbrl", "calculation"]
    assert body["verification"]["verdict"] == "grounded"
    assert body["trace"]


def test_agentic_ask_hides_agent_results_by_default(client, supervisor):
    """Specialist payloads are verbose; they are opt-in."""
    default = client.post("/ask/agentic", json={"question": "R&D growth?"})
    assert default.json()["agent_results"] is None

    opted_in = client.post(
        "/ask/agentic",
        json={"question": "R&D growth?", "include_agent_results": True},
    )
    assert opted_in.json()["agent_results"] == {"xbrl": {"value": 8675000000}}


def test_agentic_ask_returns_503_when_the_supervisor_is_down(client, no_supervisor):
    response = client.post("/ask/agentic", json={"question": "R&D growth?"})
    assert response.status_code == 503
    assert "edgartools" in response.json()["detail"]


def test_agentic_ask_returns_500_when_a_specialist_raises(
    client, supervisor, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("graph traversal failed")

    monkeypatch.setattr(supervisor, "ask", boom)
    response = client.post("/ask/agentic", json={"question": "R&D growth?"})

    assert response.status_code == 500
    assert "graph traversal failed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /agents/status
# ---------------------------------------------------------------------------


def test_agents_status_lists_available_specialists(client, supervisor):
    response = client.get("/agents/status")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["agents"]["graph"]["available"] is True
    assert body["agents"]["xbrl"]["supported_metrics"] == ["Revenues", "NetIncomeLoss"]


def test_agents_status_degrades_gracefully_without_a_supervisor(client, no_supervisor):
    """This route reports unavailability as data rather than as an HTTP error."""
    response = client.get("/agents/status")

    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "reason": "Supervisor not initialised",
    }


# ---------------------------------------------------------------------------
# GET /ask/stream
# ---------------------------------------------------------------------------


def test_ask_stream_emits_tokens_then_done(client, pipeline):
    response = client.get("/ask/stream", params={"question": "Apple revenue?"})

    assert response.status_code == 200
    assert "$383.3 billion." in response.text
    assert "[DONE]" in response.text


def test_ask_stream_requires_a_question(client, pipeline):
    assert client.get("/ask/stream").status_code == 422


def test_ask_stream_returns_503_when_the_pipeline_is_down(client, no_pipeline):
    response = client.get("/ask/stream", params={"question": "revenue?"})
    assert response.status_code == 503


def test_ask_stream_reports_mid_stream_errors_in_band(client, pipeline, monkeypatch):
    """
    Headers are already sent once streaming starts, so a failure cannot become
    an HTTP error code — it has to arrive as an event.
    """

    def boom(*args, **kwargs):
        raise RuntimeError("ollama died mid-stream")
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(pipeline, "ask_stream", boom)
    response = client.get("/ask/stream", params={"question": "revenue?"})

    assert response.status_code == 200
    assert "[ERROR: ollama died mid-stream]" in response.text


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------


def test_status_returns_a_system_snapshot(client, pipeline):
    response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert body["ollama_running"] is True
    assert body["total_chunks"] == 120


def test_status_returns_503_when_the_pipeline_is_down(client, no_pipeline):
    assert client.get("/status").status_code == 503


def test_status_returns_500_when_the_snapshot_fails(client, pipeline, monkeypatch):
    def boom():
        raise RuntimeError("chroma unreachable")

    monkeypatch.setattr(pipeline, "get_system_status", boom)
    response = client.get("/status")

    assert response.status_code == 500
    assert "chroma unreachable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


def test_documents_lists_ingested_filings(client, pipeline):
    response = client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["documents"][0]["ticker"] == "AAPL"


def test_documents_returns_503_when_the_pipeline_is_down(client, no_pipeline):
    assert client.get("/documents").status_code == 503


# ---------------------------------------------------------------------------
# GET /metrics/{ticker}/{year}
# ---------------------------------------------------------------------------


def test_metrics_returns_key_metrics_from_the_ingest_log(client, pipeline):
    response = client.get("/metrics/AAPL/2023")

    assert response.status_code == 200
    assert response.json()["key_metrics"] == {"revenue": "$383.3B"}


def test_metrics_is_case_insensitive_on_ticker(client, pipeline):
    assert client.get("/metrics/aapl/2023").status_code == 200


def test_metrics_returns_404_for_an_uningested_filing(client, pipeline):
    response = client.get("/metrics/TSLA/2023")

    assert response.status_code == 404
    assert "Has this filing been ingested?" in response.json()["detail"]


def test_metrics_falls_back_to_the_processed_chunks_file(
    client, pipeline, tmp_path, monkeypatch
):
    """A filing absent from the in-memory log is still served from disk."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "NVDA_2023_chunks.json").write_text(
        json.dumps({"key_metrics": {"revenue": "$26.97B"}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    response = client.get("/metrics/NVDA/2023")

    assert response.status_code == 200
    assert response.json()["key_metrics"] == {"revenue": "$26.97B"}


# ---------------------------------------------------------------------------
# POST /evaluate and /evaluate/compare
# ---------------------------------------------------------------------------


class FakeEvaluator:
    report = {"total_questions": 20, "avg_keyword_overlap": 0.42}
    comparison = {"hybrid_rerank": {"avg_mrr": 0.59}}

    def __init__(self, pipeline=None) -> None:
        pass

    def run_evaluation(self, pipeline=None):
        return dict(self.report)

    def compare_retrieval_methods(self):
        return dict(self.comparison)


@pytest.fixture
def fake_evaluator(monkeypatch):
    import src.evaluation.evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "RAGEvaluator", FakeEvaluator)
    return FakeEvaluator


def test_evaluate_returns_a_report(client, pipeline, fake_evaluator):
    response = client.post("/evaluate")

    assert response.status_code == 200
    assert response.json()["total_questions"] == 20


def test_evaluate_returns_500_when_the_run_fails(
    client, pipeline, fake_evaluator, monkeypatch
):
    def boom(self, pipeline=None):
        raise RuntimeError("eval questions missing")

    monkeypatch.setattr(FakeEvaluator, "run_evaluation", boom)
    response = client.post("/evaluate")

    assert response.status_code == 500
    assert "eval questions missing" in response.json()["detail"]


def test_evaluate_returns_503_when_the_pipeline_is_down(client, no_pipeline):
    assert client.post("/evaluate").status_code == 503


def test_evaluate_compare_returns_per_method_metrics(client, pipeline, fake_evaluator):
    response = client.post("/evaluate/compare")

    assert response.status_code == 200
    assert response.json()["hybrid_rerank"]["avg_mrr"] == 0.59


def test_evaluate_compare_returns_500_when_retrieval_fails(
    client, pipeline, fake_evaluator, monkeypatch
):
    def boom(self):
        raise RuntimeError("bm25 index not loaded")

    monkeypatch.setattr(FakeEvaluator, "compare_retrieval_methods", boom)
    response = client.post("/evaluate/compare")

    assert response.status_code == 500
    assert "bm25 index not loaded" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


def test_models_lists_available_ollama_models(client, monkeypatch):
    monkeypatch.setattr(
        api_main._ollama,
        "list",
        lambda: {"models": [{"name": "llama3.2", "size": 2019393189}]},
    )
    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["models"][0]["name"] == "llama3.2"


def test_models_reports_an_unreachable_ollama_without_failing(client, monkeypatch):
    """
    A dead Ollama returns 200 with an error field rather than a 5xx, so the
    dashboard can render a degraded state instead of an error page.
    """

    def boom():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(api_main._ollama, "list", boom)
    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == []
    assert "connection refused" in body["error"]
