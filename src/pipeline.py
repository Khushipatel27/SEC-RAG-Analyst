"""
SEC RAG Pipeline
Top-level orchestrator wiring ingestion, retrieval, and generation together.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Generator, Optional

from loguru import logger
from tqdm import tqdm

from src.config import settings
from src.generation.chain import FinancialRAGChain
from src.generation.reranker import CrossEncoderReranker
from src.ingestion.chunker import FinancialChunker
from src.ingestion.parser import FinancialDocumentParser
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import OllamaEmbedder
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.vector_store import ChromaVectorStore


class SECRAGPipeline:
    """
    End-to-end pipeline for ingesting SEC 10-K filings and answering
    financial questions using hybrid retrieval + cross-encoder reranking.
    """

    def __init__(self) -> None:
        logger.info("Initialising SECRAGPipeline components...")

        self.parser = FinancialDocumentParser()
        self.chunker = FinancialChunker()
        self.embedder = OllamaEmbedder()
        self.vector_store = ChromaVectorStore()
        self.bm25_store = BM25Store()
        self.reranker = CrossEncoderReranker()
        self.chain = FinancialRAGChain()

        # Try loading an existing BM25 index from disk
        loaded = self.bm25_store.load_index()
        if not loaded:
            logger.info("No existing BM25 index found; it will be built on first ingest.")

        self.hybrid_searcher = HybridSearcher(
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
            embedder=self.embedder,
        )

        # In-memory cache of ingested document metadata
        self._ingested_docs: list[dict] = self._load_ingested_docs_log()

        logger.success("SECRAGPipeline fully initialised")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        file_path: Path,
        ticker: Optional[str] = None,
        year: Optional[str] = None,
    ) -> dict:
        """
        Parse, chunk, embed, and index a single SEC filing document.

        Steps
        -----
        1. Parse PDF / HTM / TXT → text blocks + table blocks
        2. Chunk document
        3. Embed all chunks via Ollama
        4. Add to ChromaDB
        5. Rebuild BM25 index with new chunks appended
        6. Extract key financial metrics
        7. Save processed chunk JSON to data/processed/

        Parameters
        ----------
        file_path : Path
        ticker : str, optional  – overrides filename-based detection
        year : str, optional    – overrides filename-based detection

        Returns
        -------
        dict
            {num_chunks, table_chunks, text_chunks, key_metrics, processing_time_seconds}
        """
        file_path = Path(file_path)
        start_time = time.perf_counter()

        logger.info(f"Ingesting document: {file_path}")

        # --- Step 1: Parse ---
        parsed_doc = self.parser.parse_document(file_path)
        if ticker:
            parsed_doc["ticker"] = ticker
            parsed_doc["metadata"]["ticker"] = ticker
        if year:
            parsed_doc["year"] = str(year)
            parsed_doc["metadata"]["year"] = str(year)

        # --- Step 2: Chunk ---
        chunks = self.chunker.chunk_document(parsed_doc)
        if not chunks:
            logger.warning(f"No chunks produced for {file_path.name}")
            return {
                "num_chunks": 0,
                "table_chunks": 0,
                "text_chunks": 0,
                "key_metrics": {},
                "processing_time_seconds": 0,
            }

        # --- Step 3: Embed ---
        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_batch(texts, batch_size=10)

        # --- Step 4: Add to ChromaDB ---
        self.vector_store.add_chunks(chunks, embeddings)

        # --- Step 5: Update BM25 index ---
        existing_chunks: list[dict] = list(self.bm25_store._chunks)
        all_chunks = existing_chunks + chunks
        self.bm25_store.build_index(all_chunks)

        # --- Step 6: Extract key metrics ---
        key_metrics = parsed_doc.get("metadata", {}).get("key_metrics", {})
        if not key_metrics:
            key_metrics = self.parser.extract_key_metrics(parsed_doc)

        # --- Step 7: Save processed chunks to disk ---
        self._save_processed_chunks(parsed_doc, chunks, key_metrics)

        # --- Log ingest event ---
        doc_ticker = parsed_doc.get("ticker", "UNKNOWN")
        doc_year = parsed_doc.get("year", "0000")
        table_chunks = sum(1 for c in chunks if c["block_type"] == "table")
        text_chunks = sum(1 for c in chunks if c["block_type"] == "text")

        elapsed = round(time.perf_counter() - start_time, 2)

        ingest_record = {
            "file_path": str(file_path),
            "ticker": doc_ticker,
            "year": doc_year,
            "num_chunks": len(chunks),
            "table_chunks": table_chunks,
            "text_chunks": text_chunks,
            "key_metrics": key_metrics,
            "processing_time_seconds": elapsed,
        }
        self._ingested_docs.append(ingest_record)
        self._save_ingested_docs_log()

        logger.success(
            f"Ingestion complete for {file_path.name} in {elapsed}s | "
            f"{len(chunks)} chunks ({table_chunks} table, {text_chunks} text)"
        )

        return {
            "num_chunks": len(chunks),
            "table_chunks": table_chunks,
            "text_chunks": text_chunks,
            "key_metrics": key_metrics,
            "processing_time_seconds": elapsed,
        }

    def batch_ingest(self, file_paths: list[Path]) -> dict:
        """
        Ingest multiple documents with a progress bar.

        Parameters
        ----------
        file_paths : list[Path]

        Returns
        -------
        dict  – {total_files, successful, failed, total_chunks, total_time_seconds}
        """
        logger.info(f"Starting batch ingest of {len(file_paths)} files...")
        successful = 0
        failed = 0
        total_chunks = 0
        batch_start = time.perf_counter()

        for fp in tqdm(file_paths, desc="Ingesting documents", unit="file"):
            try:
                result = self.ingest_document(fp)
                total_chunks += result.get("num_chunks", 0)
                successful += 1
            except Exception as exc:
                logger.error(f"Failed to ingest {fp}: {exc}")
                failed += 1

        elapsed = round(time.perf_counter() - batch_start, 2)
        summary = {
            "total_files": len(file_paths),
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks,
            "total_time_seconds": elapsed,
        }
        logger.success(f"Batch ingest complete: {summary}")
        return summary

    # ------------------------------------------------------------------
    # Retrieval + Generation
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        ticker_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
    ) -> dict:
        """
        Answer a financial question using hybrid retrieval and LLM generation.

        Steps
        -----
        1. Apply financial query routing (auto-detect filters)
        2. Hybrid search (vector + BM25 + RRF)
        3. Cross-encoder rerank
        4. LLM generation

        Parameters
        ----------
        question : str
        ticker_filter : str, optional
        year_filter : str, optional

        Returns
        -------
        dict  – full response dict from FinancialRAGChain.generate()
        """
        logger.info(f"Pipeline.ask: '{question[:80]}'")

        # --- Step 1: Query routing ---
        auto_filters = self.hybrid_searcher.apply_financial_query_routing(question)

        # Explicit filters take precedence over auto-detected ones
        filters: dict = {}
        if ticker_filter:
            filters["ticker"] = ticker_filter
        elif auto_filters.get("ticker"):
            filters["ticker"] = auto_filters["ticker"]

        if year_filter:
            filters["year"] = str(year_filter)
        elif auto_filters.get("year"):
            filters["year"] = auto_filters["year"]

        if auto_filters.get("block_type") and "block_type" not in filters:
            filters["block_type"] = auto_filters["block_type"]

        effective_filters = filters if filters else None

        # --- Step 2: Hybrid search ---
        retrieved_chunks = self.hybrid_searcher.search(
            query=question,
            k_final=settings.top_k_vector,
            filters=effective_filters,
        )

        if not retrieved_chunks:
            logger.warning("No chunks retrieved; generating with empty context")

        # --- Step 3: Rerank ---
        reranked_chunks = self.reranker.rerank(
            query=question,
            chunks=retrieved_chunks,
            top_k=settings.top_k_rerank,
        )

        # --- Step 4: Generate ---
        response = self.chain.generate(question, reranked_chunks)
        response["filters_applied"] = effective_filters
        response["retrieved_before_rerank"] = len(retrieved_chunks)

        return response

    def ask_stream(
        self,
        question: str,
        ticker_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming version of ask(). Yields tokens as they arrive.
        """
        # Routing and retrieval (same as ask())
        auto_filters = self.hybrid_searcher.apply_financial_query_routing(question)
        filters: dict = {}
        if ticker_filter:
            filters["ticker"] = ticker_filter
        elif auto_filters.get("ticker"):
            filters["ticker"] = auto_filters["ticker"]

        if year_filter:
            filters["year"] = str(year_filter)
        elif auto_filters.get("year"):
            filters["year"] = auto_filters["year"]

        effective_filters = filters if filters else None

        retrieved_chunks = self.hybrid_searcher.search(
            query=question,
            k_final=settings.top_k_vector,
            filters=effective_filters,
        )
        reranked_chunks = self.reranker.rerank(
            query=question,
            chunks=retrieved_chunks,
            top_k=settings.top_k_rerank,
        )

        yield from self.chain.generate_stream(question, reranked_chunks)

    # ------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------

    def get_system_status(self) -> dict:
        """
        Return a comprehensive status snapshot of the pipeline.

        Returns
        -------
        dict
            {
                ollama_running, models_available, documents_indexed,
                companies_available, years_available, total_chunks,
                table_chunks, chroma_db_size_mb
            }
        """
        # Check Ollama — handle both old dict API and new object API (ollama >= 0.2)
        ollama_running = False
        models_available: list[str] = []
        try:
            import ollama as _ollama
            response = _ollama.list()
            ollama_running = True

            # New API (>= 0.2.0): ListResponse with .models list of Model objects
            if hasattr(response, "models"):
                for m in response.models:
                    # Model object has .model or .name attribute
                    name = getattr(m, "model", None) or getattr(m, "name", None)
                    if name:
                        models_available.append(str(name))
            # Old API: plain dict {"models": [{"name": "...", ...}]}
            elif isinstance(response, dict):
                for m in response.get("models", []):
                    name = m.get("name") or m.get("model", "")
                    if name:
                        models_available.append(str(name))

            logger.info(f"Ollama models found: {models_available}")
        except Exception as exc:
            logger.warning(f"Ollama health check failed: {exc}")

        # ChromaDB stats
        chroma_stats = self.vector_store.get_collection_stats()
        total_chunks = chroma_stats.get("num_chunks", 0)
        table_chunks = chroma_stats.get("table_chunks", 0)
        companies_available = chroma_stats.get("companies", [])
        years_available = chroma_stats.get("years", [])

        # ChromaDB disk size
        chroma_dir = Path(settings.chroma_dir)
        chroma_size_mb = 0.0
        if chroma_dir.exists():
            total_bytes = sum(
                f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file()
            )
            chroma_size_mb = round(total_bytes / (1024 * 1024), 2)

        return {
            "ollama_running": ollama_running,
            "models_available": models_available,
            "documents_indexed": len(self._ingested_docs),
            "companies_available": companies_available,
            "years_available": years_available,
            "total_chunks": total_chunks,
            "table_chunks": table_chunks,
            "chroma_db_size_mb": chroma_size_mb,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_processed_chunks(
        self, parsed_doc: dict, chunks: list[dict], key_metrics: dict
    ) -> None:
        """Persist processed chunks to data/processed/ as JSON."""
        processed_dir = Path(settings.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        ticker = parsed_doc.get("ticker", "UNKNOWN")
        year = parsed_doc.get("year", "0000")
        out_path = processed_dir / f"{ticker}_{year}_chunks.json"

        payload = {
            "ticker": ticker,
            "year": year,
            "num_chunks": len(chunks),
            "key_metrics": key_metrics,
            "chunks": chunks,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved processed chunks to {out_path}")

    def _ingested_docs_log_path(self) -> Path:
        processed_dir = Path(settings.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir / "ingested_docs.json"

    def _load_ingested_docs_log(self) -> list[dict]:
        log_path = self._ingested_docs_log_path()
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_ingested_docs_log(self) -> None:
        log_path = self._ingested_docs_log_path()
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self._ingested_docs, f, indent=2, ensure_ascii=False)
