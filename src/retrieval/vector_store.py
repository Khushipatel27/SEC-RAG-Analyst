"""
ChromaDB Vector Store
Persistent vector store for SEC filing chunks using cosine similarity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from src.config import settings

_COLLECTION_NAME = "sec_filings"
_BATCH_SIZE = 100


class ChromaVectorStore:
    """
    Persistent Chroma vector store for SEC filing chunks.

    All chunks are stored with their full metadata, enabling
    filtered retrieval by ticker, year, block_type, and section.
    """

    def __init__(self) -> None:
        chroma_dir = Path(settings.chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaVectorStore initialized | "
            f"path={chroma_dir} | "
            f"collection={_COLLECTION_NAME} | "
            f"existing_docs={self._collection.count()}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add chunks and their pre-computed embeddings to ChromaDB.

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts (must each have "chunk_id", "text", and metadata keys).
        embeddings : list[list[float]]
            Corresponding embedding vectors (same length as *chunks*).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        if not chunks:
            logger.warning("add_chunks called with empty list; nothing to do.")
            return

        total = len(chunks)
        added = 0

        for start in range(0, total, _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, total)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]
            metadatas = [self._build_metadata(c) for c in batch_chunks]

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )
            added += len(batch_chunks)
            logger.debug(f"Added batch {start}-{end} ({len(batch_chunks)} chunks)")

        logger.success(
            f"Added {added} chunks to ChromaDB collection '{_COLLECTION_NAME}'"
        )

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Nearest-neighbour search using cosine distance.

        Parameters
        ----------
        query_embedding : list[float]
            Query vector.
        k : int
            Number of results to return.
        filters : dict, optional
            Supported keys: ticker, year, block_type, section.
            Multiple keys are AND-combined.

        Returns
        -------
        list[dict]
            Each element: {text, metadata, distance, score}
        """
        where_clause = self._build_where_clause(filters) if filters else None

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, max(self._collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        try:
            raw = self._collection.query(**query_kwargs)
        except Exception as exc:
            logger.error(f"ChromaDB query failed: {exc}")
            return []

        results: list[dict] = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert cosine distance → similarity score (1 = identical)
            score = 1.0 - float(dist)
            results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "score": score,
                }
            )

        logger.debug(
            f"ChromaDB search returned {len(results)} results "
            f"(filters={filters})"
        )
        return results

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the stored collection.

        Returns
        -------
        dict
            {num_chunks, companies, years, table_chunks, text_chunks}
        """
        count = self._collection.count()
        if count == 0:
            return {
                "num_chunks": 0,
                "companies": [],
                "years": [],
                "table_chunks": 0,
                "text_chunks": 0,
            }

        # Fetch all metadatas in pages to avoid SQLite "too many SQL variables" error
        metadatas: list[dict] = []
        page_size = 500
        offset = 0
        while offset < count:
            result = self._collection.get(
                include=["metadatas"],
                limit=page_size,
                offset=offset,
            )
            batch = result.get("metadatas", [])
            if not batch:
                break
            metadatas.extend(batch)
            offset += page_size

        companies: set[str] = set()
        years: set[str] = set()
        table_chunks = 0
        text_chunks = 0

        for meta in metadatas:
            if meta.get("ticker"):
                companies.add(meta["ticker"])
            if meta.get("year"):
                years.add(meta["year"])
            block_type = meta.get("block_type", "text")
            if block_type == "table":
                table_chunks += 1
            else:
                text_chunks += 1

        return {
            "num_chunks": count,
            "companies": sorted(companies),
            "years": sorted(years),
            "table_chunks": table_chunks,
            "text_chunks": text_chunks,
        }

    def delete_collection(self) -> None:
        """Delete the entire collection (for cleanup / re-indexing)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{_COLLECTION_NAME}' deleted and recreated.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(chunk: dict) -> dict:
        """Extract a flat, Chroma-compatible metadata dict from a chunk."""
        return {
            "chunk_id": str(chunk.get("chunk_id", "")),
            "ticker": str(chunk.get("ticker", "")),
            "year": str(chunk.get("year", "")),
            "page_num": int(chunk.get("page_num", 0)),
            "section": str(chunk.get("section", "")),
            "block_type": str(chunk.get("block_type", "text")),
            "contains_numbers": bool(chunk.get("contains_numbers", False)),
            "source_file": str(chunk.get("source_file", "")),
            "char_count": int(chunk.get("char_count", 0)),
        }

    @staticmethod
    def _build_where_clause(filters: dict) -> Optional[dict]:
        """
        Build a Chroma $and where clause from a filters dict.

        Supported keys: ticker, year, block_type, section
        """
        conditions: list[dict] = []

        for key in ("ticker", "year", "block_type", "section"):
            value = filters.get(key)
            if value is not None:
                conditions.append({key: {"$eq": str(value)}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
