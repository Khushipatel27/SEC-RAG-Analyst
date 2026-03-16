"""
Ollama Embedder
Generates text embeddings using a locally running Ollama instance.
"""
from __future__ import annotations

import time
from typing import Any

import ollama
from loguru import logger
from tqdm import tqdm

from src.config import settings


class OllamaEmbedder:
    """
    Thin wrapper around the Ollama embeddings API.

    Supports single-text embedding and batched embedding with
    automatic retry (exponential back-off, up to 3 attempts).
    """

    def __init__(self) -> None:
        self.model: str = settings.embedding_model
        self.base_url: str = settings.ollama_base_url
        logger.info(
            f"OllamaEmbedder initialized | model={self.model} | base_url={self.base_url}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single string.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list[float]
            The embedding vector.

        Raises
        ------
        RuntimeError
            If all retry attempts fail.
        """
        return self._embed_with_retry(text)

    def embed_batch(
        self, texts: list[str], batch_size: int = 10
    ) -> list[list[float]]:
        """
        Embed a list of strings in batches.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        batch_size : int
            Number of texts per batch (default 10).

        Returns
        -------
        list[list[float]]
            Embeddings in the same order as *texts*.
        """
        all_embeddings: list[list[float]] = []

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.info(
            f"Embedding {len(texts)} texts in {len(batches)} batches "
            f"(batch_size={batch_size})"
        )

        for batch in tqdm(batches, desc="Embedding batches", unit="batch"):
            for text in batch:
                embedding = self._embed_with_retry(text)
                all_embeddings.append(embedding)

        logger.success(f"Embedded {len(all_embeddings)} texts successfully")
        return all_embeddings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_with_retry(
        self,
        text: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> list[float]:
        """
        Call ollama.embeddings() with exponential back-off retry.

        Parameters
        ----------
        text : str
            Text to embed.
        max_retries : int
            Maximum number of attempts.
        base_delay : float
            Initial delay in seconds (doubles each retry).

        Returns
        -------
        list[float]
        """
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response: Any = ollama.embeddings(
                    model=self.model,
                    prompt=text,
                )
                embedding: list[float] = response["embedding"]
                if not embedding:
                    raise ValueError("Received empty embedding from Ollama")
                return embedding

            except Exception as exc:
                exc_str = str(exc).lower()

                # Model-not-found is a configuration error — retrying won't help.
                # Fail immediately with a clear, actionable message.
                if "not found" in exc_str or "404" in exc_str or "model" in exc_str and "pull" in exc_str:
                    raise RuntimeError(
                        f"Embedding model '{self.model}' is not available in Ollama. "
                        f"Fix: run  ollama pull {self.model}  in a terminal, "
                        f"then restart the API.\n"
                        f"Original error: {exc}"
                    ) from exc

                last_exc = exc
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Embedding attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                if attempt < max_retries:
                    time.sleep(delay)

        raise RuntimeError(
            f"All {max_retries} embedding attempts failed. "
            f"Last error: {last_exc}"
        )
