"""
Shared test fixtures.

Everything here is deterministic and offline: no Ollama, no ChromaDB server, no
model downloads. Dense retrieval is simulated with a bag-of-words embedder and
an in-memory store that implements the same interface as ChromaVectorStore, so
the retrieval *pipeline* is exercised for real even though the production
embedding model is not.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.retrieval.bm25_store import BM25Store


# ---------------------------------------------------------------------------
# Fixture corpus — 10-K style snippets with known ground truth
# ---------------------------------------------------------------------------

FIXTURE_DOCUMENTS: list[dict] = [
    {
        "chunk_id": "aapl-rev-2023",
        "text": (
            "Total net sales for fiscal 2023 were $383.3 billion, a decrease of "
            "3% compared to $394.3 billion in fiscal 2022, driven primarily by "
            "lower iPhone and Mac net sales."
        ),
        "ticker": "AAPL", "year": "2023", "section": "ITEM 7",
        "page_num": 28, "block_type": "table", "contains_numbers": True,
        "source_file": "AAPL_2023_10K.pdf",
    },
    {
        "chunk_id": "aapl-rnd-2023",
        "text": (
            "Research and development expense was $29.9 billion in fiscal 2023, "
            "an increase of 14% year over year, reflecting continued investment "
            "in silicon and services engineering headcount."
        ),
        "ticker": "AAPL", "year": "2023", "section": "ITEM 7",
        "page_num": 31, "block_type": "table", "contains_numbers": True,
        "source_file": "AAPL_2023_10K.pdf",
    },
    {
        "chunk_id": "aapl-risk-supply",
        "text": (
            "The Company depends on component and product manufacturing and "
            "logistical services provided by outsourcing partners, many of which "
            "are located outside of the United States, concentrating supply "
            "chain risk in a small number of suppliers."
        ),
        "ticker": "AAPL", "year": "2023", "section": "ITEM 1A",
        "page_num": 12, "block_type": "text", "contains_numbers": False,
        "source_file": "AAPL_2023_10K.pdf",
    },
    {
        "chunk_id": "msft-rev-2023",
        "text": (
            "Total revenue was $211.9 billion for fiscal year 2023, an increase "
            "of 7%, driven by growth in the Intelligent Cloud segment and Azure "
            "consumption."
        ),
        "ticker": "MSFT", "year": "2023", "section": "ITEM 7",
        "page_num": 40, "block_type": "table", "contains_numbers": True,
        "source_file": "MSFT_2023_10K.pdf",
    },
    {
        "chunk_id": "msft-cloud",
        "text": (
            "Microsoft Cloud revenue increased 22% to $111.6 billion, with Azure "
            "and other cloud services revenue growing 27% for the fiscal year."
        ),
        "ticker": "MSFT", "year": "2023", "section": "ITEM 7",
        "page_num": 42, "block_type": "table", "contains_numbers": True,
        "source_file": "MSFT_2023_10K.pdf",
    },
    {
        "chunk_id": "msft-risk-ai",
        "text": (
            "Issues in the development and use of artificial intelligence may "
            "result in reputational or competitive harm, regulatory scrutiny, "
            "and legal liability for the company."
        ),
        "ticker": "MSFT", "year": "2023", "section": "ITEM 1A",
        "page_num": 20, "block_type": "text", "contains_numbers": False,
        "source_file": "MSFT_2023_10K.pdf",
    },
    {
        "chunk_id": "nvda-rev-2023",
        "text": (
            "Revenue for fiscal year 2023 was $26.97 billion, flat compared with "
            "the prior year, as Data Center growth was offset by a decline in "
            "Gaming revenue."
        ),
        "ticker": "NVDA", "year": "2023", "section": "ITEM 7",
        "page_num": 35, "block_type": "table", "contains_numbers": True,
        "source_file": "NVDA_2023_10K.pdf",
    },
    {
        "chunk_id": "nvda-datacenter",
        "text": (
            "Data Center revenue was $15.01 billion, up 41%, reflecting strong "
            "demand for accelerated computing platforms from hyperscale "
            "customers."
        ),
        "ticker": "NVDA", "year": "2023", "section": "ITEM 7",
        "page_num": 36, "block_type": "table", "contains_numbers": True,
        "source_file": "NVDA_2023_10K.pdf",
    },
    {
        "chunk_id": "nvda-risk-fab",
        "text": (
            "We depend on foundries such as TSMC to manufacture our "
            "semiconductor products, and long manufacturing lead times expose us "
            "to supply chain risk and inventory obsolescence."
        ),
        "ticker": "NVDA", "year": "2023", "section": "ITEM 1A",
        "page_num": 18, "block_type": "text", "contains_numbers": False,
        "source_file": "NVDA_2023_10K.pdf",
    },
    {
        "chunk_id": "amzn-rev-2022",
        "text": (
            "Net sales increased 9% to $513.98 billion in 2022, compared with "
            "$469.82 billion in 2021, with growth driven by AWS and advertising "
            "services."
        ),
        "ticker": "AMZN", "year": "2022", "section": "ITEM 7",
        "page_num": 25, "block_type": "table", "contains_numbers": True,
        "source_file": "AMZN_2022_10K.pdf",
    },
    {
        "chunk_id": "amzn-aws",
        "text": (
            "AWS segment net sales were $80.1 billion in 2022, an increase of "
            "29% year over year, and operating income for the segment was $22.8 "
            "billion."
        ),
        "ticker": "AMZN", "year": "2022", "section": "ITEM 7",
        "page_num": 26, "block_type": "table", "contains_numbers": True,
        "source_file": "AMZN_2022_10K.pdf",
    },
    {
        "chunk_id": "amzn-tech-content",
        "text": (
            "Technology and content costs were $73.2 billion, which includes "
            "payroll for engineering staff and infrastructure spend; Amazon does "
            "not separately report a research and development line item."
        ),
        "ticker": "AMZN", "year": "2022", "section": "ITEM 7",
        "page_num": 27, "block_type": "text", "contains_numbers": True,
        "source_file": "AMZN_2022_10K.pdf",
    },
    {
        "chunk_id": "googl-rev-2023",
        "text": (
            "Total revenues were $307.4 billion for 2023, an increase of 9% "
            "compared to 2022, led by Google Search and YouTube advertising "
            "revenues."
        ),
        "ticker": "GOOGL", "year": "2023", "section": "ITEM 7",
        "page_num": 33, "block_type": "table", "contains_numbers": True,
        "source_file": "GOOGL_2023_10K.pdf",
    },
    {
        "chunk_id": "googl-cloud",
        "text": (
            "Google Cloud revenues were $33.1 billion for 2023, up 26%, and the "
            "segment recorded its first full year of operating profitability."
        ),
        "ticker": "GOOGL", "year": "2023", "section": "ITEM 7",
        "page_num": 34, "block_type": "table", "contains_numbers": True,
        "source_file": "GOOGL_2023_10K.pdf",
    },
    {
        "chunk_id": "googl-risk-antitrust",
        "text": (
            "We are subject to ongoing antitrust and competition investigations "
            "and litigation in the United States and the European Union that "
            "could result in substantial fines or changes to our business "
            "practices."
        ),
        "ticker": "GOOGL", "year": "2023", "section": "ITEM 1A",
        "page_num": 15, "block_type": "text", "contains_numbers": False,
        "source_file": "GOOGL_2023_10K.pdf",
    },
    {
        "chunk_id": "aapl-segments",
        "text": (
            "The Company's reportable segments are Americas, Europe, Greater "
            "China, Japan and Rest of Asia Pacific, and segment operating "
            "performance is evaluated on net sales and operating income."
        ),
        "ticker": "AAPL", "year": "2023", "section": "ITEM 8",
        "page_num": 55, "block_type": "text", "contains_numbers": False,
        "source_file": "AAPL_2023_10K.pdf",
    },
    {
        "chunk_id": "msft-segments",
        "text": (
            "We report our financial performance in three segments: Productivity "
            "and Business Processes, Intelligent Cloud, and More Personal "
            "Computing."
        ),
        "ticker": "MSFT", "year": "2023", "section": "ITEM 8",
        "page_num": 60, "block_type": "text", "contains_numbers": False,
        "source_file": "MSFT_2023_10K.pdf",
    },
    {
        "chunk_id": "generic-forward-looking",
        "text": (
            "This report contains forward-looking statements within the meaning "
            "of the Private Securities Litigation Reform Act of 1995 that "
            "involve risks and uncertainties."
        ),
        "ticker": "AAPL", "year": "2023", "section": "ITEM 1",
        "page_num": 3, "block_type": "text", "contains_numbers": False,
        "source_file": "AAPL_2023_10K.pdf",
    },
    {
        "chunk_id": "generic-accounting",
        "text": (
            "The preparation of financial statements in conformity with GAAP "
            "requires management to make estimates and assumptions that affect "
            "the reported amounts of assets and liabilities."
        ),
        "ticker": "MSFT", "year": "2023", "section": "ITEM 8",
        "page_num": 58, "block_type": "text", "contains_numbers": False,
        "source_file": "MSFT_2023_10K.pdf",
    },
    {
        "chunk_id": "generic-employees",
        "text": (
            "As of the end of the fiscal year the company had approximately "
            "161,000 full-time equivalent employees across its global "
            "operations."
        ),
        "ticker": "AAPL", "year": "2023", "section": "ITEM 1",
        "page_num": 8, "block_type": "text", "contains_numbers": True,
        "source_file": "AAPL_2023_10K.pdf",
    },

    # -----------------------------------------------------------------------
    # Near-miss distractors.
    #
    # Prior-year and sibling-segment restatements of the documents above. They
    # share almost all of their wording with a correct answer and differ only in
    # the fiscal year or the line item, which is exactly the confusion a
    # financial retriever has to resolve. Without these the fixture set is
    # trivially separable and Recall@5 saturates at 1.0 for every strategy,
    # making the accuracy tests worthless as a regression guard.
    # -----------------------------------------------------------------------
    {
        "chunk_id": "aapl-rev-2022",
        "text": (
            "Total net sales for fiscal 2022 were $394.3 billion, an increase of "
            "8% compared to $365.8 billion in fiscal 2021, driven by higher "
            "iPhone and Services net sales."
        ),
        "ticker": "AAPL", "year": "2022", "section": "ITEM 7",
        "page_num": 27, "block_type": "table", "contains_numbers": True,
        "source_file": "AAPL_2022_10K.pdf",
    },
    {
        "chunk_id": "aapl-rnd-2022",
        "text": (
            "Research and development expense was $26.3 billion in fiscal 2022, "
            "an increase of 20% year over year, reflecting higher headcount "
            "related expenses."
        ),
        "ticker": "AAPL", "year": "2022", "section": "ITEM 7",
        "page_num": 30, "block_type": "table", "contains_numbers": True,
        "source_file": "AAPL_2022_10K.pdf",
    },
    {
        "chunk_id": "aapl-sga-2023",
        "text": (
            "Selling, general and administrative expense was $24.9 billion in "
            "fiscal 2023, roughly flat compared with the prior year."
        ),
        "ticker": "AAPL", "year": "2023", "section": "ITEM 7",
        "page_num": 32, "block_type": "table", "contains_numbers": True,
        "source_file": "AAPL_2023_10K.pdf",
    },
    {
        "chunk_id": "msft-rev-2022",
        "text": (
            "Total revenue was $198.3 billion for fiscal year 2022, an increase "
            "of 18%, driven by growth across all three of our reportable "
            "segments."
        ),
        "ticker": "MSFT", "year": "2022", "section": "ITEM 7",
        "page_num": 39, "block_type": "table", "contains_numbers": True,
        "source_file": "MSFT_2022_10K.pdf",
    },
    {
        "chunk_id": "msft-cloud-2022",
        "text": (
            "Microsoft Cloud revenue increased 32% to $91.2 billion in fiscal "
            "2022, with Azure and other cloud services revenue growing 45%."
        ),
        "ticker": "MSFT", "year": "2022", "section": "ITEM 7",
        "page_num": 41, "block_type": "table", "contains_numbers": True,
        "source_file": "MSFT_2022_10K.pdf",
    },
    {
        "chunk_id": "nvda-rev-2022",
        "text": (
            "Revenue for fiscal year 2022 was $26.91 billion, up 61% from the "
            "prior year, with record results in both Data Center and Gaming."
        ),
        "ticker": "NVDA", "year": "2022", "section": "ITEM 7",
        "page_num": 34, "block_type": "table", "contains_numbers": True,
        "source_file": "NVDA_2022_10K.pdf",
    },
    {
        "chunk_id": "nvda-gaming",
        "text": (
            "Gaming revenue was $9.07 billion, down 27%, reflecting lower "
            "shipments of GeForce GPUs as channel partners reduced inventory."
        ),
        "ticker": "NVDA", "year": "2023", "section": "ITEM 7",
        "page_num": 37, "block_type": "table", "contains_numbers": True,
        "source_file": "NVDA_2023_10K.pdf",
    },
    {
        "chunk_id": "amzn-rev-2021",
        "text": (
            "Net sales increased 22% to $469.82 billion in 2021, compared with "
            "$386.06 billion in 2020, driven by increased unit sales and "
            "subscription services."
        ),
        "ticker": "AMZN", "year": "2021", "section": "ITEM 7",
        "page_num": 24, "block_type": "table", "contains_numbers": True,
        "source_file": "AMZN_2021_10K.pdf",
    },
    {
        "chunk_id": "amzn-aws-2021",
        "text": (
            "AWS segment net sales were $62.2 billion in 2021, an increase of "
            "37% year over year, and operating income for the segment was $18.5 "
            "billion."
        ),
        "ticker": "AMZN", "year": "2021", "section": "ITEM 7",
        "page_num": 25, "block_type": "table", "contains_numbers": True,
        "source_file": "AMZN_2021_10K.pdf",
    },
    {
        "chunk_id": "googl-rev-2022",
        "text": (
            "Total revenues were $282.8 billion for 2022, an increase of 10% "
            "compared to 2021, led by Google Search and cloud revenues."
        ),
        "ticker": "GOOGL", "year": "2022", "section": "ITEM 7",
        "page_num": 32, "block_type": "table", "contains_numbers": True,
        "source_file": "GOOGL_2022_10K.pdf",
    },
    {
        "chunk_id": "googl-cloud-2022",
        "text": (
            "Google Cloud revenues were $26.3 billion for 2022, up 37%, and the "
            "segment recorded an operating loss for the full year."
        ),
        "ticker": "GOOGL", "year": "2022", "section": "ITEM 7",
        "page_num": 33, "block_type": "table", "contains_numbers": True,
        "source_file": "GOOGL_2022_10K.pdf",
    },
    {
        "chunk_id": "amzn-risk-fulfillment",
        "text": (
            "We may be unable to adequately staff and operate our fulfillment "
            "network during peak season, which could harm customer experience "
            "and increase shipping costs."
        ),
        "ticker": "AMZN", "year": "2022", "section": "ITEM 1A",
        "page_num": 14, "block_type": "text", "contains_numbers": False,
        "source_file": "AMZN_2022_10K.pdf",
    },
]


# Ground truth: query -> chunk_ids that genuinely answer it.
# Used by the Recall@5 tests in test_retrieval_accuracy.py.
LABELLED_QUERIES: list[dict] = [
    {
        "query": "What was Apple's total net sales in fiscal 2023?",
        "relevant": {"aapl-rev-2023"},
    },
    {
        "query": "How much did Apple spend on research and development in 2023?",
        "relevant": {"aapl-rnd-2023"},
    },
    {
        "query": "What was Microsoft total revenue for fiscal year 2023?",
        "relevant": {"msft-rev-2023"},
    },
    {
        "query": "How fast did Azure and Microsoft Cloud revenue grow?",
        "relevant": {"msft-cloud"},
    },
    {
        "query": "What was NVIDIA Data Center revenue?",
        "relevant": {"nvda-datacenter"},
    },
    {
        "query": "What were Amazon AWS segment net sales and operating income?",
        "relevant": {"amzn-aws"},
    },
    {
        "query": "What were Alphabet total revenues in 2023?",
        "relevant": {"googl-rev-2023"},
    },
    {
        "query": "Which companies disclose supply chain risk from manufacturing partners?",
        "relevant": {"aapl-risk-supply", "nvda-risk-fab"},
    },
    {
        "query": "What antitrust litigation does Google face?",
        "relevant": {"googl-risk-antitrust"},
    },
    {
        # The year is explicit because the corpus holds both the 2022 and 2023
        # Google Cloud disclosures. Without it the label would be ambiguous and
        # the "failure" would be the fixture's, not the retriever's.
        "query": "What were Google Cloud revenues in 2023 and was the segment profitable?",
        "relevant": {"googl-cloud"},
    },
]


_TOKEN_RE = re.compile(r"[a-z0-9$%.]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Offline stand-ins for the production backends
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """
    Deterministic bag-of-words embedder.

    Not a semantic model — it produces an L2-normalised term-frequency vector
    over a fixed vocabulary. That is enough for the in-memory store to rank by
    cosine similarity the way ChromaDB does, without a network call.
    """

    def __init__(self, vocabulary: list[str]) -> None:
        self._vocab = {term: i for i, term in enumerate(vocabulary)}

    @property
    def dim(self) -> int:
        return len(self._vocab)

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * len(self._vocab)
        for token in _tokens(text):
            idx = self._vocab.get(token)
            if idx is not None:
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class InMemoryVectorStore:
    """
    Stands in for ChromaVectorStore with the same search() contract.

    Implements cosine ranking and the same AND-combined metadata filtering, so
    VectorRetriever is exercised against realistic behaviour including filters.
    """

    def __init__(self, documents: list[dict], embedder: FakeEmbedder) -> None:
        self._documents = documents
        self._embeddings = [embedder.embed_text(d["text"]) for d in documents]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _matches(doc: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        for key in ("ticker", "year", "block_type", "section"):
            wanted = filters.get(key)
            if wanted is not None and str(doc.get(key, "")) != str(wanted):
                return False
        return True

    def search(self, query_embedding, k=10, filters=None) -> list[dict]:
        scored = []
        for doc, emb in zip(self._documents, self._embeddings):
            if not self._matches(doc, filters):
                continue
            score = self._cosine(query_embedding, emb)
            if score <= 0.0:
                continue
            scored.append((score, doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            {
                "text": doc["text"],
                "metadata": {
                    "chunk_id": doc["chunk_id"],
                    "ticker": doc["ticker"],
                    "year": doc["year"],
                    "section": doc["section"],
                    "page_num": doc["page_num"],
                    "block_type": doc["block_type"],
                },
                "distance": 1.0 - score,
                "score": score,
            }
            for score, doc in scored[:k]
        ]


class KeywordReranker:
    """
    Deterministic stand-in for CrossEncoderReranker.

    Scores by the fraction of distinct query terms present in the passage, which
    mimics the cross-encoder's job (precision re-scoring of an existing
    candidate set) without loading a 90 MB transformer in unit tests.
    """

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []
        if not query.strip():
            return chunks[:top_k]

        stop = {"the", "a", "an", "was", "were", "is", "are", "what", "how",
                "much", "did", "does", "do", "in", "for", "of", "and", "on",
                "its", "it", "which", "that", "s"}
        terms = {t for t in _tokens(query) if t not in stop and len(t) > 1}

        scored = []
        for i, chunk in enumerate(chunks):
            passage = set(_tokens(chunk.get("text", "")))
            hits = len(terms & passage)
            score = hits / len(terms) if terms else 0.0
            # index breaks ties so ordering is stable and reproducible
            scored.append((-score, i, chunk, score))
        scored.sort(key=lambda x: (x[0], x[1]))

        out = []
        for _, _, chunk, score in scored[:top_k]:
            copy = dict(chunk)
            copy["rerank_score"] = float(score)
            out.append(copy)
        return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def documents() -> list[dict]:
    """The fixture corpus. Session-scoped and treated as read-only."""
    return [dict(d) for d in FIXTURE_DOCUMENTS]


@pytest.fixture(scope="session")
def vocabulary(documents) -> list[str]:
    """Vocabulary spanning the corpus plus the labelled query terms."""
    vocab: set[str] = set()
    for doc in documents:
        vocab.update(_tokens(doc["text"]))
    for item in LABELLED_QUERIES:
        vocab.update(_tokens(item["query"]))
    return sorted(vocab)


@pytest.fixture(scope="session")
def embedder(vocabulary) -> FakeEmbedder:
    return FakeEmbedder(vocabulary)


@pytest.fixture(scope="session")
def vector_store(documents, embedder) -> InMemoryVectorStore:
    return InMemoryVectorStore(documents, embedder)


@pytest.fixture
def bm25_store(documents, tmp_path, monkeypatch) -> BM25Store:
    """
    A real BM25Okapi index over the fixture corpus.

    rank-bm25 is pure Python, so this is the genuine production class rather
    than a mock. The index path is redirected into tmp_path so building it can
    never overwrite the developer's real data/bm25_index.pkl.
    """
    monkeypatch.setattr(settings, "bm25_index_path", tmp_path / "bm25_index.pkl")
    store = BM25Store()
    store.build_index(documents)
    return store


@pytest.fixture
def reranker() -> KeywordReranker:
    return KeywordReranker()
