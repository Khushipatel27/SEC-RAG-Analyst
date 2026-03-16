"""
RAG Evaluator
Measures retrieval and generation quality across a standardised question set.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

_EVAL_QUESTIONS_PATH = Path("data/eval/eval_questions.json")
_EVAL_RESULTS_PATH = Path("data/eval/eval_results.json")
_RETRIEVAL_COMPARISON_PATH = Path("data/eval/retrieval_comparison.json")


class RAGEvaluator:
    """
    Evaluates a SECRAGPipeline instance on a curated set of 20 questions.

    Metrics computed
    ----------------
    - Keyword Overlap (F1-like)
    - Numerical Accuracy (fraction of key numbers present in answer)
    - Citation Rate (fraction of answers that reference a source)
    - Retrieval Recall@k, Precision@k, MRR
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        logger.info("RAGEvaluator initialized")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_eval_questions(self) -> list[dict]:
        """
        Load evaluation questions from data/eval/eval_questions.json.

        Returns
        -------
        list[dict]  – each dict has keys: question, reference_answer,
                      ticker, year, question_type, difficulty.
        """
        path = Path(_EVAL_QUESTIONS_PATH)
        if not path.exists():
            logger.error(f"Eval questions file not found: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            questions: list[dict] = json.load(f)

        logger.info(f"Loaded {len(questions)} eval questions from {path}")
        return questions

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def compute_keyword_overlap(self, reference: str, generated: str) -> float:
        """
        Compute token-level F1 overlap between reference and generated answers.

        Parameters
        ----------
        reference : str
        generated : str

        Returns
        -------
        float  – F1 score in [0, 1].
        """
        def tokenize(text: str) -> set[str]:
            return set(re.findall(r"\w+", text.lower()))

        ref_tokens = tokenize(reference)
        gen_tokens = tokenize(generated)

        if not ref_tokens or not gen_tokens:
            return 0.0

        common = ref_tokens & gen_tokens
        if not common:
            return 0.0

        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(ref_tokens)

        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 4)

    def compute_numerical_accuracy(self, reference: str, generated: str) -> float:
        """
        Compute the fraction of key numbers in *reference* that also appear
        in *generated*.

        Parameters
        ----------
        reference : str
        generated : str

        Returns
        -------
        float  – fraction in [0, 1], or 1.0 if no numbers in reference.
        """
        number_pattern = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

        ref_numbers = set(number_pattern.findall(reference))
        if not ref_numbers:
            return 1.0  # no numbers to check → vacuously correct

        gen_numbers = set(number_pattern.findall(generated))
        matched = ref_numbers & gen_numbers
        score = len(matched) / len(ref_numbers)
        return round(score, 4)

    def compute_citation_rate(self, answers: list[dict]) -> float:
        """
        Fraction of generated answers that include at least one cited source.

        A citation is detected by the presence of a non-empty "sources" list.

        Parameters
        ----------
        answers : list[dict]  – generated answer dicts from pipeline.ask()

        Returns
        -------
        float  – citation rate in [0, 1].
        """
        if not answers:
            return 0.0

        cited = sum(
            1 for a in answers if a.get("sources") and len(a["sources"]) > 0
        )
        rate = cited / len(answers)
        logger.debug(f"Citation rate: {cited}/{len(answers)} = {rate:.4f}")
        return round(rate, 4)

    def compute_retrieval_metrics(
        self,
        retrieved_chunks: list[dict],
        relevant_tickers: list[str],
        relevant_years: list[str],
    ) -> dict:
        """
        Compute Recall@k, Precision@k, and MRR for a single query.

        "Relevant" is defined as a chunk whose ticker is in *relevant_tickers*
        AND whose year is in *relevant_years*.

        Parameters
        ----------
        retrieved_chunks : list[dict]
        relevant_tickers : list[str]
        relevant_years : list[str]

        Returns
        -------
        dict  – {recall_at_k, precision_at_k, mrr, num_relevant, k}
        """
        k = len(retrieved_chunks)
        if k == 0:
            return {
                "recall_at_k": 0.0,
                "precision_at_k": 0.0,
                "mrr": 0.0,
                "num_relevant": 0,
                "k": 0,
            }

        relevant_tickers_set = {t.upper() for t in relevant_tickers}
        relevant_years_set = set(relevant_years)

        num_relevant = 0
        first_relevant_rank: Optional[int] = None

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata") or {}
            ticker = (chunk.get("ticker") or meta.get("ticker", "")).upper()
            year = str(chunk.get("year") or meta.get("year", ""))

            is_relevant = (
                ticker in relevant_tickers_set and year in relevant_years_set
            )
            if is_relevant:
                num_relevant += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        precision_at_k = num_relevant / k
        # Assume there is at least 1 true relevant document in the corpus
        recall_at_k = min(num_relevant, 1) / max(1, min(num_relevant, 1))
        mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

        return {
            "recall_at_k": round(recall_at_k, 4),
            "precision_at_k": round(precision_at_k, 4),
            "mrr": round(mrr, 4),
            "num_relevant": num_relevant,
            "k": k,
        }

    # ------------------------------------------------------------------
    # Full evaluation run
    # ------------------------------------------------------------------

    def run_evaluation(self, pipeline: Any) -> dict:
        """
        Run all 20 evaluation questions and compute aggregated metrics.

        Saves results to data/eval/eval_results.json and prints a report.

        Parameters
        ----------
        pipeline : SECRAGPipeline

        Returns
        -------
        dict  – full evaluation report.
        """
        questions = self.load_eval_questions()
        if not questions:
            logger.error("No eval questions found; aborting evaluation.")
            return {}

        results: list[dict] = []
        all_answers: list[dict] = []
        keyword_scores: list[float] = []
        numerical_scores: list[float] = []
        retrieval_metrics_list: list[dict] = []

        logger.info(f"Running evaluation over {len(questions)} questions...")

        for i, q in enumerate(questions, start=1):
            question = q["question"]
            reference = q.get("reference_answer", "")
            ticker = q.get("ticker", None)
            year = q.get("year", None)

            logger.info(f"[{i}/{len(questions)}] {question[:80]}")

            start = time.perf_counter()
            try:
                answer_dict = pipeline.ask(
                    question=question,
                    ticker_filter=ticker,
                    year_filter=year,
                )
            except Exception as exc:
                logger.error(f"Pipeline.ask() failed for Q{i}: {exc}")
                answer_dict = {
                    "answer": f"[ERROR: {exc}]",
                    "sources": [],
                    "latency_ms": 0,
                }

            elapsed = (time.perf_counter() - start) * 1000
            all_answers.append(answer_dict)

            generated = answer_dict.get("answer", "")
            kw_score = self.compute_keyword_overlap(reference, generated)
            num_score = self.compute_numerical_accuracy(reference, generated)

            keyword_scores.append(kw_score)
            numerical_scores.append(num_score)

            # Retrieval metrics (approximate from sources)
            sources = answer_dict.get("sources", [])
            ret_metrics = self.compute_retrieval_metrics(
                retrieved_chunks=sources,
                relevant_tickers=[ticker] if ticker else [],
                relevant_years=[str(year)] if year else [],
            )
            retrieval_metrics_list.append(ret_metrics)

            results.append(
                {
                    "question_id": i,
                    "question": question,
                    "question_type": q.get("question_type", ""),
                    "difficulty": q.get("difficulty", ""),
                    "ticker": ticker,
                    "year": year,
                    "reference_answer": reference,
                    "generated_answer": generated,
                    "keyword_overlap": kw_score,
                    "numerical_accuracy": num_score,
                    "retrieval_metrics": ret_metrics,
                    "latency_ms": elapsed,
                    "num_sources": len(sources),
                }
            )

        # --- Aggregate metrics ---
        citation_rate = self.compute_citation_rate(all_answers)
        avg_keyword = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
        avg_numerical = sum(numerical_scores) / len(numerical_scores) if numerical_scores else 0
        avg_mrr = (
            sum(m["mrr"] for m in retrieval_metrics_list) / len(retrieval_metrics_list)
            if retrieval_metrics_list else 0
        )
        avg_precision = (
            sum(m["precision_at_k"] for m in retrieval_metrics_list) / len(retrieval_metrics_list)
            if retrieval_metrics_list else 0
        )

        report = {
            "total_questions": len(questions),
            "avg_keyword_overlap": round(avg_keyword, 4),
            "avg_numerical_accuracy": round(avg_numerical, 4),
            "citation_rate": citation_rate,
            "avg_mrr": round(avg_mrr, 4),
            "avg_precision_at_k": round(avg_precision, 4),
            "per_question_results": results,
        }

        # Save results
        _EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.success(f"Evaluation results saved to {_EVAL_RESULTS_PATH}")

        # Print report
        self._print_report(report)

        # Also run retrieval method comparison so the dashboard chart populates
        try:
            self.compare_retrieval_methods()
        except Exception as exc:
            logger.warning(f"Retrieval comparison skipped: {exc}")

        return report

    def compare_retrieval_methods(self) -> dict:
        """
        Compare vector-only, BM25-only, hybrid, and hybrid+rerank retrieval.

        Runs the same questions with each method and computes MRR and P@k.
        Saves to data/eval/retrieval_comparison.json.

        Returns
        -------
        dict  – comparison results by method.
        """
        questions = self.load_eval_questions()
        if not questions:
            logger.error("No eval questions found; aborting comparison.")
            return {}

        methods = ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]
        comparison: dict[str, dict] = {m: {"mrr_scores": [], "precision_scores": []} for m in methods}

        for q in questions:
            question = q["question"]
            ticker = q.get("ticker", None)
            year = q.get("year", None)
            relevant_tickers = [ticker] if ticker else []
            relevant_years = [str(year)] if year else []

            for method in methods:
                try:
                    chunks = self._retrieve_with_method(question, method, ticker, year)
                    metrics = self.compute_retrieval_metrics(
                        retrieved_chunks=chunks,
                        relevant_tickers=relevant_tickers,
                        relevant_years=relevant_years,
                    )
                    comparison[method]["mrr_scores"].append(metrics["mrr"])
                    comparison[method]["precision_scores"].append(metrics["precision_at_k"])
                except Exception as exc:
                    logger.warning(f"Method '{method}' failed for '{question[:40]}': {exc}")
                    comparison[method]["mrr_scores"].append(0.0)
                    comparison[method]["precision_scores"].append(0.0)

        # Aggregate
        results: dict[str, dict] = {}
        for method, data in comparison.items():
            mrr_scores = data["mrr_scores"]
            p_scores = data["precision_scores"]
            results[method] = {
                "avg_mrr": round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else 0,
                "avg_precision_at_k": round(sum(p_scores) / len(p_scores), 4) if p_scores else 0,
                "num_questions": len(mrr_scores),
            }

        _RETRIEVAL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_RETRIEVAL_COMPARISON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.success(f"Retrieval comparison saved to {_RETRIEVAL_COMPARISON_PATH}")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_with_method(
        self,
        query: str,
        method: str,
        ticker: Optional[str],
        year: Optional[str],
    ) -> list[dict]:
        """Route retrieval to a specific method for comparison purposes."""
        pipeline = self._pipeline
        filters: dict = {}
        if ticker:
            filters["ticker"] = ticker
        if year:
            filters["year"] = str(year)

        if method == "vector_only":
            embedding = pipeline.embedder.embed_text(query)
            return pipeline.vector_store.search(embedding, k=10, filters=filters or None)

        elif method == "bm25_only":
            return pipeline.bm25_store.search(query, k=10)

        elif method == "hybrid":
            return pipeline.hybrid_searcher.search(query, k_final=10, filters=filters or None)

        elif method == "hybrid_rerank":
            chunks = pipeline.hybrid_searcher.search(query, k_final=10, filters=filters or None)
            return pipeline.reranker.rerank(query, chunks, top_k=5)

        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    @staticmethod
    def _print_report(report: dict) -> None:
        """Print a formatted evaluation report to stdout."""
        print("\n" + "=" * 60)
        print("  SEC RAG EVALUATION REPORT")
        print("=" * 60)
        print(f"  Total Questions      : {report['total_questions']}")
        print(f"  Avg Keyword Overlap  : {report['avg_keyword_overlap']:.4f}")
        print(f"  Avg Numerical Acc.   : {report['avg_numerical_accuracy']:.4f}")
        print(f"  Citation Rate        : {report['citation_rate']:.4f}")
        print(f"  Avg MRR              : {report['avg_mrr']:.4f}")
        print(f"  Avg Precision@k      : {report['avg_precision_at_k']:.4f}")
        print("=" * 60)

        # Per-type breakdown
        per_q = report.get("per_question_results", [])
        type_scores: dict[str, list[float]] = {}
        for r in per_q:
            qt = r.get("question_type", "unknown")
            type_scores.setdefault(qt, []).append(r.get("keyword_overlap", 0))

        if type_scores:
            print("\n  By Question Type (Keyword Overlap):")
            for qt, scores in sorted(type_scores.items()):
                avg = sum(scores) / len(scores)
                print(f"    {qt:<25} : {avg:.4f} ({len(scores)} Qs)")

        print("=" * 60 + "\n")
