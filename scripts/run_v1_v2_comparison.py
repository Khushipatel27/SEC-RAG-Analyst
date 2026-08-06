"""
run_v1_v2_comparison.py
───────────────────────
Runs the agentic question set through both the v1 pipeline and the v2
supervisor, and writes data/eval/agentic_comparison.json.

Run:  python scripts/run_v1_v2_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.agents.supervisor import SupervisorAgent  # noqa: E402
from src.evaluation.evaluator import RAGEvaluator  # noqa: E402
from src.pipeline import SECRAGPipeline  # noqa: E402


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    print("Initialising v1 pipeline...")
    pipeline = SECRAGPipeline()

    print("Initialising v2 supervisor...")
    supervisor = SupervisorAgent(pipeline=pipeline)

    evaluator = RAGEvaluator(pipeline=pipeline)
    evaluator.compare_v1_v2(supervisor=supervisor, pipeline=pipeline)


if __name__ == "__main__":
    main()
