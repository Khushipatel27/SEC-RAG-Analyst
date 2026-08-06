"""
Agentic layer (v2).

The v1 system was a single retrieval path: question → hybrid search → rerank →
LLM. That works for narrative questions ("what did Amazon say about AI?") but
fails on questions that are not retrieval problems at all — exact figures,
arithmetic across companies, and relationship traversal.

This package adds four specialists and a supervisor that routes between them:

    xbrl_agent          exact SEC-tagged figures, no LLM in the loop
    calc_agent          arithmetic over those figures, in plain Python
    graph_agent         entity/relationship traversal over the filing corpus
    narrative_agent     thin wrapper around the existing v1 pipeline
    verification_agent  checks the drafted answer is grounded in the evidence
    supervisor          LangGraph orchestrator tying it together
"""
