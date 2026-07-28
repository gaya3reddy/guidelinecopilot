"""
GraphState definition for GuidelineCopilot's agentic retrieval loop.

This is the object that flows through every LangGraph node. Each node
reads what it needs from state and returns a dict of updates — LangGraph
merges those updates back into state automatically between node calls.
"""

from typing import Any, Dict, TypedDict


class GraphState(TypedDict):
    """
    Attributes:
        question: The current query used for retrieval. This gets
            overwritten on each rewrite iteration, so retrieval always
            uses the latest version.
        original_question: The user's question, untouched. Needed so
            generate() answers what the user actually asked, not the
            (possibly narrower) rewritten retrieval query.
        documents: Retrieved + reranked chunks from the current attempt.
            Each item is a dict shaped like HybridRetriever.search()'s
            output: {"text", "meta", "distance", "bm25_score", "rrf_score"}.
            Same shape your _build_context() in pipeline.py already expects.
        generation: The LLM's answer, once produced.
        retry_count: How many times we've rewritten the query. Capped
            in the routing logic to prevent infinite loops on queries
            that genuinely have no good answer in the corpus.
        grade: "yes"/"no" verdict from grade_documents on whether the
            current retrieval is sufficient. Consumed by the routing
            function to decide generate vs. rewrite_query.
    """

    question: str
    original_question: str
    documents: list[Dict[str, Any]]
    generation: str
    retry_count: int
    grade: str


# Ablation (eval/reports/latency_comparison_retries2.json vs retries=1 run):
# retries=1 matched or beat retries=2 on all 4 RAGAS metrics while cutting
# unanswerable-question latency ~28%. Second retry wasn't earning its cost.
MAX_RETRIES = 1
