"""
Shared context-formatting logic for RAG prompts.

Extracted from core/rag/pipeline.py's private _build_context() so it can
be reused by the LangGraph generate() node without reaching into another
module's private function. Behavior is unchanged from the original.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_context(citations: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into numbered, citable context blocks.

    Each citation dict must have "meta" (with "doc_id", "page") and "text",
    matching HybridRetriever.search()'s output shape.
    """
    blocks = []
    for i, c in enumerate(citations, start=1):
        meta = c["meta"]
        doc_id = meta.get("doc_id")
        page = meta.get("page")
        text = c["text"]
        blocks.append(f"[{i}] ({doc_id} p.{page})\n{text}")
    return "\n\n".join(blocks)
