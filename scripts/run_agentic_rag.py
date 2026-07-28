"""
Manual smoke test for the agentic retrieval graph.

Not a pytest test — this makes real OpenAI + ChromaDB calls against
your live index, same as running the API by hand. Use it to see the
retry loop actually fire before wiring this into apps/api.

Usage:
    python -m scripts.run_agentic_rag "your question here"
"""

from __future__ import annotations

import sys

from core.graph.graph import build_graph, build_initial_state


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is hand hygiene?"

    app = build_graph()
    initial_state = build_initial_state(question)

    print(f"Question: {question}\n")

    final_state = app.invoke(initial_state)

    print(f"Retries used: {final_state['retry_count']}")
    print(f"Final grade: {final_state['grade']}")
    print(f"Final query used for retrieval: {final_state['question']}\n")
    print(f"Answer:\n{final_state['generation']}")


if __name__ == "__main__":
    main()
