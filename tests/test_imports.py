"""
Smoke tests: verify that key modules import without errors.
If any of these fail it usually means a broken dependency or circular import.
"""


def test_pipeline_imports():
    from core.rag import pipeline  # noqa: F401


def test_chunker_imports():
    from core.ingestion import chunker  # noqa: F401


def test_config_imports():
    from apps.api import config  # noqa: F401


def test_vectorstore_imports():
    from core.retrieval import vectorstore  # noqa: F401


def test_embedder_imports():
    from core.retrieval import embedder  # noqa: F401
