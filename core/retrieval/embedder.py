from __future__ import annotations

from typing import List, Optional

from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: Optional[str], model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI embeddings endpoint
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
