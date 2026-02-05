from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    page: int
    text: str


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, text in pages:
        if not text:
            continue

        start = 0
        idx = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(chunk_id=f"p{page_num}_c{idx}", page=page_num, text=piece))
            idx += 1

            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start

    return chunks
