from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader


@dataclass
class PageText:
    page: int  # 1-based page number
    text: str


def extract_pages(pdf_bytes: bytes) -> List[PageText]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append(PageText(page=i + 1, text=txt.strip()))
    return pages
