from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class PaperChunk:
    chunk_id: str
    paper_id: str
    page: int
    text: str
    section: str | None = None


_SECTION_RE = re.compile(r"^(?:\d+(?:\.\d+)*)?\s*[A-Z][A-Za-z0-9 ,:\-]{2,80}$")


def parse_pdf(path: str | Path, paper_id: str, *, chunk_chars: int = 1800, overlap: int = 250) -> list[PaperChunk]:
    doc = fitz.open(str(path))
    chunks: list[PaperChunk] = []
    current_section: str | None = None

    for page_idx, page in enumerate(doc):
        raw = page.get_text("text")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for line in lines:
            if _SECTION_RE.match(line) and len(line.split()) <= 12:
                current_section = line

        text = "\n".join(lines)
        if not text:
            continue

        start = 0
        local_idx = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    PaperChunk(
                        chunk_id=f"{paper_id}:p{page_idx + 1}:c{local_idx}",
                        paper_id=paper_id,
                        page=page_idx + 1,
                        text=piece,
                        section=current_section,
                    )
                )
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
            local_idx += 1

    return chunks
