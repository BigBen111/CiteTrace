from __future__ import annotations

import re
import unicodedata
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


def clean_extracted_text(value: str) -> str:
    """Normalize PDF text and remove control-glyph noise without losing line breaks."""
    normalized = unicodedata.normalize("NFKC", value)
    printable = "".join(
        char if char in "\n\t" or char.isprintable() else " " for char in normalized
    )
    lines = [" ".join(line.split()) for line in printable.splitlines()]
    return "\n".join(line for line in lines if line)


def parse_pdf(path: str | Path, paper_id: str, *, chunk_chars: int = 1800, overlap: int = 250) -> list[PaperChunk]:
    doc = fitz.open(str(path))
    chunks: list[PaperChunk] = []
    current_section: str | None = None

    for page_idx, page in enumerate(doc):
        raw = clean_extracted_text(page.get_text("text"))
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


def build_pdf_metadata(path: str | Path, chunks: list[PaperChunk]) -> dict:
    with fitz.open(str(path)) as doc:
        total_pages = doc.page_count
    text_pages = sorted({chunk.page for chunk in chunks})
    reference_start = next(
        (
            index
            for index, chunk in enumerate(chunks)
            if re.search(r"(?im)^\s*(references|bibliography)\s*$", chunk.text)
        ),
        None,
    )
    if reference_start is None:
        reference_pages = sorted(
            {
                chunk.page
                for chunk in chunks
                if "reference" in (chunk.section or "").casefold()
                or "bibliograph" in (chunk.section or "").casefold()
            }
        )
    else:
        reference_pages = sorted({chunk.page for chunk in chunks[reference_start:]})
    return {
        "total_pages": total_pages,
        "pages_with_text": len(text_pages),
        "text_pages": text_pages,
        "references_detected": bool(reference_pages),
        "reference_pages": reference_pages,
    }
