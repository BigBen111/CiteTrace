from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from threading import Lock

from citetrace.citation import CitationScanResult, exact_name_pattern, normalized_name
from citetrace.paper.parser import PaperChunk, clean_extracted_text
from citetrace.schemas import Evidence
from citetrace.utils.files import atomic_json


_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._+\-/][A-Za-z0-9]+)*")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_EN_STOP = {
    "a", "an", "and", "are", "article", "cite", "cited", "cites", "does",
    "has", "in", "is", "it", "of", "paper", "reference", "references",
    "the", "this", "to", "was", "were", "with",
}
_ZH_STOP = {"文章", "论文", "里面", "是否", "有没有", "有", "引用", "提到", "吗", "的", "了"}


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = [token for token in _LATIN_TOKEN_RE.findall(normalized) if token not in _EN_STOP]
    for run in _CJK_RUN_RE.findall(normalized):
        if run not in _ZH_STOP:
            tokens.append(run)
        tokens.extend(
            run[index : index + 2]
            for index in range(max(0, len(run) - 1))
            if run[index : index + 2] not in _ZH_STOP
        )
    return tokens


def _entity_terms(query: str) -> list[str]:
    return [
        token.casefold()
        for token in _LATIN_TOKEN_RE.findall(unicodedata.normalize("NFKC", query))
        if len(token) >= 3 and token.casefold() not in _EN_STOP
    ]


class JsonPaperStore:
    """Small transparent lexical store suitable for course-scale papers.

    The interface can later be replaced by FAISS/BM25/vector retrieval without
    changing retriever or pipeline contracts.
    """

    def __init__(self, root: str | Path = "data/index") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[int, list[PaperChunk], list[str], list[list[str]], Counter, float]] = {}
        self._cache_lock = Lock()

    def save(
        self,
        paper_id: str,
        chunks: list[PaperChunk],
        metadata: dict | None = None,
        *,
        pdf_filename: str | None = None,
    ) -> None:
        path = self.root / f"{paper_id}.json"
        # One commit point for chunks, coverage metadata and the immutable PDF.
        # Legacy list-based indexes remain readable; old PDFs are never overwritten.
        atomic_json(path, {
            "version": 2,
            "chunks": [c.__dict__ for c in chunks],
            "metadata": metadata,
            "pdf_filename": pdf_filename,
        })
        with self._cache_lock:
            self._cache.pop(paper_id, None)

    def load(self, paper_id: str) -> list[PaperChunk]:
        rows = json.loads((self.root / f"{paper_id}.json").read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows["chunks"]
        return [PaperChunk(**row) for row in rows]

    def search(self, paper_id: str, query: str, *, top_k: int = 6) -> list[Evidence]:
        chunks, cleaned, document_tokens, document_frequency, average_length = self._search_index(paper_id)
        q = Counter(_tokens(query))
        if not q:
            return []
        entity_terms = _entity_terms(query)
        document_count = len(document_tokens)
        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, PaperChunk]] = []
        for chunk, text, tokens in zip(chunks, cleaned, document_tokens):
            if not text or not tokens:
                continue
            frequencies = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for token, query_frequency in q.items():
                frequency = frequencies[token]
                if not frequency:
                    continue
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency[token] + 0.5)
                    / (document_frequency[token] + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * length / max(1.0, average_length)
                )
                score += query_frequency * inverse_document_frequency * (
                    frequency * (k1 + 1) / denominator
                )

            folded_text = text.casefold()
            for entity in entity_terms:
                if entity in folded_text:
                    score += 5.0
            if score > 0:
                scored.append(
                    (
                        score,
                        PaperChunk(
                            chunk_id=chunk.chunk_id,
                            paper_id=chunk.paper_id,
                            page=chunk.page,
                            text=text,
                            section=chunk.section,
                        ),
                    )
                )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Evidence(
                text=c.text,
                page=c.page,
                section=c.section,
                score=score,
                source=c.chunk_id,
            )
            for score, c in scored[:top_k]
        ]

    def validate_evidence(self, paper_id: str, evidence: Evidence) -> bool:
        if not evidence.source:
            return False
        chunks = {chunk.chunk_id: chunk for chunk in self.load(paper_id)}
        source = chunks.get(evidence.source)
        return bool(source and clean_extracted_text(evidence.text) in clean_extracted_text(source.text))

    def scan_citation(self, paper_id: str, target: str) -> CitationScanResult:
        chunks = self.load(paper_id)
        metadata = self._load_metadata(paper_id, chunks)
        exact_pattern = exact_name_pattern(target)
        compact_target = normalized_name(target)
        evidences: list[Evidence] = []
        reference_pages: set[int] = set(metadata.get("reference_pages", []))
        reference_sources = _reference_chunk_ids(chunks)

        for chunk in chunks:
            exact = exact_pattern.search(chunk.text)
            match = exact
            match_type = "exact"
            if match is None and compact_target:
                folded = unicodedata.normalize("NFKC", chunk.text).casefold()
                compact = normalized_name(folded)
                if compact_target not in compact:
                    continue
                match_type = "normalized"
            elif match is None:
                continue

            is_reference = chunk.chunk_id in reference_sources
            if is_reference:
                reference_pages.add(chunk.page)
            excerpt = _match_excerpt(chunk.text, match.start() if match else None)
            evidences.append(
                Evidence(
                    text=excerpt,
                    page=chunk.page,
                    section=chunk.section,
                    score=10.0 if match_type == "exact" else 7.0,
                    source=chunk.chunk_id,
                    agent="citation_rule",
                    matched_term=target,
                    match_type=match_type,
                    relation="cites" if is_reference else "mentions",
                    validated=excerpt in chunk.text,
                )
            )

        for index, evidence in enumerate(evidences, 1):
            evidence.evidence_id = f"E{index}"
        coverage = {
            **metadata,
            "chunks_scanned": len(chunks),
            "target": target,
            "exact_hits": sum(item.match_type == "exact" for item in evidences),
            "normalized_hits": sum(item.match_type == "normalized" for item in evidences),
            "reference_hits": sum(item.relation == "cites" for item in evidences),
            "reference_pages": sorted(reference_pages),
        }
        coverage["complete_for_absence"] = bool(
            coverage.get("total_pages")
            and coverage.get("pages_with_text") == coverage.get("total_pages")
            and coverage.get("references_detected")
        )
        return CitationScanResult(target=target, evidences=evidences, coverage=coverage)

    def _search_index(
        self, paper_id: str
    ) -> tuple[list[PaperChunk], list[str], list[list[str]], Counter, float]:
        path = self.root / f"{paper_id}.json"
        stamp = path.stat().st_mtime_ns
        with self._cache_lock:
            cached = self._cache.get(paper_id)
            if cached and cached[0] == stamp:
                return cached[1:]
            chunks = self.load(paper_id)
            cleaned = [clean_extracted_text(chunk.text) for chunk in chunks]
            document_tokens = [_tokens(text) for text in cleaned]
            document_frequency = Counter(
                token for tokens in document_tokens for token in set(tokens)
            )
            average_length = sum(map(len, document_tokens)) / max(1, len(document_tokens))
            self._cache[paper_id] = (
                stamp,
                chunks,
                cleaned,
                document_tokens,
                document_frequency,
                average_length,
            )
            return chunks, cleaned, document_tokens, document_frequency, average_length

    def _load_metadata(self, paper_id: str, chunks: list[PaperChunk]) -> dict:
        path = self.root / f"{paper_id}.meta.json"
        try:
            payload = json.loads((self.root / f"{paper_id}.json").read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if isinstance(payload.get("metadata"), dict):
                    return payload["metadata"]
                # Never reuse coverage metadata from an older index version.
                raise ValueError("No coverage metadata")
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            pages = sorted({chunk.page for chunk in chunks})
            reference_sources = _reference_chunk_ids(chunks)
            reference_pages = sorted(
                {chunk.page for chunk in chunks if chunk.chunk_id in reference_sources}
            )
            return {
                "total_pages": max(pages, default=0),
                "pages_with_text": len(pages),
                "text_pages": pages,
                "references_detected": bool(reference_pages),
                "reference_pages": reference_pages,
            }


def _is_reference_chunk(chunk: PaperChunk) -> bool:
    section = (chunk.section or "").casefold()
    if "reference" in section or "bibliograph" in section:
        return True
    return bool(re.search(r"(?im)^\s*(references|bibliography)\s*$", chunk.text))


def _reference_chunk_ids(chunks: list[PaperChunk]) -> set[str]:
    marker_index = next(
        (
            index
            for index, chunk in enumerate(chunks)
            if re.search(r"(?im)^\s*(references|bibliography)\s*$", chunk.text)
        ),
        None,
    )
    if marker_index is not None:
        return {chunk.chunk_id for chunk in chunks[marker_index:]}
    return {chunk.chunk_id for chunk in chunks if _is_reference_chunk(chunk)}


def _match_excerpt(text: str, position: int | None, radius: int = 320) -> str:
    if position is None:
        return text[: radius * 2].strip()
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    return text[start:end].strip()
