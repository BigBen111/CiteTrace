from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from citetrace.paper.parser import PaperChunk
from citetrace.schemas import Evidence


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class JsonPaperStore:
    """Small transparent lexical store suitable for course-scale papers.

    The interface can later be replaced by FAISS/BM25/vector retrieval without
    changing retriever or pipeline contracts.
    """

    def __init__(self, root: str | Path = "data/index") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, paper_id: str, chunks: list[PaperChunk]) -> None:
        path = self.root / f"{paper_id}.json"
        path.write_text(
            json.dumps([c.__dict__ for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, paper_id: str) -> list[PaperChunk]:
        rows = json.loads((self.root / f"{paper_id}.json").read_text(encoding="utf-8"))
        return [PaperChunk(**row) for row in rows]

    def search(self, paper_id: str, query: str, *, top_k: int = 6) -> list[Evidence]:
        chunks = self.load(paper_id)
        q = Counter(_tokens(query))
        if not q:
            return []

        scored: list[tuple[float, PaperChunk]] = []
        for chunk in chunks:
            d = Counter(_tokens(chunk.text))
            dot = sum(q[t] * d[t] for t in q)
            qn = math.sqrt(sum(v * v for v in q.values()))
            dn = math.sqrt(sum(v * v for v in d.values()))
            score = dot / (qn * dn) if qn and dn else 0.0
            if score > 0:
                scored.append((score, chunk))

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
