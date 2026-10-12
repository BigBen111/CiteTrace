from __future__ import annotations

import re
from typing import Callable, Sequence

from citetrace.llm.base import BaseLLM
from citetrace.paper.store import JsonPaperStore
from citetrace.retrieval.base import BaseRetriever
from citetrace.schemas import Evidence
from citetrace.tracing import TraceCollector
from citetrace.utils.json import parse_json_object


_SYSTEM = """You plan lexical searches over one research paper.
Generate short, high-precision queries that target direct textual evidence.

Rules:
- Preserve every proper noun, acronym, model name, number, and quoted phrase exactly.
- Add useful English variants when the question is not English.
- For citation questions, search the title/name itself plus bibliography terms.
- Do not answer the question and do not add facts not present in the question.
- Return strict JSON only: {"queries": ["...", "..."]}.
"""


class LLMRetrievalAgent(BaseRetriever):
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        store: JsonPaperStore,
        *,
        queries_per_agent: int = 4,
        top_k_per_query: int = 4,
        max_results: int = 8,
        trace_collector: TraceCollector | None = None,
        min_direct_results: int = 4,
    ) -> None:
        self.name = name
        self.llm = llm
        self.store = store
        self.queries_per_agent = queries_per_agent
        self.top_k_per_query = top_k_per_query
        self.max_results = max_results
        self.trace_collector = trace_collector
        self.min_direct_results = max(1, min_direct_results)
        self._progress: Callable[[str, str], None] | None = None

    def set_progress(
        self, progress: Callable[[str, str], None] | None
    ) -> None:
        self._progress = progress
        setter = getattr(self.llm, "set_progress", None)
        if callable(setter):
            setter(progress)

    def retrieve(self, question: str, paper_id: str) -> Sequence[Evidence]:
        entities = re.findall(r"[A-Za-z][A-Za-z0-9._+\-/]{2,}", question)
        by_source: dict[str, Evidence] = {}
        query_results: list[dict] = []

        def search_query(query: str) -> None:
            hits = self.store.search(
                paper_id, query, top_k=self.top_k_per_query
            )
            query_results.append(
                {
                    "query": query,
                    "hit_count": len(hits),
                    "hits": [
                        {"source": item.source, "page": item.page, "score": item.score}
                        for item in hits
                    ],
                }
            )
            for evidence in hits:
                evidence.agent = self.name
                key = evidence.source or f"{evidence.page}:{evidence.text}"
                current = by_source.get(key)
                if current is None or (evidence.score or 0.0) > (current.score or 0.0):
                    by_source[key] = evidence

        direct_queries = _unique([question, *entities])
        for query in direct_queries:
            search_query(query)

        planned: list[str] = []
        planner_called = len(by_source) < self.min_direct_results
        if planner_called:
            if self._progress is not None:
                self._progress(
                    "query_planner",
                    f"直接检索仅得到 {len(by_source)} 条候选，正在生成补充查询",
                )
            try:
                raw = self.llm.complete(
                    _SYSTEM,
                    f"Question: {question}\nGenerate at most {self.queries_per_agent} retrieval queries.",
                )
                planned = _parse_queries(raw)
            except Exception:
                if not by_source:
                    raise
            already_searched = {query.casefold() for query in direct_queries}
            for query in _unique(planned)[: self.queries_per_agent]:
                if query.casefold() not in already_searched:
                    search_query(query)

        final_queries = [item["query"] for item in query_results]
        result = sorted(
            by_source.values(), key=lambda item: item.score or 0.0, reverse=True
        )[: self.max_results]
        if self.trace_collector is not None:
            self.trace_collector.record(
                kind="preprocessing",
                agent=self.name,
                stage="retrieval_plan",
                entities=entities,
                model_queries=planned,
                final_queries=final_queries,
                planner_called=planner_called,
                per_query=query_results,
                output_count=len(result),
            )
        return result


def _parse_queries(raw: str) -> list[str]:
    try:
        obj = parse_json_object(raw)
        return [str(x).strip() for x in obj.get("queries", []) if str(x).strip()]
    except Exception:
        return []


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = " ".join(item.split()).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
