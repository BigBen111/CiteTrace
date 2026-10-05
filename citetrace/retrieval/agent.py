from __future__ import annotations

from typing import Sequence

from citetrace.llm.base import BaseLLM
from citetrace.paper.store import JsonPaperStore
from citetrace.retrieval.base import BaseRetriever
from citetrace.schemas import Evidence
from citetrace.utils.json import parse_json_object


_SYSTEM = """You are a retrieval planner for research-paper factual QA.
Generate several short search queries that independently target the evidence
needed to answer the user's question. Expand terminology when useful, but do
not answer the question. Return strict JSON: {"queries": ["...", "..."]}.
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
    ) -> None:
        self.name = name
        self.llm = llm
        self.store = store
        self.queries_per_agent = queries_per_agent
        self.top_k_per_query = top_k_per_query

    def retrieve(self, question: str, paper_id: str) -> Sequence[Evidence]:
        raw = self.llm.complete(
            _SYSTEM,
            f"Question: {question}\nGenerate at most {self.queries_per_agent} retrieval queries.",
        )
        queries = _parse_queries(raw) or [question]
        results: list[Evidence] = []
        for query in queries[: self.queries_per_agent]:
            for evidence in self.store.search(paper_id, query, top_k=self.top_k_per_query):
                evidence.agent = self.name
                results.append(evidence)
        return results


def _parse_queries(raw: str) -> list[str]:
    try:
        obj = parse_json_object(raw)
        return [str(x).strip() for x in obj.get("queries", []) if str(x).strip()]
    except Exception:
        return []
