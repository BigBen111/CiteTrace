from __future__ import annotations

from typing import Iterable, Sequence

from citetrace.aggregation.base import BaseAggregator
from citetrace.answer.base import BaseAnswerAgent
from citetrace.retrieval.base import BaseRetriever
from citetrace.schemas import Evidence, FinalResult, VerificationResult
from citetrace.verification.base import BaseVerifier


class CiteTracePipeline:
    """Model-agnostic orchestration for CiteTrace's first architecture."""

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        answer_agent: BaseAnswerAgent,
        verifiers: Sequence[BaseVerifier],
        aggregator: BaseAggregator,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("CiteTrace expects at least two independent retrievers.")
        if len(verifiers) < 2:
            raise ValueError("CiteTrace expects at least two independent verifiers.")
        self.retrievers = retrievers
        self.answer_agent = answer_agent
        self.verifiers = verifiers
        self.aggregator = aggregator

    def run(self, question: str, paper_id: str) -> FinalResult:
        evidence_pool = self._cross_retrieve(question, paper_id)
        draft = self.answer_agent.answer(question, evidence_pool)

        verifications: list[VerificationResult] = []
        for claim in draft.claims:
            for verifier in self.verifiers:
                verifications.append(verifier.verify(claim, evidence_pool))

        return self.aggregator.resolve(draft, evidence_pool, verifications)

    def _cross_retrieve(self, question: str, paper_id: str) -> list[Evidence]:
        pooled: list[Evidence] = []
        for retriever in self.retrievers:
            pooled.extend(retriever.retrieve(question, paper_id))
        return _deduplicate_evidence(pooled)


def _deduplicate_evidence(items: Iterable[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, int | None]] = set()
    result: list[Evidence] = []
    for item in items:
        key = (" ".join(item.text.split()), item.page)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
