from __future__ import annotations

from dataclasses import dataclass

from citetrace.answer.agent import EvidenceAnswerAgent
from citetrace.paper.store import JsonPaperStore
from citetrace.schemas import AnswerDraft


@dataclass
class BaselineResult:
    answer: str
    claims: list[str]


class RagBaseline:
    """Single-path RAG baseline without cross retrieval or verification."""

    def __init__(self, store: JsonPaperStore, answer_agent: EvidenceAnswerAgent, *, top_k: int = 8) -> None:
        self.store = store
        self.answer_agent = answer_agent
        self.top_k = top_k

    def run(self, question: str, paper_id: str) -> BaselineResult:
        evidences = self.store.search(paper_id, question, top_k=self.top_k)
        draft: AnswerDraft = self.answer_agent.answer(question, evidences)
        return BaselineResult(answer=draft.text, claims=draft.claims)
