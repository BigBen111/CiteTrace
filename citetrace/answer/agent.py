from __future__ import annotations

from typing import Sequence

from citetrace.answer.base import BaseAnswerAgent
from citetrace.llm.base import BaseLLM
from citetrace.schemas import AnswerDraft, Evidence
from citetrace.utils.json import parse_json_object


_SYSTEM = """You answer factual questions about one research paper.
Use ONLY the supplied evidence. Do not use outside knowledge.
If evidence is insufficient, say so.
Return strict JSON:
{
  "answer": "...",
  "claims": ["one atomic factual claim", "..."]
}
Every claim must be independently checkable against the supplied evidence.
Do not include claims that are merely rhetorical or subjective.
"""


class EvidenceAnswerAgent(BaseAnswerAgent):
    def __init__(self, name: str, llm: BaseLLM) -> None:
        self.name = name
        self.llm = llm

    def answer(self, question: str, evidences: Sequence[Evidence]) -> AnswerDraft:
        if not evidences:
            return AnswerDraft(
                text="Unable to confirm from the currently retrieved paper evidence.",
                claims=[],
            )

        evidence_text = "\n\n".join(
            f"[E{i}] page={e.page} section={e.section or '-'} source={e.source}\n{e.text}"
            for i, e in enumerate(evidences, 1)
        )
        raw = self.llm.complete(
            _SYSTEM,
            f"Question:\n{question}\n\nEvidence:\n{evidence_text}",
        )
        try:
            obj = parse_json_object(raw)
            return AnswerDraft(
                text=str(obj.get("answer", "")).strip(),
                claims=[str(x).strip() for x in obj.get("claims", []) if str(x).strip()],
            )
        except Exception:
            return AnswerDraft(
                text="Unable to produce a structured evidence-grounded answer.",
                claims=[],
            )
