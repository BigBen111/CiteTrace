from __future__ import annotations

import re
from typing import Callable, Sequence

from citetrace.answer.base import BaseAnswerAgent
from citetrace.llm.base import BaseLLM
from citetrace.schemas import AnswerDraft, Evidence
from citetrace.utils.json import parse_json_object


_SYSTEM = """You answer factual questions about exactly one research paper.
Use ONLY the numbered evidence supplied below; never use outside knowledge.

Evidence policy:
- Cite supporting evidence IDs inline, for example [E2].
- A bibliography entry proves that the paper cites a work, but not that it discusses,
  uses, compares, or evaluates that work.
- Absence from retrieved evidence does not prove absence from the full paper.
- If evidence is insufficient, state that clearly instead of guessing.
- Answer in the same language as the question and include the useful identifying detail,
  not merely "yes" or "no".

Return strict JSON only:
{
  "answer": "concise answer with [E#] citations",
  "sufficient": true,
  "claims": [{"text": "one atomic factual claim", "evidence_ids": ["E#"]}]
}
Set sufficient=false whenever the evidence cannot directly answer any material
part of the question. Every factual or absence assertion in the answer must have
an equivalent item in claims.
Every claim must be independently checkable against the supplied evidence.
Do not include claims that are merely rhetorical or subjective.
"""


class EvidenceAnswerAgent(BaseAnswerAgent):
    def __init__(self, name: str, llm: BaseLLM) -> None:
        self.name = name
        self.llm = llm

    def set_progress(
        self, progress: Callable[[str, str], None] | None
    ) -> None:
        setter = getattr(self.llm, "set_progress", None)
        if callable(setter):
            setter(progress)

    def answer(self, question: str, evidences: Sequence[Evidence]) -> AnswerDraft:
        if not evidences:
            return AnswerDraft(
                text="Unable to confirm from the currently retrieved paper evidence.",
                claims=[],
                insufficient=True,
            )

        evidence_text = "\n\n".join(
            f"[{e.evidence_id or f'E{i}'}] page={e.page} section={e.section or '-'} "
            f"source={e.source} validated={e.validated}\n{e.text}"
            for i, e in enumerate(evidences, 1)
        )
        raw = self.llm.complete(
            _SYSTEM,
            f"Question:\n{question}\n\nEvidence:\n{evidence_text}",
        )
        try:
            obj = parse_json_object(raw)
            claims = []
            claim_evidence_ids: dict[str, list[str]] = {}
            valid_evidence_ids = {
                evidence.evidence_id for evidence in evidences if evidence.evidence_id
            }
            for item in obj.get("claims", []):
                text = item.get("text", "") if isinstance(item, dict) else item
                if str(text).strip():
                    claim = str(text).strip()
                    claims.append(claim)
                    ids = item.get("evidence_ids", []) if isinstance(item, dict) else []
                    claim_evidence_ids[claim] = [
                        str(evidence_id).strip()
                        for evidence_id in ids
                        if str(evidence_id).strip() in valid_evidence_ids
                    ]
            answer = str(obj.get("answer", "")).strip()
            links_complete = bool(claims) and all(
                claim_evidence_ids.get(claim) for claim in claims
            )
            return AnswerDraft(
                text=answer,
                claims=claims,
                insufficient=(
                    not bool(obj.get("sufficient", bool(claims)))
                    or _signals_insufficient_evidence(answer)
                    or not links_complete
                ),
                claim_evidence_ids=claim_evidence_ids,
            )
        except Exception:
            return AnswerDraft(
                text="Unable to produce a structured evidence-grounded answer.",
                claims=[],
                insufficient=True,
            )


def _signals_insufficient_evidence(answer: str) -> bool:
    patterns = (
        r"证据不足",
        r"无法(?:从|根据|据此)?.{0,12}(?:确认|判断|得出)",
        r"没有(?:足够|相关)?.{0,10}(?:证据|信息)(?:表明|证明|显示)",
        r"insufficient evidence",
        r"unable to (?:confirm|determine)",
        r"no (?:relevant |sufficient )?(?:evidence|information) (?:shows|supports|indicates)",
    )
    return any(re.search(pattern, answer, re.I) for pattern in patterns)
