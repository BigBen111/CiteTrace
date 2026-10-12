from __future__ import annotations

from typing import Callable, Sequence

from citetrace.llm.base import BaseLLM
from citetrace.schemas import Evidence, VerificationResult, Verdict
from citetrace.utils.json import parse_json_object
from citetrace.verification.base import BaseVerifier


_SYSTEM = """You are a strict, independent claim-evidence verifier.
Judge ONLY whether the supplied evidence supports the atomic claim.
Do not use outside knowledge and do not infer that missing evidence proves absence.

Important distinctions:
- A bibliography entry supports "the paper cites X" only.
- It does not support stronger claims that the paper uses, discusses, compares, or
  evaluates X unless other evidence explicitly says so.
- Verify every material detail, including numbers, names, dates, and relationship verbs.
- Prefer unknown over unsupported when relevant evidence is simply missing.

Verdicts:
- supported: evidence directly supports the claim.
- partially_supported: evidence supports only a weaker or narrower version.
- unsupported: evidence conflicts with the claim or fails to support a material assertion.
- unknown: there is not enough relevant evidence to decide.

Return strict JSON:
{"verdict":"supported|partially_supported|unsupported|unknown",
 "confidence":0.0,
 "reason":"brief reason naming the decisive [E#] evidence or the missing support"}
"""


class LLMVerifier(BaseVerifier):
    def __init__(self, name: str, llm: BaseLLM) -> None:
        self.name = name
        self.llm = llm

    def set_progress(
        self, progress: Callable[[str, str], None] | None
    ) -> None:
        setter = getattr(self.llm, "set_progress", None)
        if callable(setter):
            setter(progress)

    def verify(self, claim: str, evidences: Sequence[Evidence]) -> VerificationResult:
        evidence_text = "\n\n".join(
            f"[{e.evidence_id or f'E{i}'}] page={e.page} source={e.source} "
            f"validated={e.validated}\n{e.text}"
            for i, e in enumerate(evidences, 1)
        )
        raw = self.llm.complete(
            _SYSTEM,
            f"Claim:\n{claim}\n\nEvidence:\n{evidence_text}",
        )
        try:
            obj = parse_json_object(raw)
            verdict = Verdict(str(obj["verdict"]).lower())
            confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
            reason = str(obj.get("reason", "")).strip()
        except Exception:
            verdict = Verdict.UNKNOWN
            confidence = 0.0
            reason = "Verifier returned invalid structured output."

        return VerificationResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            verifier=self.name,
            evidence_ids=[
                evidence.evidence_id for evidence in evidences if evidence.evidence_id
            ],
        )
