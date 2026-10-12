from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from citetrace.aggregation.base import BaseAggregator
from citetrace.schemas import AnswerDraft, Evidence, FinalResult, VerificationResult, Verdict


class ConservativeAggregator(BaseAggregator):
    """First-pass resolver: disagreement is treated as uncertainty, not majority vote."""

    def resolve(
        self,
        draft: AnswerDraft,
        evidences: Sequence[Evidence],
        verifications: Sequence[VerificationResult],
    ) -> FinalResult:
        if draft.insufficient or not draft.claims:
            return FinalResult(
                answer=draft.text
                or "Unable to confirm from the currently retrieved paper evidence.",
                evidences=list(evidences),
                verifications=list(verifications),
                abstained=True,
                note=(
                    "The answer agent marked the retrieved evidence as insufficient."
                    if draft.insufficient
                    else "No independently verifiable factual claim was produced."
                ),
            )

        grouped: dict[str, list[VerificationResult]] = defaultdict(list)
        for item in verifications:
            grouped[item.claim].append(item)

        risky_claims: list[str] = []
        for claim in draft.claims:
            results = grouped.get(claim, [])
            if not results:
                risky_claims.append(claim)
                continue
            verdicts = {r.verdict for r in results}
            if verdicts - {Verdict.SUPPORTED}:
                risky_claims.append(claim)

        if risky_claims:
            return FinalResult(
                answer="Unable to provide a fully supported factual answer from the current evidence.",
                evidences=list(evidences),
                verifications=list(verifications),
                abstained=True,
                note="Verifier disagreement or insufficient support detected for: " + "; ".join(risky_claims),
            )

        return FinalResult(
            answer=draft.text,
            evidences=list(evidences),
            verifications=list(verifications),
            abstained=False,
        )
