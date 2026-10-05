from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from citetrace.schemas import AnswerDraft, FinalResult, Evidence, VerificationResult


class BaseAggregator(ABC):
    """Resolve verifier agreement/conflict into the final system output."""

    @abstractmethod
    def resolve(
        self,
        draft: AnswerDraft,
        evidences: Sequence[Evidence],
        verifications: Sequence[VerificationResult],
    ) -> FinalResult:
        raise NotImplementedError
