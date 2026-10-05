from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from citetrace.schemas import Evidence, VerificationResult


class BaseVerifier(ABC):
    """Independently judge whether evidence supports one atomic claim."""

    name: str

    @abstractmethod
    def verify(self, claim: str, evidences: Sequence[Evidence]) -> VerificationResult:
        raise NotImplementedError
