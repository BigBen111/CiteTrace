from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from citetrace.schemas import Evidence


class BaseRetriever(ABC):
    """Independent evidence-retrieval agent interface."""

    name: str

    @abstractmethod
    def retrieve(self, question: str, paper_id: str) -> Sequence[Evidence]:
        """Return evidence candidates for a factual question."""
        raise NotImplementedError
