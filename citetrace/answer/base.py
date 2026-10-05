from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from citetrace.schemas import AnswerDraft, Evidence


class BaseAnswerAgent(ABC):
    """Generate an evidence-grounded draft answer."""

    name: str

    @abstractmethod
    def answer(self, question: str, evidences: Sequence[Evidence]) -> AnswerDraft:
        raise NotImplementedError
