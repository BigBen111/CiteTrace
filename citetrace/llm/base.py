from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    name: str

    @abstractmethod
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        raise NotImplementedError
