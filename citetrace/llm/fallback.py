from __future__ import annotations

import logging
from typing import Callable, Sequence

from citetrace.llm.base import BaseLLM
from citetrace.llm.validation import InvalidModelOutput


logger = logging.getLogger(__name__)


class FallbackLLM(BaseLLM):
    """Try each provider once, including when its output is unusable."""

    def __init__(
        self,
        name: str,
        providers: Sequence[BaseLLM],
        *,
        progress_stage: str = "fallback",
        validator: Callable[[str], None] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("FallbackLLM requires at least one provider")
        self.name = name
        self.providers = list(providers)
        self.progress_stage = progress_stage
        self.validator = validator
        self._progress: Callable[[str, str], None] | None = None

    def set_progress(
        self, progress: Callable[[str, str], None] | None
    ) -> None:
        self._progress = progress

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        last_error: Exception | None = None
        for index, provider in enumerate(self.providers):
            try:
                raw = provider.complete(system, user, temperature=temperature)
                try:
                    if not isinstance(raw, str) or not raw.strip():
                        raise InvalidModelOutput("Empty model output")
                    if self.validator is not None:
                        self.validator(raw)
                except InvalidModelOutput as exc:
                    recorder = getattr(provider, "record_invalid_output", None)
                    if callable(recorder):
                        recorder(str(exc))
                    raise
                return raw
            except Exception as exc:
                last_error = exc
                if self._progress is not None and index + 1 < len(self.providers):
                    next_provider = self.providers[index + 1]
                    self._progress(
                        self.progress_stage,
                        f"{provider.name} 调用失败，正在切换到 {next_provider.name}",
                    )
                logger.warning(
                    "LLM %s failed (%d/%d), trying fallback: %s",
                    provider.name,
                    index + 1,
                    len(self.providers),
                    exc,
                )
        assert last_error is not None
        raise last_error
