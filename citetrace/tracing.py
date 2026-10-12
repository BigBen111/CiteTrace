from __future__ import annotations

import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from citetrace.llm.base import BaseLLM


class TraceCollector:
    """Thread-safe per-run collector for auditable agent and model events."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._lock = Lock()

    def record(self, *, kind: str, agent: str, **details: Any) -> None:
        with self._lock:
            self._entries.append(
                {
                    "sequence": len(self._entries) + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "agent": agent,
                    **details,
                }
            )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(entry) for entry in self._entries]


class TracingLLM(BaseLLM):
    """Record prompts, raw responses, latency, and provider failures."""

    def __init__(self, delegate: BaseLLM, agent: str, collector: TraceCollector) -> None:
        self.delegate = delegate
        self.agent = agent
        self.collector = collector
        self.name = delegate.name
        self.model = getattr(delegate, "model", delegate.name)

    def record_invalid_output(self, reason: str) -> None:
        self.collector.record(
            kind="llm_validation", agent=self.agent, status="error",
            provider=self.name, model=self.model, error=reason,
        )

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        started = time.monotonic()
        common = {
            "provider": self.delegate.name,
            "model": self.model,
            "system_prompt": system,
            "user_prompt": user,
            "temperature": temperature,
        }
        try:
            raw = self.delegate.complete(system, user, temperature=temperature)
        except Exception as exc:
            self.collector.record(
                kind="llm_call",
                agent=self.agent,
                status="error",
                duration_seconds=round(time.monotonic() - started, 3),
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
                **common,
            )
            raise
        self.collector.record(
            kind="llm_call",
            agent=self.agent,
            status="success",
            duration_seconds=round(time.monotonic() - started, 3),
            raw_output=raw,
            **common,
        )
        return raw
