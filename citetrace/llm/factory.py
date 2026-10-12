from __future__ import annotations

import os
from collections.abc import Mapping

from citetrace.llm.openai_compatible import OpenAICompatibleLLM


def build_llm(
    provider: str, config: Mapping[str, str] | None = None
) -> OpenAICompatibleLLM:
    p = provider.upper()
    values = config or os.environ
    try:
        api_key = values[f"{p}_API_KEY"].strip()
        base_url = values[f"{p}_BASE_URL"].strip()
        model = values[f"{p}_MODEL"].strip()
    except KeyError as exc:
        raise ValueError(f"Missing configuration for {p}") from exc
    if not api_key or not base_url or not model:
        raise ValueError(f"Incomplete configuration for {p}")
    timeout = float(values.get("CITETRACE_LLM_TIMEOUT_SECONDS", "60"))
    max_retries = int(values.get("CITETRACE_LLM_MAX_RETRIES", "1"))
    if not 5 <= timeout <= 300:
        raise ValueError("CITETRACE_LLM_TIMEOUT_SECONDS must be between 5 and 300")
    if not 0 <= max_retries <= 5:
        raise ValueError("CITETRACE_LLM_MAX_RETRIES must be between 0 and 5")
    return OpenAICompatibleLLM(
        name=provider.lower(),
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
