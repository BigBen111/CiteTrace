from __future__ import annotations

import os

from citetrace.llm.openai_compatible import OpenAICompatibleLLM


def build_llm(provider: str) -> OpenAICompatibleLLM:
    p = provider.upper()
    api_key = os.environ[f"{p}_API_KEY"]
    base_url = os.environ[f"{p}_BASE_URL"]
    model = os.environ[f"{p}_MODEL"]
    return OpenAICompatibleLLM(
        name=provider.lower(),
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
