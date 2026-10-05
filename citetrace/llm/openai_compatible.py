from __future__ import annotations

from openai import OpenAI

from citetrace.llm.base import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, name: str, model: str, api_key: str, base_url: str) -> None:
        self.name = name
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
