from __future__ import annotations

from openai import OpenAI

from citetrace.llm.base import BaseLLM
from citetrace.llm.validation import InvalidModelOutput


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        self.name = name
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if not response.choices:
            raise InvalidModelOutput("Model returned no choices")
        choice = response.choices[0]
        if choice.finish_reason in {"length", "content_filter"}:
            raise InvalidModelOutput(f"Model output incomplete: {choice.finish_reason}")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise InvalidModelOutput("Empty model output")
        return content
