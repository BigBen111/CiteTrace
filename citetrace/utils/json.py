from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = _THINK_RE.sub("", raw).strip()
    text = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Expected a JSON object")
        return obj
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise
