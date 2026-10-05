from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Expected a JSON object")
        return obj
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        raise
