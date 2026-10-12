from __future__ import annotations

import math

from citetrace.utils.json import parse_json_object


class InvalidModelOutput(ValueError):
    """A successful HTTP response did not contain usable model output."""


def validate_output(raw: str, kind: str) -> None:
    try:
        obj = parse_json_object(raw)
        if kind == "retrieval":
            queries = obj["queries"]
            valid = isinstance(queries, list) and all(
                isinstance(q, str) and q.strip() for q in queries
            )
        elif kind == "answer":
            claims = obj["claims"]
            valid = (
                isinstance(obj["answer"], str) and bool(obj["answer"].strip())
                and isinstance(obj["sufficient"], bool)
                and isinstance(claims, list)
                and all(
                    isinstance(c, dict)
                    and isinstance(c.get("text"), str) and c["text"].strip()
                    and isinstance(c.get("evidence_ids"), list)
                    and all(isinstance(e, str) and e.strip() for e in c["evidence_ids"])
                    for c in claims
                )
                and (not obj["sufficient"] or bool(claims))
            )
        elif kind == "verification":
            confidence = obj["confidence"]
            valid = (
                obj["verdict"] in ("supported", "partially_supported", "unsupported", "unknown")
                and type(confidence) in (int, float)
                and math.isfinite(confidence) and 0 <= confidence <= 1
                and isinstance(obj["reason"], str) and bool(obj["reason"].strip())
            )
        else:
            raise ValueError(f"Unknown output schema: {kind}")
        if not valid:
            raise ValueError("Invalid fields")
    except (ValueError, TypeError, KeyError) as exc:
        raise InvalidModelOutput(f"Invalid {kind} structured output") from exc
