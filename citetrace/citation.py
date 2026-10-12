from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from citetrace.schemas import Evidence


_CITATION_INTENT_RE = re.compile(
    r"引用|参考文献|文献中|\bcit(?:e|ed|es|ing|ation)\b|\breference(?:d|s)?\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"[A-Za-z\u0370-\u03ff][A-Za-z0-9\u0370-\u03ff._+\-/]{1,}")
_QUOTED_RE = re.compile(r"[\"“‘']([^\"”’']{3,160})[\"”’']")
_GENERIC_TERMS = {
    "article",
    "cite",
    "cited",
    "cites",
    "citation",
    "does",
    "paper",
    "reference",
    "referenced",
    "references",
    "this",
}


@dataclass
class CitationScanResult:
    target: str
    evidences: list[Evidence]
    coverage: dict


def extract_citation_target(question: str) -> str | None:
    """Return a named citation target only for explicit citation questions."""
    if not _CITATION_INTENT_RE.search(question):
        return None
    quoted = [value.strip() for value in _QUOTED_RE.findall(question) if value.strip()]
    if quoted:
        return quoted[-1]
    candidates = [
        value
        for value in _ENTITY_RE.findall(question)
        if value.casefold() not in _GENERIC_TERMS
    ]
    return candidates[-1] if candidates else None


def normalized_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in normalized if char.isalnum())


def exact_name_pattern(target: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9._+\-/]){re.escape(target)}(?![A-Za-z0-9._+\-/])",
        re.IGNORECASE,
    )
