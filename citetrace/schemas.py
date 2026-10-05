from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass
class Evidence:
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None
    agent: Optional[str] = None


@dataclass
class AnswerDraft:
    text: str
    claims: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    claim: str
    verdict: Verdict
    confidence: float
    reason: str
    verifier: str


@dataclass
class FinalResult:
    answer: str
    evidences: List[Evidence]
    verifications: List[VerificationResult]
    abstained: bool = False
    note: Optional[str] = None
