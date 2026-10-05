from citetrace.aggregation.conservative import ConservativeAggregator
from citetrace.answer.base import BaseAnswerAgent
from citetrace.pipeline import CiteTracePipeline
from citetrace.retrieval.base import BaseRetriever
from citetrace.schemas import AnswerDraft, Evidence, VerificationResult, Verdict
from citetrace.verification.base import BaseVerifier


class DummyRetriever(BaseRetriever):
    def __init__(self, name, evidence):
        self.name = name
        self.evidence = evidence

    def retrieve(self, question, paper_id):
        return [self.evidence]


class DummyAnswer(BaseAnswerAgent):
    name = "answer"

    def answer(self, question, evidences):
        return AnswerDraft(
            text="The paper uses 1,000 trajectories.",
            claims=["The paper uses 1,000 trajectories."],
        )


class DummyVerifier(BaseVerifier):
    def __init__(self, name, verdict):
        self.name = name
        self.verdict = verdict

    def verify(self, claim, evidences):
        return VerificationResult(
            claim=claim,
            verdict=self.verdict,
            confidence=0.9,
            reason="test",
            verifier=self.name,
        )


def build(verdict_a, verdict_b):
    evidence = Evidence(text="We collect approximately 1,000 trajectories.", page=6)
    return CiteTracePipeline(
        retrievers=[DummyRetriever("r1", evidence), DummyRetriever("r2", evidence)],
        answer_agent=DummyAnswer(),
        verifiers=[DummyVerifier("v1", verdict_a), DummyVerifier("v2", verdict_b)],
        aggregator=ConservativeAggregator(),
    )


def test_agreement_allows_answer():
    result = build(Verdict.SUPPORTED, Verdict.SUPPORTED).run("How much data?", "paper")
    assert result.abstained is False


def test_conflict_causes_abstention():
    result = build(Verdict.SUPPORTED, Verdict.UNSUPPORTED).run("How much data?", "paper")
    assert result.abstained is True
