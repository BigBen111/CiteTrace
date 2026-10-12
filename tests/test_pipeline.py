from citetrace.aggregation.conservative import ConservativeAggregator
from citetrace.answer.base import BaseAnswerAgent
from citetrace.pipeline import CiteTracePipeline, _prioritize_for_question
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


class EmptyAnswer(BaseAnswerAgent):
    name = "empty-answer"

    def answer(self, question, evidences):
        return AnswerDraft(text="The current evidence is insufficient.", claims=[])


class InsufficientAnswer(BaseAnswerAgent):
    name = "insufficient-answer"

    def answer(self, question, evidences):
        return AnswerDraft(
            text="The evidence does not establish the requested comparison.",
            claims=["The paper cites the named work."],
            insufficient=True,
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


class FailingRetriever(BaseRetriever):
    name = "failing-retriever"

    def retrieve(self, question, paper_id):
        raise RuntimeError("retriever unavailable")


class FailingVerifier(BaseVerifier):
    name = "failing-verifier"

    def verify(self, claim, evidences):
        raise RuntimeError("verifier unavailable")


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


def test_partial_support_causes_abstention():
    result = build(Verdict.PARTIALLY_SUPPORTED, Verdict.PARTIALLY_SUPPORTED).run(
        "How much data?", "paper"
    )
    assert result.abstained is True


def test_citation_question_keeps_exact_entity_evidence():
    exact = Evidence(text="[42] OpenVLA: An open-source VLA model.", page=14, score=8)
    broad = Evidence(text="We compare several vision-language-action systems.", page=8, score=20)
    similarly_named = Evidence(text="OpenVLA-mini is a smaller model.", page=12, score=10)

    result = _prioritize_for_question(
        "文章里面有引用 OpenVLA 吗？", [broad, similarly_named, exact]
    )

    assert result == [exact]


def test_answer_without_verifiable_claim_abstains():
    pipeline = build(Verdict.SUPPORTED, Verdict.SUPPORTED)
    pipeline.answer_agent = EmptyAnswer()

    result = pipeline.run("Unknown question", "paper")

    assert result.abstained is True
    assert result.note == "No independently verifiable factual claim was produced."


def test_explicitly_insufficient_answer_abstains_even_with_supported_claims():
    pipeline = build(Verdict.SUPPORTED, Verdict.SUPPORTED)
    pipeline.answer_agent = InsufficientAnswer()

    result = pipeline.run("Was it used as a baseline?", "paper")

    assert result.abstained is True
    assert result.note == "The answer agent marked the retrieved evidence as insufficient."


def test_pipeline_reports_real_stage_progress():
    pipeline = build(Verdict.SUPPORTED, Verdict.SUPPORTED)
    events = []

    pipeline.run("How much data?", "paper", progress=lambda stage, message: events.append(stage))

    assert events == [
        "question_routing",
        "direct_retrieval",
        "evidence_validation",
        "answer_generation",
        "claim_binding",
        "source_validation",
        "semantic_verification",
        "semantic_verification",
        "semantic_verification",
        "aggregation",
        "completed",
    ]


def test_provider_failures_degrade_to_conservative_result():
    evidence = Evidence(text="We collect approximately 1,000 trajectories.", page=6)
    pipeline = CiteTracePipeline(
        retrievers=[DummyRetriever("healthy", evidence), FailingRetriever()],
        answer_agent=DummyAnswer(),
        verifiers=[
            DummyVerifier("healthy-verifier", Verdict.SUPPORTED),
            FailingVerifier(),
        ],
        aggregator=ConservativeAggregator(),
    )

    result = pipeline.run("How much data?", "paper")

    assert result.abstained is True
    assert len(result.evidences) == 1
    assert {item.verdict for item in result.verifications} == {
        Verdict.SUPPORTED,
        Verdict.UNKNOWN,
    }
