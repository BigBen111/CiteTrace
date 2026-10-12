from citetrace.answer.agent import EvidenceAnswerAgent, _signals_insufficient_evidence
from citetrace.llm.base import BaseLLM
from citetrace.schemas import Evidence


class StubLLM(BaseLLM):
    name = "stub"

    def __init__(self, response):
        self.response = response

    def complete(self, system, user, *, temperature=0.0):
        return self.response


def test_detects_chinese_insufficient_evidence_language():
    assert _signals_insufficient_evidence("没有信息表明该模型被用作对比基线。")


def test_direct_supported_answer_is_not_marked_insufficient():
    assert not _signals_insufficient_evidence("文章在参考文献 [42] 中引用了 OpenVLA [E1]。")


def test_answer_agent_preserves_claim_evidence_links():
    llm = StubLLM(
        '{"answer":"The paper uses ACT [E1].","sufficient":true,'
        '"claims":[{"text":"The paper uses ACT.","evidence_ids":["E1"]}]}'
    )
    agent = EvidenceAnswerAgent("answer", llm)
    evidence = Evidence(
        text="We instantiate the method with ACT.",
        page=3,
        source="paper:p3:c0",
        evidence_id="E1",
        validated=True,
    )

    draft = agent.answer("Which policy is used?", [evidence])

    assert draft.insufficient is False
    assert draft.claim_evidence_ids == {"The paper uses ACT.": ["E1"]}
