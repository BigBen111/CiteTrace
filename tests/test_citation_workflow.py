from citetrace.aggregation.conservative import ConservativeAggregator
from citetrace.answer.base import BaseAnswerAgent
from citetrace.citation import extract_citation_target
from citetrace.paper.parser import PaperChunk
from citetrace.paper.store import JsonPaperStore
from citetrace.pipeline import CiteTracePipeline
from citetrace.retrieval.base import BaseRetriever
from citetrace.verification.base import BaseVerifier


class MustNotRunRetriever(BaseRetriever):
    name = "must-not-run-retriever"

    def retrieve(self, question, paper_id):
        raise AssertionError("citation workflow must bypass LLM retrieval")


class MustNotRunAnswer(BaseAnswerAgent):
    name = "must-not-run-answer"

    def answer(self, question, evidences):
        raise AssertionError("citation workflow must bypass answer LLM")


class MustNotRunVerifier(BaseVerifier):
    name = "must-not-run-verifier"

    def verify(self, claim, evidences):
        raise AssertionError("citation workflow must bypass verifier LLM")


def _metadata():
    return {
        "total_pages": 2,
        "pages_with_text": 2,
        "text_pages": [1, 2],
        "references_detected": True,
        "reference_pages": [2],
    }


def test_citation_absence_uses_complete_scan_without_agents(tmp_path):
    store = JsonPaperStore(tmp_path)
    store.save(
        "paper",
        [
            PaperChunk("paper:p1:c0", "paper", 1, "Main text about robot learning."),
            PaperChunk("paper:p2:c0", "paper", 2, "REFERENCES\n[1] Other Work."),
        ],
        _metadata(),
    )
    pipeline = CiteTracePipeline(
        retrievers=[MustNotRunRetriever(), MustNotRunRetriever()],
        answer_agent=MustNotRunAnswer(),
        verifiers=[MustNotRunVerifier(), MustNotRunVerifier()],
        aggregator=ConservativeAggregator(),
        citation_scanner=store.scan_citation,
    )

    result = pipeline.run("本文有引用 ActProbe 吗？", "paper")

    assert result.workflow == "citation_rule"
    assert result.abstained is False
    assert "未发现" in result.answer
    assert result.coverage["complete_for_absence"] is True


def test_citation_hit_returns_validated_evidence_table(tmp_path):
    store = JsonPaperStore(tmp_path)
    store.save(
        "paper",
        [
            PaperChunk("paper:p1:c0", "paper", 1, "Main text."),
            PaperChunk(
                "paper:p2:c0",
                "paper",
                2,
                "REFERENCES\n[1] Smith et al. ActProbe: Runtime Detection.",
            ),
        ],
        _metadata(),
    )

    scan = store.scan_citation("paper", "ActProbe")

    assert scan.coverage["reference_hits"] == 1
    assert scan.evidences[0].evidence_id == "E1"
    assert scan.evidences[0].relation == "cites"
    assert scan.evidences[0].validated is True


def test_extract_citation_target_ignores_generic_question_words():
    assert extract_citation_target("Does this paper cite ActProbe?") == "ActProbe"
    assert extract_citation_target("本文有引用actprobe吗") == "actprobe"
    assert extract_citation_target("本文有引用π0吗") == "π0"
    assert extract_citation_target("How many trajectories were used?") is None
