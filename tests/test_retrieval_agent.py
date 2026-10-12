from citetrace.llm.base import BaseLLM
from citetrace.retrieval.agent import LLMRetrievalAgent
from citetrace.schemas import Evidence
from citetrace.tracing import TraceCollector


class MustNotRunLLM(BaseLLM):
    name = "must-not-run"

    def complete(self, system, user, *, temperature=0.0):
        raise AssertionError("direct retrieval should have skipped the query planner")


class DirectHitStore:
    def search(self, paper_id, query, *, top_k=6):
        return [
            Evidence(text=f"evidence {index}", page=index, source=f"p{index}", score=5 - index)
            for index in range(1, 5)
        ]


def test_direct_retrieval_skips_llm_query_planner():
    trace = TraceCollector()
    retriever = LLMRetrievalAgent(
        "hybrid",
        MustNotRunLLM(),
        DirectHitStore(),
        trace_collector=trace,
    )

    result = retriever.retrieve("ACT baseline", "paper")

    assert len(result) == 4
    preprocessing = trace.snapshot()[0]
    assert preprocessing["planner_called"] is False
    assert preprocessing["model_queries"] == []
