from citetrace.llm.base import BaseLLM
from citetrace.tracing import TraceCollector, TracingLLM


class StubLLM(BaseLLM):
    name = "stub-provider"
    model = "stub-model"

    def complete(self, system, user, *, temperature=0.0):
        return '{"queries":["ActProbe"]}'


def test_tracing_llm_records_prompts_output_and_latency():
    collector = TraceCollector()
    llm = TracingLLM(StubLLM(), "retriever", collector)

    output = llm.complete("system prompt", "user prompt")

    assert output == '{"queries":["ActProbe"]}'
    trace = collector.snapshot()
    assert len(trace) == 1
    assert trace[0]["agent"] == "retriever"
    assert trace[0]["model"] == "stub-model"
    assert trace[0]["raw_output"] == output
    assert trace[0]["duration_seconds"] >= 0
