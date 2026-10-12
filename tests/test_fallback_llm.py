from citetrace.llm.base import BaseLLM
from citetrace.llm.fallback import FallbackLLM
from citetrace.llm.validation import InvalidModelOutput, validate_output
from functools import partial
import pytest


class StubLLM(BaseLLM):
    def __init__(self, name, response=None, error=None):
        self.name = name
        self.response = response
        self.error = error
        self.calls = 0

    def complete(self, system, user, *, temperature=0.0):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def test_fallback_llm_uses_second_provider_after_failure():
    primary = StubLLM("primary", error=RuntimeError("down"))
    fallback = StubLLM("fallback", response='{"answer":"ok"}')
    llm = FallbackLLM("resilient", [primary, fallback])

    result = llm.complete("system", "user")

    assert result == '{"answer":"ok"}'
    assert primary.calls == 1
    assert fallback.calls == 1


def test_fallback_llm_reports_real_fallback_transition():
    primary = StubLLM("primary", error=RuntimeError("down"))
    fallback = StubLLM("fallback", response="ok")
    events = []
    llm = FallbackLLM(
        "resilient", [primary, fallback], progress_stage="answer_fallback"
    )
    llm.set_progress(lambda stage, message: events.append((stage, message)))

    llm.complete("system", "user")

    assert events == [
        ("answer_fallback", "primary 调用失败，正在切换到 fallback")
    ]


@pytest.mark.parametrize("response", ["", "   ", None])
def test_empty_output_uses_backup(response):
    primary = StubLLM("primary", response=response)
    backup = StubLLM("backup", response="usable")
    assert FallbackLLM("test", [primary, backup]).complete("s", "u") == "usable"
    assert (primary.calls, backup.calls) == (1, 1)


@pytest.mark.parametrize("response", ["not JSON", '{"answer":', '{}',
                                      '{"answer":"x","sufficient":"false","claims":[]}'])
def test_malformed_output_uses_backup(response):
    backup_output = '{"answer":"证据不足","sufficient":false,"claims":[]}'
    primary = StubLLM("primary", response=response)
    backup = StubLLM("backup", response=backup_output)
    llm = FallbackLLM("test", [primary, backup], validator=partial(validate_output, kind="answer"))
    assert llm.complete("s", "u") == backup_output
    assert backup.calls == 1


def test_valid_abstention_does_not_trigger_fallback():
    primary = StubLLM("primary", response='{"answer":"证据不足","sufficient":false,"claims":[]}')
    backup = StubLLM("backup", response="must not be used")
    llm = FallbackLLM("test", [primary, backup], validator=partial(validate_output, kind="answer"))
    assert llm.complete("s", "u") == primary.response
    assert backup.calls == 0


def test_all_empty_providers_fail_instead_of_retrying_forever():
    providers = [StubLLM("a", response=""), StubLLM("b", response="")]
    with pytest.raises(InvalidModelOutput):
        FallbackLLM("test", providers).complete("s", "u")
    assert [p.calls for p in providers] == [1, 1]


@pytest.mark.parametrize("raw", [
    '{"verdict":"supported","confidence":NaN,"reason":"x"}',
    '{"verdict":"supported","confidence":true,"reason":"x"}',
    '{"verdict":"invalid","confidence":0.9,"reason":"x"}',
    '{"queries":"not a list"}',
])
def test_invalid_schema_is_rejected(raw):
    with pytest.raises(InvalidModelOutput):
        validate_output(raw, "retrieval" if "queries" in raw else "verification")


def test_rejected_output_is_visible_in_trace():
    from citetrace.tracing import TraceCollector, TracingLLM
    trace = TraceCollector()
    primary = TracingLLM(StubLLM("a", response="bad json"), "answer", trace)
    backup = StubLLM("b", response='{"answer":"证据不足","sufficient":false,"claims":[]}')
    FallbackLLM("test", [primary, backup], validator=partial(validate_output, kind="answer")).complete("s", "u")
    assert any(e["kind"] == "llm_validation" and e["status"] == "error" for e in trace.snapshot())
