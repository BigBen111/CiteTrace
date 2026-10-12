from citetrace.utils.json import parse_json_object


def test_parse_json_object_ignores_thinking_wrapper():
    result = parse_json_object(
        '<think>{"discard": true}</think>\n{"answer": "yes", "claims": []}'
    )

    assert result["answer"] == "yes"
