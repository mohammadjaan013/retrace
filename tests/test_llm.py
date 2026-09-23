from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from retrace.errors import LLMOutputError, LLMRefusalError
from retrace.llm import AnthropicLLM, CallableLLM, as_llm, parse_json_object

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


class FakeMessages:
    """Records every create() call and replays a scripted response."""

    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(*, text='{"x": 1}', stop_reason="end_turn", iterations=(), stop_details=None):
    response = SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=stop_details,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=120, output_tokens=30, iterations=list(iterations)),
    )
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


class TestParseJsonObject:
    def test_plain(self):
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_tolerates_markdown_fence(self):
        assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_rejects_prose(self):
        with pytest.raises(LLMOutputError, match="not valid JSON"):
            parse_json_object('Sure! Here is the JSON: {"a": 1}')

    def test_rejects_non_object(self):
        with pytest.raises(LLMOutputError, match="expected a JSON object"):
            parse_json_object("[1, 2]")


class TestCallableLLM:
    def test_accepts_dict_return(self):
        llm = as_llm(lambda system, prompt, schema: {"x": 2})
        assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"x": 2}

    def test_accepts_string_return(self):
        llm = as_llm(lambda system, prompt, schema: '{"x": 3}')
        assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"x": 3}

    def test_rejects_other_return_types(self):
        llm = CallableLLM(lambda s, p, sc: 42)
        with pytest.raises(LLMOutputError, match="expected dict or str"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)

    def test_counts_calls(self):
        llm = CallableLLM(lambda s, p, sc: {})
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        assert llm.usage.calls == 2

    def test_passes_through_an_existing_llm(self):
        llm = CallableLLM(lambda s, p, sc: {})
        assert as_llm(llm) is llm

    def test_rejects_unusable(self):
        with pytest.raises(TypeError, match="not usable as an LLM"):
            as_llm("nope")


class TestAnthropicLLM:
    def test_request_shape(self):
        client, messages = fake_client()
        llm = AnthropicLLM(client=client)
        assert llm.complete_json(system="SYS", prompt="PROMPT", schema=SCHEMA) == {"x": 1}

        sent = messages.calls[0]
        assert sent["model"] == "claude-opus-5"
        assert sent["system"] == "SYS"
        assert sent["messages"] == [{"role": "user", "content": "PROMPT"}]
        assert sent["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
        assert "effort" not in sent["output_config"]
        # Refusal fallback is on by default.
        assert sent["fallbacks"] == "default"
        assert sent["betas"] == ["server-side-fallback-2026-07-01"]
        # Adaptive thinking is the model default; nothing else may be sent.
        assert "thinking" not in sent
        assert "temperature" not in sent

    def test_effort_is_forwarded(self):
        client, messages = fake_client()
        llm = AnthropicLLM(client=client, effort="low")
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        assert messages.calls[0]["output_config"]["effort"] == "low"

    def test_fallback_can_be_disabled(self):
        client, messages = fake_client()
        AnthropicLLM(client=client, refusal_fallback=False).complete_json(
            system="s", prompt="p", schema=SCHEMA
        )
        assert "fallbacks" not in messages.calls[0]
        assert "betas" not in messages.calls[0]

    def test_skips_thinking_blocks_to_find_text(self):
        client, _ = fake_client(text='{"x": 7}')
        llm = AnthropicLLM(client=client)
        assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"x": 7}

    def test_refusal_is_checked_before_content(self):
        details = SimpleNamespace(category="cyber", explanation="...")
        client, _ = fake_client(stop_reason="refusal", text="", stop_details=details)
        llm = AnthropicLLM(client=client)
        with pytest.raises(LLMRefusalError, match="cyber"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        assert llm.usage.refusals == 1

    def test_truncation_is_an_output_error(self):
        client, _ = fake_client(stop_reason="max_tokens", text='{"x":')
        with pytest.raises(LLMOutputError, match="truncated"):
            AnthropicLLM(client=client).complete_json(system="s", prompt="p", schema=SCHEMA)

    def test_usage_accumulates_and_counts_fallbacks(self):
        client, _ = fake_client(iterations=[SimpleNamespace(type="fallback_message")])
        llm = AnthropicLLM(client=client)
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        assert llm.usage.as_dict() == {
            "calls": 2,
            "input_tokens": 240,
            "output_tokens": 60,
            "refusals": 0,
            "fallbacks": 2,
        }

    def test_name(self):
        client, _ = fake_client()
        llm = AnthropicLLM(client=client, model="claude-sonnet-5")
        assert llm.name == "anthropic:claude-sonnet-5"

    def test_missing_text_block(self):
        client, messages = fake_client()
        messages.response.content = [SimpleNamespace(type="thinking", thinking="")]
        with pytest.raises(LLMOutputError, match="no text block"):
            AnthropicLLM(client=client).complete_json(system="s", prompt="p", schema=SCHEMA)


def test_schema_is_json_serialisable():
    from retrace.synthesis import QUESTION_SCHEMA

    json.dumps(QUESTION_SCHEMA)
    assert QUESTION_SCHEMA["additionalProperties"] is False
    assert set(QUESTION_SCHEMA["required"]) == set(QUESTION_SCHEMA["properties"])
