"""OpenAICompatibleLLM against a scripted fake server. No network, no key."""

from __future__ import annotations

import json
import os

import pytest

from retrace.errors import LLMOutputError, LLMRefusalError, RetraceError
from retrace.llm import (
    PROVIDERS,
    AnthropicLLM,
    OpenAICompatibleLLM,
    build_llm,
    load_env_file,
)

SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
    "additionalProperties": False,
}


def ok(content='{"x": 1}', finish="stop", prompt_tokens=100, completion_tokens=20):
    body = {
        "choices": [{"message": {"role": "assistant", "content": content,
                                 "reasoning": "thinking that must be ignored"},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return 200, {}, json.dumps(body).encode()


def err(status, message="boom", headers=None):
    return status, headers or {}, json.dumps({"error": {"message": message}}).encode()


class FakeServer:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": headers, "body": json.loads(body)})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_real")


def groq(server, **kw):
    slept = []
    llm = OpenAICompatibleLLM.preset("groq", transport=server, sleep=slept.append, **kw)
    return llm, slept


class TestRequest:
    def test_shape(self, key):
        server = FakeServer(ok())
        llm, _ = groq(server)
        assert llm.complete_json(system="SYS", prompt="P", schema=SCHEMA) == {"x": 1}

        req = server.requests[0]
        assert req["url"] == "https://api.groq.com/openai/v1/chat/completions"
        assert req["headers"]["Authorization"] == "Bearer gsk_test_not_real"
        body = req["body"]
        assert body["model"] == PROVIDERS["groq"]["model"]
        assert body["messages"] == [{"role": "system", "content": "SYS"},
                                    {"role": "user", "content": "P"}]
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "retrace_output", "strict": True, "schema": SCHEMA},
        }
        # Groq's default of 1024 truncates reasoning models.
        assert body["max_completion_tokens"] >= 4096
        assert "reasoning_effort" not in body

    def test_reasoning_effort_and_model_override(self, key):
        server = FakeServer(ok())
        llm, _ = groq(server, model="openai/gpt-oss-20b", reasoning_effort="low")
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        assert server.requests[0]["body"]["model"] == "openai/gpt-oss-20b"
        assert server.requests[0]["body"]["reasoning_effort"] == "low"

    def test_json_object_mode_puts_the_schema_in_the_prompt(self, key):
        server = FakeServer(ok())
        llm, _ = groq(server, strict_schema=False)
        llm.complete_json(system="SYS", prompt="p", schema=SCHEMA)
        body = server.requests[0]["body"]
        assert body["response_format"] == {"type": "json_object"}
        assert '"required": ["x"]' in body["messages"][0]["content"]

    def test_reasoning_field_is_ignored(self, key):
        llm, _ = groq(FakeServer(ok(content='{"x": 5}')))
        assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"x": 5}

    def test_usage(self, key):
        llm, _ = groq(FakeServer(ok(), ok()))
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        assert (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens) == (2, 200, 40)


class TestKeyHandling:
    def test_missing_key_is_a_clear_setup_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        llm, _ = groq(FakeServer(ok()))
        with pytest.raises(RetraceError, match="GROQ_API_KEY is not set"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)

    def test_key_never_appears_in_repr_or_name(self, key):
        llm, _ = groq(FakeServer())
        assert "gsk_" not in repr(llm) and "gsk_" not in llm.name
        assert llm.name == "groq:openai/gpt-oss-120b"

    def test_auth_failure_stops_the_run(self, key):
        llm, _ = groq(FakeServer(err(401, "Invalid API Key")))
        with pytest.raises(RetraceError, match=r"authentication failed \(401\).*GROQ_API_KEY"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)


class TestRateLimits:
    def test_waits_for_retry_after_then_succeeds(self, key):
        server = FakeServer(err(429, "slow down", {"retry-after": "7"}), ok())
        llm, slept = groq(server)
        assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"x": 1}
        assert slept == [7.0]
        assert (llm.usage.rate_limit_waits, llm.usage.waited_s) == (1, 7.0)

    def test_backs_off_without_retry_after(self, key):
        llm, slept = groq(FakeServer(err(429), err(429), ok()))
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        assert slept == [2.0, 4.0]

    def test_a_daily_quota_stops_instead_of_sleeping_for_hours(self, key):
        llm, slept = groq(FakeServer(err(429, "TPD", {"retry-after": "5400"})))
        with pytest.raises(RetraceError, match="daily quota"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)
        assert slept == []

    def test_gives_up_after_max_retries(self, key):
        llm, _ = groq(FakeServer(*[err(429, headers={"retry-after": "1"})] * 3), max_retries=2)
        with pytest.raises(RetraceError, match="still rate limited"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)


class TestFailures:
    def test_server_errors_are_retried(self, key):
        llm, slept = groq(FakeServer(err(503), ok()))
        assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"x": 1}
        assert slept == [2.0]

    def test_persistent_server_error(self, key):
        llm, _ = groq(FakeServer(err(500), err(500)), max_retries=1)
        with pytest.raises(RetraceError, match="server error 500"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)

    def test_network_errors_are_retried(self, key):
        llm, slept = groq(FakeServer(ConnectionResetError("reset"), ok()))
        assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"x": 1}
        assert slept == [2.0]

    def test_truncation_rejects_the_passage(self, key):
        llm, _ = groq(FakeServer(ok(content='{"x":', finish="length")))
        with pytest.raises(LLMOutputError, match="truncated"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)

    def test_schema_mismatch_rejects_the_passage(self, key):
        llm, _ = groq(FakeServer(err(400, "Generated JSON does not match the expected schema. "
                                          "Please adjust your prompt.")))
        with pytest.raises(LLMOutputError, match="schema validation"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)

    def test_bad_model_stops_the_run(self, key):
        llm, _ = groq(FakeServer(err(404, "The model `nope` does not exist")))
        with pytest.raises(RetraceError, match="HTTP 404: The model `nope`"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)

    def test_content_filter_is_a_refusal(self, key):
        llm, _ = groq(FakeServer(ok(content="", finish="content_filter")))
        with pytest.raises(LLMRefusalError):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)

    def test_empty_content(self, key):
        llm, _ = groq(FakeServer(ok(content="")))
        with pytest.raises(LLMOutputError, match="no content"):
            llm.complete_json(system="s", prompt="p", schema=SCHEMA)


class TestBuildLLM:
    def test_groq_preset(self):
        llm = build_llm("groq", effort="low")
        assert isinstance(llm, OpenAICompatibleLLM)
        assert llm.base_url == "https://api.groq.com/openai/v1"
        assert llm.api_key_env == "GROQ_API_KEY"
        assert llm.reasoning_effort == "low"

    def test_rejects_efforts_groq_does_not_have(self):
        with pytest.raises(ValueError, match="xhigh is not supported by groq"):
            build_llm("groq", effort="xhigh")

    def test_openai_compatible_needs_url_and_model(self):
        with pytest.raises(ValueError, match="--base-url and --model"):
            build_llm("openai-compatible")
        llm = build_llm("openai-compatible", base_url="http://localhost:11434/v1",
                        model="llama3", api_key_env="OLLAMA_KEY")
        assert (llm.base_url, llm.model, llm.api_key_env) == (
            "http://localhost:11434/v1", "llama3", "OLLAMA_KEY")

    def test_anthropic(self):
        llm = build_llm("anthropic", model="claude-sonnet-5", effort="low")
        assert isinstance(llm, AnthropicLLM) and llm.model == "claude-sonnet-5"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="unknown provider"):
            build_llm("nope")


class TestEnvFile:
    def test_loads_without_overriding(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RT_A", raising=False)
        monkeypatch.setenv("RT_B", "already-set")
        monkeypatch.delenv("RT_C", raising=False)
        path = tmp_path / ".env"
        path.write_text(
            "# comment\n\nRT_A=plain\nexport RT_C=\"quoted value\"\nRT_B=ignored\nnot a pair\n",
            encoding="utf-8",
        )
        loaded = load_env_file(str(path))
        assert sorted(loaded) == ["RT_A", "RT_C"]
        assert os.environ["RT_A"] == "plain"
        assert os.environ["RT_C"] == "quoted value"
        assert os.environ["RT_B"] == "already-set"

    def test_missing_file(self, tmp_path):
        assert load_env_file(str(tmp_path / "nope.env")) == []


def test_synth_saves_partial_results_when_the_provider_stops_it(tmp_path, monkeypatch, capsys):
    """A daily quota mid-run must not discard candidates that were already produced."""
    from retrace import read_fixture
    from retrace.cli import main

    passage = ("Radiator area is almost always sized by the worst-case hot condition rather "
               "than the average, because a design that survives the hottest attitude will "
               "generally tolerate the cold case with modest heater power. ")
    (tmp_path / "docs.json").write_text(json.dumps({"d1": passage * 6}), encoding="utf-8")
    good = {"answerable": True,
            "question": "What sets the size of a spacecraft's heat-dumping panels?",
            "answer": "The worst-case hot condition.",
            "evidence": "sized by the worst-case hot condition rather than the average"}
    server = FakeServer(ok(content=json.dumps(good)),
                        err(429, "tokens per day", {"retry-after": "9000"}))
    monkeypatch.setattr("retrace.llm._urllib_transport", server)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_real")
    monkeypatch.chdir(tmp_path)

    code = main(["synth", "--docs", "docs.json", "--out", "cand.jsonl", "--n", "3",
                 "--chunk-size", "300", "--provider", "groq"])
    assert code == 2
    assert len(read_fixture(tmp_path / "cand.jsonl")) == 1
    out = capsys.readouterr().out
    assert "stopped:" in out and "daily quota" in out
    assert "gsk_test_not_real" not in out
