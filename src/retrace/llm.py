"""The language-model seam.

Retrace needs a model for exactly two jobs: writing candidate questions (Week 2)
and, optionally, judging answers (Week 3). Neither justifies a hard dependency,
so the core defines one small protocol and every provider lives behind it.

* `LLM` -- the protocol. One method: structured JSON in, dict out.
* `as_llm()` -- wraps any plain callable, so a team already calling Groq,
  OpenAI, Gemini or a local model through their own client plugs in with a
  three-line function instead of a new dependency.
* `AnthropicLLM` -- the shipped adapter, built on the official ``anthropic``
  SDK behind the ``retrace-rag[anthropic]`` extra and imported lazily, so the core
  install stays dependency-free.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .errors import LLMError, LLMOutputError, LLMRefusalError


@runtime_checkable
class LLM(Protocol):
    """Anything that can answer a prompt with JSON matching a schema."""

    def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a dict conforming to `schema`.

        Raise `LLMRefusalError` when the model declines, and `LLMOutputError`
        when it answers in an unusable shape. Anything else propagates -- a
        network failure is not a per-item problem and should stop the run.
        """
        ...


@dataclass
class Usage:
    """Running token totals, so a run can report what it cost."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    refusals: int = 0
    fallbacks: int = 0
    rate_limit_waits: int = 0
    waited_s: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "refusals": self.refusals,
            "fallbacks": self.fallbacks,
            "rate_limit_waits": self.rate_limit_waits,
            "waited_s": round(self.waited_s, 1),
        }


# --------------------------------------------------------------------------
# Plain callables
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model text, tolerating a Markdown fence.

    Needed only for providers without native structured output. Deliberately
    strict beyond the fence: no regex rescue of JSON embedded in prose, because
    a salvaged half-object is worse than a clean rejection.
    """
    stripped = _FENCE_RE.sub("", text.strip())
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise LLMOutputError(f"expected a JSON object, got {type(value).__name__}")
    return value


class CallableLLM:
    """Adapts ``fn(system, prompt, schema) -> dict | str`` to `LLM`."""

    __slots__ = ("_fn", "name", "usage")

    def __init__(self, fn: Callable[..., Any], name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "callable_llm")
        self.usage = Usage()

    def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.usage.calls += 1
        out = self._fn(system, prompt, schema)
        if isinstance(out, dict):
            return out
        if isinstance(out, str):
            return parse_json_object(out)
        raise LLMOutputError(f"{self.name} returned {type(out).__name__}, expected dict or str")

    def __repr__(self) -> str:
        return f"CallableLLM({self.name})"


def as_llm(obj: Any) -> LLM:
    """Accept an `LLM`, or a callable taking ``(system, prompt, schema)``."""
    if hasattr(obj, "complete_json") and callable(obj.complete_json):
        return obj
    if callable(obj):
        return CallableLLM(obj)
    raise TypeError(
        f"{obj!r} is not usable as an LLM: expected .complete_json(...) or a "
        "callable taking (system, prompt, schema)"
    )


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"

# Server-side refusal fallback. When the requested model's safety classifiers
# decline a request, the API re-runs it on a fallback model inside the same
# call instead of returning the refusal. "default" lets Anthropic route by
# refusal category, so there is no pinned fallback model to go stale.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AnthropicLLM:
    """`LLM` backed by the official Anthropic SDK.

    Install with ``pip install "retrace-rag[anthropic]"``. Credentials resolve the
    way the SDK always resolves them (``ANTHROPIC_API_KEY`` and friends); pass
    `client` to inject a configured one.

    The SDK already retries rate limits, overloads and connection errors with
    exponential backoff, so this class does not add a second retry layer --
    tune `max_retries` instead.
    """

    model: str = DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = 16_000
    effort: str | None = None
    """Passed as ``output_config.effort`` when set. Question synthesis is a
    bulk job: measure ``low`` or ``medium`` on your own corpus before paying
    for the default."""

    refusal_fallback: bool = True
    max_retries: int = 2
    client: Any = None
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if self.client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised manually
                raise ImportError(
                    'AnthropicLLM needs the Anthropic SDK: pip install "retrace-rag[anthropic]"'
                ) from exc
            self.client = anthropic.Anthropic(max_retries=self.max_retries)

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if self.refusal_fallback:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        response = self.client.beta.messages.create(**kwargs)
        self._record_usage(response)

        # Check stop_reason before reading content: a refusal can arrive as a
        # normal 200 with nothing usable in it.
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            self.usage.refusals += 1
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusalError(
                f"model declined the request (category: {category or 'unspecified'})"
            )
        if stop == "max_tokens":
            raise LLMOutputError(
                f"output truncated at max_tokens={self.max_tokens}; raise max_tokens"
            )

        text = next(
            (block.text for block in response.content if getattr(block, "type", None) == "text"),
            None,
        )
        if text is None:
            raise LLMOutputError("response contained no text block")
        return parse_json_object(text)

    def _record_usage(self, response: Any) -> None:
        self.usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.usage.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        iterations = getattr(usage, "iterations", None) or []
        if any(getattr(it, "type", None) == "fallback_message" for it in iterations):
            self.usage.fallbacks += 1


# --------------------------------------------------------------------------
# OpenAI-compatible endpoints (Groq, OpenAI, Together, Ollama, vLLM, ...)
# --------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        # Supports strict json_schema output; free tier is limited by tokens per
        # minute, which the retry logic below absorbs. Verified live.
        "model": "openai/gpt-oss-120b",
    },
    "deepseek": {
        # From DeepSeek's API docs (read 2026-11): OpenAI-compatible, but
        # - only JSON-object mode, no json_schema -> schema goes in the prompt,
        #   which must mention "json" and include an example;
        # - the output limit is `max_tokens`, not `max_completion_tokens`;
        # - thinking is ON by default and the docs do not say JSON mode works
        #   while thinking, so it is switched off unless an effort is asked for;
        # - JSON mode may return whitespace-only content -> retried.
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-flash",
        "strict_schema": False,
        "max_tokens_param": "max_tokens",
        "no_effort_body": {"thinking": {"type": "disabled"}},
    },
}

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]


def _urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, dict[str, str], bytes]:
    """POST with the standard library. Returns (status, headers, body) for any status."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


@dataclass
class OpenAICompatibleLLM:
    """`LLM` for any endpoint speaking the OpenAI chat-completions protocol.

    Standard library only -- no SDK -- so one class covers Groq, OpenAI,
    Together, Ollama, vLLM and LM Studio without adding a dependency. Use
    `OpenAICompatibleLLM.preset("groq")` for a ready configuration.

    The key is read from the environment variable named by `api_key_env` at
    call time and never stored in logs, reprs or run files.

    Failure handling, from the cheapest to the most fatal:

    * 429 rate limit  -- wait for ``retry-after`` and retry. Free tiers are
      limited by tokens per minute, so this is routine, not exceptional.
    * 5xx / network   -- exponential backoff, then give up.
    * truncated / unparseable output, schema mismatch -- `LLMOutputError`:
      that one passage is rejected and the run continues.
    * 401/403, bad model, other 4xx, or a wait longer than `max_wait_s` (a
      daily quota) -- `RetraceError`: a setup problem that should stop the run
      immediately rather than burn through every passage.
    """

    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 4096
    """Reasoning models spend output tokens thinking; Groq's default of 1024
    truncates them."""

    reasoning_effort: str | None = None
    strict_schema: bool = True
    """Constrained json_schema decoding. Set False for models that only support
    JSON-object mode; the schema then goes into the system prompt."""

    max_tokens_param: str = "max_completion_tokens"
    """Name of the output-limit field. OpenAI and Groq use max_completion_tokens;
    DeepSeek documents only max_tokens."""

    no_effort_body: dict[str, Any] = field(default_factory=dict)
    """Extra request fields sent only when no reasoning effort is set -- e.g.
    DeepSeek's switch that turns its default-on thinking mode off."""

    max_retries: int = 6
    max_wait_s: float = 120.0
    timeout_s: float = 120.0
    provider: str = "openai-compatible"
    transport: Transport | None = None
    sleep: Callable[[float], None] | None = None
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def preset(cls, provider: str, **overrides: Any) -> OpenAICompatibleLLM:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
        config = dict(PROVIDERS[provider])
        config.update({k: v for k, v in overrides.items() if v is not None})
        return cls(provider=provider, **config)

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:  # never include the key
        return f"OpenAICompatibleLLM({self.name}, base_url={self.base_url!r})"

    def _api_key(self) -> str:
        import os

        from .errors import RetraceError

        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RetraceError(
                f"{self.api_key_env} is not set. Put it in your environment or in a .env "
                f"file next to where you run retrace ({self.api_key_env}=...)."
            )
        return key

    def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        import time

        from .errors import RetraceError

        if self.strict_schema:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": "retrace_output", "strict": True, "schema": schema},
            }
        else:
            response_format = {"type": "json_object"}
            system = (f"{system}\n\nRespond with a single JSON object matching this JSON "
                      f"schema, and nothing else:\n{json.dumps(schema)}\n\n"
                      f"Example of the required json shape:\n"
                      f"{json.dumps(_example_from_schema(schema))}")

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            self.max_tokens_param: self.max_tokens,
            "response_format": response_format,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body.update(self.no_effort_body)

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "User-Agent": "retrace-rag",
        }
        payload = json.dumps(body).encode("utf-8")
        send = self.transport or _urllib_transport
        sleep = self.sleep or time.sleep

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                status, resp_headers, raw = send(url, headers, payload, self.timeout_s)
            except OSError as exc:  # DNS, connection reset, timeout
                if attempt == self.max_retries:
                    raise RetraceError(f"{self.name}: network error after retries: {exc}") from exc
                sleep(delay)
                delay = min(delay * 2, 60.0)
                continue

            if status == 200:
                try:
                    return self._parse(raw)
                except _TransientOutput as exc:
                    # Whitespace-only JSON-mode output, or generation interrupted
                    # by the provider: worth another attempt, not a rejection yet.
                    if attempt == self.max_retries:
                        raise LLMOutputError(str(exc)) from exc
                    sleep(delay if exc.backoff else 0.0)
                    delay = min(delay * 2, 60.0) if exc.backoff else delay
                    continue

            text = raw.decode("utf-8", errors="replace")
            message = _error_message(text)

            if status == 429:
                wait = _retry_after(resp_headers) or delay
                if wait > self.max_wait_s:
                    raise RetraceError(
                        f"{self.name}: rate limited for {wait:.0f}s -- probably a daily quota. "
                        "Wait it out or use a different model."
                    )
                if attempt == self.max_retries:
                    raise RetraceError(f"{self.name}: still rate limited after {attempt} retries")
                self.usage.rate_limit_waits += 1
                self.usage.waited_s += wait
                sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            if status >= 500:
                if attempt == self.max_retries:
                    raise RetraceError(f"{self.name}: server error {status}: {message}")
                sleep(delay)
                delay = min(delay * 2, 60.0)
                continue

            if status == 400 and _is_generation_failure(text):
                # The model produced output the provider could not validate:
                # bad luck on one passage, not a broken setup. Groq's docs name
                # one wording; the live API also returned "Failed to validate
                # JSON ... see 'failed_generation'", so match on structure.
                self.usage.calls += 1
                raise LLMOutputError(f"output failed validation: {message}")
            if status in (401, 403):
                raise RetraceError(
                    f"{self.name}: authentication failed ({status}). Check {self.api_key_env}."
                )
            if status == 402:
                raise RetraceError(f"{self.name}: insufficient account balance (402). "
                                   "Top up the account before running again.")
            raise RetraceError(f"{self.name}: HTTP {status}: {message}")

        raise RetraceError(f"{self.name}: request failed")  # pragma: no cover - loop always returns

    def _parse(self, raw: bytes) -> dict[str, Any]:
        self.usage.calls += 1
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"response is not JSON: {exc.msg}") from exc
        usage = data.get("usage") or {}
        self.usage.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.usage.output_tokens += int(usage.get("completion_tokens") or 0)

        choices = data.get("choices") or []
        if not choices:
            raise LLMOutputError("response contained no choices")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish == "content_filter":
            self.usage.refusals += 1
            raise LLMRefusalError("provider content filter blocked the response")
        if finish == "length":
            raise LLMOutputError(
                f"output truncated at max_tokens={self.max_tokens}; raise max_tokens"
            )
        if finish in ("insufficient_system_resource", "aborted"):
            raise _TransientOutput(f"generation interrupted by the provider ({finish})",
                                   backoff=True)
        content = (choice.get("message") or {}).get("content")
        if content is not None and not content.strip():
            # DeepSeek's JSON mode documents this: non-JSON output is replaced
            # with spaces, so a "successful" reply can be blank. Its docs say retry.
            raise _TransientOutput("response content was blank", backoff=False)
        if not content:
            raise LLMOutputError("response message has no content")
        return parse_json_object(content)


class _TransientOutput(Exception):
    """A 200 response worth retrying rather than rejecting the passage."""

    def __init__(self, message: str, *, backoff: bool) -> None:
        super().__init__(message)
        self.backoff = backoff


def _example_from_schema(schema: dict[str, Any]) -> Any:
    """A placeholder instance of a JSON schema, for providers that ask for an example."""
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind == "object":
        return {k: _example_from_schema(v) for k, v in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [_example_from_schema(schema.get("items") or {"type": "string"})]
    if "enum" in schema:
        return schema["enum"][0]
    return {"boolean": True, "integer": 0, "number": 0.0}.get(kind, "...")


_GENERATION_FAILURE_PHRASES = (
    "does not match the expected schema",
    "failed to validate json",
    "failed_generation",
)


def _is_generation_failure(text: str) -> bool:
    """A 400 caused by what the model generated, rather than by the request."""
    try:
        err = json.loads(text).get("error")
    except (json.JSONDecodeError, AttributeError):
        err = None
    if isinstance(err, dict):
        if "failed_generation" in err:
            return True
        if "json_validate" in str(err.get("code", "")).lower():
            return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in _GENERATION_FAILURE_PHRASES)


def _error_message(text: str) -> str:
    try:
        err = json.loads(text).get("error")
    except (json.JSONDecodeError, AttributeError):
        return text[:300]
    if isinstance(err, dict):
        return str(err.get("message", ""))[:300]
    return str(err)[:300]


def _retry_after(headers: dict[str, str]) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def build_llm(
    provider: str,
    *,
    model: str | None = None,
    effort: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> Any:
    """One place that turns CLI choices into an `LLM`."""
    if provider == "anthropic":
        return AnthropicLLM(model=model or DEFAULT_ANTHROPIC_MODEL, effort=effort)
    if effort not in (None, "low", "medium", "high"):
        raise ValueError(f"--effort {effort} is not supported by {provider}; use low/medium/high")
    if provider == "openai-compatible":
        if not (base_url and model):
            raise ValueError("--provider openai-compatible needs --base-url and --model")
        return OpenAICompatibleLLM(model=model, base_url=base_url,
                                   api_key_env=api_key_env or "OPENAI_API_KEY",
                                   reasoning_effort=effort)
    return OpenAICompatibleLLM.preset(provider, model=model, base_url=base_url,
                                      api_key_env=api_key_env, reasoning_effort=effort)


def load_env_file(path: str = ".env") -> list[str]:
    """Load ``KEY=VALUE`` lines into the environment, never overriding what is set.

    Returns the names loaded -- never the values. Lets a key live in a
    gitignored file instead of being typed into a shell history or a chat.
    """
    import os
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "LLM",
    "PROVIDERS",
    "AnthropicLLM",
    "CallableLLM",
    "LLMError",
    "OpenAICompatibleLLM",
    "Usage",
    "as_llm",
    "build_llm",
    "load_env_file",
    "parse_json_object",
]
