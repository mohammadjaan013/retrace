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

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "refusals": self.refusals,
            "fallbacks": self.fallbacks,
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


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "LLM",
    "AnthropicLLM",
    "CallableLLM",
    "LLMError",
    "Usage",
    "as_llm",
    "parse_json_object",
]
