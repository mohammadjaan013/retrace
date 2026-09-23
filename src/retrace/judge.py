"""Answer judges: did the generated answer convey the expected answer?

Only the last stage of the cascade needs one. The first three verdicts are
decided by set membership -- was the gold chunk indexed, retrieved, kept -- and
involve no judgement at all. That is a deliberate strength: three of the four
verdicts are exact, and only GENERATION_MISS rests on an opinion.

Every judge may answer "I can't tell" (`correct=None`). The cascade turns that
into UNSCORABLE rather than guessing, because a judge that is forced to pick a
side is the most common way generation metrics turn into noise.

Two judges ship:

* `TokenRecallJudge` -- free, instant, dependency-free. Blind to negation and
  paraphrase; its uncertain middle band exists so it declines instead of being
  confidently wrong. Good enough to triage; not good enough to quote.
* `LLMJudge` -- any `LLM`. Handles paraphrase and contradiction. Use it before
  putting a GENERATION_MISS number in front of anyone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .errors import LLMError
from .llm import as_llm
from .text import content_tokens, tokenize
from .types import Chunk


@dataclass(frozen=True, slots=True)
class Judgement:
    correct: bool | None
    """True / False, or None when the judge cannot tell."""

    score: float | None = None
    reason: str = ""
    judge: str = ""


@runtime_checkable
class AnswerJudge(Protocol):
    def judge(
        self, question: str, expected: str, actual: str, context: Sequence[Chunk]
    ) -> Judgement: ...


class TokenRecallJudge:
    """Fraction of the expected answer's content words present in the actual answer.

    Recall rather than F1 because generated answers are wordy: "The worst-case
    hot condition drives radiator sizing, because..." is a correct answer to an
    expected "worst-case hot condition", and F1 would punish it for length.

    Known blind spots, stated rather than hidden: "it is NOT the average" and "it
    is the average" share every content word, and synonyms score zero. The band
    between the thresholds returns None for exactly that reason.
    """

    name = "token_recall"

    def __init__(self, correct_at: float = 0.75, wrong_below: float = 0.25) -> None:
        if not 0.0 <= wrong_below < correct_at <= 1.0:
            raise ValueError("need 0 <= wrong_below < correct_at <= 1")
        self.correct_at = correct_at
        self.wrong_below = wrong_below

    def judge(
        self, question: str, expected: str, actual: str, context: Sequence[Chunk]
    ) -> Judgement:
        expected_terms = content_tokens(expected)
        if not expected_terms:
            return Judgement(None, reason="expected answer has no content words", judge=self.name)
        actual_terms = set(tokenize(actual))
        recall = len(expected_terms & actual_terms) / len(expected_terms)

        if recall >= self.correct_at:
            verdict: bool | None = True
        elif recall < self.wrong_below:
            verdict = False
        else:
            verdict = None
        missing = sorted(expected_terms - actual_terms)
        reason = f"{recall:.0%} of expected content words present"
        if missing and verdict is not True:
            reason += f"; missing: {', '.join(missing[:6])}"
        return Judgement(verdict, score=round(recall, 3), reason=reason, judge=self.name)


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect", "unsure"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """\
You grade answers produced by a question-answering system against a reference \
answer written by a person.

Mark the answer `correct` if it conveys the facts in the reference answer. \
Paraphrase, extra detail and a different level of formality are all fine.

Mark it `incorrect` if it contradicts the reference, omits the fact the question \
was asking for, or says it cannot answer.

Mark it `unsure` only when the reference itself is ambiguous or the question \
could reasonably have more than one right answer. Do not use `unsure` to avoid \
a hard call.

Keep `reason` to one sentence naming the fact that decided it."""

JUDGE_PROMPT = """\
<question>{question}</question>
<reference_answer>{expected}</reference_answer>
<system_answer>{actual}</system_answer>"""


class LLMJudge:
    """Grades with a language model. Any `LLM`, or a callable, is accepted."""

    def __init__(self, llm: Any) -> None:
        self._llm = as_llm(llm)
        self.name = f"llm:{getattr(self._llm, 'name', type(self._llm).__name__)}"

    def judge(
        self, question: str, expected: str, actual: str, context: Sequence[Chunk]
    ) -> Judgement:
        try:
            out = self._llm.complete_json(
                system=JUDGE_SYSTEM,
                prompt=JUDGE_PROMPT.format(question=question, expected=expected, actual=actual),
                schema=JUDGE_SCHEMA,
            )
        except LLMError as exc:
            return Judgement(None, reason=f"judge failed: {exc}", judge=self.name)

        verdict = str(out.get("verdict", "")).lower()
        mapping = {"correct": True, "incorrect": False, "unsure": None}
        if verdict not in mapping:
            return Judgement(None, reason=f"judge returned {verdict!r}", judge=self.name)
        return Judgement(mapping[verdict], reason=str(out.get("reason", "")), judge=self.name)
