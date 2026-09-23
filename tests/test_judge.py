from __future__ import annotations

import pytest

from retrace.errors import LLMRefusalError
from retrace.judge import JUDGE_SCHEMA, LLMJudge, TokenRecallJudge


class TestTokenRecallJudge:
    judge = TokenRecallJudge()

    def test_verbose_correct_answer_passes(self):
        j = self.judge.judge(
            "What drives radiator sizing?",
            "the worst-case hot condition",
            "Radiators are sized for the worst-case hot condition, since surviving it "
            "means the cold case only needs heater power.",
            (),
        )
        assert j.correct is True
        assert j.score == 1.0

    def test_unrelated_answer_fails(self):
        j = self.judge.judge("?", "the worst-case hot condition", "Solar array degradation.", ())
        assert j.correct is False
        assert "missing" in j.reason

    def test_partial_answer_declines_to_decide(self):
        j = self.judge.judge("?", "multilayer insulation and surface coatings",
                             "Mostly insulation.", ())
        assert j.correct is None

    def test_negation_is_a_known_blind_spot(self):
        """Documented limitation: use LLMJudge before quoting GENERATION_MISS."""
        j = self.judge.judge("?", "the worst-case hot condition",
                             "It is not the worst-case hot condition.", ())
        assert j.correct is True

    def test_expected_with_no_content_words(self):
        assert self.judge.judge("?", "it is the one", "anything", ()).correct is None

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            TokenRecallJudge(correct_at=0.2, wrong_below=0.5)


class TestLLMJudge:
    def test_maps_verdicts(self):
        for word, expected in (("correct", True), ("incorrect", False), ("unsure", None)):
            judge = LLMJudge(lambda s, p, sc, w=word: {"verdict": w, "reason": "because"})
            j = judge.judge("q", "exp", "act", ())
            assert j.correct is expected
            assert j.reason == "because"

    def test_prompt_contains_all_three_texts(self):
        seen = {}

        def fn(system, prompt, schema):
            seen["prompt"] = prompt
            seen["schema"] = schema
            return {"verdict": "correct", "reason": ""}

        LLMJudge(fn).judge("THE QUESTION", "THE REFERENCE", "THE ANSWER", ())
        assert all(s in seen["prompt"] for s in ("THE QUESTION", "THE REFERENCE", "THE ANSWER"))
        assert seen["schema"] is JUDGE_SCHEMA

    def test_unexpected_verdict_is_undecided(self):
        j = LLMJudge(lambda s, p, sc: {"verdict": "maybe", "reason": ""}).judge("q", "e", "a", ())
        assert j.correct is None

    def test_model_failure_is_undecided_not_fatal(self):
        def fn(system, prompt, schema):
            raise LLMRefusalError("declined")

        j = LLMJudge(fn).judge("q", "e", "a", ())
        assert j.correct is None
        assert "judge failed" in j.reason

    def test_name_includes_model(self):
        def my_model(system, prompt, schema):
            return {}

        assert LLMJudge(my_model).name == "llm:my_model"
