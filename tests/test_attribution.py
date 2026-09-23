"""The cascade, one verdict at a time, against hand-built scenarios."""

from __future__ import annotations

from retrace import Chunk, EvalCase, InMemoryCorpus, Provenance, RetrievalResult, Span, Verdict
from retrace.attribution import (
    CHUNK_ID,
    DOCUMENT,
    SPAN,
    attribute,
    error_attribution,
    gold_matcher,
)
from retrace.judge import Judgement

DOC = "x" * 100 + "RADIATOR SIZING IS DRIVEN BY THE WORST CASE" + "y" * 100
GOLD_SPAN = Span("d1", 100, 143)


def c(cid, start, end, doc="d1", span=True):
    return Chunk(
        id=cid, doc_id=doc, text=DOC[start:end] if doc == "d1" else "other",
        span=Span(doc, start, end) if span else None,
    )


GOLD = c("gold", 90, 160)
NOISE_A = c("a", 0, 90)
NOISE_B = c("b", 160, 243)
OTHER = c("o", 0, 5, doc="d2")


def corpus(*chunks):
    return InMemoryCorpus(chunks, documents={"d1": DOC})


def case(**kw):
    defaults = {
        "id": "q1",
        "question": "What drives radiator sizing?",
        "answer": "the worst case",
        "gold_spans": (GOLD_SPAN,),
        "provenance": Provenance.HUMAN,
    }
    defaults.update(kw)
    return EvalCase(**defaults)


def result(candidates, context=None, answer=None):
    return RetrievalResult(
        query="q", candidates=tuple(candidates),
        context=tuple(candidates if context is None else context), answer=answer,
    )


class FixedJudge:
    name = "fixed"

    def __init__(self, correct):
        self.correct = correct
        self.calls = 0

    def judge(self, question, expected, actual, context):
        self.calls += 1
        return Judgement(self.correct, reason=f"fixed={self.correct}", judge=self.name)


class TestCorpusStage:
    def test_gap_when_evidence_is_not_indexed(self):
        a = attribute(case(), result([NOISE_A]), corpus=corpus(NOISE_A, NOISE_B))
        assert a.verdict is Verdict.CORPUS_GAP
        assert a.stage == "corpus"
        assert a.gold_in_corpus == 0
        assert a.is_failure

    def test_gap_short_circuits_later_stages(self):
        # Even with the right chunk retrieved *by text*, an unindexed gold is a gap.
        a = attribute(case(gold_doc_ids=("d_missing",), gold_spans=()),
                      result([NOISE_A]), corpus=corpus(NOISE_A))
        assert a.verdict is Verdict.CORPUS_GAP
        assert a.candidate_rank is None

    def test_without_corpus_gap_is_not_detectable_and_says_so(self):
        a = attribute(case(), result([NOISE_A]))
        assert a.verdict is Verdict.RETRIEVAL_MISS
        assert a.gold_in_corpus is None
        assert "may also be a corpus gap" in a.reason
        assert any("CORPUS_GAP cannot be detected" in n for n in a.notes)


class TestRetrievalStage:
    def test_miss_when_gold_indexed_but_not_retrieved(self):
        a = attribute(case(), result([NOISE_A, NOISE_B]), corpus=corpus(GOLD, NOISE_A, NOISE_B))
        assert a.verdict is Verdict.RETRIEVAL_MISS
        assert a.gold_in_corpus == 1
        assert a.n_candidates == 2

    def test_hit_records_rank(self):
        a = attribute(case(), result([NOISE_A, NOISE_B, GOLD]), corpus=corpus(GOLD, NOISE_A))
        assert a.verdict is Verdict.OK
        assert a.candidate_rank == 3
        assert a.context_rank == 3
        assert a.match_granularity == SPAN

    def test_gold_injected_into_context_is_not_a_retrieval_miss(self):
        a = attribute(case(), result([NOISE_A], context=[GOLD, NOISE_A]),
                      corpus=corpus(GOLD, NOISE_A))
        assert a.verdict is Verdict.OK
        assert any("without appearing among the candidates" in n for n in a.notes)


class TestRankingStage:
    def test_miss_when_retrieved_but_cut(self):
        a = attribute(case(), result([NOISE_A, GOLD, NOISE_B], context=[NOISE_A]),
                      corpus=corpus(GOLD, NOISE_A, NOISE_B))
        assert a.verdict is Verdict.RANKING_MISS
        assert a.candidate_rank == 2
        assert a.context_rank is None
        assert "candidate rank 2" in a.reason

    def test_tier_one_never_emits_ranking_miss(self):
        a = attribute(case(), result([GOLD]), corpus=corpus(GOLD))
        assert a.verdict is Verdict.OK
        assert any("ranking stage not observable" in n for n in a.notes)

    def test_tier_two_carries_no_tier_one_note(self):
        a = attribute(case(), result([NOISE_A, GOLD], context=[GOLD]), corpus=corpus(GOLD))
        assert not any("ranking stage not observable" in n for n in a.notes)
        assert a.context_rank == 1


class TestGenerationStage:
    def test_generation_miss_when_judged_wrong(self):
        a = attribute(case(), result([GOLD], answer="the average"),
                      corpus=corpus(GOLD), judge=FixedJudge(False))
        assert a.verdict is Verdict.GENERATION_MISS
        assert a.generation_evaluated
        assert "position 1" in a.reason

    def test_ok_when_judged_right(self):
        a = attribute(case(), result([GOLD], answer="the worst case"),
                      corpus=corpus(GOLD), judge=FixedJudge(True))
        assert a.verdict is Verdict.OK
        assert a.generation_evaluated
        assert "judged correct" in a.reason

    def test_unscorable_when_judge_cannot_decide(self):
        a = attribute(case(), result([GOLD], answer="hmm"),
                      corpus=corpus(GOLD), judge=FixedJudge(None))
        assert a.verdict is Verdict.UNSCORABLE
        assert not a.is_failure

    def test_no_answer_means_not_evaluated_rather_than_passed(self):
        a = attribute(case(), result([GOLD]), corpus=corpus(GOLD), judge=FixedJudge(False))
        assert a.verdict is Verdict.OK
        assert not a.generation_evaluated
        assert any("no generated answer" in n for n in a.notes)

    def test_no_expected_answer(self):
        a = attribute(case(answer=None), result([GOLD], answer="x"),
                      corpus=corpus(GOLD), judge=FixedJudge(False))
        assert not a.generation_evaluated
        assert any("no expected answer" in n for n in a.notes)

    def test_no_judge(self):
        a = attribute(case(), result([GOLD], answer="x"), corpus=corpus(GOLD))
        assert not a.generation_evaluated
        assert any("no judge configured" in n for n in a.notes)


class TestLucky:
    def test_right_answer_without_evidence_is_flagged_not_passed(self):
        judge = FixedJudge(True)
        a = attribute(case(), result([NOISE_A], answer="the worst case"),
                      corpus=corpus(GOLD, NOISE_A), judge=judge)
        assert a.verdict is Verdict.RETRIEVAL_MISS
        assert a.lucky
        assert any("answered from memory" in n for n in a.notes)

    def test_wrong_answer_without_evidence_is_not_lucky(self):
        a = attribute(case(), result([NOISE_A], answer="nope"),
                      corpus=corpus(GOLD, NOISE_A), judge=FixedJudge(False))
        assert not a.lucky

    def test_lucky_is_not_checked_on_success(self):
        judge = FixedJudge(True)
        attribute(case(), result([GOLD], answer="the worst case"), corpus=corpus(GOLD), judge=judge)
        assert judge.calls == 1  # generation stage only, no second lucky probe


class TestGoldMatcher:
    def test_span_match(self):
        assert gold_matcher(case())(GOLD) == SPAN

    def test_span_mismatch(self):
        assert gold_matcher(case())(NOISE_A) is None

    def test_boundary_clip_is_not_gold(self):
        clipper = c("clip", 0, 104)  # overlaps the gold span by 4 chars
        assert gold_matcher(case())(clipper) is None

    def test_chunk_id_match_without_spans(self):
        bare = Chunk(id="gold", doc_id="d1", text="...")
        m = gold_matcher(case(), corpus(GOLD))
        assert m(bare) == CHUNK_ID

    def test_document_fallback_only_when_chunk_has_no_span(self):
        bare = Chunk(id="unknown", doc_id="d1", text="...")
        assert gold_matcher(case())(bare) == DOCUMENT
        a = attribute(case(), result([bare]))
        assert a.match_granularity == DOCUMENT
        assert any("document granularity" in n for n in a.notes)

    def test_other_document_is_never_gold(self):
        assert gold_matcher(case())(OTHER) is None


def test_untrusted_cases_are_marked():
    a = attribute(case(provenance=Provenance.MODEL), result([GOLD]), corpus=corpus(GOLD))
    assert a.trusted is False


def test_error_attribution():
    a = error_attribution(case(), TimeoutError("upstream timed out"))
    assert a.verdict is Verdict.UNSCORABLE
    assert a.stage == "error"
    assert a.error == "TimeoutError: upstream timed out"
