from __future__ import annotations

import pytest

from retrace import Chunk, EvalCase, Provenance, RetrievalResult, Span, stable_id


class TestSpan:
    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="must be >"):
            Span("d1", 100, 50)

    def test_rejects_empty_range(self):
        with pytest.raises(ValueError, match="must be >"):
            Span("d1", 10, 10)

    def test_rejects_negative_start(self):
        with pytest.raises(ValueError, match=">= 0"):
            Span("d1", -1, 10)

    def test_overlap_within_document(self):
        assert Span("d1", 0, 100).overlaps(Span("d1", 50, 150))
        assert not Span("d1", 0, 50).overlaps(Span("d1", 50, 100))

    def test_never_overlaps_across_documents(self):
        assert not Span("d1", 0, 100).overlaps(Span("d2", 0, 100))
        assert Span("d1", 0, 100).overlap_chars(Span("d2", 0, 100)) == 0

    def test_overlap_chars_counts_shared_region(self):
        assert Span("d1", 0, 100).overlap_chars(Span("d1", 80, 200)) == 20
        assert Span("d1", 0, 100).overlap_chars(Span("d1", 200, 300)) == 0


class TestChunk:
    def test_requires_ids(self):
        with pytest.raises(ValueError, match="chunk id"):
            Chunk(id="", doc_id="d1", text="x")
        with pytest.raises(ValueError, match="doc_id"):
            Chunk(id="c1", doc_id="", text="x")

    def test_span_must_belong_to_same_document(self):
        with pytest.raises(ValueError, match="points at"):
            Chunk(id="c1", doc_id="d1", text="x", span=Span("d2", 0, 10))

    def test_with_score_returns_a_copy(self):
        original = Chunk(id="c1", doc_id="d1", text="hello", meta={"k": "v"})
        scored = original.with_score(0.9)
        assert original.score is None
        assert scored.score == 0.9
        assert scored.text == original.text
        assert scored.meta == {"k": "v"}
        assert scored.meta is not original.meta

    def test_hashable_despite_dict_field(self):
        # `meta` is a dict, so the default frozen-dataclass hash would raise
        # TypeError. The custom __hash__ uses identity fields only.
        chunk = Chunk(id="c1", doc_id="d1", text="x", meta={"deep": {"nested": 1}})
        assert isinstance(hash(chunk), int)
        assert chunk in {chunk}

    def test_equal_chunks_hash_equal(self):
        # The contract that must hold: equal objects share a hash.
        a = Chunk(id="c1", doc_id="d1", text="x", meta={"k": "v"})
        b = Chunk(id="c1", doc_id="d1", text="x", meta={"k": "v"})
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_same_identity_different_payload_collides_but_stays_distinct(self):
        # Permitted and intended: unequal objects may share a hash. Corpus
        # de-duplication keys on chunk.id explicitly rather than relying on this.
        a = Chunk(id="c1", doc_id="d1", text="x", meta={"a": 1})
        b = Chunk(id="c1", doc_id="d1", text="x", meta={"b": 2})
        assert hash(a) == hash(b)
        assert a != b
        assert len({a, b}) == 2


class TestEvalCase:
    def test_requires_some_gold_evidence(self):
        with pytest.raises(ValueError, match="no gold evidence"):
            EvalCase(id="q1", question="What?")

    def test_accepts_doc_ids_alone(self):
        case = EvalCase(id="q1", question="What?", gold_doc_ids=("d1",))
        assert case.implied_doc_ids == {"d1"}

    def test_rejects_blank_question(self):
        with pytest.raises(ValueError, match="empty question"):
            EvalCase(id="q1", question="   ", gold_doc_ids=("d1",))

    def test_implied_docs_unions_both_sources(self):
        case = EvalCase(
            id="q1",
            question="What?",
            gold_doc_ids=("d1",),
            gold_spans=(Span("d2", 0, 10),),
        )
        assert case.implied_doc_ids == {"d1", "d2"}

    @pytest.mark.parametrize(
        ("provenance", "trusted"),
        [
            (Provenance.HUMAN, True),
            (Provenance.MODEL_VERIFIED, True),
            (Provenance.MODEL, False),
            (Provenance.IMPORTED, False),
        ],
    )
    def test_trust_follows_provenance(self, provenance, trusted):
        case = EvalCase(
            id="q1", question="What?", gold_doc_ids=("d1",), provenance=provenance
        )
        assert case.is_trusted is trusted


class TestRetrievalResult:
    def test_detects_collapsed_ranking_stage(self):
        chunks = (Chunk(id="c1", doc_id="d1", text="a"),)
        tier1 = RetrievalResult(query="q", candidates=chunks, context=chunks)
        assert not tier1.exposes_ranking_stage

    def test_detects_real_ranking_stage(self):
        c1 = Chunk(id="c1", doc_id="d1", text="a")
        c2 = Chunk(id="c2", doc_id="d1", text="b")
        tier2 = RetrievalResult(query="q", candidates=(c1, c2), context=(c1,))
        assert tier2.exposes_ranking_stage

    def test_reordering_alone_counts_as_a_ranking_stage(self):
        c1 = Chunk(id="c1", doc_id="d1", text="a")
        c2 = Chunk(id="c2", doc_id="d1", text="b")
        reranked = RetrievalResult(query="q", candidates=(c1, c2), context=(c2, c1))
        assert reranked.exposes_ranking_stage


class TestStableId:
    def test_is_deterministic(self):
        assert stable_id("a", "b") == stable_id("a", "b")

    def test_separates_parts_unambiguously(self):
        # Without a separator, ("ab","c") and ("a","bc") would collide.
        assert stable_id("ab", "c") != stable_id("a", "bc")

    def test_prefix_and_length(self):
        got = stable_id("x", prefix="c_", length=8)
        assert got.startswith("c_")
        assert len(got) == len("c_") + 8
