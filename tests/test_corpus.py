"""Tests for gold-span resolution.

This is the file that matters most in Week 1. If span resolution is wrong, every
number Retrace ever reports is wrong, and the failure is silent.
"""

from __future__ import annotations

import pytest

from retrace import (
    Chunk,
    CorpusError,
    EvalCase,
    InMemoryCorpus,
    Span,
    chunk_documents,
    corpus_covers,
    coverage_warning,
    resolve_gold_chunks,
    retrieval_coverage,
)

DOC = (
    "Spacecraft thermal control maintains equipment within survivable limits. "  # 0-75
    "Passive techniques include multilayer insulation and surface coatings. "  # 75-146
    "Active techniques include fluid loops, heaters, and deployable radiators. "  # 146-220
    "Radiator sizing is driven by the worst-case hot attitude."  # 220-277
)


def build_corpus(size: int = 80, overlap: int = 0) -> InMemoryCorpus:
    chunks = chunk_documents({"d1": DOC}, size=size, overlap=overlap)
    corpus = InMemoryCorpus(chunks)
    corpus.add_document("d1", DOC)
    return corpus


class TestInMemoryCorpus:
    def test_membership_and_lookup(self):
        corpus = build_corpus()
        chunk = next(corpus.iter_chunks())
        assert chunk.id in corpus
        assert corpus.get(chunk.id) == chunk
        assert corpus.get("nope") is None
        assert "nope" not in corpus

    def test_rejects_conflicting_id(self):
        corpus = InMemoryCorpus([Chunk(id="c1", doc_id="d1", text="original")])
        with pytest.raises(CorpusError, match="different text"):
            corpus.add(Chunk(id="c1", doc_id="d1", text="different"))

    def test_reindexing_identical_chunk_is_idempotent(self):
        chunk = Chunk(id="c1", doc_id="d1", text="same")
        corpus = InMemoryCorpus([chunk])
        corpus.add(chunk)
        assert len(corpus) == 1
        assert len(corpus.chunks_in("d1")) == 1


class TestSpanResolution:
    def test_resolves_to_the_chunk_holding_the_answer(self):
        corpus = build_corpus(size=80)
        # "multilayer insulation" lives around offset 110.
        case = EvalCase(
            id="q1",
            question="What passive thermal techniques are used?",
            gold_spans=(Span("d1", 100, 140),),
        )
        resolved = resolve_gold_chunks(case, corpus)
        assert resolved
        for chunk_id in resolved:
            chunk = corpus.get(chunk_id)
            assert chunk.span.overlaps(Span("d1", 100, 140))

    def test_survives_rechunking(self):
        """The whole point: same label, different chunking, still resolves."""
        case = EvalCase(
            id="q1",
            question="What passive thermal techniques are used?",
            gold_spans=(Span("d1", 100, 140),),
        )

        coarse = build_corpus(size=200)
        fine = build_corpus(size=50)

        coarse_ids = resolve_gold_chunks(case, coarse)
        fine_ids = resolve_gold_chunks(case, fine)

        assert coarse_ids and fine_ids
        # Different chunkings produce different ids -- that is expected, and is
        # exactly why cached chunk ids cannot be the source of truth.
        assert coarse_ids != fine_ids

        for ids, corpus in ((coarse_ids, coarse), (fine_ids, fine)):
            covered = "".join(sorted(corpus.get(i).text for i in ids))
            assert "insulation" in covered or "Passive" in covered

    def test_short_span_lowers_its_own_threshold(self):
        corpus = build_corpus(size=80)
        # A 6-char span is shorter than DEFAULT_MIN_OVERLAP_CHARS (32); the
        # threshold must drop to the span length or it could never resolve.
        case = EvalCase(
            id="q1", question="Which?", gold_spans=(Span("d1", 100, 106),)
        )
        assert resolve_gold_chunks(case, corpus)

    def test_boundary_clip_below_threshold_is_excluded(self):
        left = Chunk(id="a", doc_id="d1", text="x" * 100, span=Span("d1", 0, 100))
        right = Chunk(id="b", doc_id="d1", text="y" * 100, span=Span("d1", 100, 200))
        corpus = InMemoryCorpus([left, right])

        # Overlaps "a" by 95 chars and "b" by only 5 -- below the 32 threshold.
        case = EvalCase(id="q1", question="?", gold_spans=(Span("d1", 5, 105),))
        assert resolve_gold_chunks(case, corpus) == {"a"}

    def test_multiple_valid_gold_chunks(self):
        corpus = build_corpus(size=60)
        case = EvalCase(
            id="q1",
            question="What thermal techniques exist?",
            gold_spans=(Span("d1", 75, 146), Span("d1", 146, 220)),
        )
        assert len(resolve_gold_chunks(case, corpus)) >= 2


class TestResolutionFallbacks:
    def test_falls_back_to_doc_ids_when_no_spans(self):
        corpus = build_corpus()
        case = EvalCase(id="q1", question="?", gold_doc_ids=("d1",))
        assert len(resolve_gold_chunks(case, corpus)) == len(corpus)

    def test_stale_cached_chunk_ids_are_ignored(self):
        corpus = build_corpus()
        case = EvalCase(
            id="q1",
            question="?",
            gold_doc_ids=("d1",),
            gold_chunk_ids=("c_deadbeef", "c_notreal"),
        )
        # Cached ids are gone; resolution must fall through to doc ids rather
        # than reporting a spurious CORPUS_GAP.
        assert len(resolve_gold_chunks(case, corpus)) == len(corpus)

    def test_spans_take_priority_over_doc_ids(self):
        corpus = build_corpus(size=60)
        case = EvalCase(
            id="q1",
            question="?",
            gold_spans=(Span("d1", 0, 40),),
            gold_doc_ids=("d1",),
        )
        # The precise label must not be diluted to "every chunk in the doc".
        assert len(resolve_gold_chunks(case, corpus)) < len(corpus)

    def test_missing_document_resolves_to_nothing(self):
        corpus = build_corpus()
        case = EvalCase(id="q1", question="?", gold_doc_ids=("d_absent",))
        assert resolve_gold_chunks(case, corpus) == frozenset()
        assert corpus_covers(case, corpus) is False

    def test_span_in_unindexed_document_resolves_to_nothing(self):
        corpus = build_corpus()
        case = EvalCase(id="q1", question="?", gold_spans=(Span("d_absent", 0, 50),))
        assert not corpus_covers(case, corpus)


class TestCoverageGuard:
    """Regression guard for the inflated-recall trap.

    Found while building the Week 1 baseline: k=5 over a 5-chunk corpus scored
    100% recall because retrieval returned the entire corpus.
    """

    def _corpus(self, n: int) -> InMemoryCorpus:
        return InMemoryCorpus(
            Chunk(id=f"c{i}", doc_id="d1", text=f"chunk {i}") for i in range(n)
        )

    def test_coverage_fraction(self):
        assert retrieval_coverage(5, self._corpus(100)) == 0.05
        assert retrieval_coverage(5, self._corpus(10)) == 0.5

    def test_coverage_is_clamped_when_k_exceeds_corpus(self):
        assert retrieval_coverage(50, self._corpus(5)) == 1.0

    def test_empty_corpus_has_zero_coverage(self):
        assert retrieval_coverage(5, self._corpus(0)) == 0.0

    def test_silent_when_ratio_is_safe(self):
        assert coverage_warning(5, self._corpus(1000)) is None

    def test_warns_when_k_swallows_the_corpus(self):
        warning = coverage_warning(5, self._corpus(5))
        assert warning is not None
        assert "100%" in warning
        assert "inflated" in warning

    def test_warns_at_the_exact_threshold(self):
        # 10 of 100 is exactly 10%, which must warn rather than squeak through.
        assert coverage_warning(10, self._corpus(100)) is not None
        assert coverage_warning(9, self._corpus(100)) is None
