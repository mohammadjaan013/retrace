"""Tests for the naive baseline and chunker.

Several of these assert that the baseline is *bad* in specific ways. That is
deliberate: those weaknesses are what the attribution cascade will be measured
against in Week 3, and a silent improvement here would quietly invalidate the
Week 5 before/after numbers.
"""

from __future__ import annotations

import pytest

from retrace import (
    Chunk,
    InMemoryCorpus,
    NaiveRetriever,
    RetrieverPipeline,
    as_retriever,
    chunk_documents,
    fixed_size_chunks,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_strips_punctuation(self):
        assert tokenize("Thermal-Control, v2.0!") == ["thermal", "control", "v2", "0"]

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("!!!") == []


class TestFixedSizeChunks:
    def test_covers_the_whole_document(self):
        text = "abcdefghij" * 10  # 100 chars
        chunks = fixed_size_chunks("d1", text, size=30)
        assert "".join(c.text for c in chunks) == text

    def test_spans_are_contiguous_and_correct(self):
        text = "abcdefghij" * 10
        chunks = fixed_size_chunks("d1", text, size=30)
        for chunk in chunks:
            assert text[chunk.span.start : chunk.span.end] == chunk.text
        assert chunks[0].span.start == 0
        assert chunks[-1].span.end == len(text)

    def test_overlap_repeats_content(self):
        text = "abcdefghij" * 10
        chunks = fixed_size_chunks("d1", text, size=30, overlap=10)
        assert chunks[1].span.start == 20
        assert chunks[0].text[-10:] == chunks[1].text[:10]

    def test_ids_are_stable_across_runs(self):
        text = "some stable content here"
        first = fixed_size_chunks("d1", text, size=10)
        second = fixed_size_chunks("d1", text, size=10)
        assert [c.id for c in first] == [c.id for c in second]

    def test_different_documents_get_different_ids(self):
        a = fixed_size_chunks("d1", "identical text", size=100)
        b = fixed_size_chunks("d2", "identical text", size=100)
        assert a[0].id != b[0].id

    def test_empty_document_yields_nothing(self):
        assert fixed_size_chunks("d1", "") == []

    def test_whitespace_only_window_is_dropped(self):
        chunks = fixed_size_chunks("d1", "abc" + " " * 50, size=10)
        assert all(c.text.strip() for c in chunks)

    @pytest.mark.parametrize(
        ("size", "overlap", "match"),
        [(0, 0, "size"), (10, -1, "overlap"), (10, 10, "smaller")],
    )
    def test_rejects_bad_parameters(self, size, overlap, match):
        with pytest.raises(ValueError, match=match):
            fixed_size_chunks("d1", "text", size=size, overlap=overlap)

    def test_splits_mid_word_on_purpose(self):
        chunks = fixed_size_chunks("d1", "thermalcontrol", size=7)
        assert [c.text for c in chunks] == ["thermal", "control"]


class TestNaiveRetriever:
    @pytest.fixture
    def retriever(self):
        docs = {
            "d1": "Spacecraft thermal control uses multilayer insulation blankets.",
            "d2": "Radiator sizing depends on the worst case hot attitude analysis.",
            "d3": "Attitude determination uses star trackers and sun sensors.",
        }
        return NaiveRetriever(chunk_documents(docs, size=500))

    def test_finds_the_obvious_match(self, retriever):
        top = retriever.retrieve("multilayer insulation", k=1)
        assert top[0].doc_id == "d1"

    def test_respects_k(self, retriever):
        assert len(retriever.retrieve("attitude", k=2)) <= 2

    def test_scores_are_attached_and_descending(self, retriever):
        results = retriever.retrieve("attitude radiator", k=3)
        scores = [c.score for c in results]
        assert all(s is not None for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_no_overlap_returns_nothing(self, retriever):
        assert retriever.retrieve("photosynthesis chlorophyll", k=5) == []

    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.retrieve("", k=5) == []

    def test_rejects_bad_k(self, retriever):
        with pytest.raises(ValueError, match="k must be"):
            retriever.retrieve("anything", k=0)

    def test_ties_are_broken_deterministically(self):
        # Two chunks with identical content must always rank in the same order,
        # or every before/after comparison becomes noise.
        chunks = [
            Chunk(id="c_zzz", doc_id="d1", text="identical text"),
            Chunk(id="c_aaa", doc_id="d2", text="identical text"),
        ]
        retriever = NaiveRetriever(chunks)
        first = [c.id for c in retriever.retrieve("identical text", k=2)]
        assert first == ["c_aaa", "c_zzz"]
        assert first == [c.id for c in retriever.retrieve("identical text", k=2)]

    def test_has_no_idf_and_this_is_the_known_weakness(self):
        """Stopwords are weighted like technical terms. Week 5 fixes this."""
        chunks = [
            Chunk(id="c1", doc_id="d1", text="the the the the the the the the"),
            Chunk(id="c2", doc_id="d2", text="cryogenic propellant boiloff mitigation"),
        ]
        retriever = NaiveRetriever(chunks)
        top = retriever.retrieve("the cryogenic", k=1)
        assert top[0].id == "c1"

    def test_cannot_match_synonyms(self):
        """Vocabulary mismatch looks identical to a corpus gap until attribution."""
        chunks = [Chunk(id="c1", doc_id="d1", text="heat rejection subsystem")]
        retriever = NaiveRetriever(chunks)
        assert retriever.retrieve("thermal control", k=1) == []


class TestAdapters:
    def test_wraps_a_plain_function(self):
        def my_retriever(query: str, k: int):
            return [Chunk(id="c1", doc_id="d1", text=query)][:k]

        wrapped = as_retriever(my_retriever)
        assert wrapped.retrieve("hello", 1)[0].text == "hello"

    def test_passes_through_an_object_with_retrieve(self):
        retriever = NaiveRetriever([Chunk(id="c1", doc_id="d1", text="x")])
        assert as_retriever(retriever) is retriever

    def test_rejects_something_unusable(self):
        with pytest.raises(TypeError, match="not usable as a retriever"):
            as_retriever(42)

    def test_tier_one_pipeline_collapses_the_ranking_stage(self):
        corpus_chunks = chunk_documents({"d1": "thermal control systems"}, size=500)
        pipeline = RetrieverPipeline(NaiveRetriever(corpus_chunks), k=5)
        result = pipeline.run("thermal control")

        assert not result.exposes_ranking_stage
        assert result.meta["tier"] == 1
        assert result.latency_ms is not None
        assert result.answer is None

    def test_pipeline_uses_a_generator_when_given_one(self):
        class EchoGenerator:
            def generate(self, query, context):
                return f"{query}|{len(context)}"

        chunks = chunk_documents({"d1": "thermal control systems"}, size=500)
        pipeline = RetrieverPipeline(
            NaiveRetriever(chunks), k=5, generator=EchoGenerator()
        )
        assert pipeline.run("thermal").answer == "thermal|1"

    def test_pipeline_rejects_bad_k(self):
        with pytest.raises(ValueError, match="k must be"):
            RetrieverPipeline(NaiveRetriever([]), k=0)


class TestEndToEnd:
    def test_naive_pipeline_over_a_corpus(self):
        docs = {
            "d1": "Spacecraft thermal control maintains equipment within limits.",
            "d2": "Radiator sizing is driven by the worst-case hot attitude.",
        }
        chunks = chunk_documents(docs, size=200)
        corpus = InMemoryCorpus(chunks, documents=docs)
        pipeline = RetrieverPipeline(NaiveRetriever(corpus.iter_chunks()), k=3)

        result = pipeline.run("radiator sizing")
        assert result.context
        assert result.context[0].doc_id == "d2"
        assert all(c.id in corpus for c in result.context)
