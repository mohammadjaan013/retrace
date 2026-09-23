from __future__ import annotations

import json

import pytest

from retrace import Chunk, EvalCase, Span, Verdict, evaluate
from retrace.chunking import sentence_chunks
from retrace.retrievers import (
    BM25Retriever,
    CallableEmbedder,
    CrossEncoderReranker,
    DenseRetriever,
    EmbeddingCache,
    HybridRetriever,
    RerankingPipeline,
    SentenceTransformerEmbedder,
    analyze,
)


def chunk(cid, text, doc="d1", start=0):
    return Chunk(id=cid, doc_id=doc, text=text, span=Span(doc, start, start + len(text)))


CORPUS = [
    chunk("thermal", "Radiators reject waste heat to deep space as infrared radiation."),
    chunk("power", "Batteries carry the spacecraft through eclipse; arrays degrade over time."),
    chunk("adcs", "Star trackers give accurate attitude but can be blinded by the Sun."),
    chunk("noise", "The the the of of a an is is the of."),
]


class TestAnalyze:
    def test_drops_stopwords_and_folds_plurals(self):
        assert analyze("What are the star trackers?") == ["star", "tracker"]


class TestBM25:
    def test_rare_term_wins(self):
        top = BM25Retriever(CORPUS).retrieve("what blinds star trackers", k=1)
        assert top[0].id == "adcs"

    def test_stopwords_alone_match_nothing(self):
        # The naive baseline's core weakness: BM25 must not reward "the of is".
        assert BM25Retriever(CORPUS).retrieve("the of is", k=5) == []

    def test_only_matching_chunks_are_returned(self):
        results = BM25Retriever(CORPUS).retrieve("batteries eclipse", k=10)
        assert [c.id for c in results] == ["power"]

    def test_scores_descend_and_ties_are_deterministic(self):
        twins = [chunk("b", "radiator"), chunk("a", "radiator")]
        results = BM25Retriever(twins).retrieve("radiator", k=2)
        assert [c.id for c in results] == ["a", "b"]

    def test_rejects_bad_k(self):
        with pytest.raises(ValueError):
            BM25Retriever(CORPUS).retrieve("x", k=0)

    def test_empty_corpus(self):
        assert BM25Retriever([]).retrieve("anything", k=3) == []


def keyword_embedder(texts):
    """A fake embedding model with a tiny, fixed concept vocabulary."""
    concepts = [
        ("heat", "warmth", "thermal", "radiator"),
        ("battery", "batteries", "power", "eclipse"),
        ("star", "tracker", "trackers", "attitude"),
    ]
    out = []
    for text in texts:
        lower = text.lower()
        out.append([float(sum(word in lower for word in group)) for group in concepts] + [0.01])
    return out


class TestDense:
    def test_bridges_a_vocabulary_mismatch_lexical_search_cannot(self):
        query = "how do you get rid of warmth"
        assert BM25Retriever(CORPUS).retrieve(query, k=5) == []
        top = DenseRetriever(CORPUS, keyword_embedder).retrieve(query, k=1)
        assert top[0].id == "thermal"

    def test_accepts_an_embedder_object(self):
        emb = CallableEmbedder(keyword_embedder, name="kw")
        assert DenseRetriever(CORPUS, emb).retrieve("battery", k=1)[0].id == "power"

    def test_sentence_transformer_adapter(self):
        class FakeST:
            def encode(self, texts, batch_size, normalize_embeddings):
                assert normalize_embeddings is True
                return keyword_embedder(texts)

        emb = SentenceTransformerEmbedder("fake", model=FakeST())
        assert emb.name == "st:fake"
        assert DenseRetriever(CORPUS, emb).retrieve("star tracker", k=1)[0].id == "adcs"


class TestEmbeddingCache:
    def test_second_pass_is_free(self):
        calls = []

        def counting(texts):
            calls.append(len(texts))
            return keyword_embedder(texts)

        cache = EmbeddingCache()
        emb = CallableEmbedder(counting, name="counting")
        cache.get_many(emb, ["a", "b", "c"])
        cache.get_many(emb, ["a", "b", "c", "d"])
        assert calls == [3, 1]
        assert (cache.hits, cache.misses) == (3, 4)

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "vectors.jsonl"
        emb = CallableEmbedder(keyword_embedder, name="kw")
        EmbeddingCache(path).get_many(emb, ["heat", "battery"])

        fresh = EmbeddingCache(path)
        fresh.get_many(emb, ["heat", "battery"])
        assert fresh.misses == 0
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    def test_model_name_is_part_of_the_key(self):
        cache = EmbeddingCache()
        cache.get_many(CallableEmbedder(keyword_embedder, name="a"), ["x"])
        cache.get_many(CallableEmbedder(keyword_embedder, name="b"), ["x"])
        assert cache.misses == 2

    def test_rejects_a_short_response(self):
        with pytest.raises(ValueError, match="returned 1 vectors for 2"):
            EmbeddingCache().get_many(CallableEmbedder(lambda t: [[1.0]], name="bad"), ["a", "b"])


class TestHybrid:
    def test_records_which_leg_found_each_chunk(self):
        hybrid = HybridRetriever({
            "bm25": BM25Retriever(CORPUS),
            "dense": DenseRetriever(CORPUS, keyword_embedder),
        })
        results = hybrid.retrieve("how do you get rid of warmth", k=2)
        thermal = next(c for c in results if c.id == "thermal")
        assert thermal.meta["legs"] == {"bm25": None, "dense": 1}

    def test_agreement_between_legs_ranks_first(self):
        hybrid = HybridRetriever({
            "bm25": BM25Retriever(CORPUS),
            "dense": DenseRetriever(CORPUS, keyword_embedder),
        })
        top = hybrid.retrieve("star trackers attitude", k=1)[0]
        assert top.id == "adcs"
        assert top.meta["legs"] == {"bm25": 1, "dense": 1}

    def test_needs_a_leg(self):
        with pytest.raises(ValueError):
            HybridRetriever({})


class TestReranking:
    def test_pipeline_is_tier_two(self):
        pipe = RerankingPipeline(BM25Retriever(CORPUS), candidates_k=3, context_k=1)
        result = pipe.run("radiators batteries")
        assert result.exposes_ranking_stage
        assert len(result.context) == 1
        assert result.meta["tier"] == 2

    def test_cross_encoder_reorders(self):
        class FakeCE:
            def predict(self, pairs):
                return [1.0 if "Sun" in text else 0.0 for _, text in pairs]

        reranked = CrossEncoderReranker("fake", model=FakeCE()).rerank("q", CORPUS[:3])
        assert reranked[0].id == "adcs"
        assert reranked[0].meta["reranked_from"] == 3

    def test_empty_candidates(self):
        assert CrossEncoderReranker("fake", model=object()).rerank("q", []) == []

    def test_rejects_bad_sizes(self):
        with pytest.raises(ValueError):
            RerankingPipeline(BM25Retriever(CORPUS), candidates_k=3, context_k=5)

    def test_ranking_miss_becomes_observable(self):
        """Gold retrieved at rank 2, context keeps 1: the cascade names the stage."""
        class Worst:
            name = "worst"

            def rerank(self, query, chunks):
                return list(reversed(chunks))

        docs = [chunk("a", "radiator radiator radiator"), chunk("b", "radiator", start=100)]
        case = EvalCase(id="q", question="radiator", gold_spans=(Span("d1", 0, 26),))
        pipe = RerankingPipeline(BM25Retriever(docs), reranker=Worst(), candidates_k=2, context_k=1)
        (a,) = evaluate(pipe, [case]).attributions
        assert a.verdict is Verdict.RANKING_MISS
        assert a.candidate_rank == 1


class TestSentenceChunks:
    TEXT = (
        "First sentence is here. Second one follows it. Third is short.\n\n"
        "A new paragraph starts. It has two sentences."
    )

    def test_spans_are_exact(self):
        for c in sentence_chunks("d", self.TEXT, max_chars=50):
            assert self.TEXT[c.span.start:c.span.end] == c.text

    def test_never_crosses_a_paragraph(self):
        for c in sentence_chunks("d", self.TEXT, max_chars=1000):
            assert "\n\n" not in c.text

    def test_never_splits_a_sentence(self):
        for c in sentence_chunks("d", self.TEXT, max_chars=10):
            assert c.text.rstrip().endswith((".", "!", "?"))

    def test_overlap_repeats_the_last_sentence(self):
        chunks = sentence_chunks("d", self.TEXT, max_chars=50, overlap_sentences=1)
        assert chunks[0].text.endswith("Second one follows it.")
        assert chunks[1].text.startswith("Second one follows it.")

    def test_no_chunk_is_a_subset_of_another(self):
        long = "Short one. " + "A very long sentence " * 20 + "ends here. Tail."
        chunks = sentence_chunks("d", long, max_chars=60)
        for a in chunks:
            for b in chunks:
                if a is not b:
                    assert not (b.span.start <= a.span.start and a.span.end <= b.span.end)

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            sentence_chunks("d", "x", max_chars=0)
        with pytest.raises(ValueError):
            sentence_chunks("d", "x", overlap_sentences=-1)


def test_trace_legs_render_in_the_report():
    from retrace.report import render_report

    hybrid = HybridRetriever({"bm25": BM25Retriever(CORPUS),
                              "dense": DenseRetriever(CORPUS, keyword_embedder)})
    case = EvalCase(id="q", question="get rid of warmth",
                    gold_spans=(Span("d1", 0, 30),))
    page = render_report(evaluate(hybrid, [case], k=3))
    assert "dense #1" in page
    assert json.dumps("bm25")[1:-1] in page
