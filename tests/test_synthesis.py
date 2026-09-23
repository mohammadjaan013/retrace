from __future__ import annotations

import pytest

from retrace import Chunk, Provenance, Span, chunk_documents, resolve_gold_chunks
from retrace.corpus import InMemoryCorpus
from retrace.errors import LLMRefusalError
from retrace.synthesis import (
    lexical_overlap,
    locate_evidence,
    sample_chunks,
    synthesize,
)

PASSAGE = (
    "Radiator area is almost always sized by the worst-case hot condition rather "
    "than the average, because a design that survives the hottest attitude will "
    "generally tolerate the cold case."
)


def scripted(*responses):
    """An LLM that returns the given dicts in order (or raises them if exceptions)."""
    it = iter(responses)

    def fn(system, prompt, schema):
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    return fn


def good(question="What sets the size of a spacecraft's heat-dumping panels?",
         evidence="sized by the worst-case hot condition rather than the average",
         answer="The worst-case hot condition."):
    return {"answerable": True, "question": question, "answer": answer, "evidence": evidence}


def chunk(text=PASSAGE, start=1000, doc_id="d1", cid="c1"):
    return Chunk(id=cid, doc_id=doc_id, text=text, span=Span(doc_id, start, start + len(text)))


class TestLocateEvidence:
    def test_exact(self):
        start, end = locate_evidence(PASSAGE, "worst-case hot condition")
        assert PASSAGE[start:end] == "worst-case hot condition"

    def test_tolerates_collapsed_whitespace(self):
        passage = "sized by the\n   worst-case hot\ncondition"
        start, end = locate_evidence(passage, "the worst-case hot condition")
        assert passage[start:end] == "the\n   worst-case hot\ncondition"

    def test_tolerates_case(self):
        start, end = locate_evidence(PASSAGE, "radiator area is almost always sized")
        assert PASSAGE[start:end] == "Radiator area is almost always sized"

    def test_rejects_paraphrase(self):
        assert locate_evidence(PASSAGE, "radiators are sized for the hottest case") is None

    def test_rejects_empty(self):
        assert locate_evidence(PASSAGE, "   ") is None


class TestLexicalOverlap:
    def test_copied_question_scores_high(self):
        q = "Is radiator area sized by the worst-case hot condition?"
        assert lexical_overlap(q, PASSAGE) == 1.0

    def test_paraphrased_question_scores_low(self):
        q = "What sets the size of heat-dumping panels on a spacecraft?"
        assert lexical_overlap(q, PASSAGE) < 0.3

    def test_stopwords_are_ignored(self):
        assert lexical_overlap("what is the", PASSAGE) == 0.0


class TestSampleChunks:
    def _chunks(self):
        docs = {f"d{i}": ("x" * 50 + " ") * 30 for i in range(4)}
        docs["huge"] = ("y" * 50 + " ") * 400
        return chunk_documents(docs, size=300)

    def test_is_deterministic(self):
        a = [c.id for c in sample_chunks(self._chunks(), 8, seed=3)]
        b = [c.id for c in sample_chunks(list(reversed(self._chunks())), 8, seed=3)]
        assert a == b

    def test_spreads_across_documents(self):
        picked = sample_chunks(self._chunks(), 5, seed=0)
        # One huge document must not supply everything.
        assert len({c.doc_id for c in picked}) == 5

    def test_skips_short_chunks(self):
        chunks = [Chunk(id="s", doc_id="d", text="tiny"), chunk()]
        assert [c.id for c in sample_chunks(chunks, 5, min_chars=50)] == ["c1"]

    def test_returns_fewer_when_corpus_is_small(self):
        assert len(sample_chunks([chunk()], 10, min_chars=50)) == 1

    def test_rejects_bad_n(self):
        with pytest.raises(ValueError):
            sample_chunks([chunk()], 0)


class TestSynthesize:
    def test_produces_a_grounded_untrusted_case(self):
        result = synthesize([chunk()], scripted(good()))
        assert not result.rejections
        (case,) = result.cases
        assert case.provenance is Provenance.MODEL
        assert not case.is_trusted
        assert case.source_chunk_id == "c1"
        assert case.gold_doc_ids == ("d1",)
        # The span is in *document* coordinates, offset by the chunk's start.
        (span,) = case.gold_spans
        local = PASSAGE.find("sized by the worst-case")
        assert span.start == 1000 + local
        assert span.length == len("sized by the worst-case hot condition rather than the average")

    def test_span_resolves_against_a_different_chunking(self):
        doc = "Preamble. " * 40 + PASSAGE + " Postscript." * 40
        coarse = chunk_documents({"d1": doc}, size=900)
        passage_chunk = next(c for c in coarse if "worst-case hot" in c.text)
        result = synthesize([passage_chunk], scripted(good()))
        (case,) = result.cases

        fine = InMemoryCorpus(chunk_documents({"d1": doc}, size=120), documents={"d1": doc})
        gold = resolve_gold_chunks(case, fine)
        assert gold
        assert any("worst-case" in fine.get(g).text for g in gold)

    def test_hallucinated_evidence_is_rejected(self):
        result = synthesize([chunk()], scripted(good(evidence="radiators are sized for peak heat")))
        assert not result.cases
        assert result.rejections[0].reason == "evidence_not_verbatim"

    def test_leaky_question_is_rejected(self):
        # "radiator area" is outside the evidence quote but inside the passage --
        # still leakage, because retrieval matches the whole chunk.
        leaky = good(question="Is radiator area sized by the worst-case hot condition?")
        result = synthesize([chunk()], scripted(leaky))
        assert result.rejections[0].reason == "lexical_leakage"

    def test_leakage_filter_can_be_disabled(self):
        leaky = good(question="Is radiator area sized by the worst-case hot condition?")
        result = synthesize([chunk()], scripted(leaky), max_overlap=None)
        assert result.cases[0].labels["lexical_overlap"] == 1.0

    def test_unanswerable_passage(self):
        result = synthesize([chunk()], scripted({"answerable": False, "question": "",
                                                 "answer": "", "evidence": ""}))
        assert result.rejections[0].reason == "not_question_worthy"

    def test_duplicate_questions_are_rejected(self):
        c1, c2 = chunk(cid="c1"), chunk(cid="c2", start=5000)
        result = synthesize([c1, c2], scripted(good(), good()))
        assert len(result.cases) == 1
        assert result.rejections[0].reason == "duplicate_question"

    def test_model_refusal_becomes_a_rejection_and_the_run_continues(self):
        c1, c2 = chunk(cid="c1"), chunk(cid="c2", start=5000)
        result = synthesize(
            [c1, c2],
            scripted(LLMRefusalError("declined"), good()),
        )
        assert len(result.cases) == 1
        assert result.rejections[0].reason == "model_error"

    def test_non_model_errors_propagate(self):
        with pytest.raises(ConnectionError):
            synthesize([chunk()], scripted(ConnectionError("network down")))

    def test_chunk_without_span_falls_back_to_document_label(self):
        bare = Chunk(id="c9", doc_id="d9", text=PASSAGE)
        (case,) = synthesize([bare], scripted(good())).cases
        assert case.gold_spans == ()
        assert case.gold_doc_ids == ("d9",)
        assert case.gold_chunk_ids == ("c9",)
        assert case.labels["span_unavailable"] is True

    def test_rejection_counts_are_sorted(self):
        chunks = [chunk(cid=f"c{i}", start=i * 1000) for i in range(3)]
        bad = good(evidence="not in the passage at all")
        result = synthesize(chunks, scripted(bad, bad, {"answerable": False, "question": "",
                                                        "answer": "", "evidence": ""}))
        assert list(result.rejection_counts.items()) == [
            ("evidence_not_verbatim", 2),
            ("not_question_worthy", 1),
        ]

    def test_ids_are_stable(self):
        a = synthesize([chunk()], scripted(good())).cases[0].id
        b = synthesize([chunk()], scripted(good())).cases[0].id
        assert a == b and a.startswith("q_")
