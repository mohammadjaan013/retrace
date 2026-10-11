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


OCR_PASSAGE = (
    "In their simplest form, an electric probe\n(specifically, a Langmuir probe) "
    "consists of an insulated\n(perhaps with a small metaIIic sphere at the end) wire "
    "inserted into the plasma."
)


class TestMatchEvidence:
    def test_exact(self):
        from retrace.synthesis import match_evidence

        m = match_evidence(PASSAGE, "worst-case hot condition")
        assert (m.kind, m.similarity) == ("exact", 1.0)

    def test_normalised_whitespace_and_case(self):
        from retrace.synthesis import match_evidence

        m = match_evidence("sized by the\n   worst-case hot\ncondition", "Sized by the worst-case")
        assert m.kind == "normalised"

    def test_tolerates_an_ocr_correction(self):
        """Source says 'metaIIic' (OCR); the model quoted 'metallic'."""
        from retrace.synthesis import match_evidence

        quote = ("an electric probe (specifically, a Langmuir probe) consists of an "
                 "insulated (perhaps with a small metallic sphere at the end) wire")
        m = match_evidence(OCR_PASSAGE, quote)
        assert m is not None and m.kind == "fuzzy"
        assert m.similarity >= 0.95
        found = OCR_PASSAGE[m.start:m.end]
        assert found.startswith("an electric probe") and found.endswith("wire")
        assert "metaIIic" in found  # the span points at the real source text

    def test_rejects_a_paraphrase(self):
        from retrace.synthesis import match_evidence

        quote = "a Langmuir probe is basically an insulated wire with a little metal ball on it"
        assert match_evidence(OCR_PASSAGE, quote) is None

    def test_short_quotes_never_match_fuzzily(self):
        from retrace.synthesis import match_evidence

        assert match_evidence(OCR_PASSAGE, "metallic sphere") is None

    FOOTER_PASSAGE = (
        "Comparing the results, the limiting link is\n\nNASA/TM—2019-220203\n\n12\n\n"
        "the DSG to the lunar surface. Therefore the design is driven by it."
    )

    def test_skips_a_page_footer_inside_the_sentence(self):
        """The real NASA case: footer and page number interrupt the quoted sentence."""
        from retrace.synthesis import match_evidence

        quote = "the limiting link is\n\nthe DSG to the lunar surface."
        m = match_evidence(self.FOOTER_PASSAGE, quote)
        assert m is not None and m.kind == "gapped"
        found = self.FOOTER_PASSAGE[m.start:m.end]
        assert found.startswith("the limiting link is")
        assert found.endswith("lunar surface.")
        assert "NASA/TM" in found  # the span covers the footer; evidence stays inside it

    def test_gap_must_be_short(self):
        from retrace.synthesis import match_evidence

        long_gap = "the limiting link is " + "x" * 200 + " the DSG to the lunar surface."
        quote = "the limiting link is the DSG to the lunar surface."
        assert match_evidence(long_gap, quote) is None

    def test_gap_pieces_must_be_substantial(self):
        from retrace.synthesis import _match_with_gap

        # "the DSG" alone is too short to anchor a piece.
        assert _match_with_gap("the dsg | footer | to the lunar surface.",
                               "the dsg to the lunar surface.") is None

    def test_strict_locate_evidence_is_unchanged(self):
        assert locate_evidence(OCR_PASSAGE, "a small metallic sphere at the end") is None


class TestQuestionQuality:
    """Patterns taken from real candidates generated from the NASA corpus."""

    @pytest.mark.parametrize("question", [
        "What is the operating rate of the green laser mentioned in the passage?",
        "What is the main component of EPIC according to the passage?",
        "What is the maximum peak radiance value specified in the document?",
        "Which semiconductor material is used in the calculations described in the memorandum?",
        "What kind of propulsion system is highlighted in the passage?",
        "Which figure number corresponds to the simplified shunt regulation illustration?",
        "What figure caption mentions the MEMS-actuated wavefront controller?",
    ])
    def test_context_dependent(self, question):
        from retrace.synthesis import is_context_dependent

        assert is_context_dependent(question)

    @pytest.mark.parametrize("question", [
        "What sets the size of a spacecraft's heat-dumping panels?",
        "How many packets did Goldstone receive from the satellites when DTN was used?",
        "Which propellant shows lower specific mass below 4000 seconds of specific impulse?",
        "What does a Langmuir probe consist of?",
        "Which table salt substitute was tested?",  # 'table' mid-sentence is fine
    ])
    def test_ordinary_questions_pass(self, question):
        from retrace.synthesis import is_context_dependent

        assert not is_context_dependent(question)

    def test_answer_in_question(self):
        from retrace.synthesis import answer_in_question

        assert answer_in_question(
            "Which company is developing the 12 kW Aerojet thruster for the PPE?", "Aerojet")
        assert answer_in_question(
            "What pre-combustion pressure is specified for a BMEP of 200 psi?", "200 PSI")
        assert not answer_in_question("Which figure shows the regulator?", "Figure 6-5")
        assert not answer_in_question("What sets radiator size?", "the worst-case hot condition")
        assert not answer_in_question("Anything?", None)

    def test_context_dependent_candidates_are_rejected(self):
        bad = good(question="What drives radiator sizing according to the passage?")
        result = synthesize([chunk()], scripted(bad))
        assert result.rejections[0].reason == "context_dependent"

    def test_answer_leak_is_flagged_not_rejected(self):
        leak = good(question="Is the worst-case hot condition what sizes the panels?",
                    answer="the worst-case hot condition")
        (case,) = synthesize([chunk()], scripted(leak), max_overlap=None).cases
        assert case.labels["answer_in_question"] is True


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

    BIBLIOGRAPHY = (
        "18. Smith, J., \"Characteristics of Ka Band Networks,\" AIAA-94-0987-CP, 1993.\n"
        "19. Pike, C. P., et al., \"Spacecraft Charging Control,\" Journal of Spacecraft, "
        "vol. 12, pp. 417-429, 1975.\n"
        "20. Boyce, R., et al., \"Development of 12 kW Hall Thruster,\" Proceedings of the "
        "AIAA Propulsion and Energy Conference, 2020.\n"
        "21. Deutschmann, J., \"Comparison of Extended Kalman Filters,\" Journal of Guidance, "
        "vol. 5, no. 5, pp. 417-429, 1982.\n"
    )

    def test_detects_a_bibliography(self):
        from retrace.synthesis import citation_density, looks_like_references

        assert looks_like_references(self.BIBLIOGRAPHY)
        assert not looks_like_references(PASSAGE)
        assert citation_density("") == 0.0

    def test_prose_mentioning_one_citation_survives(self):
        from retrace.synthesis import looks_like_references

        prose = PASSAGE * 4 + " This follows the approach of Fischer et al. (1993)."
        assert not looks_like_references(prose)

    def test_reference_chunks_are_never_sampled(self):
        bib = Chunk(id="bib", doc_id="d2", text=self.BIBLIOGRAPHY)
        picked = sample_chunks([bib, chunk()], 5, min_chars=50)
        assert [c.id for c in picked] == ["c1"]
        kept = sample_chunks([bib, chunk()], 5, min_chars=50, skip_references=False)
        assert {c.id for c in kept} == {"bib", "c1"}


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
        # The full quote is kept so the rejection can be diagnosed later.
        assert result.rejections[0].evidence == "radiators are sized for peak heat"

    def test_ocr_corrected_evidence_is_kept_and_labelled(self):
        ocr = chunk(text=OCR_PASSAGE, cid="ocr")
        quote = ("an electric probe (specifically, a Langmuir probe) consists of an "
                 "insulated (perhaps with a small metallic sphere at the end) wire")
        result = synthesize([ocr], scripted(good(
            question="What does the simplest plasma diagnostic instrument look like?",
            evidence=quote, answer="An insulated wire, possibly with a small sphere.",
        )))
        (case,) = result.cases
        assert case.labels["evidence_match"] == "fuzzy"
        assert case.labels["evidence_similarity"] >= 0.95

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
