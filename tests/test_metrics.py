from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from retrace import (
    Chunk,
    EvalCase,
    NaiveRetriever,
    Provenance,
    RetrievalResult,
    Span,
    Verdict,
    build_corpus,
    evaluate,
    fixed_size_chunks,
    load_documents,
    read_fixture,
)
from retrace.cli import main
from retrace.metrics import compute_metrics, fix_first, percentile
from retrace.summary import print_summary
from retrace.text import content_terms, fold_plural

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture(scope="module")
def sample_run():
    docs = load_documents(SAMPLE / "documents.json")
    corpus = build_corpus(docs, chunker=lambda d, t: fixed_size_chunks(d, t, size=300))
    golden = read_fixture(SAMPLE / "golden.jsonl")
    return evaluate(NaiveRetriever(corpus.iter_chunks()), golden, corpus=corpus, k=5)


class TestFoldPlural:
    @pytest.mark.parametrize(
        ("word", "folded"),
        [("trackers", "tracker"), ("batteries", "battery"), ("radiators", "radiator"),
         ("class", "class"), ("bus", "bus"), ("analysis", "analysis"), ("gas", "gas"),
         ("its", "its"), ("heat", "heat")],
    )
    def test_cases(self, word, folded):
        assert fold_plural(word) == folded

    def test_content_terms_fold_and_drop_stopwords(self):
        assert content_terms("What are the star trackers?") == {"star", "tracker"}


class TestPercentile:
    def test_nearest_rank(self):
        assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
        assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10

    def test_empty(self):
        assert percentile([], 50) is None


class TestComputeMetrics:
    def test_sample_baseline(self, sample_run):
        m = compute_metrics(sample_run)
        assert m.cases == 12
        assert m.answerable == 10  # the 2 corpus gaps are excluded
        assert m.hit_rate[1] == 0.6
        assert m.hit_rate[5] == 0.7
        assert m.mrr == 0.65
        assert m.verdicts["CORPUS_GAP"] == 2
        assert m.errors == 0

    def test_every_baseline_miss_is_recoverable_by_keyword_search(self, sample_run):
        """The Week 5 finding: the naive baseline loses to BM25 on all three.

        An earlier word-overlap heuristic called two of these vocabulary
        mismatches. The before/after experiment falsified it; the probe
        measures the counterfactual directly.
        """
        m = compute_metrics(sample_run)
        assert m.findable_but_missed == 3
        assert m.vocabulary_mismatch == 0
        ranks = {a.case_id: a.lexical_probe_rank for a in sample_run.attributions
                 if a.verdict is Verdict.RETRIEVAL_MISS}
        assert all(r is not None and r <= 5 for r in ranks.values())

    def test_genuine_vocabulary_mismatch_is_missed_by_the_probe_too(self):
        docs = {"d1": "Radiators reject heat to deep space as infrared.",
                "d2": "Batteries store energy for eclipse periods."}
        corpus = build_corpus(docs, chunker=lambda d, t: fixed_size_chunks(d, t, size=500))
        case = EvalCase(id="q", question="How do you get rid of warmth?",
                        gold_spans=(Span("d1", 0, 48),))
        run = evaluate(lambda q, k: [], [case], corpus=corpus)
        (a,) = run.attributions
        assert a.verdict is Verdict.RETRIEVAL_MISS
        assert a.lexical_probe_rank is None
        assert compute_metrics(run).vocabulary_mismatch == 1
        assert any("vocabulary mismatch" in n for n in a.notes)

    def test_probe_can_be_disabled(self, sample_run):
        docs = load_documents(SAMPLE / "documents.json")
        corpus = build_corpus(docs, chunker=lambda d, t: fixed_size_chunks(d, t, size=300))
        run = evaluate(NaiveRetriever(corpus.iter_chunks()), read_fixture(SAMPLE / "golden.jsonl"),
                       corpus=corpus, k=5, lexical_probe=False)
        assert all(a.lexical_probe_k is None for a in run.attributions)

    def test_ks_never_exceed_candidates_returned(self, sample_run):
        assert max(compute_metrics(sample_run).hit_rate) == 5

    def test_trusted_only_filter(self):
        gold = Chunk(id="g", doc_id="d1", text="g", span=Span("d1", 0, 50))
        cases = [
            EvalCase(id="t", question="q", gold_spans=(Span("d1", 0, 50),),
                     provenance=Provenance.HUMAN),
            EvalCase(id="u", question="q", gold_spans=(Span("d1", 100, 150),),
                     provenance=Provenance.MODEL),
        ]
        run = evaluate(lambda q, k: [gold], cases)
        assert compute_metrics(run).cases == 2
        trusted = compute_metrics(run, trusted_only=True)
        assert trusted.cases == 1
        assert trusted.hit_rate[1] == 1.0

    def test_precision_counts_every_gold_chunk(self):
        g1 = Chunk(id="g1", doc_id="d1", text="a", span=Span("d1", 0, 60))
        g2 = Chunk(id="g2", doc_id="d1", text="b", span=Span("d1", 40, 100))
        noise = Chunk(id="n", doc_id="d1", text="c", span=Span("d1", 500, 560))
        case = EvalCase(id="q", question="q", gold_spans=(Span("d1", 10, 90),))
        run = evaluate(lambda q, k: [g1, noise, g2], [case])
        m = compute_metrics(run)
        assert run.attributions[0].gold_ranks == (1, 3)
        assert m.precision[1] == 1.0
        assert m.precision[3] == round(2 / 3, 4)

    def test_generation_accuracy(self):
        class Judge:
            name = "j"

            def judge(self, q, e, a, c):
                from retrace import Judgement

                return Judgement(a == "right")

        gold = Chunk(id="g", doc_id="d1", text="g", span=Span("d1", 0, 50))

        class P:
            answers = iter(["right", "wrong"])

            def run(self, q):
                return RetrievalResult(q, (gold,), (gold,), answer=next(self.answers))

        cases = [EvalCase(id=f"q{i}", question="q", answer="x",
                          gold_spans=(Span("d1", 0, 50),)) for i in range(2)]
        m = compute_metrics(evaluate(P(), cases, judge=Judge()))
        assert m.generation_evaluated == 2
        assert m.generation_accuracy == 0.5
        assert m.verdicts["GENERATION_MISS"] == 1


class TestFixFirst:
    def test_largest_bucket_first(self, sample_run):
        recs = fix_first(sample_run)
        assert [r.verdict for r in recs] == [Verdict.RETRIEVAL_MISS, Verdict.CORPUS_GAP]
        assert recs[0].count == 3 and recs[0].share == 0.6
        assert recs[1].case_ids == ("q_hall_thruster", "q_hohmann")

    def test_retrieval_advice_leads_with_the_dominant_subtype(self, sample_run):
        rec = fix_first(sample_run)[0]
        assert "3 found by a reference keyword retriever" in rec.detail
        # Losing to keyword search: lead with the lexical fixes, not embeddings.
        assert rec.actions[0].startswith("Check chunking first")

    def test_detail_is_plain_ascii(self, sample_run):
        for rec in fix_first(sample_run):
            rec.detail.encode("ascii")

    def test_no_failures(self):
        gold = Chunk(id="g", doc_id="d1", text="g", span=Span("d1", 0, 50))
        run = evaluate(lambda q, k: [gold],
                       [EvalCase(id="q", question="q", gold_spans=(Span("d1", 0, 50),))])
        assert fix_first(run) == []


class TestSummary:
    def test_sections(self, sample_run):
        buf = io.StringIO()
        print_summary(sample_run, buf)
        text = buf.getvalue()
        assert "VERDICTS  12 cases" in text
        assert "not observable (tier 1 pipeline)" in text
        assert "2 corpus gaps excluded" in text
        assert "FIX FIRST" in text
        assert "k=5 returns 31%" in text

    def test_falls_back_to_ascii_on_legacy_consoles(self, sample_run):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        print_summary(sample_run, stream)  # must not raise UnicodeEncodeError
        stream.flush()
        assert b"#" in raw.getvalue()

    def test_warns_when_corpus_missing_and_cases_unverified(self):
        case = EvalCase(id="q", question="q", gold_spans=(Span("d1", 0, 50),),
                        provenance=Provenance.MODEL)
        run = evaluate(lambda q, k: [], [case])
        buf = io.StringIO()
        print_summary(run, buf)
        text = buf.getvalue()
        assert "No corpus supplied" in text
        assert "unverified model output" in text


class TestRunCommand:
    def test_demo_writes_a_run_file(self, tmp_path, capsys):
        out = tmp_path / "run.json"
        code = main([
            "run", "--demo", "--docs", str(SAMPLE / "documents.json"),
            "--fixture", str(SAMPLE / "golden.jsonl"),
            "--chunk-size", "300", "--k", "5", "--out", str(out),
        ])
        assert code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["verdicts"]["RETRIEVAL_MISS"] == 3
        assert data["meta"]["k"] == 5
        assert "FIX FIRST" in capsys.readouterr().out

    def test_pipeline_factory_returning_pipeline_and_corpus(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "my_rag.py").write_text(
            "from retrace import NaiveRetriever, build_corpus, load_documents, fixed_size_chunks\n"
            f"DOCS = load_documents(r'{SAMPLE / 'documents.json'}')\n"
            "def build():\n"
            "    chunker = lambda d, t: fixed_size_chunks(d, t, size=300)\n"
            "    corpus = build_corpus(DOCS, chunker=chunker)\n"
            "    return NaiveRetriever(corpus.iter_chunks()), corpus\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        code = main(["run", "--pipeline", "my_rag:build()", "--fixture",
                     str(SAMPLE / "golden.jsonl"), "--k", "5", "--out", "r.json",
                     "--judge", "none"])
        assert code == 0
        data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        assert data["corpus_checked"] is True
        assert data["verdicts"]["CORPUS_GAP"] == 2
        assert data["judge"] is None

    def test_demo_requires_docs(self, tmp_path, capsys):
        code = main(["run", "--demo", "--fixture", str(SAMPLE / "golden.jsonl"),
                     "--out", str(tmp_path / "r.json")])
        assert code == 2
        assert "--demo needs --docs" in capsys.readouterr().err

    def test_needs_a_target(self, tmp_path, capsys):
        code = main(["run", "--fixture", str(SAMPLE / "golden.jsonl"),
                     "--out", str(tmp_path / "r.json")])
        assert code == 2
        assert "--pipeline" in capsys.readouterr().err
