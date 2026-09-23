from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrace import (
    Chunk,
    EvalCase,
    NaiveRetriever,
    RetrievalResult,
    Span,
    Verdict,
    build_corpus,
    fixed_size_chunks,
    load_documents,
    read_fixture,
)
from retrace.errors import RetraceError
from retrace.judge import TokenRecallJudge
from retrace.runner import (
    RunAbortedError,
    as_pipeline,
    evaluate,
    load_run,
    run_from_dict,
    run_to_dict,
    save_run,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture(scope="module")
def sample():
    docs = load_documents(SAMPLE / "documents.json")
    corpus = build_corpus(docs, chunker=lambda d, t: fixed_size_chunks(d, t, size=300))
    return docs, corpus, read_fixture(SAMPLE / "golden.jsonl")


def cases(n=3):
    return [
        EvalCase(id=f"q{i}", question=f"question {i}", gold_spans=(Span("d1", 0, 50),))
        for i in range(n)
    ]


class TestEvaluateOnSample:
    def test_reproduces_the_week_one_baseline(self, sample):
        _, corpus, golden = sample
        run = evaluate(NaiveRetriever(corpus.iter_chunks()), golden, corpus=corpus, k=5)
        counts = run.verdict_counts
        # Matches examples/quickstart.py: 7 hits, 3 retrieval misses, 2 gaps.
        assert counts[Verdict.OK] == 7
        assert counts[Verdict.RETRIEVAL_MISS] == 3
        assert counts[Verdict.CORPUS_GAP] == 2
        assert counts[Verdict.RANKING_MISS] == 0
        assert run.tier == 1
        assert run.corpus_checked
        assert run.coverage_warning is not None
        assert len(run.traces) == len(golden)

    def test_without_corpus_gaps_become_retrieval_misses(self, sample):
        _, corpus, golden = sample
        run = evaluate(NaiveRetriever(corpus.iter_chunks()), golden, k=5)
        assert run.verdict_counts[Verdict.CORPUS_GAP] == 0
        assert run.verdict_counts[Verdict.RETRIEVAL_MISS] == 5
        assert not run.corpus_checked
        assert run.coverage_warning is None

    def test_progress_callback(self, sample):
        _, corpus, golden = sample
        seen = []
        evaluate(NaiveRetriever(corpus.iter_chunks()), golden, corpus=corpus, k=5,
                 progress=lambda i, n, a: seen.append((i, n, a.case_id)))
        assert [s[0] for s in seen] == list(range(1, len(golden) + 1))
        assert all(s[1] == len(golden) for s in seen)


class TestErrorHandling:
    def test_isolated_failures_become_unscorable(self):
        calls = {"n": 0}

        def flaky(query, k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("one malformed document")
            return [Chunk(id="c", doc_id="d1", text="t", span=Span("d1", 0, 50))]

        run = evaluate(flaky, cases(3))
        verdicts = [a.verdict for a in run.attributions]
        assert verdicts == [Verdict.OK, Verdict.UNSCORABLE, Verdict.OK]
        assert run.attributions[1].error == "ValueError: one malformed document"
        assert "q1" not in run.traces

    def test_systematic_failure_aborts_early(self):
        def broken(query, k):
            raise PermissionError("invalid API key")

        with pytest.raises(RunAbortedError, match="setup problem"):
            evaluate(broken, cases(10))

    def test_mixed_leading_errors_do_not_abort(self):
        errors = iter([ValueError("a"), KeyError("b"), ValueError("c")])

        def mixed(query, k):
            raise next(errors)

        run = evaluate(mixed, cases(3))
        assert all(a.verdict is Verdict.UNSCORABLE for a in run.attributions)

    def test_failures_after_a_success_never_abort(self):
        calls = {"n": 0}

        def degrades(query, k):
            calls["n"] += 1
            if calls["n"] == 1:
                return []
            raise TimeoutError("slow")

        run = evaluate(degrades, cases(5))
        assert sum(a.verdict is Verdict.UNSCORABLE for a in run.attributions) == 4


class TestTier:
    def test_tier_two_detected(self):
        class Reranking:
            def run(self, query):
                a = Chunk(id="a", doc_id="d1", text="a", span=Span("d1", 0, 50))
                b = Chunk(id="b", doc_id="d1", text="b", span=Span("d1", 100, 150))
                return RetrievalResult(query=query, candidates=(a, b), context=(b,))

        run = evaluate(Reranking(), cases(1))
        assert run.tier == 2
        assert run.attributions[0].verdict is Verdict.RANKING_MISS

    def test_as_pipeline_passes_pipelines_through(self):
        class P:
            def run(self, q):
                return None

        p = P()
        assert as_pipeline(p) is p


class TestPersistence:
    def test_round_trip(self, sample, tmp_path):
        _, corpus, golden = sample
        run = evaluate(NaiveRetriever(corpus.iter_chunks()), golden, corpus=corpus, k=5,
                       judge=TokenRecallJudge())
        path = save_run(run, tmp_path / "runs" / "run.json")
        loaded = load_run(path)

        assert [a.verdict for a in loaded.attributions] == [a.verdict for a in run.attributions]
        assert [a.candidate_rank for a in loaded.attributions] == [
            a.candidate_rank for a in run.attributions
        ]
        assert loaded.cases == run.cases
        assert loaded.tier == run.tier
        assert loaded.judge == "token_recall"
        assert loaded.coverage_warning == run.coverage_warning
        some = golden[0].id
        assert [c.id for c in loaded.traces[some].candidates] == [
            c.id for c in run.traces[some].candidates
        ]
        assert loaded.traces[some].candidates[0].span == run.traces[some].candidates[0].span

    def test_file_is_plain_json_with_verdict_totals(self, sample, tmp_path):
        _, corpus, golden = sample
        run = evaluate(NaiveRetriever(corpus.iter_chunks()), golden, corpus=corpus, k=5)
        data = json.loads(save_run(run, tmp_path / "r.json").read_text(encoding="utf-8"))
        assert data["retrace_run"] == 1
        assert data["verdicts"]["CORPUS_GAP"] == 2
        assert sum(data["verdicts"].values()) == len(golden)

    def test_long_chunk_text_is_truncated_in_traces(self):
        long_chunk = Chunk(id="c", doc_id="d1", text="z" * 5000, span=Span("d1", 0, 5000))
        run = evaluate(lambda q, k: [long_chunk], cases(1))
        text = run_to_dict(run)["traces"]["q0"]["candidates"][0]["text"]
        assert len(text) < 700 and text.endswith("…")

    def test_rejects_unknown_version(self):
        with pytest.raises(RetraceError, match="unsupported run file"):
            run_from_dict({"retrace_run": 99})

    def test_missing_file(self, tmp_path):
        with pytest.raises(RetraceError, match="not found"):
            load_run(tmp_path / "nope.json")
