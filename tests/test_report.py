from __future__ import annotations

import json
import re

import pytest

from retrace import SAMPLE_DIR as SAMPLE
from retrace import (
    Chunk,
    EvalCase,
    NaiveRetriever,
    Provenance,
    RetrievalResult,
    Span,
    build_corpus,
    evaluate,
    fixed_size_chunks,
    load_documents,
    read_fixture,
    save_run,
)
from retrace.cli import main
from retrace.report import render_report, write_report


@pytest.fixture(scope="module")
def sample_run():
    docs = load_documents(SAMPLE / "documents.json")
    corpus = build_corpus(docs, chunker=lambda d, t: fixed_size_chunks(d, t, size=300))
    golden = read_fixture(SAMPLE / "golden.jsonl")
    return evaluate(NaiveRetriever(corpus.iter_chunks()), golden, corpus=corpus, k=5)


@pytest.fixture(scope="module")
def html(sample_run):
    return render_report(sample_run)


def embedded_run(page: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="retrace-run">(.*?)</script>', page, re.S
    )
    assert match, "run JSON not embedded"
    return json.loads(match.group(1))


class TestContent:
    def test_is_a_complete_document(self, html):
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
        assert "<title>Retrace report" in html

    def test_leads_with_the_failure_summary(self, html):
        assert "5 of 12 questions failed: 3 retrieval misses and 2 corpus gaps." in html
        assert "cannot be fixed by any retrieval change" in html
        # The headline comes before the fix list, which comes before the cases.
        assert html.index('class="lede"') < html.index("Fix first") < html.index("Every question")

    def test_fix_first_links_to_cases(self, html):
        assert 'href="#case-q_hall_thruster"' in html
        assert 'id="case-q_hall_thruster"' in html

    def test_unobservable_stages_are_labelled_not_zeroed(self, html):
        assert "not observed" in html
        assert "tier 1 pipeline" in html

    def test_metrics_exclude_corpus_gaps_and_say_so(self, html):
        assert "Over 10 answerable questions" in html
        assert "2 corpus gaps excluded" in html

    def test_gold_evidence_is_highlighted_in_the_trace(self, html):
        assert "<mark>" in html
        assert 'class="gold"' in html

    def test_tier_and_coverage_warnings(self, html):
        assert "Before you quote these numbers" in html
        assert "k=5 returns 31%" in html
        assert "cannot be told apart" in html

    def test_every_case_is_present(self, html, sample_run):
        for case in sample_run.cases:
            assert f'id="case-{case.id}"' in html


class TestSelfContained:
    def test_no_external_resources(self, html):
        assert not re.search(r"""(src|href)=["']https?://""", html)
        assert "@import" not in html
        assert "fonts.googleapis" not in html

    def test_embeds_the_full_run(self, html, sample_run):
        data = embedded_run(html)
        assert data["retrace_run"] == 1
        assert len(data["attributions"]) == len(sample_run.attributions)
        assert data["verdicts"]["CORPUS_GAP"] == 2

    def test_supports_both_themes(self, html):
        assert "prefers-color-scheme: dark" in html
        assert ':root[data-theme="dark"]' in html


class TestEscaping:
    """User documents are third-party input. The report must never execute them."""

    @pytest.fixture
    def hostile(self):
        payload = '<script>alert("x")</script><img src=x onerror=alert(1)>'
        chunk = Chunk(id="c1", doc_id="d1", text=f"evidence {payload}", span=Span("d1", 0, 60))
        case = EvalCase(
            id="q1", question=f"What is {payload}?", answer=payload,
            gold_spans=(Span("d1", 0, 60),), provenance=Provenance.HUMAN,
        )

        class P:
            def run(self, q):
                return RetrievalResult(q, (chunk,), (chunk,), answer=payload)

        return render_report(evaluate(P(), [case]))

    def test_markup_in_questions_and_chunks_is_escaped(self, hostile):
        body = hostile.split('<script type="application/json"')[0]
        assert "<img src=x" not in body
        assert '<script>alert("x")' not in body
        assert "&lt;script&gt;" in body

    def test_embedded_json_contains_no_markup_at_all(self, hostile):
        start = hostile.index('id="retrace-run">') + len('id="retrace-run">')
        payload = hostile[start:hostile.index("</script>", start)]
        assert "<" not in payload and ">" not in payload and "&" not in payload
        # ...and the original text survives the round trip exactly.
        assert embedded_run(hostile)["cases"][0]["question"].startswith("What is <script>")

    def test_comment_opener_in_data_cannot_derail_the_parser(self):
        from retrace.report import _json_island

        island = _json_island({"q": "<!--<script>"})
        assert "<" not in island
        assert json.loads(island) == {"q": "<!--<script>"}

    def test_only_two_script_elements(self, hostile):
        assert hostile.count("<script") == 2  # the JSON island and the filter


def test_empty_trace_and_no_failures():
    gold = Chunk(id="g", doc_id="d1", text="g" * 60, span=Span("d1", 0, 60))
    run = evaluate(lambda q, k: [gold],
                   [EvalCase(id="q", question="q?", gold_spans=(Span("d1", 0, 60),))])
    page = render_report(run)
    assert "All 1 questions passed." in page
    assert "Fix first" not in page


def test_write_report_and_cli(tmp_path, sample_run, capsys):
    path = write_report(sample_run, tmp_path / "out" / "r.html")
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")

    run_file = save_run(sample_run, tmp_path / "run.json")
    assert main(["report", str(run_file)]) == 0
    assert (tmp_path / "run.html").exists()
    assert "report ->" in capsys.readouterr().out


def test_run_with_report_flag(tmp_path, capsys):
    out = tmp_path / "run.json"
    code = main([
        "run", "--demo", "--docs", str(SAMPLE / "documents.json"),
        "--fixture", str(SAMPLE / "golden.jsonl"), "--chunk-size", "300", "--k", "5",
        "--out", str(out), "--report",
    ])
    assert code == 0
    assert (tmp_path / "run.html").exists()


def test_numeric_headers_align_with_their_columns(html):
    # Found from a rendered screenshot: headers were left-aligned over
    # right-aligned numbers, so every metrics column read as misaligned.
    assert "th.num" in html
