from __future__ import annotations

import json
import sys

import pytest

from retrace import read_fixture
from retrace.cli import load_object, main
from retrace.corpus import build_corpus, load_documents
from retrace.errors import CorpusError, RetraceError

DOCS = {
    "d1": (
        "Radiator area is almost always sized by the worst-case hot condition rather "
        "than the average, because a design that survives the hottest attitude will "
        "generally tolerate the cold case with modest heater power. " * 3
    ),
}

FAKE_LLM_MODULE = '''
def llm(system, prompt, schema):
    return {
        "answerable": True,
        "question": "What sets the size of a spacecraft's heat-dumping panels?",
        "answer": "The worst-case hot condition.",
        "evidence": "sized by the worst-case hot condition rather than the average",
    }
'''


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "docs.json").write_text(json.dumps(DOCS), encoding="utf-8")
    (tmp_path / "fake_llm.py").write_text(FAKE_LLM_MODULE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fake_llm", None)
    return tmp_path


class TestLoadDocuments:
    def test_json_mapping(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"a": "one", "b": "two"}), encoding="utf-8")
        assert load_documents(path) == {"a": "one", "b": "two"}

    def test_jsonl_records(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            '{"id": "a", "text": "one"}\n\n{"id": "b", "text": "two"}\n', encoding="utf-8"
        )
        assert load_documents(path) == {"a": "one", "b": "two"}

    def test_preserves_text_exactly(self, tmp_path):
        # Spans are offsets; line endings and whitespace must survive untouched.
        path = tmp_path / "d.jsonl"
        text = "line one\r\nline  two\n\tindented"
        path.write_text(json.dumps({"id": "a", "text": text}) + "\n", encoding="utf-8")
        assert load_documents(path)["a"] == text

    def test_duplicate_ids_rejected(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"id": "a", "text": "1"}\n{"id": "a", "text": "2"}\n', encoding="utf-8")
        with pytest.raises(CorpusError, match="duplicate"):
            load_documents(path)

    def test_bad_record(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"id": "a"}\n', encoding="utf-8")
        with pytest.raises(CorpusError, match=":1:"):
            load_documents(path)

    def test_wrong_json_shape(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(CorpusError, match="expected a JSON object"):
            load_documents(path)

    def test_missing(self, tmp_path):
        with pytest.raises(CorpusError, match="not found"):
            load_documents(tmp_path / "nope.json")

    def test_build_corpus_keeps_source_text(self):
        corpus = build_corpus({"a": "x" * 3000})
        assert corpus.document_text("a") == "x" * 3000
        assert len(corpus) == 3


class TestLoadObject:
    def test_imports_attribute(self, workspace):
        assert callable(load_object("fake_llm:llm"))

    def test_calls_factory(self, workspace):
        (workspace / "factory_mod.py").write_text("def build():\n    return 42\n", encoding="utf-8")
        assert load_object("factory_mod:build()") == 42

    @pytest.mark.parametrize("spec", ["no_colon", ":attr", "mod:"])
    def test_rejects_bad_spec(self, spec):
        with pytest.raises(RetraceError, match="module:attribute"):
            load_object(spec)

    def test_missing_module(self):
        with pytest.raises(RetraceError, match="could not import"):
            load_object("definitely_not_a_module_xyz:thing")

    def test_missing_attribute(self, workspace):
        with pytest.raises(RetraceError, match="no attribute"):
            load_object("fake_llm:nope")


class TestSynthCommand:
    def test_end_to_end_with_a_user_supplied_llm(self, workspace, capsys):
        code = main([
            "synth", "--docs", "docs.json", "--out", "cand.jsonl",
            "--n", "3", "--chunk-size", "300", "--llm", "fake_llm:llm",
        ])
        assert code == 0
        cases = read_fixture(workspace / "cand.jsonl")
        # The fake returns the same question every time: one kept, rest duplicates.
        assert len(cases) == 1
        rejections = (workspace / "cand.rejections.jsonl").read_text(encoding="utf-8").splitlines()
        reasons = {json.loads(line)["reason"] for line in rejections}
        assert reasons <= {"duplicate_question", "evidence_not_verbatim"}
        out = capsys.readouterr().out
        assert "retrace label --candidates" in out

    def test_nothing_to_sample(self, workspace, capsys):
        code = main(["synth", "--docs", "docs.json", "--out", "c.jsonl",
                     "--min-chunk-chars", "100000", "--llm", "fake_llm:llm"])
        assert code == 2

    def test_bad_overlap_argument(self, workspace):
        with pytest.raises(SystemExit):
            main(["synth", "--docs", "docs.json", "--out", "c.jsonl", "--max-overlap", "7"])


class TestStatsCommand:
    def test_reports_trust(self, workspace, capsys):
        main(["synth", "--docs", "docs.json", "--out", "cand.jsonl", "--n", "1",
              "--chunk-size", "300", "--llm", "fake_llm:llm"])
        capsys.readouterr()
        assert main(["stats", "cand.jsonl"]) == 0
        out = capsys.readouterr().out
        assert "trusted    0" in out
        assert "unverified" in out


def test_errors_are_reported_not_raised(workspace, capsys):
    assert main(["stats", "missing.jsonl"]) == 2
    assert "retrace: error:" in capsys.readouterr().err
