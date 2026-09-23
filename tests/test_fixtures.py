from __future__ import annotations

import json

import pytest

from retrace import (
    EvalCase,
    FixtureError,
    Provenance,
    SchemaVersionError,
    Span,
    case_from_dict,
    case_to_dict,
    fixture_stats,
    read_fixture,
    write_fixture,
)


def make_case(case_id: str = "q1", **kwargs) -> EvalCase:
    defaults = {
        "question": "What drives radiator sizing?",
        "answer": "The worst-case hot attitude.",
        "gold_spans": (Span("d1", 220, 277),),
        "provenance": Provenance.HUMAN,
    }
    defaults.update(kwargs)
    return EvalCase(id=case_id, **defaults)


class TestRoundTrip:
    def test_preserves_every_field(self):
        case = make_case(
            source_chunk_id="c_abc",
            labels={"difficulty": "hard", "hard_negative": True},
            gold_doc_ids=("d1",),
            gold_chunk_ids=("c_abc",),
        )
        assert case_from_dict(case_to_dict(case)) == case

    def test_omits_empty_collections(self):
        payload = case_to_dict(make_case())
        assert "gold_doc_ids" not in payload
        assert "gold_chunk_ids" not in payload
        assert "labels" not in payload

    def test_file_round_trip(self, tmp_path):
        cases = [make_case("q1"), make_case("q2"), make_case("q3")]
        path = tmp_path / "golden.jsonl"
        assert write_fixture(path, cases) == 3
        assert read_fixture(path) == cases

    def test_unicode_survives(self, tmp_path):
        case = make_case("q1", question="What is the Δv budget for a 300 km orbit?")
        path = tmp_path / "golden.jsonl"
        write_fixture(path, [case])
        assert read_fixture(path)[0].question == case.question


class TestValidation:
    def test_rejects_unknown_schema_version(self):
        payload = case_to_dict(make_case())
        payload["v"] = 99
        with pytest.raises(SchemaVersionError) as exc:
            case_from_dict(payload)
        assert exc.value.found == 99

    def test_rejects_unknown_provenance(self):
        payload = case_to_dict(make_case())
        payload["provenance"] = "vibes"
        with pytest.raises(FixtureError, match="unknown provenance"):
            case_from_dict(payload)

    def test_rejects_malformed_span(self):
        payload = case_to_dict(make_case())
        payload["gold_spans"] = [{"doc_id": "d1", "start": 100}]
        with pytest.raises(FixtureError, match="malformed gold_span"):
            case_from_dict(payload)

    def test_rejects_inverted_span(self):
        payload = case_to_dict(make_case())
        payload["gold_spans"] = [{"doc_id": "d1", "start": 100, "end": 50}]
        with pytest.raises(FixtureError):
            case_from_dict(payload)

    def test_reports_line_number_on_bad_json(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text(
            json.dumps(case_to_dict(make_case("q1"))) + "\n" + "{not json\n",
            encoding="utf-8",
        )
        with pytest.raises(FixtureError, match=r":2: invalid JSON"):
            read_fixture(path)

    def test_rejects_duplicate_ids(self, tmp_path):
        path = tmp_path / "dupes.jsonl"
        write_fixture(path, [make_case("q1"), make_case("q1")])
        with pytest.raises(FixtureError, match="duplicate case id"):
            read_fixture(path)

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(FixtureError, match="not found"):
            read_fixture(tmp_path / "absent.jsonl")


class TestComments:
    def test_comments_and_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "annotated.jsonl"
        path.write_text(
            "# reviewed by MJ on 2026-09-21\n"
            "\n"
            + json.dumps(case_to_dict(make_case("q1")))
            + "\n"
            "# the case below is a known hard negative\n"
            + json.dumps(case_to_dict(make_case("q2")))
            + "\n",
            encoding="utf-8",
        )
        assert [c.id for c in read_fixture(path)] == ["q1", "q2"]


class TestAtomicWrite:
    def test_original_survives_a_failed_write(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        write_fixture(path, [make_case("q1")])
        original = path.read_text(encoding="utf-8")

        def exploding():
            yield make_case("q2")
            raise RuntimeError("labelling tool crashed mid-export")

        with pytest.raises(RuntimeError):
            write_fixture(path, exploding())

        assert path.read_text(encoding="utf-8") == original
        assert len(read_fixture(path)) == 1

    def test_no_temp_files_left_behind(self, tmp_path):
        path = tmp_path / "golden.jsonl"

        def exploding():
            yield make_case("q1")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            write_fixture(path, exploding())

        assert list(tmp_path.glob(".*tmp")) == []


class TestStats:
    def test_separates_trusted_from_total(self):
        cases = [
            make_case("q1", provenance=Provenance.HUMAN),
            make_case("q2", provenance=Provenance.MODEL_VERIFIED),
            make_case("q3", provenance=Provenance.MODEL),
            make_case("q4", provenance=Provenance.MODEL),
        ]
        stats = fixture_stats(cases)
        assert stats["total"] == 4
        assert stats["trusted"] == 2
        assert stats["by_provenance"]["model"] == 2
        assert stats["with_spans"] == 4
        assert stats["documents"] == 1
