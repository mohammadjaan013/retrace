from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from retrace import EvalCase, Provenance, Span, read_fixture, write_fixture
from retrace.labelling import (
    ACCEPT,
    REJECT,
    SKIP,
    UNDO,
    Decision,
    DecisionLog,
    LabelSession,
    decisions_path_for,
    materialise,
    replay,
)

DOC = "Radiator area is sized by the worst-case hot condition. " * 3


def candidates(n=4):
    return [
        EvalCase(
            id=f"q{i}",
            question=f"Question {i}?",
            answer=f"Answer {i}",
            gold_spans=(Span("d1", 0, 20),),
            provenance=Provenance.MODEL,
            labels={"lexical_overlap": 0.2},
        )
        for i in range(n)
    ]


class Clock:
    def __init__(self, step=3.0):
        self.t = 0.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


def session(tmp_path, keys, lines=(), n=4):
    out = tmp_path / "golden.jsonl"
    key_iter = iter(keys)
    line_iter = iter(lines)
    buf = io.StringIO()
    s = LabelSession(
        candidates=candidates(n),
        documents={"d1": DOC},
        log=DecisionLog(decisions_path_for(out)),
        output=out,
        key_reader=lambda: next(key_iter),
        line_reader=lambda _prompt: next(line_iter),
        out=buf,
        clock=Clock(),
    )
    return s, out, buf


class TestReplay:
    def test_last_decision_wins(self):
        state = replay([Decision("a", REJECT), Decision("a", ACCEPT)])
        assert state["a"].action == ACCEPT

    def test_undo_pops_one_decision(self):
        state = replay([Decision("a", REJECT), Decision("a", ACCEPT), Decision("a", UNDO)])
        assert state["a"].action == REJECT

    def test_undo_everything_leaves_undecided(self):
        state = replay([Decision("a", ACCEPT), Decision("a", UNDO)])
        assert "a" not in state

    def test_undo_with_nothing_to_undo_is_harmless(self):
        assert replay([Decision("a", UNDO)]) == {}


class TestMaterialise:
    def test_only_accepted_cases_are_promoted(self):
        golden = materialise(candidates(3), [Decision("q0", ACCEPT), Decision("q1", REJECT)])
        assert [c.id for c in golden] == ["q0"]
        assert golden[0].provenance is Provenance.MODEL_VERIFIED
        assert golden[0].is_trusted
        assert golden[0].labels["reviewed"] is True

    def test_edits_are_applied_and_recorded(self):
        golden = materialise(
            candidates(1),
            [Decision("q0", ACCEPT, question="Better question?", answer="Better answer")],
        )
        assert golden[0].question == "Better question?"
        assert golden[0].answer == "Better answer"
        assert golden[0].labels["question_edited"] is True
        assert golden[0].labels["answer_edited"] is True

    def test_original_labels_survive(self):
        golden = materialise(candidates(1), [Decision("q0", ACCEPT)])
        assert golden[0].labels["lexical_overlap"] == 0.2


class TestDecisionLog:
    def test_round_trip(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.append(Decision("q1", ACCEPT, seconds=2.5, question="edited?"))
        (d,) = log.read()
        assert (d.case_id, d.action, d.seconds, d.question) == ("q1", ACCEPT, 2.5, "edited?")

    def test_torn_final_line_is_tolerated(self, tmp_path):
        path = tmp_path / "d.jsonl"
        log = DecisionLog(path)
        log.append(Decision("q1", ACCEPT))
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"case_id": "q2", "act')  # crash mid-write
        assert [d.case_id for d in log.read()] == ["q1"]

    def test_corruption_elsewhere_is_an_error(self, tmp_path):
        from retrace.errors import FixtureError

        path = tmp_path / "d.jsonl"
        path.write_text('garbage\n{"case_id": "q1", "action": "accept"}\n', encoding="utf-8")
        with pytest.raises(FixtureError, match="corrupt"):
            DecisionLog(path).read()

    def test_path_convention(self):
        assert decisions_path_for("out/golden.jsonl") == Path("out/golden.decisions.jsonl")


class TestSession:
    def test_accept_reject_flow(self, tmp_path):
        s, out, _ = session(tmp_path, ["y", "n", "y", "n"])
        summary = s.run()
        assert summary["accepted"] == 2
        assert summary["rejected"] == 2
        assert [c.id for c in read_fixture(out)] == ["q0", "q2"]

    def test_quit_saves_and_resume_continues(self, tmp_path):
        s, out, _ = session(tmp_path, ["y", "q"])
        first = s.run()
        assert first["accepted"] == 1
        assert first["undecided"] == 3
        assert len(read_fixture(out)) == 1

        s2, out2, _ = session(tmp_path, ["y", "y", "n"])
        second = s2.run()
        assert second["accepted"] == 3
        assert second["undecided"] == 0
        assert [c.id for c in read_fixture(out2)] == ["q0", "q1", "q2"]

    def test_skipped_cases_come_back_last(self, tmp_path):
        s, _, buf = session(tmp_path, ["s", "y", "y", "y", "y"])
        summary = s.run()
        assert summary["accepted"] == 4
        shown = [line for line in buf.getvalue().splitlines() if line.startswith("Q: ")]
        assert shown == ["Q: Question 0?", "Q: Question 1?", "Q: Question 2?",
                         "Q: Question 3?", "Q: Question 0?"]

    def test_undo_reshows_the_undone_case_immediately(self, tmp_path):
        s, out, buf = session(tmp_path, ["n", "u", "y", "y", "y", "y"])
        summary = s.run()
        assert summary["accepted"] == 4
        shown = [line for line in buf.getvalue().splitlines() if line.startswith("Q: ")]
        # "n" rejects q0 and q1 appears; "u" is pressed at q1's prompt, undoing
        # q0 -- which must jump the queue ahead of q1.
        assert shown[:4] == ["Q: Question 0?", "Q: Question 1?",
                             "Q: Question 0?", "Q: Question 1?"]
        assert "q0" in {c.id for c in read_fixture(out)}

    def test_undo_with_empty_history(self, tmp_path):
        s, _, buf = session(tmp_path, ["u", "y", "y", "y", "y"])
        s.run()
        assert "nothing to undo" in buf.getvalue()

    def test_edit_question(self, tmp_path):
        s, out, _ = session(tmp_path, ["e", "y", "q"], lines=["A sharper question?"])
        s.run()
        (case,) = read_fixture(out)
        assert case.question == "A sharper question?"
        assert case.labels["question_edited"] is True

    def test_blank_edit_leaves_question_unchanged(self, tmp_path):
        s, out, _ = session(tmp_path, ["e", "y", "q"], lines=[""])
        s.run()
        (case,) = read_fixture(out)
        assert case.question == "Question 0?"
        assert "question_edited" not in case.labels

    def test_unknown_key_prompts_again(self, tmp_path):
        s, _, buf = session(tmp_path, ["x", "y", "y", "y", "y"])
        assert s.run()["accepted"] == 4
        assert "y / n / e / a / s / u / q" in buf.getvalue()

    def test_evidence_is_highlighted(self, tmp_path):
        s, _, buf = session(tmp_path, ["q"])
        s.run()
        assert f">>>{DOC[0:20]}<<<" in buf.getvalue()

    def test_measures_labelling_speed(self, tmp_path):
        s, _, buf = session(tmp_path, ["y", "n", "y", "n"])
        summary = s.run()
        # The fake clock advances 3s per reading, i.e. 3s per decision.
        assert summary["seconds_per_label"] == 3.0
        assert summary["minutes_per_50"] == 2.5
        assert "50 labels in ~2.5 min" in buf.getvalue()

    def test_warns_when_acceptance_is_low(self, tmp_path):
        s, _, buf = session(tmp_path, ["n"] * 11 + ["y"], n=12)
        s.run()
        assert "wasting your time" in buf.getvalue()

    def test_every_decision_is_on_disk_before_the_next_prompt(self, tmp_path):
        out = tmp_path / "golden.jsonl"
        log_path = decisions_path_for(out)
        seen_counts = []

        def key_reader():
            seen_counts.append(
                len(log_path.read_text(encoding="utf-8").splitlines())
                if log_path.exists()
                else 0
            )
            return "y"

        LabelSession(
            candidates=candidates(3),
            documents={"d1": DOC},
            log=DecisionLog(log_path),
            output=out,
            key_reader=key_reader,
            out=io.StringIO(),
            clock=Clock(),
        ).run()
        assert seen_counts == [0, 1, 2]
        assert all(json.loads(line)["action"] == "accept"
                   for line in log_path.read_text(encoding="utf-8").splitlines())


def test_fixture_round_trip_through_label(tmp_path):
    from retrace.labelling import label

    cand_path = tmp_path / "candidates.jsonl"
    write_fixture(cand_path, candidates(2))
    keys = iter(["y", "n"])
    summary = label(
        cand_path, {"d1": DOC}, tmp_path / "golden.jsonl",
        key_reader=lambda: next(keys), out=io.StringIO(), clock=Clock(),
    )
    assert summary["accepted"] == 1
    assert SKIP not in {d.action for d in DecisionLog(tmp_path / "golden.decisions.jsonl").read()}
