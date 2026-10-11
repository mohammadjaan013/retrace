"""The labelling loop: turn model-proposed candidates into a trusted golden set.

The product claim this module has to earn is *"a trustworthy eval set for your
corpus in under an hour"*. That means the loop is optimised for one thing --
decisions per minute -- and it measures itself so the claim can be quoted with a
real number.

DURABILITY
----------
Every decision is appended to a log file and fsynced before the next candidate
is shown. The golden fixture is *derived* from candidates + log, never edited in
place. So a crash, a closed laptop or a Ctrl-C loses at most the decision being
typed, and re-running the same command resumes exactly where it stopped.

Undo appends an ``undo`` record rather than rewriting history, which keeps the
log append-only and therefore crash-safe.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TextIO

from .errors import FixtureError
from .fixtures import read_fixture, write_fixture
from .types import EvalCase, Provenance

ACCEPT = "accept"
REJECT = "reject"
SKIP = "skip"
UNDO = "undo"

CONTEXT_CHARS = 220


@dataclass(frozen=True, slots=True)
class Decision:
    case_id: str
    action: str
    seconds: float = 0.0
    question: str | None = None
    answer: str | None = None
    at: float = 0.0

    def to_json(self) -> str:
        record: dict[str, Any] = {"case_id": self.case_id, "action": self.action}
        if self.seconds:
            record["seconds"] = round(self.seconds, 2)
        if self.question is not None:
            record["question"] = self.question
        if self.answer is not None:
            record["answer"] = self.answer
        record["at"] = round(self.at or time.time(), 3)
        return json.dumps(record, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        return cls(
            case_id=data["case_id"],
            action=data["action"],
            seconds=float(data.get("seconds", 0.0)),
            question=data.get("question"),
            answer=data.get("answer"),
            at=float(data.get("at", 0.0)),
        )


def decisions_path_for(output: str | os.PathLike[str]) -> Path:
    """The log lives next to the golden set: ``golden.jsonl`` -> ``golden.decisions.jsonl``."""
    out = Path(output)
    return out.with_name(f"{out.stem}.decisions.jsonl")


class DecisionLog:
    """Append-only, fsynced decision record."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def read(self) -> list[Decision]:
        if not self.path.exists():
            return []
        out: list[Decision] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    out.append(Decision.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    # A torn final line from a crash mid-write is expected and
                    # harmless; corruption anywhere else is not.
                    if lineno == _line_count(self.path):
                        break
                    raise FixtureError(f"{self.path}:{lineno}: corrupt decision record") from exc
        return out

    def append(self, decision: Decision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(decision.to_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def replay(decisions: Iterable[Decision]) -> dict[str, Decision]:
    """Collapse the log into the current decision per case, honouring undo."""
    history: dict[str, list[Decision]] = {}
    for d in decisions:
        if d.action == UNDO:
            if history.get(d.case_id):
                history[d.case_id].pop()
        else:
            history.setdefault(d.case_id, []).append(d)
    return {cid: stack[-1] for cid, stack in history.items() if stack}


def materialise(candidates: Iterable[EvalCase], decisions: Iterable[Decision]) -> list[EvalCase]:
    """Build the golden set: accepted candidates, promoted and edited.

    Accepted cases become `Provenance.MODEL_VERIFIED`. A human edit to the
    question is recorded as a label, because an edited question has left the
    distribution the synthesiser produces -- worth knowing when comparing sets.
    """
    current = replay(decisions)
    golden: list[EvalCase] = []
    for case in candidates:
        decision = current.get(case.id)
        if decision is None or decision.action != ACCEPT:
            continue
        labels = dict(case.labels)
        labels["reviewed"] = True
        if decision.seconds:
            labels["review_seconds"] = round(decision.seconds, 1)
        question = case.question
        answer = case.answer
        if decision.question and decision.question != case.question:
            labels["question_edited"] = True
            question = decision.question
        if decision.answer and decision.answer != case.answer:
            labels["answer_edited"] = True
            answer = decision.answer
        golden.append(
            replace(
                case,
                question=question,
                answer=answer,
                provenance=Provenance.MODEL_VERIFIED,
                labels=labels,
            )
        )
    return golden


# --------------------------------------------------------------------------
# Terminal I/O
# --------------------------------------------------------------------------


def read_key() -> str:  # pragma: no cover - needs a real terminal
    """One keypress, no Enter. Falls back to a line read when stdin is piped."""
    if not sys.stdin.isatty():
        return (sys.stdin.readline() or "q").strip()[:1].lower() or "?"
    try:
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # arrow / function key prefix
            msvcrt.getwch()
            return "?"
        return ch.lower()
    except ImportError:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x03":
            raise KeyboardInterrupt from None
        return ch.lower()


def _supports_colour(stream: TextIO) -> bool:
    return hasattr(stream, "isatty") and stream.isatty() and os.environ.get("NO_COLOR") is None


@dataclass
class LabelSession:
    """One labelling run. All I/O is injected so the loop is testable end to end."""

    candidates: list[EvalCase]
    documents: dict[str, str]
    log: DecisionLog
    output: Path
    key_reader: Callable[[], str] = read_key
    line_reader: Callable[[str], str] = input
    out: TextIO = field(default_factory=lambda: sys.stdout)
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._colour = _supports_colour(self.out)
        self._session_decisions: list[Decision] = []

    # -- rendering ---------------------------------------------------------

    def _hl(self, text: str) -> str:
        return f"\x1b[1;7m{text}\x1b[0m" if self._colour else f">>>{text}<<<"

    def _dim(self, text: str) -> str:
        return f"\x1b[2m{text}\x1b[0m" if self._colour else text

    def _evidence_view(self, case: EvalCase) -> str:
        if not case.gold_spans:
            return self._dim("(no span -- document-level label only)")
        span = case.gold_spans[0]
        text = self.documents.get(span.doc_id)
        if text is None:
            return self._dim(f"(document {span.doc_id!r} not loaded)")
        lo = max(0, span.start - CONTEXT_CHARS)
        hi = min(len(text), span.end + CONTEXT_CHARS)
        before = text[lo : span.start].replace("\n", " ")
        gold = text[span.start : span.end].replace("\n", " ")
        after = text[span.end : hi].replace("\n", " ")
        prefix = "..." if lo > 0 else ""
        suffix = "..." if hi < len(text) else ""
        return f"{prefix}{self._dim(before)}{self._hl(gold)}{self._dim(after)}{suffix}"

    def _show(self, case: EvalCase, position: int, total: int) -> None:
        w = self.out.write
        w("\n" + "-" * 72 + "\n")
        overlap = case.labels.get("lexical_overlap")
        meta = f"  overlap {overlap:.0%}" if isinstance(overlap, (int, float)) else ""
        match = case.labels.get("evidence_match")
        if match in ("fuzzy", "gapped"):
            # The highlight shows the source text; the model's quote differed --
            # an OCR correction (fuzzy) or a page footer it skipped (gapped).
            meta += f"  {match} evidence {case.labels.get('evidence_similarity', 0):.0%}"
        if case.labels.get("answer_in_question"):
            meta += "  ! answer appears in the question"
        doc = next(iter(case.implied_doc_ids), "?")
        w(f"[{position}/{total}]  {doc}{meta}\n\n")
        w(f"Q: {case.question}\n")
        if case.answer:
            w(f"A: {case.answer}\n")
        w("\n" + self._evidence_view(case) + "\n\n")
        w(self._dim("[y] accept  [n] reject  [e] edit question  [a] edit answer  "
                    "[s] skip  [u] undo  [q] quit") + "\n> ")
        self.out.flush()

    # -- the loop ----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Label until every candidate is decided or the user quits.

        Returns a summary including the measured labelling rate.
        """
        prior = self.log.read()
        state = replay(prior)
        undo_stack: list[str] = [d.case_id for d in prior if d.action != UNDO]
        priority: str | None = None  # an undone case jumps the queue

        def pending() -> list[EvalCase]:
            # Undecided first, in candidate order; skipped cases come back round
            # only after everything else has been seen once.
            fresh = [c for c in self.candidates if c.id not in state]
            skipped = [
                c for c in self.candidates if c.id in state and state[c.id].action == SKIP
            ]
            return fresh + skipped

        total = len(self.candidates)
        quit_requested = False

        while not quit_requested:
            queue = pending()
            if not queue:
                break
            case = next((c for c in queue if c.id == priority), queue[0])
            priority = None
            decided = total - len(queue) + 1
            self._show(case, decided, total)

            started = self.clock()
            question: str | None = None
            answer: str | None = None
            while True:
                key = self.key_reader()
                if key == "y":
                    action = ACCEPT
                    break
                if key == "n":
                    action = REJECT
                    break
                if key == "s":
                    action = SKIP
                    break
                if key == "e":
                    self.out.write("\nnew question: ")
                    self.out.flush()
                    edited = self.line_reader("").strip()
                    if edited:
                        question = edited
                        self.out.write(f"  -> {edited}\n  [y] accept edited  [n] reject > ")
                    else:
                        self.out.write("  (unchanged) > ")
                    self.out.flush()
                    continue
                if key == "a":
                    self.out.write("\nnew answer: ")
                    self.out.flush()
                    edited = self.line_reader("").strip()
                    if edited:
                        answer = edited
                        self.out.write(f"  -> {edited}\n  [y] accept edited  [n] reject > ")
                    else:
                        self.out.write("  (unchanged) > ")
                    self.out.flush()
                    continue
                if key == "u":
                    if not undo_stack:
                        self.out.write("\n  nothing to undo > ")
                        self.out.flush()
                        continue
                    target = undo_stack.pop()
                    self.log.append(Decision(target, UNDO, at=time.time()))
                    state = replay(self.log.read())
                    self._session_decisions = [
                        d for d in self._session_decisions if d.case_id != target
                    ] + [d for d in self._session_decisions if d.case_id == target][:-1]
                    priority = target
                    self.out.write(f"\n  undid decision on {target}\n")
                    action = ""
                    break
                if key == "q":
                    quit_requested = True
                    action = ""
                    break
                self.out.write(self._dim("\n  y / n / e / a / s / u / q > "))
                self.out.flush()

            if not action:
                continue

            decision = Decision(
                case_id=case.id,
                action=action,
                seconds=self.clock() - started,
                question=question,
                answer=answer,
                at=time.time(),
            )
            self.log.append(decision)
            self._session_decisions.append(decision)
            undo_stack.append(case.id)
            state[case.id] = decision

        golden = materialise(self.candidates, self.log.read())
        write_fixture(self.output, golden)
        summary = self.summary(golden)
        self._print_summary(summary)
        return summary

    def summary(self, golden: list[EvalCase]) -> dict[str, Any]:
        final = replay(self.log.read())
        counts = {ACCEPT: 0, REJECT: 0, SKIP: 0}
        for d in final.values():
            counts[d.action] = counts.get(d.action, 0) + 1

        timed = [d for d in self._session_decisions if d.action in (ACCEPT, REJECT)]
        seconds = sum(d.seconds for d in timed)
        per_label = seconds / len(timed) if timed else None
        return {
            "candidates": len(self.candidates),
            "accepted": counts[ACCEPT],
            "rejected": counts[REJECT],
            "skipped": counts[SKIP],
            "undecided": len(self.candidates) - sum(counts.values()),
            "golden_written": len(golden),
            "output": str(self.output),
            "session_labels": len(timed),
            "session_seconds": round(seconds, 1),
            "seconds_per_label": round(per_label, 1) if per_label else None,
            "minutes_per_50": round(per_label * 50 / 60, 1) if per_label else None,
        }

    def _print_summary(self, s: dict[str, Any]) -> None:
        w = self.out.write
        w("\n\n" + "=" * 72 + "\n")
        w(f"accepted {s['accepted']}  rejected {s['rejected']}  skipped {s['skipped']}"
          f"  undecided {s['undecided']}\n")
        w(f"golden set: {s['golden_written']} cases -> {s['output']}\n")
        if s["seconds_per_label"]:
            w(f"this session: {s['session_labels']} labels, "
              f"{s['seconds_per_label']}s each -> 50 labels in ~{s['minutes_per_50']} min\n")
        acceptance = s["accepted"] / max(1, s["accepted"] + s["rejected"])
        if s["accepted"] + s["rejected"] >= 10 and acceptance < 0.5:
            w(self._dim(
                f"acceptance is {acceptance:.0%}: the synthesiser is wasting your time. "
                "Try a smaller chunk size or a stricter --max-overlap.\n"
            ))
        self.out.flush()


def label(
    candidates_path: str | os.PathLike[str],
    documents: dict[str, str],
    output: str | os.PathLike[str],
    **io: Any,
) -> dict[str, Any]:
    """Convenience entry point used by the CLI."""
    candidates = read_fixture(candidates_path)
    session = LabelSession(
        candidates=candidates,
        documents=documents,
        log=DecisionLog(decisions_path_for(output)),
        output=Path(output),
        **io,
    )
    return session.run()
