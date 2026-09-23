"""Run a pipeline over a golden set, attribute every case, persist the run.

A run is a self-contained JSON document: the cases, the verdicts, and a trace of
what the pipeline returned for each query. Reports are rendered *from* the run
file, so a run can be re-rendered, diffed against a later run, or attached to a
CI artefact without re-querying anybody's pipeline.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .attribution import Attribution, attribute, error_attribution
from .corpus import coverage_warning
from .errors import RetraceError
from .fixtures import case_from_dict, case_to_dict
from .judge import AnswerJudge, Judgement
from .protocols import RetrieverPipeline
from .types import Chunk, EvalCase, RetrievalResult, Span, Verdict

RUN_SCHEMA_VERSION = 1
TRACE_TEXT_CHARS = 600
ABORT_AFTER_CONSECUTIVE_ERRORS = 3


class RunAbortedError(RetraceError):
    """The pipeline failed the same way on the first cases -- a setup problem, not data."""


@dataclass
class RunResult:
    cases: list[EvalCase]
    attributions: list[Attribution]
    traces: dict[str, RetrievalResult] = field(default_factory=dict)
    tier: int = 1
    corpus_checked: bool = False
    judge: str | None = None
    coverage_warning: str | None = None
    pipeline: str = ""
    started_at: str = ""
    duration_s: float = 0.0
    retrace_version: str = __version__
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict_counts(self) -> dict[Verdict, int]:
        counts = Counter(a.verdict for a in self.attributions)
        return {v: counts.get(v, 0) for v in Verdict}

    def case(self, case_id: str) -> EvalCase:
        return self._case_index()[case_id]

    def _case_index(self) -> dict[str, EvalCase]:
        return {c.id: c for c in self.cases}


def as_pipeline(obj: Any, k: int = 10):
    """Accept a Tier 2 pipeline, or lift a Tier 1 retriever into one."""
    if hasattr(obj, "run") and callable(obj.run):
        return obj
    return RetrieverPipeline(obj, k=k)


def evaluate(
    pipeline: Any,
    cases: Iterable[EvalCase],
    *,
    corpus: Any = None,
    judge: AnswerJudge | None = None,
    k: int = 10,
    progress: Callable[[int, int, Attribution], None] | None = None,
    keep_traces: bool = True,
) -> RunResult:
    """Attribute every case.

    A case whose pipeline call raises becomes UNSCORABLE and the run continues --
    one malformed document should not throw away four hundred good results. But
    if the *first* cases all fail with the same exception, the problem is setup
    (bad key, wrong URL, index not built), and the run stops immediately rather
    than producing a report that is 100% UNSCORABLE.
    """
    cases = list(cases)
    runner = as_pipeline(pipeline, k=k)
    started = time.perf_counter()
    started_at = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")

    attributions: list[Attribution] = []
    traces: dict[str, RetrievalResult] = {}
    tiers: set[int] = set()
    leading_errors: list[type[BaseException]] = []  # only until the first success
    any_success = False

    for i, case in enumerate(cases, start=1):
        try:
            result = runner.run(case.question)
        except Exception as exc:  # per-case isolation is the point
            if not any_success:
                leading_errors.append(type(exc))
                if (
                    len(leading_errors) == ABORT_AFTER_CONSECUTIVE_ERRORS
                    and len(set(leading_errors)) == 1
                ):
                    raise RunAbortedError(
                        f"the pipeline raised {type(exc).__name__} on each of the first "
                        f"{ABORT_AFTER_CONSECUTIVE_ERRORS} cases -- this looks like a "
                        f"setup problem, not a data problem: {exc}"
                    ) from exc
            attribution = error_attribution(case, exc)
        else:
            any_success = True
            tiers.add(2 if result.exposes_ranking_stage else 1)
            attribution = attribute(case, result, corpus=corpus, judge=judge)
            if keep_traces:
                traces[case.id] = result

        attributions.append(attribution)
        if progress is not None:
            progress(i, len(cases), attribution)

    # A pipeline whose reranker happens to preserve order on every query still
    # counts as Tier 1 for this run: the ranking stage was never observed.
    tier = 2 if 2 in tiers else 1

    return RunResult(
        cases=cases,
        attributions=attributions,
        traces=traces,
        tier=tier,
        corpus_checked=corpus is not None,
        judge=getattr(judge, "name", type(judge).__name__) if judge is not None else None,
        coverage_warning=coverage_warning(k, corpus) if corpus is not None else None,
        pipeline=repr(runner),
        started_at=started_at,
        duration_s=round(time.perf_counter() - started, 3),
    )


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    out: dict[str, Any] = {"id": chunk.id, "doc_id": chunk.doc_id}
    text = chunk.text
    out["text"] = text if len(text) <= TRACE_TEXT_CHARS else text[:TRACE_TEXT_CHARS] + "…"
    if chunk.score is not None:
        out["score"] = chunk.score
    if chunk.span is not None:
        out["span"] = [chunk.span.doc_id, chunk.span.start, chunk.span.end]
    if chunk.meta:
        out["meta"] = _json_safe(chunk.meta)
    return out


def _chunk_from_dict(d: dict[str, Any]) -> Chunk:
    span = Span(d["span"][0], d["span"][1], d["span"][2]) if d.get("span") else None
    return Chunk(
        id=d["id"], doc_id=d["doc_id"], text=d.get("text", ""),
        score=d.get("score"), span=span, meta=d.get("meta", {}),
    )


def _attribution_to_dict(a: Attribution) -> dict[str, Any]:
    out: dict[str, Any] = {
        "case_id": a.case_id,
        "verdict": a.verdict.value,
        "reason": a.reason,
        "stage": a.stage,
        "gold_in_corpus": a.gold_in_corpus,
        "candidate_rank": a.candidate_rank,
        "context_rank": a.context_rank,
        "gold_ranks": list(a.gold_ranks),
        "query_gold_overlap": a.query_gold_overlap,
        "n_candidates": a.n_candidates,
        "n_context": a.n_context,
        "match_granularity": a.match_granularity,
        "generation_evaluated": a.generation_evaluated,
        "lucky": a.lucky,
        "trusted": a.trusted,
        "notes": list(a.notes),
    }
    if a.judgement is not None:
        out["judgement"] = {
            "correct": a.judgement.correct,
            "score": a.judgement.score,
            "reason": a.judgement.reason,
            "judge": a.judgement.judge,
        }
    if a.error:
        out["error"] = a.error
    return out


def _attribution_from_dict(d: dict[str, Any]) -> Attribution:
    j = d.get("judgement")
    return Attribution(
        case_id=d["case_id"],
        verdict=Verdict(d["verdict"]),
        reason=d.get("reason", ""),
        stage=d.get("stage", ""),
        gold_in_corpus=d.get("gold_in_corpus"),
        candidate_rank=d.get("candidate_rank"),
        context_rank=d.get("context_rank"),
        gold_ranks=tuple(d.get("gold_ranks", ())),
        query_gold_overlap=d.get("query_gold_overlap"),
        n_candidates=d.get("n_candidates", 0),
        n_context=d.get("n_context", 0),
        match_granularity=d.get("match_granularity"),
        generation_evaluated=d.get("generation_evaluated", False),
        judgement=Judgement(j["correct"], j.get("score"), j.get("reason", ""), j.get("judge", ""))
        if j else None,
        lucky=d.get("lucky", False),
        trusted=d.get("trusted", True),
        notes=tuple(d.get("notes", ())),
        error=d.get("error"),
    )


def run_to_dict(run: RunResult) -> dict[str, Any]:
    traces = {}
    for case_id, r in run.traces.items():
        traces[case_id] = {
            "query": r.query,
            "answer": r.answer,
            "latency_ms": r.latency_ms,
            "candidates": [_chunk_to_dict(c) for c in r.candidates],
            "context": [_chunk_to_dict(c) for c in r.context],
            "meta": _json_safe(r.meta),
        }
    return {
        "retrace_run": RUN_SCHEMA_VERSION,
        "retrace_version": run.retrace_version,
        "started_at": run.started_at,
        "duration_s": run.duration_s,
        "pipeline": run.pipeline,
        "tier": run.tier,
        "corpus_checked": run.corpus_checked,
        "judge": run.judge,
        "coverage_warning": run.coverage_warning,
        "meta": _json_safe(run.meta),
        "verdicts": {v.value: n for v, n in run.verdict_counts.items()},
        "cases": [case_to_dict(c) for c in run.cases],
        "attributions": [_attribution_to_dict(a) for a in run.attributions],
        "traces": traces,
    }


def run_from_dict(data: dict[str, Any]) -> RunResult:
    version = data.get("retrace_run")
    if version != RUN_SCHEMA_VERSION:
        raise RetraceError(f"unsupported run file version {version!r}")
    traces = {}
    for case_id, t in data.get("traces", {}).items():
        traces[case_id] = RetrievalResult(
            query=t["query"],
            candidates=tuple(_chunk_from_dict(c) for c in t["candidates"]),
            context=tuple(_chunk_from_dict(c) for c in t["context"]),
            answer=t.get("answer"),
            latency_ms=t.get("latency_ms"),
            meta=t.get("meta", {}),
        )
    return RunResult(
        cases=[case_from_dict(c) for c in data["cases"]],
        attributions=[_attribution_from_dict(a) for a in data["attributions"]],
        traces=traces,
        tier=data.get("tier", 1),
        corpus_checked=data.get("corpus_checked", False),
        judge=data.get("judge"),
        coverage_warning=data.get("coverage_warning"),
        pipeline=data.get("pipeline", ""),
        started_at=data.get("started_at", ""),
        duration_s=data.get("duration_s", 0.0),
        retrace_version=data.get("retrace_version", ""),
        meta=data.get("meta", {}),
    )


def save_run(run: RunResult, path: str | os.PathLike[str]) -> Path:
    """Write a run file atomically (temp file, fsync, rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(run_to_dict(run), handle, ensure_ascii=False, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_run(path: str | os.PathLike[str]) -> RunResult:
    path = Path(path)
    if not path.exists():
        raise RetraceError(f"run file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RetraceError(f"{path}: invalid JSON: {exc.msg}") from exc
    return run_from_dict(data)
