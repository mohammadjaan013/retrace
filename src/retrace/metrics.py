"""Retrieval metrics, and the ranked "fix this first" list.

The metrics are deliberately boring -- hit@k, MRR, precision@k, recall@k -- so a
team already using another tool can line the numbers up. Two things are not
boring:

1. **Corpus gaps are excluded from retrieval metrics.** A question whose answer
   is not indexed is not a retrieval failure, and counting it as one is how
   teams end up tuning embeddings for questions their documents cannot answer.
   The excluded count is always reported next to the metric.

2. **`fix_first()` turns verdicts into an ordered to-do list.** The largest
   failure bucket goes first, each with the concrete changes that address that
   stage and the case ids to look at.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .attribution import Attribution
from .types import Verdict

STANDARD_KS: tuple[int, ...] = (1, 3, 5, 10, 20)



@dataclass
class Metrics:
    cases: int
    trusted: int
    scored: int
    """Cases the pipeline answered without raising."""

    answerable: int
    """Scored cases whose evidence is indexed -- the denominator for retrieval metrics."""

    verdicts: dict[str, int]
    hit_rate: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float | None = None
    context_hit_rate: float | None = None
    generation_evaluated: int = 0
    generation_accuracy: float | None = None
    lucky: int = 0
    errors: int = 0
    vocabulary_mismatch: int = 0
    """Retrieval misses a reference keyword retriever also misses."""

    findable_but_missed: int = 0
    """Retrieval misses a reference keyword retriever finds: the pipeline is
    losing to plain keyword search."""
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "trusted": self.trusted,
            "scored": self.scored,
            "answerable": self.answerable,
            "verdicts": self.verdicts,
            "hit_rate": {str(k): v for k, v in self.hit_rate.items()},
            "precision": {str(k): v for k, v in self.precision.items()},
            "recall": {str(k): v for k, v in self.recall.items()},
            "mrr": self.mrr,
            "context_hit_rate": self.context_hit_rate,
            "generation_evaluated": self.generation_evaluated,
            "generation_accuracy": self.generation_accuracy,
            "lucky": self.lucky,
            "errors": self.errors,
            "vocabulary_mismatch": self.vocabulary_mismatch,
            "findable_but_missed": self.findable_but_missed,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
        }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile. Small samples make fancier interpolation fake precision."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, min(len(ordered), math.ceil(pct / 100 * len(ordered))))
    return round(ordered[rank - 1], 2)


def compute_metrics(run: Any, *, trusted_only: bool = False) -> Metrics:
    attributions: list[Attribution] = [
        a for a in run.attributions if a.trusted or not trusted_only
    ]
    scored = [a for a in attributions if a.stage != "error"]
    answerable = [a for a in scored if a.verdict is not Verdict.CORPUS_GAP]

    max_candidates = max((a.n_candidates for a in answerable), default=0)
    ks = [k for k in STANDARD_KS if k <= max(max_candidates, 1)]

    hit_rate, precision, recall = {}, {}, {}
    for k in ks:
        hits = [1.0 if a.candidate_rank is not None and a.candidate_rank <= k else 0.0
                for a in answerable]
        hit_rate[k] = _mean(hits)
        precision[k] = _mean([sum(1 for r in a.gold_ranks if r <= k) / k for a in answerable])
        with_gold_count = [a for a in answerable if a.gold_in_corpus]
        if with_gold_count:
            recall[k] = _mean([
                min(1.0, sum(1 for r in a.gold_ranks if r <= k) / a.gold_in_corpus)
                for a in with_gold_count
            ])

    evaluated = [a for a in scored if a.generation_evaluated and a.judgement is not None]
    decided = [a for a in evaluated if a.judgement.correct is not None]

    misses = [a for a in scored if a.verdict is Verdict.RETRIEVAL_MISS]
    probed = [a for a in misses if a.lexical_probe_k is not None]
    findable = sum(1 for a in probed if a.lexical_probe_rank is not None)
    vocab = len(probed) - findable

    ids = {a.case_id for a in attributions}
    latencies = [
        t.latency_ms for cid, t in run.traces.items()
        if cid in ids and t.latency_ms is not None
    ]

    counts = {v.value: 0 for v in Verdict}
    for a in attributions:
        counts[a.verdict.value] += 1

    return Metrics(
        cases=len(attributions),
        trusted=sum(1 for a in attributions if a.trusted),
        scored=len(scored),
        answerable=len(answerable),
        verdicts=counts,
        hit_rate=hit_rate,
        precision=precision,
        recall=recall,
        mrr=_mean([1.0 / a.candidate_rank if a.candidate_rank else 0.0 for a in answerable]),
        context_hit_rate=_mean([1.0 if a.context_rank else 0.0 for a in answerable]),
        generation_evaluated=len(evaluated),
        generation_accuracy=_mean([1.0 if a.judgement.correct else 0.0 for a in decided]),
        lucky=sum(1 for a in attributions if a.lucky),
        errors=len(attributions) - len(scored),
        vocabulary_mismatch=vocab,
        findable_but_missed=findable,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )


# --------------------------------------------------------------------------
# What to fix first
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    verdict: Verdict
    count: int
    share: float
    """Share of all failures."""

    headline: str
    actions: tuple[str, ...]
    case_ids: tuple[str, ...]
    detail: str = ""


_PLAYBOOK: dict[Verdict, tuple[str, tuple[str, ...]]] = {
    Verdict.CORPUS_GAP: (
        "The answer is not in the indexed documents. Stop tuning retrieval for these.",
        (
            "Find out whether the source documents exist at all -- this is a content "
            "problem, and no retrieval change can fix it.",
            "If they exist, check the ingestion pipeline for silently dropped files "
            "(failed PDF extraction, size limits, unsupported formats).",
            "If they do not, either add the sources or scope the product so users are "
            "told these questions are out of coverage.",
        ),
    ),
    Verdict.RETRIEVAL_MISS: (
        "The evidence is indexed but retrieval never returns it.",
        (
            "Check chunking first: answers split across a chunk boundary are unfindable. "
            "Try structure-aware chunks with overlap.",
            "Add keyword search (BM25) alongside vector search and fuse the results -- "
            "it recovers exact terms, codes and names that embeddings blur.",
            "For vocabulary mismatches, add dense retrieval if you are lexical-only, or "
            "rewrite queries into the corpus's own terminology.",
        ),
    ),
    Verdict.RANKING_MISS: (
        "The evidence is retrieved, then cut before the model sees it.",
        (
            "Add or retune a reranker over a wider candidate set -- usually the "
            "cheapest win in the whole pipeline.",
            "Check whether the context budget is too tight for your chunk size.",
            "Inspect the fusion weights if you merge several retrievers.",
        ),
    ),
    Verdict.GENERATION_MISS: (
        "The model had the evidence in context and still answered wrong.",
        (
            "Confirm with an LLM judge before acting: the free token-recall judge is "
            "blind to negation and paraphrase.",
            "Check where the evidence sits in the context -- models under-use material "
            "buried in the middle of long contexts.",
            "Tighten the prompt to answer only from the provided passages, then "
            "compare models.",
        ),
    ),
}


def fix_first(run: Any, *, trusted_only: bool = False, examples: int = 5) -> list[Recommendation]:
    """Failure buckets, largest first, each with the changes that address it."""
    failures = [
        a for a in run.attributions
        if a.is_failure and (a.trusted or not trusted_only)
    ]
    if not failures:
        return []

    by_verdict: dict[Verdict, list[Attribution]] = {}
    for a in failures:
        by_verdict.setdefault(a.verdict, []).append(a)

    metrics = compute_metrics(run, trusted_only=trusted_only)
    recs = []
    for verdict, items in sorted(by_verdict.items(), key=lambda kv: (-len(kv[1]), kv[0].value)):
        headline, actions = _PLAYBOOK[verdict]
        detail = ""
        if verdict is Verdict.RETRIEVAL_MISS:
            parts = []
            if metrics.vocabulary_mismatch:
                parts.append(f"{metrics.vocabulary_mismatch} vocabulary mismatch")
            if metrics.findable_but_missed:
                parts.append(f"{metrics.findable_but_missed} found by a reference keyword "
                             "retriever (losing to BM25)")
            detail = "; ".join(parts)
            if metrics.vocabulary_mismatch > metrics.findable_but_missed:
                # Lead with the fix for the dominant sub-type: lexical tuning
                # cannot reach a vocabulary mismatch.
                actions = (actions[2], actions[0], actions[1])
        if verdict in (Verdict.CORPUS_GAP, Verdict.RETRIEVAL_MISS, Verdict.RANKING_MISS):
            lucky = sum(1 for a in items if a.lucky)
            if lucky:
                note = (f"{lucky} answered correctly anyway -- from model memory, "
                        "or an incomplete label")
                detail = f"{detail}; {note}" if detail else note

        recs.append(Recommendation(
            verdict=verdict,
            count=len(items),
            share=round(len(items) / len(failures), 3),
            headline=headline,
            actions=actions,
            case_ids=tuple(a.case_id for a in items[:examples]),
            detail=detail,
        ))
    return recs
