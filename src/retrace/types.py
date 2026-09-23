"""Core value types.

Everything here is a frozen dataclass with no dependencies outside the standard
library. These types are the contract between Retrace and somebody else's RAG
pipeline, so they change slowly and deliberately.

The one non-obvious design decision is how ground truth is stored -- see `Span`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Bumped only when a persisted record's shape changes incompatibly.
# Every EvalCase written to disk carries this, per record rather than per file,
# so a fixture can be migrated case by case and a partially migrated file is
# still readable.
FIXTURE_SCHEMA_VERSION = 1
SUPPORTED_FIXTURE_VERSIONS: tuple[int, ...] = (1,)


class Verdict(StrEnum):
    """Which stage of the pipeline is responsible for a failure.

    Assigned by the attribution cascade (Week 3). Defined here in Week 1 because
    the vocabulary belongs to the data model: reports, fixtures and the CLI all
    name these, and renaming them later would break persisted runs.
    """

    OK = "OK"
    """The gold evidence reached the model and the answer was right."""

    CORPUS_GAP = "CORPUS_GAP"
    """The answer is not in the indexed corpus at all. No pipeline change helps."""

    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    """The gold chunk is indexed but was never returned as a candidate."""

    RANKING_MISS = "RANKING_MISS"
    """The gold chunk was retrieved, then ranked below the context cutoff."""

    GENERATION_MISS = "GENERATION_MISS"
    """The gold chunk reached the model, which ignored or contradicted it."""

    UNSCORABLE = "UNSCORABLE"
    """Retrace cannot honestly decide. A legitimate outcome, never a silent pass.

    Reported separately and never folded into a pass rate: a measurement tool
    that hides its own uncertainty is worse than one that admits it.
    """


class Provenance(StrEnum):
    """Where an evaluation label came from.

    Carried on every case because a golden set is only as trustworthy as its
    weakest label, and a buyer will ask. Model-generated labels that no human
    has confirmed must never be reported as ground truth.
    """

    HUMAN = "human"
    """A person wrote or explicitly confirmed this label."""

    MODEL = "model"
    """A model proposed it and nobody has checked it yet."""

    MODEL_VERIFIED = "model_verified"
    """A model proposed it and a person accepted it in the labelling loop."""

    IMPORTED = "imported"
    """Came from an external dataset. Trust is that source's problem."""


@dataclass(frozen=True, slots=True)
class Span:
    """A character range in a source document.

    This is the reason gold labels survive re-chunking.

    The obvious design is to record which chunk ids should have been retrieved.
    That breaks the first time the chunking strategy changes -- and changing the
    chunking is the single most common fix Retrace will recommend, so the obvious
    design would invalidate the golden set precisely when it is needed most.

    Character offsets into the document are stable across every chunking. At
    evaluation time `resolve_gold_chunks()` maps spans onto whichever chunks the
    pipeline currently produces.
    """

    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be > start ({self.start})")

    def overlaps(self, other: Span) -> bool:
        """True when both spans are in the same document and share any character."""
        if self.doc_id != other.doc_id:
            return False
        return self.start < other.end and other.start < self.end

    def overlap_chars(self, other: Span) -> int:
        """Number of characters the two spans share. Zero if they do not overlap."""
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Chunk:
    """One unit of retrievable text.

    `span` is optional because a pipeline being audited may not expose character
    offsets. Without it Retrace can still score at document granularity; with it
    gold-span resolution becomes exact. The adapter docs say so plainly, because
    the accuracy difference is worth the integration effort.
    """

    id: str
    doc_id: str
    text: str
    score: float | None = None
    span: Span | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("chunk id must be a non-empty string")
        if not self.doc_id:
            raise ValueError("chunk doc_id must be a non-empty string")
        if self.span is not None and self.span.doc_id != self.doc_id:
            raise ValueError(
                f"chunk {self.id!r} is in document {self.doc_id!r} "
                f"but its span points at {self.span.doc_id!r}"
            )

    def __hash__(self) -> int:
        # Hash on identity fields only: `meta` is a dict and unhashable, and two
        # equal chunks always agree on (id, doc_id), so the hash/eq contract holds.
        return hash((self.id, self.doc_id))

    def with_score(self, score: float) -> Chunk:
        """Return a copy carrying a retrieval score. Chunks are immutable."""
        return Chunk(
            id=self.id,
            doc_id=self.doc_id,
            text=self.text,
            score=score,
            span=self.span,
            meta=dict(self.meta),
        )


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One question, with enough ground truth to attribute a failure to a stage.

    Gold evidence is recorded at three granularities, most to least precise.
    At least one of `gold_spans` or `gold_doc_ids` must be present, or the case
    cannot distinguish CORPUS_GAP from RETRIEVAL_MISS and is not worth storing.
    """

    id: str
    question: str
    answer: str | None = None
    gold_spans: tuple[Span, ...] = ()
    gold_doc_ids: tuple[str, ...] = ()

    gold_chunk_ids: tuple[str, ...] = ()
    """Only meaningful for one specific chunking. Treat as a cache, never as truth."""

    provenance: Provenance = Provenance.MODEL

    source_chunk_id: str | None = None
    """When synthesised, the chunk the question was generated from."""

    labels: dict[str, Any] = field(default_factory=dict)
    """Free-form tags: difficulty, hard_negative, topic, reviewer, ..."""

    schema_version: int = FIXTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("eval case id must be a non-empty string")
        if not self.question.strip():
            raise ValueError(f"case {self.id!r} has an empty question")
        if not self.gold_spans and not self.gold_doc_ids:
            raise ValueError(
                f"case {self.id!r} has no gold evidence: set gold_spans "
                "(preferred) or at least gold_doc_ids, otherwise a failure "
                "cannot be attributed to a stage"
            )

    @property
    def is_trusted(self) -> bool:
        """True when a human stands behind this label."""
        return self.provenance in (Provenance.HUMAN, Provenance.MODEL_VERIFIED)

    @property
    def implied_doc_ids(self) -> frozenset[str]:
        """Every document this case points at, from spans and explicit ids alike."""
        return frozenset(self.gold_doc_ids) | frozenset(s.doc_id for s in self.gold_spans)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What a pipeline returned for one query, at both observable cut points.

    Keeping `candidates` and `context` separate is what makes RETRIEVAL_MISS and
    RANKING_MISS distinguishable at all. A pipeline that only exposes its final
    context can still be scored, but those two verdicts collapse into one -- the
    adapter docs say so, because it determines how much an integration is worth.
    """

    query: str
    candidates: tuple[Chunk, ...]
    """Everything retrieval returned, before any rerank or truncation."""

    context: tuple[Chunk, ...]
    """What actually reached the model, in the order it was given."""

    answer: str | None = None
    latency_ms: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def exposes_ranking_stage(self) -> bool:
        """False when candidates and context are identical, collapsing two verdicts."""
        return tuple(c.id for c in self.candidates) != tuple(c.id for c in self.context)


def stable_id(*parts: str, prefix: str = "", length: int = 12) -> str:
    """Deterministic short id built from the given parts.

    Used for chunk and case ids so that re-running ingestion over unchanged input
    produces unchanged ids -- which is what makes a fixture reusable across runs,
    and what makes re-embedding skippable.
    """
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=16).hexdigest()
    return f"{prefix}{digest[:length]}" if prefix else digest[:length]
