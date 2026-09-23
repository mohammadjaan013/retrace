"""The adapter seam.

Retrace plugs into a pipeline somebody else already built, without asking them
to change it. That makes these protocols the one interface that must not be got
wrong -- everything else in the library is replaceable.

They are structural (`typing.Protocol`), so nothing needs to import Retrace or
subclass anything. A plain function or an existing class that happens to have
the right shape already satisfies them.

INTEGRATION TIERS
-----------------
How much Retrace can tell you depends on how much the pipeline exposes.

Tier 1 -- a retrieve function.  ``retrieve(query, k) -> list[Chunk]``
    Costs about ten minutes to wire up. Yields CORPUS_GAP and RETRIEVAL_MISS.
    RANKING_MISS cannot be separated from RETRIEVAL_MISS, because there is no
    visible cut between "what was retrieved" and "what reached the model".

Tier 2 -- a pipeline.  ``run(query) -> RetrievalResult``
    Exposes the candidate set before reranking, the context after it, and the
    generated answer. Yields all four verdicts. This is the tier worth asking a
    client to reach, and the number one thing to negotiate for in an audit.

Retrace states which tier it ran at on every report. It never silently reports a
verdict the integration could not actually support.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from .types import Chunk, RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """Tier 1: anything that turns a query into ranked chunks.

    Implement as a function, a class with ``__call__``, or a class with a
    ``retrieve`` method -- `as_retriever()` normalises all three.
    """

    def retrieve(self, query: str, k: int) -> Sequence[Chunk]:
        """Return up to `k` chunks, best first.

        Returning fewer than `k` is fine and is not an error. Scores are
        optional; when present they are assumed to be higher-is-better, and
        Retrace never compares them across different retrievers.
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """Turns a query plus retrieved context into an answer.

    Only needed for GENERATION_MISS. Without one, a case whose gold evidence
    reached the context window is reported as OK at the retrieval stage, and the
    report says generation was not evaluated rather than implying it passed.
    """

    def generate(self, query: str, context: Sequence[Chunk]) -> str: ...


@runtime_checkable
class Pipeline(Protocol):
    """Tier 2: a full RAG call with its intermediate stages visible."""

    def run(self, query: str) -> RetrievalResult: ...


@runtime_checkable
class Corpus(Protocol):
    """Read access to what is actually indexed.

    Required for CORPUS_GAP, which is the verdict that cannot be derived from
    retrieval output alone: to say "the answer is not in the corpus" you have to
    be able to look in the corpus. A pipeline that cannot expose this downgrades
    CORPUS_GAP into RETRIEVAL_MISS, and the report says so.
    """

    def __contains__(self, chunk_id: str) -> bool: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def iter_chunks(self) -> Iterator[Chunk]: ...

    def document_text(self, doc_id: str) -> str | None:
        """Full source text, used to resolve gold spans onto current chunks.

        Return None when the original document is not retained. Span resolution
        then falls back to matching against chunk spans alone, which is less
        precise but still chunking-independent.
        """
        ...


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------


class _CallableRetriever:
    """Wraps a bare ``f(query, k)`` callable so it satisfies `Retriever`."""

    __slots__ = ("__name__", "_fn")

    def __init__(self, fn) -> None:
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "callable_retriever")

    def retrieve(self, query: str, k: int) -> Sequence[Chunk]:
        return self._fn(query, k)

    def __repr__(self) -> str:
        return f"<retriever {self.__name__}>"


def as_retriever(obj) -> Retriever:
    """Accept the three shapes people actually have, return a `Retriever`.

    Deliberately permissive. Every minute somebody spends reshaping their code
    to fit an evaluation tool is a minute they spend deciding not to use it.

    Accepts an object with ``.retrieve(query, k)``, an object with
    ``__call__(query, k)``, or a plain function.
    """
    if hasattr(obj, "retrieve") and callable(obj.retrieve):
        return obj
    if callable(obj):
        return _CallableRetriever(obj)
    raise TypeError(
        f"{obj!r} is not usable as a retriever: expected a .retrieve(query, k) "
        "method or a callable taking (query, k)"
    )


class RetrieverPipeline:
    """Lifts a Tier 1 `Retriever` into the `Pipeline` shape.

    `candidates` and `context` are the same sequence, which is exactly what
    `RetrievalResult.exposes_ranking_stage` detects. The attribution cascade
    reads that flag and declines to emit RANKING_MISS rather than guessing.
    """

    __slots__ = ("_generator", "_k", "_retriever")

    def __init__(self, retriever, k: int = 10, generator: Generator | None = None) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._retriever = as_retriever(retriever)
        self._k = k
        self._generator = generator

    def run(self, query: str) -> RetrievalResult:
        import time

        started = time.perf_counter()
        chunks = tuple(self._retriever.retrieve(query, self._k))
        answer = self._generator.generate(query, chunks) if self._generator else None
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return RetrievalResult(
            query=query,
            candidates=chunks,
            context=chunks,
            answer=answer,
            latency_ms=elapsed_ms,
            meta={"tier": 1, "k": self._k},
        )

    def __repr__(self) -> str:
        return f"RetrieverPipeline({self._retriever!r}, k={self._k})"
