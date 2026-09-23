"""Corpus storage, and the resolution of gold spans onto current chunks.

`resolve_gold_chunks` is the piece that makes a golden set outlive the chunking
strategy it was labelled against. Without it, recommending a chunking change --
the single most common fix Retrace produces -- would invalidate the very labels
used to prove the change helped.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .errors import CorpusError
from .types import Chunk, EvalCase, Span

# A chunk counts as gold when it shares at least this many characters with a
# gold span. Short spans lower the bar to their own length, so a 10-character
# span still resolves. Tuned to exclude chunks that merely clip a boundary.
DEFAULT_MIN_OVERLAP_CHARS = 32


class InMemoryCorpus:
    """The reference `Corpus`: a dict of chunks, optionally with source documents.

    Fine for the tens of thousands of chunks an audit actually touches. A client
    with a real vector store writes a thin adapter over their own index instead;
    the `Corpus` protocol is four methods precisely so that stays cheap.
    """

    __slots__ = ("_by_doc", "_chunks", "_docs")

    def __init__(
        self,
        chunks: Iterable[Chunk] = (),
        documents: dict[str, str] | None = None,
    ) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._docs: dict[str, str] = dict(documents or {})
        self._by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            self.add(chunk)

    def add(self, chunk: Chunk) -> None:
        existing = self._chunks.get(chunk.id)
        if existing is not None and existing.text != chunk.text:
            raise CorpusError(
                f"chunk id {chunk.id!r} already present with different text; "
                "ids must be stable and unique within a corpus"
            )
        self._chunks[chunk.id] = chunk
        bucket = self._by_doc.setdefault(chunk.doc_id, [])
        if existing is None:
            bucket.append(chunk)

    def add_document(self, doc_id: str, text: str) -> None:
        """Retain a document's full text so gold spans can be resolved exactly."""
        self._docs[doc_id] = text

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def iter_chunks(self) -> Iterator[Chunk]:
        return iter(self._chunks.values())

    def chunks_in(self, doc_id: str) -> list[Chunk]:
        return list(self._by_doc.get(doc_id, ()))

    def document_text(self, doc_id: str) -> str | None:
        return self._docs.get(doc_id)

    @property
    def doc_ids(self) -> frozenset[str]:
        return frozenset(self._by_doc)

    def __repr__(self) -> str:
        return f"InMemoryCorpus({len(self._chunks)} chunks, {len(self._by_doc)} docs)"


def resolve_gold_chunks(
    case: EvalCase,
    corpus,
    *,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> frozenset[str]:
    """Which chunks in *this* corpus hold the evidence for *this* case.

    Resolution runs most-precise-first and stops at the first tier that produces
    anything, so a precise label is never diluted by a coarse fallback:

    1. **Spans.** A chunk is gold when it overlaps a gold span by at least
       `min_overlap_chars`, or by the whole span when the span is shorter than
       that. Chunking-independent, and the tier every synthesised case gets.
    2. **Cached chunk ids.** Used only when they still exist in this corpus --
       they are stale the moment chunking changes, which is why they are a cache
       and never the source of truth.
    3. **Document ids.** Every chunk of the named documents. Coarse, but it is
       enough to separate CORPUS_GAP from a retrieval failure, which is the
       distinction that matters most.

    Returns an empty set when nothing resolves -- which is itself the signal for
    CORPUS_GAP, and is why this function never raises on a miss.
    """
    if case.gold_spans:
        resolved = _resolve_by_span(case.gold_spans, corpus, min_overlap_chars)
        if resolved:
            return resolved

    if case.gold_chunk_ids:
        live = {cid for cid in case.gold_chunk_ids if cid in corpus}
        if live:
            return frozenset(live)

    if case.gold_doc_ids:
        by_doc = {
            chunk.id
            for chunk in corpus.iter_chunks()
            if chunk.doc_id in case.gold_doc_ids
        }
        if by_doc:
            return frozenset(by_doc)

    return frozenset()


def _resolve_by_span(
    gold_spans: tuple[Span, ...],
    corpus,
    min_overlap_chars: int,
) -> frozenset[str]:
    matched: set[str] = set()
    for chunk in corpus.iter_chunks():
        if chunk.span is None:
            continue
        for gold in gold_spans:
            if chunk.span.doc_id != gold.doc_id:
                continue
            threshold = min(min_overlap_chars, gold.length)
            if chunk.span.overlap_chars(gold) >= threshold:
                matched.add(chunk.id)
                break
    return frozenset(matched)


def corpus_covers(case: EvalCase, corpus) -> bool:
    """True when the corpus holds any evidence for the case.

    False is the precondition for CORPUS_GAP. Kept as its own named function
    because it is the single most commercially important check in the library,
    and it should be readable by someone skimming the source on a sales call.
    """
    return bool(resolve_gold_chunks(case, corpus))


# Above this fraction, recall@k stops measuring retrieval quality and starts
# measuring how much of the corpus fits in k. Chosen as the point where a random
# retriever would already score well enough to look competent.
COVERAGE_WARNING_THRESHOLD = 0.10


def retrieval_coverage(k: int, corpus) -> float:
    """What fraction of the corpus a single top-k retrieval returns."""
    size = sum(1 for _ in corpus.iter_chunks())
    if size == 0:
        return 0.0
    return min(1.0, k / size)


def coverage_warning(k: int, corpus) -> str | None:
    """Warn when k is large enough relative to the corpus to inflate recall.

    Found the hard way while building the Week 1 baseline: over a four-document
    sample, k=5 against a five-chunk corpus scored 100% recall. The pipeline had
    not got anything right -- it had returned the entire corpus.

    Any measurement tool that can report a number this misleading without
    comment does not deserve to be trusted with the numbers that matter.
    Returns None when the ratio is safe.
    """
    coverage = retrieval_coverage(k, corpus)
    if coverage < COVERAGE_WARNING_THRESHOLD:
        return None
    size = sum(1 for _ in corpus.iter_chunks())
    return (
        f"k={k} returns {coverage:.0%} of a {size}-chunk corpus. "
        "Recall@k is inflated and near-meaningless at this ratio -- a retriever "
        "returning random chunks would score well. Use a larger corpus or a "
        "smaller k before quoting these numbers."
    )


# --------------------------------------------------------------------------
# Documents on disk
# --------------------------------------------------------------------------


def load_documents(path: str | os.PathLike[str]) -> dict[str, str]:
    """Load a document collection as ``{doc_id: text}``.

    Two formats, chosen by extension:

    * ``.json``  -- a single object mapping doc id to text.
    * ``.jsonl`` -- one ``{"id": ..., "text": ...}`` object per line, which is
      what streams out of most ingestion pipelines and scales past memory-sized
      JSON documents.

    Text is returned exactly as stored. Gold spans are character offsets into
    it, so this function must never normalise whitespace or line endings.
    """
    path = Path(path)
    if not path.exists():
        raise CorpusError(f"document file not found: {path}")

    if path.suffix == ".jsonl":
        docs: dict[str, str] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    doc_id, text = record["id"], record["text"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise CorpusError(
                        f"{path}:{lineno}: expected {{\"id\": ..., \"text\": ...}}: {exc}"
                    ) from exc
                if doc_id in docs:
                    raise CorpusError(f"{path}:{lineno}: duplicate document id {doc_id!r}")
                docs[str(doc_id)] = str(text)
        return docs

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise CorpusError(f"{path}: expected a JSON object mapping document id to text")
    return {str(k): v for k, v in data.items()}


def build_corpus(
    documents: dict[str, str],
    chunker=None,
) -> InMemoryCorpus:
    """Chunk a document collection into an `InMemoryCorpus` that keeps the source text.

    `chunker` is any ``fn(doc_id, text) -> list[Chunk]``; the default is the
    naive fixed-width chunker.
    """
    if chunker is None:
        from .chunking import fixed_size_chunks

        chunker = fixed_size_chunks
    corpus = InMemoryCorpus(documents=documents)
    for doc_id, text in documents.items():
        for chunk in chunker(doc_id, text):
            corpus.add(chunk)
    return corpus
