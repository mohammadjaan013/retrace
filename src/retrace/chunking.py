"""Chunking.

Week 1 ships exactly one strategy, and it is a bad one on purpose.

Retrace needs a broken pipeline to diagnose. A fixed-width character window that
splits mid-sentence and mid-word produces the failure modes the attribution
cascade exists to identify, and it is the honest "before" in every case study.
Better strategies arrive in Week 5, once there are numbers to beat.
"""

from __future__ import annotations

from collections.abc import Iterator

from .types import Chunk, Span, stable_id

DEFAULT_CHUNK_CHARS = 1200
"""Roughly 300 tokens of English prose. Characters, not tokens, so that chunking
is exactly reproducible without pulling in a tokenizer -- the core stays
dependency-free, and the approximation is stated rather than hidden."""


def fixed_size_chunks(
    doc_id: str,
    text: str,
    *,
    size: int = DEFAULT_CHUNK_CHARS,
    overlap: int = 0,
) -> list[Chunk]:
    """Split `text` into fixed-width character windows.

    Deliberately naive: no sentence awareness, no paragraph awareness, no
    structure awareness. Words and sentences are cut in half at the boundaries,
    and an answer that straddles two windows is in neither of them cleanly.
    That is the point.

    Every chunk carries its `Span`, which is what lets a golden set labelled
    against this chunking survive being re-chunked later.
    """
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    return list(_windows(doc_id, text, size, overlap))


def _windows(doc_id: str, text: str, size: int, overlap: int) -> Iterator[Chunk]:
    if not text:
        return

    stride = size - overlap
    index = 0
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        body = text[start:end]

        # A trailing window that is pure whitespace carries no evidence and only
        # adds noise to retrieval. Skip it rather than index an empty chunk.
        if body.strip():
            span = Span(doc_id=doc_id, start=start, end=end)
            yield Chunk(
                id=stable_id(doc_id, str(start), str(end), prefix="c_"),
                doc_id=doc_id,
                text=body,
                span=span,
                meta={"chunk_index": index, "strategy": "fixed_size"},
            )
            index += 1

        if end == len(text):
            break
        start += stride


def chunk_documents(
    documents: dict[str, str],
    *,
    size: int = DEFAULT_CHUNK_CHARS,
    overlap: int = 0,
) -> list[Chunk]:
    """Apply `fixed_size_chunks` across a mapping of doc_id -> text."""
    out: list[Chunk] = []
    for doc_id, text in documents.items():
        out.extend(fixed_size_chunks(doc_id, text, size=size, overlap=overlap))
    return out
