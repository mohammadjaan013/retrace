"""Chunking.

Week 1 ships exactly one strategy, and it is a bad one on purpose.

Retrace needs a broken pipeline to diagnose. A fixed-width character window that
splits mid-sentence and mid-word produces the failure modes the attribution
cascade exists to identify, and it is the honest "before" in every case study.
Better strategies arrive in Week 5, once there are numbers to beat.
"""

from __future__ import annotations

import re
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


# --------------------------------------------------------------------------
# Structure-aware chunking (Week 5)
# --------------------------------------------------------------------------

_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n+")
# A sentence ends at . ! or ? followed by whitespace and something that can
# start a sentence. Deliberately simple: abbreviations ("e.g. the") will
# occasionally split early, which costs a slightly short chunk, never lost text.
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")


def _sentences(text: str, base: int) -> list[tuple[int, int]]:
    """(start, end) offsets of sentences in `text`, shifted by `base`."""
    out: list[tuple[int, int]] = []
    pos = 0
    for match in _SENTENCE_RE.finditer(text):
        end = match.start()
        if text[pos:end].strip():
            out.append((base + pos, base + end))
        pos = match.end()
    if text[pos:].strip():
        out.append((base + pos, base + len(text.rstrip())))
    return out


def sentence_chunks(
    doc_id: str,
    text: str,
    *,
    max_chars: int = 600,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Pack whole sentences into chunks, never crossing a paragraph break.

    The fix for the fixed-width chunker's worst habit -- cutting an answer in
    half at an arbitrary character. Sentences are never split (a single
    sentence longer than `max_chars` becomes its own chunk), paragraphs are
    hard boundaries, and the last `overlap_sentences` of each chunk are
    repeated at the start of the next, so an answer that spans a boundary is
    still whole in at least one chunk.

    Spans are exact offsets into `text`, so gold labels stay valid.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    if overlap_sentences < 0:
        raise ValueError(f"overlap_sentences must be >= 0, got {overlap_sentences}")

    chunks: list[Chunk] = []
    index = 0
    para_start = 0
    boundaries = [m.span() for m in _PARAGRAPH_RE.finditer(text)] + [(len(text), len(text))]

    for para_end, next_start in boundaries:
        paragraph = text[para_start:para_end]
        sentences = _sentences(paragraph, para_start)
        para_start = next_start
        i = 0
        while i < len(sentences):
            j = i
            while j + 1 < len(sentences) and sentences[j + 1][1] - sentences[i][0] <= max_chars:
                j += 1
            start, end = sentences[i][0], sentences[j][1]
            chunks.append(Chunk(
                id=stable_id(doc_id, "s", str(start), str(end), prefix="c_"),
                doc_id=doc_id,
                text=text[start:end],
                span=Span(doc_id=doc_id, start=start, end=end),
                meta={"chunk_index": index, "strategy": "sentence"},
            ))
            index += 1
            if j + 1 >= len(sentences):
                break
            # Step forward, keeping up to `overlap_sentences` for context. The
            # overlap is dropped when the next sentence is too long to fit after
            # it: that chunk would add nothing new, only a duplicate that
            # competes with its parent in retrieval.
            nxt = max(i + 1, j + 1 - overlap_sentences)
            if nxt <= j and sentences[j + 1][1] - sentences[nxt][0] > max_chars:
                nxt = j + 1
            i = nxt
    return chunks
