"""The deliberately naive baseline retriever.

Zero dependencies, zero configuration, zero API keys -- `pip install retrace-rag`
and it runs. That matters for adoption: the quickstart must work before anybody
has decided to trust the tool enough to wire in their own stack.

It is also genuinely weak, which is the more important property. Retrace needs
something broken to diagnose, and the specific ways this fails are instructive:

* **No IDF.** Common words carry the same weight as rare ones, so a query's
  "the" and "of" drown out its one distinguishing technical term. This is the
  dominant source of RETRIEVAL_MISS in the Week 1 baseline.
* **Exact token matching only.** No stemming, no synonyms, no embeddings, so
  "thermal control" never matches "heat rejection" -- a vocabulary mismatch that
  looks exactly like a CORPUS_GAP until attribution separates them.
* **Chunk-length bias.** Cosine over raw term counts mildly favours short chunks.

Week 5 replaces this with BM25 plus vectors, fused. The delta between the two is
the case study.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from .text import tokenize
from .types import Chunk

__all__ = ["NaiveRetriever", "tokenize"]


class NaiveRetriever:
    """Bag-of-words cosine similarity over raw term frequencies.

    Builds its index eagerly at construction. For the corpus sizes an audit
    touches this is a fraction of a second and avoids a lazy-init branch in the
    hot path.
    """

    __slots__ = ("_chunks", "_norms", "_vectors")

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._chunks: list[Chunk] = list(chunks)
        self._vectors: list[Counter[str]] = []
        self._norms: list[float] = []

        for chunk in self._chunks:
            counts = Counter(tokenize(chunk.text))
            self._vectors.append(counts)
            self._norms.append(math.sqrt(sum(v * v for v in counts.values())))

    def retrieve(self, query: str, k: int) -> Sequence[Chunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        q_counts = Counter(tokenize(query))
        q_norm = math.sqrt(sum(v * v for v in q_counts.values()))
        if q_norm == 0.0:
            return []

        scored: list[tuple[float, int]] = []
        for i, (vec, norm) in enumerate(zip(self._vectors, self._norms, strict=True)):
            if norm == 0.0:
                continue
            # Iterate the query's terms, not the chunk's: queries are short.
            dot = sum(count * vec[term] for term, count in q_counts.items() if term in vec)
            if dot == 0.0:
                continue
            scored.append((dot / (q_norm * norm), i))

        if not scored:
            return []

        # Sort by score desc, then by chunk id asc so ties are deterministic --
        # a retriever whose output depends on dict ordering makes every
        # before/after comparison untrustworthy.
        scored.sort(key=lambda pair: (-pair[0], self._chunks[pair[1]].id))

        return [self._chunks[i].with_score(score) for score, i in scored[:k]]

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"NaiveRetriever({len(self._chunks)} chunks)"
