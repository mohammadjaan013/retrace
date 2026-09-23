"""Reference retrievers: the fixes the verdicts point at.

Each one answers a specific verdict from the attribution cascade:

* `BM25Retriever`     -- RETRIEVAL_MISS that was "findable by keyword": the
                         naive baseline weighs "the" like "radiator". BM25 adds
                         IDF and length normalisation. Zero dependencies.
* `DenseRetriever`    -- RETRIEVAL_MISS from vocabulary mismatch: "get rid of
                         warmth" never matches "reject heat" lexically. Needs an
                         embedding model, supplied by you (any callable).
* `HybridRetriever`   -- fuses several retrievers with reciprocal rank fusion,
                         and records each leg's rank so the report can show
                         *which* retriever found the evidence.
* `RerankingPipeline` -- a Tier 2 pipeline: retrieve wide, rerank, keep a few.
                         Exposes both lists, so RANKING_MISS becomes observable.

These are references, not a vector database. They are exact, in-memory and
readable in one sitting -- which is what you want when the question is "did this
change help?", and what you do not want for ten million chunks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .protocols import Generator, as_retriever
from .text import STOPWORDS, fold_plural, tokenize
from .types import Chunk, RetrievalResult


def analyze(text: str) -> list[str]:
    """BM25's analyser: tokenise, drop stopwords, fold regular plurals."""
    return [fold_plural(t) for t in tokenize(text) if t not in STOPWORDS]


def _with_meta(chunk: Chunk, score: float, **extra: Any) -> Chunk:
    meta = dict(chunk.meta)
    meta.update(extra)
    return Chunk(id=chunk.id, doc_id=chunk.doc_id, text=chunk.text, score=score,
                 span=chunk.span, meta=meta)


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


class BM25Retriever:
    """Okapi BM25 over an in-memory chunk list.

    ``k1`` controls term-frequency saturation and ``b`` length normalisation;
    the defaults are the textbook values and rarely worth tuning before the
    chunking is right.
    """

    name = "bm25"

    def __init__(self, chunks: Iterable[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        df: Counter[str] = Counter()
        for chunk in self._chunks:
            terms = analyze(chunk.text)
            counts = Counter(terms)
            self._tf.append(counts)
            self._len.append(len(terms))
            df.update(counts.keys())
        n = len(self._chunks)
        self._avg_len = (sum(self._len) / n) if n else 0.0
        # The "+1" variant keeps IDF positive for terms in over half the corpus.
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def retrieve(self, query: str, k: int) -> Sequence[Chunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        q_terms = [t for t in set(analyze(query)) if t in self._idf]
        if not q_terms:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg_len or 1.0))
            score = 0.0
            for term in q_terms:
                f = tf.get(term)
                if f:
                    score += self._idf[term] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda p: (-p[0], self._chunks[p[1]].id))
        return [self._chunks[i].with_score(round(s, 6)) for s, i in scored[:k]]

    def __repr__(self) -> str:
        return f"BM25Retriever({len(self._chunks)} chunks)"


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into vectors. Normalisation is not assumed."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class CallableEmbedder:
    """Adapts ``fn(texts) -> vectors`` -- wrap whatever embedding client you use."""

    def __init__(self, fn: Callable[[Sequence[str]], Any], name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "embedder")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(list(texts))]


class SentenceTransformerEmbedder:
    """Local embeddings via ``sentence-transformers`` (``retrace[dense]``).

    Imported lazily. The first use downloads the model weights from Hugging
    Face -- a few hundred megabytes for the default -- so it is never loaded
    unless asked for by name.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", model: Any = None,
                 batch_size: int = 64) -> None:
        self.name = f"st:{model_name}"
        self.batch_size = batch_size
        if model is None:  # pragma: no cover - needs the optional dependency
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    'SentenceTransformerEmbedder needs: pip install "retrace[dense]"'
                ) from exc
            model = SentenceTransformer(model_name)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True
        )
        return [list(map(float, v)) for v in vectors]


def _normalise(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else list(v)


class EmbeddingCache:
    """Content-addressed vector cache, optionally persisted as JSONL.

    Keyed by a hash of (embedder name, text), so changing either invalidates
    exactly the entries it should. Re-running an evaluation over an unchanged
    corpus then costs nothing -- which matters, because a tool that quietly
    re-spends the embedding budget on every run gets uninstalled.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else None
        self._store: dict[str, list[float]] = {}
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self._store[record["k"]] = record["v"]

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.blake2b(f"{model}\x1f{text}".encode(), digest_size=16).hexdigest()

    def get_many(self, embedder: Any, texts: Sequence[str]) -> list[list[float]]:
        model = getattr(embedder, "name", type(embedder).__name__)
        keys = [self.key(model, t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._store]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vectors = embedder.embed([texts[i] for i in missing])
            if len(vectors) != len(missing):
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for {len(missing)} texts"
                )
            new = {keys[i]: _normalise(v) for i, v in zip(missing, vectors, strict=True)}
            self._store.update(new)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    for k, v in new.items():
                        handle.write(json.dumps({"k": k, "v": v}) + "\n")
        return [self._store[k] for k in keys]


class DenseRetriever:
    """Exact cosine similarity over embedded chunks."""

    name = "dense"

    def __init__(
        self,
        chunks: Iterable[Chunk],
        embedder: Any,
        *,
        cache: EmbeddingCache | None = None,
    ) -> None:
        if not hasattr(embedder, "embed"):
            embedder = CallableEmbedder(embedder)
        self._embedder = embedder
        self._cache = cache or EmbeddingCache()
        self._chunks = list(chunks)
        self._vectors = self._cache.get_many(embedder, [c.text for c in self._chunks])

    def retrieve(self, query: str, k: int) -> Sequence[Chunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        (q,) = self._cache.get_many(self._embedder, [query])
        scored = [
            (sum(a * b for a, b in zip(q, v, strict=True)), i)
            for i, v in enumerate(self._vectors)
        ]
        scored.sort(key=lambda p: (-p[0], self._chunks[p[1]].id))
        return [self._chunks[i].with_score(round(s, 6)) for s, i in scored[:k]]

    def __repr__(self) -> str:
        return f"DenseRetriever({len(self._chunks)} chunks, {getattr(self._embedder, 'name', '?')})"


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


class HybridRetriever:
    """Reciprocal rank fusion across named retrievers.

    RRF scores a chunk ``sum(1 / (c + rank))`` over the legs that returned it.
    It ignores raw scores entirely, which is the point: BM25 scores and cosine
    similarities live on incomparable scales, and rank is the one thing they
    share. Every returned chunk carries ``meta["legs"]`` -- its rank in each leg,
    or None -- so a trace shows which retriever actually found the evidence.
    """

    def __init__(self, retrievers: dict[str, Any], *, depth: int = 50, c: int = 60) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._legs = {name: as_retriever(r) for name, r in retrievers.items()}
        self.depth, self.c = depth, c

    def retrieve(self, query: str, k: int) -> Sequence[Chunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        fused: dict[str, float] = {}
        ranks: dict[str, dict[str, int | None]] = {}
        by_id: dict[str, Chunk] = {}
        for name, leg in self._legs.items():
            for rank, chunk in enumerate(leg.retrieve(query, max(k, self.depth)), start=1):
                by_id.setdefault(chunk.id, chunk)
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.c + rank)
                ranks.setdefault(chunk.id, {})[name] = rank
        ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))[:k]
        return [
            _with_meta(
                by_id[cid], round(fused[cid], 6),
                legs={name: ranks[cid].get(name) for name in self._legs},
            )
            for cid in ordered
        ]

    def __repr__(self) -> str:
        return f"HybridRetriever({' + '.join(self._legs)})"


# --------------------------------------------------------------------------
# Reranking -- Tier 2
# --------------------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Return the chunks reordered best-first, with scores attached."""
        ...


class CrossEncoderReranker:
    """A cross-encoder via ``sentence-transformers`` (``retrace[dense]``).

    Imported lazily; the first use downloads model weights.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", model: Any = None) -> None:
        self.name = f"ce:{model_name}"
        if model is None:  # pragma: no cover - needs the optional dependency
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    'CrossEncoderReranker needs: pip install "retrace[dense]"'
                ) from exc
            model = CrossEncoder(model_name)
        self._model = model

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].id))
        return [_with_meta(chunks[i], round(float(scores[i]), 6), reranked_from=i + 1)
                for i in order]


class RerankingPipeline:
    """Retrieve ``candidates_k``, rerank, keep ``context_k``: a Tier 2 pipeline.

    With no reranker it simply truncates -- still Tier 2, because the cut
    between what was retrieved and what reached the model is now visible, which
    is exactly the observation RANKING_MISS needs.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        reranker: Reranker | None = None,
        candidates_k: int = 50,
        context_k: int = 5,
        generator: Generator | None = None,
    ) -> None:
        if not 1 <= context_k <= candidates_k:
            raise ValueError("need 1 <= context_k <= candidates_k")
        self._retriever = as_retriever(retriever)
        self._reranker = reranker
        self.candidates_k, self.context_k = candidates_k, context_k
        self._generator = generator

    def run(self, query: str) -> RetrievalResult:
        started = time.perf_counter()
        candidates = tuple(self._retriever.retrieve(query, self.candidates_k))
        ordered = self._reranker.rerank(query, candidates) if self._reranker else list(candidates)
        context = tuple(ordered[: self.context_k])
        answer = self._generator.generate(query, context) if self._generator else None
        return RetrievalResult(
            query=query,
            candidates=candidates,
            context=context,
            answer=answer,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            meta={"tier": 2, "candidates_k": self.candidates_k, "context_k": self.context_k,
                  "reranker": getattr(self._reranker, "name", None)},
        )

    def __repr__(self) -> str:
        reranker = getattr(self._reranker, "name", "truncate")
        return (f"RerankingPipeline({self._retriever!r}, {reranker}, "
                f"{self.candidates_k}->{self.context_k})")
