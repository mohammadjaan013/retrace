"""Retrace -- find out which stage of your RAG pipeline is actually broken.

Ragas tells you your RAG scores 0.72.
Retrace tells you which questions failed, why, and what to fix first.

Week 1 ships the data model, the adapter seam, fixture I/O, and a deliberately
naive baseline to diagnose. The attribution cascade that assigns `Verdict`
values lands in Week 3.
"""

from __future__ import annotations

from .baseline import NaiveRetriever, tokenize
from .chunking import chunk_documents, fixed_size_chunks
from .corpus import (
    InMemoryCorpus,
    corpus_covers,
    coverage_warning,
    resolve_gold_chunks,
    retrieval_coverage,
)
from .errors import CorpusError, FixtureError, RetraceError, SchemaVersionError
from .fixtures import (
    case_from_dict,
    case_to_dict,
    fixture_stats,
    iter_fixture,
    read_fixture,
    write_fixture,
)
from .protocols import (
    Corpus,
    Generator,
    Pipeline,
    Retriever,
    RetrieverPipeline,
    as_retriever,
)
from .types import (
    FIXTURE_SCHEMA_VERSION,
    Chunk,
    EvalCase,
    Provenance,
    RetrievalResult,
    Span,
    Verdict,
    stable_id,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "FIXTURE_SCHEMA_VERSION",
    "Chunk",
    "Corpus",
    "CorpusError",
    "EvalCase",
    "FixtureError",
    "Generator",
    "InMemoryCorpus",
    "NaiveRetriever",
    "Pipeline",
    "Provenance",
    "RetraceError",
    "RetrievalResult",
    "Retriever",
    "RetrieverPipeline",
    "SchemaVersionError",
    "Span",
    "Verdict",
    "__version__",
    "as_retriever",
    "case_from_dict",
    "case_to_dict",
    "chunk_documents",
    "corpus_covers",
    "coverage_warning",
    "fixed_size_chunks",
    "fixture_stats",
    "iter_fixture",
    "read_fixture",
    "resolve_gold_chunks",
    "retrieval_coverage",
    "stable_id",
    "tokenize",
    "write_fixture",
]
