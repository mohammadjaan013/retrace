"""Retrace -- find out which stage of your RAG pipeline is actually broken.

Ragas tells you your RAG scores 0.72.
Retrace tells you which questions failed, why, and what to fix first.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from .attribution import Attribution, attribute
from .baseline import NaiveRetriever, tokenize
from .chunking import chunk_documents, fixed_size_chunks
from .corpus import (
    InMemoryCorpus,
    build_corpus,
    corpus_covers,
    coverage_warning,
    load_documents,
    resolve_gold_chunks,
    retrieval_coverage,
)
from .errors import (
    CorpusError,
    FixtureError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    RetraceError,
    SchemaVersionError,
)
from .fixtures import (
    case_from_dict,
    case_to_dict,
    fixture_stats,
    iter_fixture,
    read_fixture,
    write_fixture,
)
from .judge import AnswerJudge, Judgement, LLMJudge, TokenRecallJudge
from .llm import LLM, AnthropicLLM, as_llm
from .protocols import (
    Corpus,
    Generator,
    Pipeline,
    Retriever,
    RetrieverPipeline,
    as_retriever,
)
from .runner import RunAbortedError, RunResult, evaluate, load_run, save_run
from .synthesis import Rejection, SynthesisResult, synthesize
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

__all__ = [
    "FIXTURE_SCHEMA_VERSION",
    "LLM",
    "AnswerJudge",
    "AnthropicLLM",
    "Attribution",
    "Chunk",
    "Corpus",
    "CorpusError",
    "EvalCase",
    "FixtureError",
    "Generator",
    "InMemoryCorpus",
    "Judgement",
    "LLMError",
    "LLMJudge",
    "LLMOutputError",
    "LLMRefusalError",
    "NaiveRetriever",
    "Pipeline",
    "Provenance",
    "Rejection",
    "RetraceError",
    "RetrievalResult",
    "Retriever",
    "RetrieverPipeline",
    "RunAbortedError",
    "RunResult",
    "SchemaVersionError",
    "Span",
    "SynthesisResult",
    "TokenRecallJudge",
    "Verdict",
    "__version__",
    "as_llm",
    "as_retriever",
    "attribute",
    "build_corpus",
    "case_from_dict",
    "case_to_dict",
    "chunk_documents",
    "corpus_covers",
    "coverage_warning",
    "evaluate",
    "fixed_size_chunks",
    "fixture_stats",
    "iter_fixture",
    "load_documents",
    "load_run",
    "read_fixture",
    "resolve_gold_chunks",
    "retrieval_coverage",
    "save_run",
    "stable_id",
    "synthesize",
    "tokenize",
    "write_fixture",
]
