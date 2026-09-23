"""Chunk-grounded question synthesis.

Every competing tool assumes the golden set already exists. This module is the
first half of making one: propose candidate questions that a person can then
accept or reject in seconds (see `labelling`).

Three design decisions carry the weight:

1. **Grounded in one chunk.** Each question is written from a single passage,
   and the model must return a *verbatim* quote of the evidence. Retrace then
   finds that quote in the passage and converts it into an exact `Span`. A quote
   that is not actually in the passage is a hallucinated citation, and the
   candidate is rejected before a human ever sees it.

2. **Lexical-leakage scoring.** Synthetic questions tend to copy phrasing
   straight out of the source ("What does the multilayer insulation blanket
   reduce?"), which makes them trivially easy for keyword retrieval and quietly
   inflates recall. Every candidate is scored for how much of its vocabulary it
   borrowed from its source passage, and the worst offenders are rejected by
   default.

3. **Nothing synthesised is trusted.** Candidates carry `Provenance.MODEL` and
   stay untrusted until a human accepts them in the labelling loop.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .baseline import tokenize
from .errors import LLMError
from .llm import LLM, as_llm
from .types import Chunk, EvalCase, Provenance, Span, stable_id

DEFAULT_MIN_CHUNK_CHARS = 200
DEFAULT_MAX_OVERLAP = 0.8

QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": "False if the passage has no factual content worth asking about.",
        },
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {
            "type": "string",
            "description": "An exact, contiguous quote copied from the passage.",
        },
    },
    "required": ["answerable", "question", "answer", "evidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You write evaluation questions for a document search system. Each question will \
be used to test whether the system can find the passage it came from.

Write one question that a real user of this document collection might plausibly \
ask, and that the passage answers directly.

What makes a good question here:
- Phrase it the way someone who has NOT read the passage would ask it. Use your \
own words rather than lifting the passage's distinctive phrases; a question that \
repeats the passage's wording tests keyword matching, not search.
- It must be answerable from this passage alone, without outside knowledge.
- It should have one clear answer. Avoid yes/no questions and questions about \
the document itself ("what does this section discuss?").

For `evidence`, copy the shortest contiguous stretch of the passage that \
contains the answer, character for character. Do not paraphrase, fix typos, \
or join separate sentences; the quote is located in the passage by exact match.

If the passage is boilerplate, a table of contents, a reference list, or \
otherwise has nothing worth asking about, set `answerable` to false and leave \
the other fields as empty strings."""

PROMPT_TEMPLATE = """\
Document: {doc_id}

<passage>
{text}
</passage>"""

_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were be been
    being it its this that these those what which who whom whose when where why how
    do does did done can could should would will shall may might must has have had
    not no nor so than then there their them they we you your our i me my he she his
    her us into onto over under about above below between during through per via
    any some each every all both either neither such same other own very too just
    also only more most much many few less least one two
    """.split()  # noqa: SIM905 - a word list reads better as prose
)


@dataclass(frozen=True, slots=True)
class Rejection:
    """A candidate that never reached a human, and why."""

    chunk_id: str
    reason: str
    detail: str = ""
    question: str | None = None


@dataclass
class SynthesisResult:
    cases: list[EvalCase] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rejections:
            counts[r.reason] = counts.get(r.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def sample_chunks(
    chunks: Iterable[Chunk],
    n: int,
    *,
    seed: int = 0,
    min_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Pick `n` chunks, spread across documents, deterministically.

    Round-robins over documents so one 400-page report cannot supply every
    question. Chunks shorter than `min_chars` are skipped: fragments, headers and
    trailing windows produce thin questions.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    rng = random.Random(seed)
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if len(chunk.text.strip()) >= min_chars:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)

    # Sort before shuffling so the sample is identical regardless of the order
    # the corpus happened to be iterated in.
    doc_ids = sorted(by_doc)
    rng.shuffle(doc_ids)
    for doc_id in doc_ids:
        by_doc[doc_id].sort(key=lambda c: c.id)
        rng.shuffle(by_doc[doc_id])

    picked: list[Chunk] = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for doc_id in doc_ids:
            bucket = by_doc[doc_id]
            if depth < len(bucket):
                picked.append(bucket[depth])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        depth += 1
    return picked


# --------------------------------------------------------------------------
# Evidence location
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def locate_evidence(passage: str, quote: str) -> tuple[int, int] | None:
    """Find `quote` in `passage`, tolerating whitespace and case differences.

    Returns ``(start, end)`` offsets into the *original* passage, or None. Models
    routinely collapse a line break into a space or change the case of a first
    letter; neither makes the quote a fabrication. Anything more than that --
    a changed word, a fixed typo, a joined sentence -- does, and fails here.
    """
    quote = quote.strip()
    if not quote:
        return None

    exact = passage.find(quote)
    if exact >= 0:
        return exact, exact + len(quote)

    # Build a whitespace-collapsed, lowercased copy of the passage, remembering
    # where each normalised character came from in the original.
    norm_chars: list[str] = []
    origin: list[int] = []
    in_space = False
    for i, ch in enumerate(passage):
        if ch.isspace():
            if not in_space and norm_chars:
                norm_chars.append(" ")
                origin.append(i)
            in_space = True
        else:
            norm_chars.append(ch.lower())
            origin.append(i)
            in_space = False
    norm_passage = "".join(norm_chars)
    norm_quote = _WS_RE.sub(" ", quote).lower()

    pos = norm_passage.find(norm_quote)
    if pos < 0:
        return None
    start = origin[pos]
    end = origin[pos + len(norm_quote) - 1] + 1
    return start, end


def lexical_overlap(question: str, evidence: str) -> float:
    """Fraction of the question's content words that appear in the evidence.

    0.0 means the question shares no distinctive vocabulary with its source;
    1.0 means every content word was lifted from it. Stopwords are ignored,
    because "what is the" carries no retrieval signal either way.
    """
    q_terms = {t for t in tokenize(question) if t not in _STOPWORDS and len(t) > 1}
    if not q_terms:
        return 0.0
    e_terms = set(tokenize(evidence))
    return len(q_terms & e_terms) / len(q_terms)


def _normalise_question(q: str) -> str:
    return " ".join(tokenize(q))


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------


def iter_synthesize(
    chunks: Iterable[Chunk],
    llm: Any,
    *,
    max_overlap: float | None = DEFAULT_MAX_OVERLAP,
) -> Iterator[EvalCase | Rejection]:
    """Yield one candidate case or one rejection per chunk, as they are produced.

    A generator so a long run can be written incrementally and interrupted
    without losing what was already paid for.

    Per-item model failures (a refusal, unusable output) become rejections.
    Anything else -- auth, network, a bug -- propagates and stops the run.
    """
    model = as_llm(llm)
    model_name = getattr(model, "name", type(model).__name__)
    seen_questions: set[str] = set()

    for chunk in chunks:
        try:
            raw = model.complete_json(
                system=SYSTEM_PROMPT,
                prompt=PROMPT_TEMPLATE.format(doc_id=chunk.doc_id, text=chunk.text),
                schema=QUESTION_SCHEMA,
            )
        except LLMError as exc:
            yield Rejection(chunk.id, "model_error", str(exc))
            continue

        yield _to_case(chunk, raw, model_name, seen_questions, max_overlap)


def _to_case(
    chunk: Chunk,
    raw: dict[str, Any],
    model_name: str,
    seen: set[str],
    max_overlap: float | None,
) -> EvalCase | Rejection:
    if not raw.get("answerable", False):
        return Rejection(chunk.id, "not_question_worthy")

    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    evidence = str(raw.get("evidence", "")).strip()

    if not question:
        return Rejection(chunk.id, "empty_question")
    if not evidence:
        return Rejection(chunk.id, "no_evidence", question=question)

    located = locate_evidence(chunk.text, evidence)
    if located is None:
        return Rejection(
            chunk.id,
            "evidence_not_verbatim",
            detail=f"quote not found in passage: {evidence[:120]!r}",
            question=question,
        )

    key = _normalise_question(question)
    if key in seen:
        return Rejection(chunk.id, "duplicate_question", question=question)

    # Measured against the whole chunk, not just the quoted evidence: retrieval
    # matches chunks, so vocabulary copied from anywhere in the passage makes the
    # question easier to find than a real user's question would be.
    overlap = lexical_overlap(question, chunk.text)
    if max_overlap is not None and overlap > max_overlap:
        return Rejection(
            chunk.id,
            "lexical_leakage",
            detail=f"{overlap:.0%} of the question's content words appear in the passage",
            question=question,
        )
    seen.add(key)

    start, end = located
    labels: dict[str, Any] = {
        "synthesized_by": model_name,
        "lexical_overlap": round(overlap, 3),
    }

    if chunk.span is not None:
        gold_spans: tuple[Span, ...] = (
            Span(chunk.doc_id, chunk.span.start + start, chunk.span.start + end),
        )
    else:
        # Without chunk offsets the span cannot be placed in the document; fall
        # back to the chunk id as a cache and the document id as truth.
        gold_spans = ()
        labels["span_unavailable"] = True

    return EvalCase(
        id=stable_id(chunk.id, key, prefix="q_"),
        question=question,
        answer=answer or None,
        gold_spans=gold_spans,
        gold_doc_ids=(chunk.doc_id,),
        gold_chunk_ids=(chunk.id,),
        provenance=Provenance.MODEL,
        source_chunk_id=chunk.id,
        labels=labels,
    )


def synthesize(
    chunks: Iterable[Chunk],
    llm: LLM | Any,
    *,
    max_overlap: float | None = DEFAULT_MAX_OVERLAP,
) -> SynthesisResult:
    """Run `iter_synthesize` to completion and collect the results."""
    result = SynthesisResult()
    for item in iter_synthesize(chunks, llm, max_overlap=max_overlap):
        if isinstance(item, Rejection):
            result.rejections.append(item)
        else:
            result.cases.append(item)
    return result
