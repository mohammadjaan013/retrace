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

from .errors import LLMError
from .llm import LLM, as_llm
from .text import content_terms, term_set, tokenize
from .types import Chunk, EvalCase, Provenance, Span, stable_id

DEFAULT_MIN_CHUNK_CHARS = 200
DEFAULT_MAX_OVERLAP = 0.9
"""Questions sharing more than this share of content words with their passage are
rejected. Was 0.8 until the first full run on the NASA corpus: 18 of 24 rejections
sat at 83-93% and read as legitimate questions -- in technical text, names and
terms with no synonyms ("Goldstone", "DTN", "Orion's unregulated bus") are
unavoidable shared vocabulary, not copied phrasing. The near-verbatim copies all
scored 100%. Every candidate keeps its overlap score for review either way."""

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



@dataclass(frozen=True, slots=True)
class Rejection:
    """A candidate that never reached a human, and why."""

    chunk_id: str
    reason: str
    detail: str = ""
    question: str | None = None
    evidence: str | None = None
    """The model's full quote, kept untruncated so rejections can be diagnosed."""


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


_CITATION_MARKERS = tuple(
    re.compile(pattern, re.MULTILINE | re.IGNORECASE)
    for pattern in (
        r"\bvol\.\s*\d", r"\bpp\.\s*\d", r"\bet al\.?", r"\bProc(?:eedings|\.)",
        r"\bJournal\b", r"\bConference\b", r"\bdoi\b", r"\bISBN\b", r"\bAIAA[- ]\d",
        r"\bNASA (?:TM|CR|TP|SP)[- ]", r"^\s*\[\d+\]",
        r"^\s*\d{1,3}\.\s+[A-Z][a-z]+,\s+[A-Z]\.",
    )
)
_CITATION_YEAR = re.compile(r"\((?:19|20)\d{2}\)|,\s*(?:19|20)\d{2}[.,]")

DEFAULT_REFERENCE_DENSITY = 4.0
"""Citation markers per 1,000 characters above which a chunk is treated as a
reference list. Calibrated on the 200-document NASA corpus: chunks at 8+ were
all bibliographies, 5-8 mostly were, 3-5 were about half. Set deliberately on
the aggressive side because the costs are lopsided -- skipping one prose chunk
out of thousands costs nothing, while synthesising from a bibliography spends a
model call and a reviewer's time on "when was this paper published?"."""


def citation_density(text: str) -> float:
    """Citation-like markers (vol., pp., et al., years in references...) per 1k chars."""
    if not text:
        return 0.0
    hits = sum(len(rx.findall(text)) for rx in _CITATION_MARKERS)
    hits += len(_CITATION_YEAR.findall(text))
    return hits / (len(text) / 1000)


def looks_like_references(text: str, threshold: float = DEFAULT_REFERENCE_DENSITY) -> bool:
    return citation_density(text) >= threshold


def sample_chunks(
    chunks: Iterable[Chunk],
    n: int,
    *,
    seed: int = 0,
    min_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    skip_references: bool = True,
) -> list[Chunk]:
    """Pick `n` chunks, spread across documents, deterministically.

    Round-robins over documents so one 400-page report cannot supply every
    question. Chunks shorter than `min_chars` are skipped: fragments, headers and
    trailing windows produce thin questions. Reference lists are skipped too:
    the first live run against NASA reports asked a reasoning model for a
    question about a bibliography entry, and the system prompt's instruction to
    decline such passages did not stop it. Filtering before the call is cheaper
    and more reliable than asking.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    rng = random.Random(seed)
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if len(chunk.text.strip()) < min_chars:
            continue
        if skip_references and looks_like_references(chunk.text):
            continue
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


DEFAULT_MIN_SIMILARITY = 0.92
FUZZY_MIN_CHARS = 30


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    start: int
    end: int
    kind: str
    """exact | normalised (whitespace/case) | gapped (page footer removed) | fuzzy (OCR)."""

    similarity: float = 1.0


def _normalise_with_origin(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed, lowercased text plus each character's original index."""
    chars: list[str] = []
    origin: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space and chars:
                chars.append(" ")
                origin.append(i)
            in_space = True
        else:
            chars.append(ch.lower())
            origin.append(i)
            in_space = False
    return "".join(chars), origin


GAP_MAX_CHARS = 80
GAP_MIN_PIECE = 15


def _match_with_gap(passage: str, quote: str) -> tuple[int, int] | None:
    """Match `quote` as two exact pieces with one short interruption between them.

    Found on the NASA corpus: the source read "the limiting link is
    NASA/TM-2019-220203 12 the DSG to the lunar surface." -- a running page
    footer and page number inside the sentence. The model quoted the sentence
    as a person reads it, footer removed. Page boundaries interrupt sentences on
    every page of every PDF, so this is common in real corpora.

    Both pieces must match exactly (after whitespace/case normalisation), split
    at a word boundary, each at least `GAP_MIN_PIECE` characters, with at most
    `GAP_MAX_CHARS` of inserted text between them. Returns normalised-text
    offsets covering both pieces and the gap.
    """
    if len(quote) < 2 * GAP_MIN_PIECE:
        return None
    for k in (i for i, ch in enumerate(quote) if ch == " "):
        left, right = quote[:k], quote[k + 1:]
        if len(left) < GAP_MIN_PIECE or len(right) < GAP_MIN_PIECE:
            continue
        start = passage.find(left)
        while start >= 0:
            after = start + len(left)
            window = passage[after: after + GAP_MAX_CHARS + len(right) + 2]
            r = window.find(right)
            if r > 0:  # r == 0 would have been a plain normalised match
                return start, after + r + len(right)
            start = passage.find(left, start + 1)
    return None


def match_evidence(
    passage: str,
    quote: str,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> EvidenceMatch | None:
    """Locate a model's evidence quote, tolerating OCR corrections but not paraphrase.

    Three tiers, strictest first. The fuzzy tier exists because of a real
    failure on the NASA corpus: the source text read "a small metaIIic sphere"
    (OCR misreading "metallic"), the model quoted the corrected word, and a
    strictly verbatim matcher threw away a correct question. Fuzzy matching
    requires a quote of at least 30 characters and `min_similarity` character
    agreement with one passage window -- a handful of corrected characters
    passes, a reworded sentence does not. Every fuzzy match is reported as
    such, so a reviewer can see it.
    """
    quote = quote.strip()
    if not quote:
        return None

    exact = passage.find(quote)
    if exact >= 0:
        return EvidenceMatch(exact, exact + len(quote), "exact")

    norm_passage, origin = _normalise_with_origin(passage)
    norm_quote = _WS_RE.sub(" ", quote).lower()
    if not norm_passage or not norm_quote:
        return None

    pos = norm_passage.find(norm_quote)
    if pos >= 0:
        return EvidenceMatch(origin[pos], origin[pos + len(norm_quote) - 1] + 1, "normalised")

    gapped = _match_with_gap(norm_passage, norm_quote)
    if gapped is not None:
        lo, hi = gapped
        return EvidenceMatch(origin[lo], origin[hi - 1] + 1, "gapped",
                             round(len(norm_quote) / (hi - lo), 3))

    if len(norm_quote) < FUZZY_MIN_CHARS:
        return None

    from difflib import SequenceMatcher

    anchor = SequenceMatcher(None, norm_passage, norm_quote, autojunk=False).find_longest_match(
        0, len(norm_passage), 0, len(norm_quote)
    )
    if anchor.size == 0:
        return None
    base = anchor.a - anchor.b
    slack = max(3, len(norm_quote) // 20)
    best: tuple[float, int, int] | None = None
    for d_start in range(-slack, slack + 1):
        lo = max(0, base + d_start)
        for d_end in range(-slack, slack + 1):
            hi = min(len(norm_passage), base + len(norm_quote) + d_end)
            if hi - lo < FUZZY_MIN_CHARS:
                continue
            ratio = SequenceMatcher(None, norm_passage[lo:hi], norm_quote, autojunk=False).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, lo, hi)
    if best is None or best[0] < min_similarity:
        return None
    ratio, lo, hi = best
    while lo < hi and norm_passage[lo] == " ":
        lo += 1
    while hi > lo and norm_passage[hi - 1] == " ":
        hi -= 1
    return EvidenceMatch(origin[lo], origin[hi - 1] + 1, "fuzzy", round(ratio, 3))


_CONTEXT_DEPENDENT = re.compile(
    r"\b(?:in|from|by|according to|per)\s+(?:the|this)\s+(?:passage|text|document|paper|"
    r"report|excerpt|section|table|figure|specifications?|memorandum|experiment|above)\b"
    r"|\b(?:mentioned|described|discussed|listed|shown|given|stated|referred to|highlighted)"
    r"\s+(?:in|above|here|earlier)\b"
    r"|\bthe (?:passage|excerpt)\b"
    r"|^\s*(?:which|what)\s+(?:figure|table|section|chapter|page|appendix)"
    r"(?:\s+(?:number|caption|title))?\s+(?:shows|illustrates|corresponds|contains|depicts|"
    r"lists|presents|describes|mentions|gives|summari[sz]es)\b",
    re.IGNORECASE,
)


def is_context_dependent(question: str) -> bool:
    """Does the question only make sense to someone holding the passage?

    "What is the operating rate of the green laser mentioned in the passage?"
    cannot be asked by a real user -- which green laser? -- and is ambiguous
    as a retrieval test. Questions about document layout ("Which figure number
    shows...") fail the same way. On the first 80 NASA candidates this pattern
    flagged 14, and every one was a question a reviewer would reject, so it
    rejects automatically.
    """
    return bool(_CONTEXT_DEPENDENT.search(question))


def answer_in_question(question: str, answer: str | None) -> bool:
    """Every meaningful token of the answer already appears in the question.

    "Which company is developing the 12 kW Aerojet thruster?" -> "Aerojet".
    A flag for the reviewer, not a rejection: on the NASA candidates it also
    caught "What material class is specified?" -> "Class I", where the
    distinguishing "I" is dropped as a stopword. Imprecise detectors flag;
    only precise ones reject.
    """
    if not answer:
        return False
    answer_tokens = {t for t in tokenize(answer) if t not in _ANSWER_FILLER}
    if not answer_tokens:
        return False
    return answer_tokens <= set(tokenize(question))


_ANSWER_FILLER = frozenset({"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is"})


def lexical_overlap(question: str, evidence: str) -> float:
    """Fraction of the question's content words that appear in the evidence.

    0.0 means the question shares no distinctive vocabulary with its source;
    1.0 means every content word was lifted from it. Stopwords are ignored,
    because "what is the" carries no retrieval signal either way.
    """
    q_terms = content_terms(question)
    if not q_terms:
        return 0.0
    e_terms = term_set(evidence)
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
    if is_context_dependent(question):
        return Rejection(chunk.id, "context_dependent",
                         detail="question only makes sense to someone holding the passage",
                         question=question, evidence=evidence or None)
    if not evidence:
        return Rejection(chunk.id, "no_evidence", question=question)

    located = match_evidence(chunk.text, evidence)
    if located is None:
        return Rejection(
            chunk.id,
            "evidence_not_verbatim",
            detail="quote not found in passage, even allowing for OCR corrections",
            question=question,
            evidence=evidence,
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

    start, end = located.start, located.end
    labels: dict[str, Any] = {
        "synthesized_by": model_name,
        "lexical_overlap": round(overlap, 3),
    }
    if located.kind in ("fuzzy", "gapped"):
        labels["evidence_match"] = located.kind
        labels["evidence_similarity"] = located.similarity
    if answer_in_question(question, answer):
        labels["answer_in_question"] = True

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
