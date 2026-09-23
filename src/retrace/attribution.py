"""The attribution cascade: which stage of the pipeline broke?

This module is the product. Everything else in Retrace exists to feed it clean
inputs or to present what it decides.

THE CASCADE
-----------
Four checks run in pipeline order, and the first one that fails names the
verdict. Order matters because each check is only meaningful if every check
before it passed -- asking "was it ranked too low?" is nonsense if it was never
retrieved at all.

    corpus      Is the gold evidence indexed at all?        no -> CORPUS_GAP
    retrieval   Is it among the candidates retrieved?        no -> RETRIEVAL_MISS
    ranking     Did it survive into the model's context?     no -> RANKING_MISS
    generation  Did the model's answer use it correctly?     no -> GENERATION_MISS
                                                            yes -> OK

Each check is a separate function in `STAGES` (Chain of Responsibility), so the
diagram above is literally the code. Three of the four are exact set-membership
tests; only generation needs a judge, and a judge may decline to decide.

HONESTY RULES
-------------
The cascade never emits a verdict the integration cannot support:

* No corpus access  -> CORPUS_GAP is impossible to detect. A missing answer is
  reported as RETRIEVAL_MISS with a note that it may be a gap.
* Tier 1 pipeline   -> candidates and context are the same list, so RANKING_MISS
  cannot occur. The report says the ranking stage was not observed.
* No answer / judge -> generation is not evaluated. The case is OK *at the
  retrieval stage*, flagged `generation_evaluated=False`, never silently passed.
* Judge unsure      -> UNSCORABLE, never a coin flip.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .corpus import DEFAULT_MIN_OVERLAP_CHARS, resolve_gold_chunks
from .judge import AnswerJudge, Judgement
from .text import content_terms, term_set
from .types import Chunk, EvalCase, RetrievalResult, Verdict

# Granularity at which a retrieved chunk was matched to the gold evidence.
SPAN = "span"
CHUNK_ID = "chunk_id"
DOCUMENT = "document"


@dataclass(frozen=True, slots=True)
class Attribution:
    """The verdict for one case, with the evidence behind it."""

    case_id: str
    verdict: Verdict
    reason: str
    stage: str
    """The cascade stage that decided: corpus, retrieval, ranking, generation, error."""

    gold_in_corpus: int | None = None
    """How many indexed chunks hold the evidence. None when the corpus was not checked."""

    candidate_rank: int | None = None
    """1-based rank of the first gold chunk among candidates, if retrieved."""

    context_rank: int | None = None
    """1-based position of the first gold chunk in the model's context, if present."""

    gold_ranks: tuple[int, ...] = ()
    """1-based candidate ranks of *every* gold chunk -- what precision@k is computed from."""

    query_gold_overlap: float | None = None
    """Share of the question's content words found in the gold evidence text.

    Separates the two kinds of RETRIEVAL_MISS that need opposite fixes: a low
    value is a vocabulary mismatch (dense retrieval, query rewriting); a high
    value means the words were there and retrieval still missed (chunking,
    ranking function). None when the evidence text is unavailable."""

    n_candidates: int = 0
    n_context: int = 0
    match_granularity: str | None = None
    generation_evaluated: bool = False
    judgement: Judgement | None = None
    lucky: bool = False
    """Right answer although the gold evidence never reached the model."""

    trusted: bool = True
    notes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.verdict not in (Verdict.OK, Verdict.UNSCORABLE)


# --------------------------------------------------------------------------
# Gold matching
# --------------------------------------------------------------------------


def gold_matcher(
    case: EvalCase,
    corpus: Any = None,
    *,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> Callable[[Chunk], str | None]:
    """Build ``is_gold(chunk) -> granularity | None`` for one case.

    Uses the most precise evidence available for each chunk:

    1. The chunk exposes a span and the case has gold spans: character overlap,
       with the same threshold `resolve_gold_chunks` uses, so "retrieved gold"
       and "indexed gold" can never disagree about what gold means.
    2. The chunk id is a known gold chunk -- resolved against the corpus when
       one is supplied, otherwise the case's cached ids.
    3. Only when the chunk carries no span: the document id. Coarse, and the
       granularity is reported so nobody mistakes it for a precise match.
    """
    resolved = resolve_gold_chunks(case, corpus) if corpus is not None else frozenset()
    known_ids = set(resolved) | set(case.gold_chunk_ids)
    doc_ids = case.implied_doc_ids

    def is_gold(chunk: Chunk) -> str | None:
        if chunk.span is not None and case.gold_spans:
            for gold in case.gold_spans:
                if chunk.span.doc_id != gold.doc_id:
                    continue
                if chunk.span.overlap_chars(gold) >= min(min_overlap_chars, gold.length):
                    return SPAN
            return None
        if chunk.id in known_ids:
            return CHUNK_ID
        if chunk.span is None and chunk.doc_id in doc_ids:
            return DOCUMENT
        return None

    return is_gold


def _first_gold(chunks: Sequence[Chunk], is_gold) -> tuple[int | None, str | None]:
    for rank, chunk in enumerate(chunks, start=1):
        granularity = is_gold(chunk)
        if granularity:
            return rank, granularity
    return None, None


# --------------------------------------------------------------------------
# The cascade
# --------------------------------------------------------------------------


@dataclass
class _Context:
    case: EvalCase
    result: RetrievalResult
    corpus: Any
    judge: AnswerJudge | None
    is_gold: Callable[[Chunk], str | None]
    gold_in_corpus: int | None = None
    candidate_rank: int | None = None
    context_rank: int | None = None
    granularity: str | None = None
    notes: list[str] = field(default_factory=list)
    judgement: Judgement | None = None
    generation_evaluated: bool = False


StageResult = tuple[Verdict, str] | None


def check_corpus(ctx: _Context) -> StageResult:
    if ctx.corpus is None:
        ctx.notes.append("corpus not supplied: CORPUS_GAP cannot be detected")
        return None
    gold = resolve_gold_chunks(ctx.case, ctx.corpus)
    ctx.gold_in_corpus = len(gold)
    if not gold:
        return (
            Verdict.CORPUS_GAP,
            "the gold evidence is not in the indexed corpus; no retrieval change can fix this",
        )
    return None


def check_retrieval(ctx: _Context) -> StageResult:
    ctx.candidate_rank, ctx.granularity = _first_gold(ctx.result.candidates, ctx.is_gold)
    if ctx.candidate_rank is not None:
        return None
    # A pipeline may inject chunks into the context that never appeared as
    # candidates (pinned documents, query-time expansion). Evidence that reached
    # the model was retrieved by *something*; do not call it a retrieval miss.
    in_context, granularity = _first_gold(ctx.result.context, ctx.is_gold)
    if in_context is not None:
        ctx.granularity = granularity
        ctx.notes.append("gold reached the context without appearing among the candidates")
        return None
    reason = f"gold evidence not among the {len(ctx.result.candidates)} retrieved candidates"
    if ctx.corpus is None:
        reason += " (may also be a corpus gap -- corpus not supplied)"
    return Verdict.RETRIEVAL_MISS, reason


def check_ranking(ctx: _Context) -> StageResult:
    ctx.context_rank, granularity = _first_gold(ctx.result.context, ctx.is_gold)
    if ctx.context_rank is not None:
        ctx.granularity = granularity or ctx.granularity
        return None
    if not ctx.result.exposes_ranking_stage:  # pragma: no cover - unreachable by construction
        return None
    return (
        Verdict.RANKING_MISS,
        f"gold retrieved at candidate rank {ctx.candidate_rank} but cut before the "
        f"{len(ctx.result.context)}-chunk context",
    )


def check_generation(ctx: _Context) -> StageResult:
    answer = ctx.result.answer
    expected = ctx.case.answer
    if not answer or not expected or ctx.judge is None:
        missing = (
            "no generated answer" if not answer
            else "no expected answer in the fixture" if not expected
            else "no judge configured"
        )
        ctx.notes.append(f"generation not evaluated: {missing}")
        return None

    ctx.generation_evaluated = True
    ctx.judgement = ctx.judge.judge(ctx.case.question, expected, answer, ctx.result.context)
    if ctx.judgement.correct is False:
        return (
            Verdict.GENERATION_MISS,
            f"gold evidence was in context at position {ctx.context_rank} but the answer "
            f"was judged wrong ({ctx.judgement.reason})",
        )
    if ctx.judgement.correct is None:
        return Verdict.UNSCORABLE, f"judge could not decide: {ctx.judgement.reason}"
    return None


STAGES: tuple[tuple[str, Callable[[_Context], StageResult]], ...] = (
    ("corpus", check_corpus),
    ("retrieval", check_retrieval),
    ("ranking", check_ranking),
    ("generation", check_generation),
)


def attribute(
    case: EvalCase,
    result: RetrievalResult,
    *,
    corpus: Any = None,
    judge: AnswerJudge | None = None,
) -> Attribution:
    """Run the cascade for one case and explain the verdict."""
    ctx = _Context(
        case=case,
        result=result,
        corpus=corpus,
        judge=judge,
        is_gold=gold_matcher(case, corpus),
    )
    if not result.exposes_ranking_stage:
        ctx.notes.append("ranking stage not observable (tier 1): RANKING_MISS cannot occur")

    verdict, reason, stage = Verdict.OK, "gold evidence retrieved and kept in context", "generation"
    for name, check in STAGES:
        outcome = check(ctx)
        if outcome is not None:
            verdict, reason = outcome
            stage = name
            break
    else:
        if ctx.judgement is not None:
            reason = f"answer judged correct ({ctx.judgement.reason})"

    lucky = False
    if verdict in (Verdict.CORPUS_GAP, Verdict.RETRIEVAL_MISS, Verdict.RANKING_MISS):
        lucky = _answered_anyway(ctx)

    gold_ranks = tuple(
        rank for rank, chunk in enumerate(result.candidates, start=1) if ctx.is_gold(chunk)
    )

    if ctx.granularity == DOCUMENT:
        ctx.notes.append("matched at document granularity: the pipeline exposes no chunk spans")

    return Attribution(
        case_id=case.id,
        verdict=verdict,
        reason=reason,
        stage=stage,
        gold_in_corpus=ctx.gold_in_corpus,
        candidate_rank=ctx.candidate_rank,
        context_rank=ctx.context_rank,
        gold_ranks=gold_ranks,
        query_gold_overlap=_query_gold_overlap(case, corpus),
        n_candidates=len(result.candidates),
        n_context=len(result.context),
        match_granularity=ctx.granularity,
        generation_evaluated=ctx.generation_evaluated,
        judgement=ctx.judgement,
        lucky=lucky,
        trusted=case.is_trusted,
        notes=tuple(ctx.notes),
    )


def _answered_anyway(ctx: _Context) -> bool:
    """A right answer without the gold evidence in context.

    A naive scorer counts this as a pass. It is not: either the model answered
    from memory (so the pipeline added nothing and will fail on anything the
    model does not already know), or the answer lives in another chunk too and
    the golden label is incomplete. Both are worth a human look.
    """
    if not ctx.result.answer or not ctx.case.answer or ctx.judge is None:
        return False
    judgement = ctx.judge.judge(
        ctx.case.question, ctx.case.answer, ctx.result.answer, ctx.result.context
    )
    if judgement.correct:
        ctx.notes.append(
            "answer judged correct without the gold evidence in context: the model "
            "answered from memory, or the golden label is incomplete"
        )
        return True
    return False


def _query_gold_overlap(case: EvalCase, corpus: Any) -> float | None:
    if corpus is None or not case.gold_spans:
        return None
    pieces = []
    for span in case.gold_spans:
        text = corpus.document_text(span.doc_id)
        if text is not None:
            pieces.append(text[span.start : span.end])
    if not pieces:
        return None
    question_terms = content_terms(case.question)
    if not question_terms:
        return None
    evidence_terms = term_set(" ".join(pieces))
    return round(len(question_terms & evidence_terms) / len(question_terms), 3)


def error_attribution(case: EvalCase, exc: BaseException) -> Attribution:
    """The pipeline itself raised. Unscorable, with the error preserved."""
    return Attribution(
        case_id=case.id,
        verdict=Verdict.UNSCORABLE,
        reason=f"pipeline raised {type(exc).__name__}",
        stage="error",
        trusted=case.is_trusted,
        error=f"{type(exc).__name__}: {exc}",
    )
