# Retrace

**Ragas tells you your RAG scores 0.72. Retrace tells you which questions failed, why, and what to fix first.**

Retrace answers one question no other RAG evaluation tool answers: *which stage of the pipeline is actually broken?*

| Verdict | What happened | What you fix |
|---|---|---|
| `CORPUS_GAP` | The answer is not in the indexed documents at all. | Nothing in the pipeline. Get more documents, or narrow the product's scope. |
| `RETRIEVAL_MISS` | The right chunk is indexed but was never returned. | Chunking, embeddings, query rewriting, hybrid search. |
| `RANKING_MISS` | It was retrieved, then ranked below the context cutoff. | Reranking, fusion weights, top-k. |
| `GENERATION_MISS` | It reached the model, which ignored or contradicted it. | Prompt, context ordering, model choice. |
| `UNSCORABLE` | Retrace cannot honestly decide. | Reported separately, never folded into a pass rate. |

`CORPUS_GAP` is the one teams most need and least often have. Weeks get spent tuning embeddings for questions the documents cannot answer, and no tool says so.

---

## Status: early, and honest about it

Built in public, one phase at a time.

| | | |
|---|---|---|
| **Week 1** | Data model, adapter seam, fixture I/O, naive baseline | ✅ done |
| Week 2 | Question synthesis + the labelling loop | — |
| Week 3 | The attribution cascade | — |
| Week 4 | Metrics + single-file HTML report | — |
| Week 5 | Retrieval traces, hybrid retrieval, reranking | — |
| Week 6 | Packaging, public case study | — |

**`Verdict` values are defined but not yet assigned** — the cascade lands in Week 3. What works today is everything underneath it: the types, the adapter, chunking-independent gold labels, fixture persistence, and a deliberately weak retriever to diagnose.

### Week 1 baseline

`python examples/quickstart.py`, against a 4-document sample corpus and a 12-case human-labelled golden set:

```
in-corpus recall@5:  7/10   (70% of answerable cases)
corpus gaps:         2/12   (17% of all cases)
```

The two misses are the ones the naive retriever was built to fail on — a synonym mismatch (`heat rejection` vs `get rid of warmth`) and a question whose common words drown out its one distinguishing term, because the baseline has no IDF.

The sample corpus is a toy and the example says so out loud: `k=5` over 16 chunks returns 31% of the corpus, and the run prints a warning that recall is inflated at that ratio. That guard exists because the first version of this baseline scored **100% recall** — k=5 against a 5-chunk corpus, where retrieval had not got anything right, it had simply returned everything.

---

## Install

```bash
pip install -e ".[dev]"
```

The core has **no required dependencies**. `pip install retrace` will never drag in torch, a vector database client, or an LLM SDK. Anything heavy lives behind an extra (`[pdf]`).

## Quickstart

```python
from retrace import (
    EvalCase, InMemoryCorpus, NaiveRetriever, RetrieverPipeline,
    Span, chunk_documents, resolve_gold_chunks,
)

docs = {"d1": "Radiator sizing is driven by the worst-case hot attitude."}

chunks = chunk_documents(docs, size=200)
corpus = InMemoryCorpus(chunks, documents=docs)
pipeline = RetrieverPipeline(NaiveRetriever(corpus.iter_chunks()), k=5)

case = EvalCase(
    id="q1",
    question="What drives radiator sizing?",
    answer="The worst-case hot attitude.",
    gold_spans=(Span("d1", 0, 56),),
)

result = pipeline.run(case.question)
gold = resolve_gold_chunks(case, corpus)

print("retrieved:", [c.id for c in result.context])
print("gold:", sorted(gold))
```

Run the worked example, which builds a corpus where the naive retriever provably fails:

```bash
python examples/quickstart.py
```

---

## Two design decisions worth knowing

### Gold labels are character spans, not chunk ids

The obvious design records which chunk ids should have been retrieved. That breaks the first time chunking changes — and changing the chunking is the most common fix Retrace recommends. The obvious design invalidates the golden set exactly when it is needed most.

Retrace stores `Span(doc_id, start, end)` into the source document. Character offsets are stable across every chunking, and `resolve_gold_chunks()` maps them onto whatever chunks the pipeline currently produces. A golden set labelled in Week 2 still scores a pipeline rebuilt in Week 5.

Chunk ids are still accepted, and treated as a cache that is silently ignored once stale.

### Integration tiers, stated on every report

How much Retrace can tell you depends on how much your pipeline exposes.

**Tier 1** — a `retrieve(query, k) -> list[Chunk]` function. About ten minutes of work. Yields `CORPUS_GAP` and `RETRIEVAL_MISS`. Cannot separate `RANKING_MISS`, because there is no visible cut between what was retrieved and what reached the model.

**Tier 2** — a `run(query) -> RetrievalResult` exposing candidates before reranking, context after it, and the answer. Yields all four verdicts.

Retrace reports which tier it ran at and never emits a verdict the integration could not support.

---

## Development

```bash
.venv/Scripts/python -m pytest        # tests
.venv/Scripts/python -m ruff check .  # lint
```

## License

MIT.
