# Retrace

**Ragas tells you your RAG scores 0.72. Retrace tells you which questions failed, why, and what to fix first.**

Most RAG evaluation tools compute a score. Retrace answers the question the score cannot: *which stage of the pipeline is actually broken?* Every failed question gets exactly one verdict:

| Verdict | What happened | What you fix |
|---|---|---|
| `CORPUS_GAP` | The answer is not in the indexed documents at all. | Nothing in the pipeline. Add the sources, or scope the product down. |
| `RETRIEVAL_MISS` | The evidence is indexed but never retrieved. | Chunking, keyword search, dense retrieval, query rewriting. |
| `RANKING_MISS` | It was retrieved, then cut before the model saw it. | Reranking, context budget, fusion weights. |
| `GENERATION_MISS` | The model had the evidence and still answered wrong. | Prompt, context order, model choice. |
| `UNSCORABLE` | Retrace could not honestly decide. | Reported on its own line. Never folded into a pass rate. |

`CORPUS_GAP` is the verdict teams most need and rarely have. Weeks go into tuning embeddings for questions the documents cannot answer, and no aggregate score will tell you that is what is happening.

---

## Try it in 30 seconds

No API key, no downloads, no dependencies:

```bash
git clone https://github.com/mohammadjaan013/retrace.git
cd retrace
pip install -e .
retrace run --demo --docs data/sample/documents.json \
    --fixture data/sample/golden.jsonl --chunk-size 300 --k 5 --report
```

```
VERDICTS  12 cases (12 trusted)
  OK               ##############              7   58%
  CORPUS_GAP       ####                        2   17%
  RETRIEVAL_MISS   ######                      3   25%
  RANKING_MISS                               not observable (tier 1 pipeline)
  GENERATION_MISS                            not evaluated (no answers or no judge)

FIX FIRST
  1. RETRIEVAL_MISS - 3 of 5 failures (60%)
     The evidence is indexed but retrieval never returns it.
     3 found by a reference keyword retriever (losing to BM25)
     -> Check chunking first: answers split across a chunk boundary are unfindable.
  2. CORPUS_GAP - 2 of 5 failures (40%)
     The answer is not in the indexed documents. Stop tuning retrieval for these.
```

`--report` also writes `runs/run.html`: a single self-contained file with the verdict breakdown, the ranked fix list, and the full retrieval trace for every question, with the gold evidence highlighted. It has no external requests and no web fonts, so it can be emailed as is.

---

## The workflow

```bash
# 1. Propose candidate questions from your documents (needs an LLM -- see below)
retrace synth --docs docs.jsonl --out candidates.jsonl --n 60

# 2. Review them. One keypress each; every decision is saved as you go.
retrace label --candidates candidates.jsonl --docs docs.jsonl --out golden.jsonl

# 3. Diagnose your pipeline against the reviewed set
retrace run --pipeline my_rag:build() --docs docs.jsonl --fixture golden.jsonl --report
```

**Step 1 is the one other tools skip.** They all assume you already have a labelled evaluation set, and almost nobody does. `synth` writes each question from a single passage and makes the model quote its evidence verbatim. Retrace then locates that quote and stores its exact character offsets, and any quote it cannot find is rejected as a hallucinated citation. Questions that copy their passage's wording are rejected too, because they make keyword retrieval look better than it is.

**Step 2 makes the set trustworthy.** Nothing a model writes counts as ground truth until a person accepts it. The labeller measures its own speed and tells you how long 50 labels take.

Documents are `.json` (`{"doc_id": "text", ...}`) or `.jsonl` (`{"id": ..., "text": ...}` per line).

## Plugging in your pipeline

Point `--pipeline` at any `module:attribute`. It can resolve to a function, a class, or a factory call returning `(pipeline, corpus)`.

**Tier 1: a retrieve function.** About ten minutes of work.

```python
# my_rag.py
from retrace import Chunk, Span

def retrieve(query: str, k: int) -> list[Chunk]:
    hits = my_vector_store.search(query, top_k=k)          # whatever you already have
    return [
        Chunk(id=h.id, doc_id=h.doc_id, text=h.text, score=h.score,
              span=Span(h.doc_id, h.start_char, h.end_char))   # offsets make matching exact
        for h in hits
    ]
```

**Tier 2: expose both stages.** Return a `RetrievalResult` with the candidates *before* reranking and the context *after*. This is the only way `RANKING_MISS` becomes observable, and it's the single most valuable thing to wire up.

```python
from retrace import RetrievalResult

class MyPipeline:
    def run(self, query: str) -> RetrievalResult:
        candidates = retrieve(query, 50)
        context = rerank(query, candidates)[:5]
        return RetrievalResult(query, tuple(candidates), tuple(context),
                               answer=generate(query, context))
```

Pass `--docs` (the documents your pipeline indexes) to enable `CORPUS_GAP` detection. Without them, a missing answer and a retrieval miss look identical, and Retrace says so rather than guessing.

### Or from Python

```python
from retrace import build_corpus, evaluate, load_documents, read_fixture
from retrace.report import write_report

corpus = build_corpus(load_documents("docs.jsonl"))
run = evaluate(my_pipeline, read_fixture("golden.jsonl"), corpus=corpus)
write_report(run, "report.html")
```

## Language models

Retrace needs a model for two optional jobs: writing candidate questions, and judging answers for `GENERATION_MISS`. The core defines a one-method protocol and depends on no provider.

- **Anthropic**: `pip install "retrace-rag[anthropic]"`. `synth` uses `claude-opus-5` by default, with structured JSON output and server-side refusal fallback. Pass `--model` and `--effort` to trade cost for quality. Synthesis is a bulk job, so measure `--effort low` on your own corpus first.
- **Anything else** (Groq, OpenAI, Gemini, a local model): write a function `(system, prompt, schema) -> dict` and pass `--llm my_module:my_function`.

Answer judging defaults to a free token-recall judge, which is instant but blind to negation and paraphrase. Use `--judge llm` before putting a `GENERATION_MISS` number in front of anyone.

## How a retrieval miss gets explained

A `RETRIEVAL_MISS` has two causes that need opposite fixes, so Retrace re-asks every missed question to a reference BM25 over your corpus:

- **The probe finds it** → *your pipeline is losing to plain keyword search.* Fix the lexical side: chunking, IDF, hybrid search.
- **The probe misses it too** → *a genuine vocabulary mismatch.* The user's words aren't the document's words, so this needs dense retrieval or query rewriting.

This probe replaced a word-overlap heuristic that turned out to be wrong. The heuristic called two of the demo's three misses vocabulary mismatches, and BM25 recovered all three. The naive baseline's real problem was stopword noise ("what", "is" and "the" matching every chunk), which counting shared words cannot see. `examples/before_after.py` runs the whole loop and says in its own output what the result does and doesn't prove.

## Also included

Reference implementations of the fixes the verdicts point at. They are exact, in memory, and readable in one sitting, not a vector database.

- `BM25Retriever`: zero dependencies.
- `DenseRetriever`: bring any embedding function, or `pip install "retrace-rag[dense]"` for local sentence-transformers. Uses a content-addressed `EmbeddingCache`, so re-running an evaluation costs nothing.
- `HybridRetriever`: reciprocal rank fusion, recording which leg found each chunk (shown in the report's trace).
- `RerankingPipeline` + `CrossEncoderReranker`: a Tier 2 pipeline.
- `sentence_chunks`: whole sentences, paragraph-bounded, with overlap.

## In CI

```bash
retrace run --pipeline my_rag:build() --docs docs.jsonl --fixture golden.jsonl \
    --max-failure-rate 0.25
```

Exits `3` when the failure rate exceeds the threshold, and `2` on a usage or setup error, so a pipeline can tell "worse" from "broken". If the first cases all fail the same way (bad key, wrong URL, empty index), the run stops immediately instead of producing a report that is 100% `UNSCORABLE`.

## Design decisions worth knowing

**Gold labels are character spans, not chunk ids.** Changing the chunking is the most common fix Retrace recommends, and chunk-id labels would be invalidated by exactly that fix. Spans into the source document survive any re-chunking. For the same reason, `.gitattributes` pins LF line endings: a CRLF checkout would shift every offset.

**It never emits a verdict the integration can't support.** No corpus means no `CORPUS_GAP`. A Tier 1 pipeline means no `RANKING_MISS`. No answers means generation is "not evaluated", never "passed". Every report says which of these applied.

**Corpus gaps are excluded from retrieval metrics**, and the excluded count is printed next to every metric. An answer that isn't indexed isn't a retrieval failure.

**It warns when a number is meaningless.** If `k` returns more than 10% of the corpus, recall is inflated and the run says so. That warning exists because the first baseline scored 100% recall by returning the entire corpus.

**Zero runtime dependencies**, checked in CI by installing into a bare environment and running the demo.

## Limitations

- The sample corpus is a toy (4 documents), and the demo's numbers demonstrate the mechanism. They are not a benchmark.
- The Anthropic adapter is tested against a recorded request shape, not yet against the live API.
- The default answer judge can't detect negation. Treat its `GENERATION_MISS` counts as triage only.
- Reference retrievers are exact and in-memory, sized for evaluation runs, not production traffic.

## Development

```bash
pip install -e ".[dev]"
python -m pytest            # 310 tests
python -m ruff check .
python examples/before_after.py
```

## License

MIT
