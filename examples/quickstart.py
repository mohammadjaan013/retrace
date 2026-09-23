"""Week 1 baseline: run the naive pipeline against the sample golden set.

The attribution cascade lands in Week 3. This script does the part that is
already possible with the Week 1 primitives -- separate the cases whose evidence
is not in the corpus at all from the cases where retrieval simply failed to find
it -- and prints the split.

That split is already the most useful thing most teams have never measured.

    python examples/quickstart.py
"""

from __future__ import annotations

import json
from pathlib import Path

from retrace import (
    InMemoryCorpus,
    NaiveRetriever,
    RetrieverPipeline,
    chunk_documents,
    corpus_covers,
    coverage_warning,
    read_fixture,
    resolve_gold_chunks,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "sample"
K = 5
CHUNK_CHARS = 300


def main() -> None:
    documents: dict[str, str] = json.loads(
        (DATA / "documents.json").read_text(encoding="utf-8")
    )
    cases = read_fixture(DATA / "golden.jsonl")

    chunks = chunk_documents(documents, size=CHUNK_CHARS)
    corpus = InMemoryCorpus(chunks, documents=documents)
    pipeline = RetrieverPipeline(NaiveRetriever(corpus.iter_chunks()), k=K)

    print(f"corpus:  {len(corpus)} chunks from {len(documents)} documents")
    print(f"golden:  {len(cases)} cases, all human-labelled")
    print("tier:    1 (retrieve-only) -- ranking stage not observable")

    warning = coverage_warning(K, corpus)
    if warning:
        print(f"\n  WARNING: {warning}")
        print("  This sample corpus is a toy. Treat the numbers below as a")
        print("  demonstration of the mechanism, not as a benchmark.")
    print()

    buckets: dict[str, list[str]] = {"hit": [], "corpus_gap": [], "retrieval_miss": []}
    rows: list[tuple[str, str, str]] = []

    for case in cases:
        gold = resolve_gold_chunks(case, corpus)

        if not gold:
            # No evidence in the index. Week 3 will call this CORPUS_GAP.
            buckets["corpus_gap"].append(case.id)
            rows.append((case.id, "gap", "evidence not indexed"))
            continue

        result = pipeline.run(case.question)
        retrieved = {c.id for c in result.context}

        if retrieved & gold:
            rank = next(
                i for i, c in enumerate(result.context, start=1) if c.id in gold
            )
            buckets["hit"].append(case.id)
            rows.append((case.id, "hit", f"gold at rank {rank}"))
        else:
            buckets["retrieval_miss"].append(case.id)
            note = case.labels.get("expect", "gold never returned")
            rows.append((case.id, "MISS", note))

    width = max(len(r[0]) for r in rows)
    for case_id, status, note in rows:
        print(f"  {case_id:<{width}}  {status:<5}  {note}")

    answerable = len(cases) - len(buckets["corpus_gap"])
    hits = len(buckets["hit"])
    print()
    print(f"  in-corpus recall@{K}:  {hits}/{answerable}"
          f"  ({hits / answerable:.0%} of answerable cases)")
    print(f"  corpus gaps:         {len(buckets['corpus_gap'])}/{len(cases)}"
          f"  ({len(buckets['corpus_gap']) / len(cases):.0%} of all cases)")
    print()
    print("  Note the second number. Those cases cannot be fixed by any amount of")
    print("  retrieval tuning -- the documents are simply not indexed. A team")
    print("  chasing an aggregate score would spend weeks on them.")

    # Sanity check that corpus_covers agrees with the resolution above.
    assert all(not corpus_covers(c, corpus) for c in cases if c.id in buckets["corpus_gap"])


if __name__ == "__main__":
    main()
