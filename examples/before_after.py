"""Week 5: the closed loop.

    diagnose -> fix only what the verdicts name -> re-measure -> check the prediction

Run it:

    python examples/before_after.py

Two rules make the result mean something:

1. **Fix only what the verdicts named.** Fixing anything else would leave no way
   to tell whether the diagnosis was right.
2. **Change one variable.** The chunking stays identical; only the retriever
   changes. Otherwise a recovered case could be credited to the wrong change.

The script checks two predictions the diagnosis made, and says plainly if
either fails:

* every RETRIEVAL_MISS the keyword probe found should be recovered by BM25;
* no CORPUS_GAP should move, because no retrieval change can fix those.

READ THIS BEFORE QUOTING THE RESULT
-----------------------------------
Both predictions are close to true by construction here. The probe *is* BM25
over these same chunks, so a BM25 fix recovering what the probe found is
expected; and a corpus gap means the evidence is not indexed, so no retriever
can move one. The probe's value is getting that answer *before* building the
fix -- not independent proof that it works.

The non-trivial result is the one that failed. The first version of the
diagnostic used a word-overlap heuristic and called two of these three misses
vocabulary mismatches that no keyword fix could reach. BM25 recovered all
three. Investigating showed the naive retriever's real failure was stopword
noise ("what", "is", "the" matching every chunk), which the heuristic could not
see. That is why the probe exists.

This sample corpus is a toy (4 documents), and the run prints the coverage
warning to say so. Treat it as a demonstration of the mechanism, not a
benchmark.
"""

from __future__ import annotations

from pathlib import Path

from retrace import (
    NaiveRetriever,
    Verdict,
    build_corpus,
    evaluate,
    fixed_size_chunks,
    load_documents,
    read_fixture,
    save_run,
)
from retrace.metrics import compute_metrics, fix_first
from retrace.report import write_report
from retrace.retrievers import BM25Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sample"
RUNS = ROOT / "runs"
K = 5


def main() -> int:
    documents = load_documents(DATA / "documents.json")
    golden = read_fixture(DATA / "golden.jsonl")
    # One chunking for both runs, so the retriever is the only variable.
    corpus = build_corpus(documents, chunker=lambda d, t: fixed_size_chunks(d, t, size=300))

    print("STEP 1  diagnose the naive baseline")
    before = evaluate(NaiveRetriever(corpus.iter_chunks()), golden, corpus=corpus, k=K)
    for rec in fix_first(before):
        print(f"  {rec.verdict.value:<15} {rec.count}  {rec.detail}")
    if before.coverage_warning:
        print(f"  ! {before.coverage_warning}")

    recoverable = {
        a.case_id for a in before.attributions
        if a.verdict is Verdict.RETRIEVAL_MISS and a.lexical_probe_rank is not None
    }
    gaps = {a.case_id for a in before.attributions if a.verdict is Verdict.CORPUS_GAP}
    print(f"\n  prediction A: BM25 recovers {len(recoverable)} of the retrieval misses")
    print(f"  prediction B: none of the {len(gaps)} corpus gaps move\n")

    print("STEP 2  apply only that fix: swap the retriever for BM25, same chunks")
    after = evaluate(BM25Retriever(corpus.iter_chunks()), golden, corpus=corpus, k=K)

    print("\nSTEP 3  re-measure")
    b, a = compute_metrics(before), compute_metrics(after)
    print(f"  {'':<18}{'before':>10}{'after':>10}")
    for label, x, y in (
        ("OK", b.verdicts["OK"], a.verdicts["OK"]),
        ("RETRIEVAL_MISS", b.verdicts["RETRIEVAL_MISS"], a.verdicts["RETRIEVAL_MISS"]),
        ("CORPUS_GAP", b.verdicts["CORPUS_GAP"], a.verdicts["CORPUS_GAP"]),
    ):
        print(f"  {label:<18}{x:>10}{y:>10}")
    for label, x, y in (
        ("hit@1", b.hit_rate[1], a.hit_rate[1]),
        (f"hit@{K}", b.hit_rate[K], a.hit_rate[K]),
        ("MRR", b.mrr, a.mrr),
    ):
        print(f"  {label:<18}{x:>10.2f}{y:>10.2f}")

    after_by_id = {x.case_id: x for x in after.attributions}
    moved = [
        (x.case_id, x.verdict.value, after_by_id[x.case_id].verdict.value)
        for x in before.attributions
        if x.verdict is not after_by_id[x.case_id].verdict
    ]
    print("\n  cases that changed:")
    for case_id, was, now in moved:
        print(f"    {case_id:<28} {was} -> {now}")

    print("\nSTEP 4  check the predictions")
    recovered = {
        cid for cid in recoverable if after_by_id[cid].verdict is Verdict.OK
    }
    gaps_moved = {cid for cid in gaps if after_by_id[cid].verdict is not Verdict.CORPUS_GAP}
    ok_a = recovered == recoverable
    ok_b = not gaps_moved
    print(f"  A: {len(recovered)}/{len(recoverable)} predicted recoveries happened  "
          f"-> {'HELD' if ok_a else 'FAILED'}")
    print(f"  B: {len(gaps_moved)} corpus gaps moved  -> {'HELD' if ok_b else 'FAILED'}")
    print("  note: both are close to true by construction -- the probe is BM25 over these")
    print("        same chunks. The probe's job is to give this answer before you build")
    print("        the fix. See the module docstring before quoting it.")

    RUNS.mkdir(exist_ok=True)
    for name, run in (("before", before), ("after", after)):
        save_run(run, RUNS / f"{name}.json")
        write_report(run, RUNS / f"{name}.html")
    print(f"\n  reports: {RUNS / 'before.html'}  {RUNS / 'after.html'}")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
