"""Command-line interface.

    retrace synth  --docs docs.json --out candidates.jsonl
    retrace label  --candidates candidates.jsonl --docs docs.json --out golden.jsonl
    retrace stats  golden.jsonl

Standard library `argparse` only, in keeping with the zero-dependency core.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .errors import RetraceError


def load_object(spec: str) -> Any:
    """Import ``package.module:attribute`` -- the uvicorn-style target syntax.

    The current directory is put on ``sys.path`` first, so a user can point at
    ``my_pipeline:build`` sitting next to them without installing anything.
    If the attribute is callable with no arguments and named like a factory
    (the user wrote ``module:factory()``), it is called.
    """
    call = spec.endswith("()")
    if call:
        spec = spec[:-2]
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise RetraceError(f"expected 'module:attribute', got {spec!r}")

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RetraceError(f"could not import {module_name!r}: {exc}") from exc

    obj: Any = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise RetraceError(f"{module_name!r} has no attribute {attr!r}") from exc
    return obj() if call else obj


PROVIDER_CHOICES = ("anthropic", "groq", "deepseek", "openai-compatible")


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=PROVIDER_CHOICES, default="anthropic",
                   help="model provider (default anthropic; groq reads GROQ_API_KEY, "
                        "deepseek reads DEEPSEEK_API_KEY)")
    p.add_argument("--model", default=None,
                   help="model id (default: claude-opus-5 for anthropic, "
                        "openai/gpt-oss-120b for groq, deepseek-flash for deepseek)")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                   help="reasoning effort; groq and openai-compatible accept low/medium/high")
    p.add_argument("--base-url", default=None, help="for --provider openai-compatible")
    p.add_argument("--api-key-env", default=None,
                   help="environment variable holding the key (openai-compatible)")


def _llm_from_args(provider, model, effort, base_url, api_key_env):
    from .llm import build_llm

    try:
        return build_llm(provider, model=model, effort=effort, base_url=base_url,
                         api_key_env=api_key_env)
    except ValueError as exc:
        raise RetraceError(str(exc)) from exc


EXIT_THRESHOLD = 3
"""Distinct from 2 (usage or runtime error) so CI can tell 'broken' from 'worse'."""


def _parse_rate(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1")
    return parsed


def _parse_overlap(value: str) -> float | None:
    if value.lower() in ("none", "off", "0"):
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1, or 'none'") from exc
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1, or 'none'")
    return parsed


# --------------------------------------------------------------------------
# synth
# --------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    from .chunking import fixed_size_chunks
    from .corpus import build_corpus, load_documents
    from .fixtures import write_fixture
    from .synthesis import Rejection, iter_synthesize, sample_chunks

    documents = load_documents(args.docs)
    corpus = build_corpus(
        documents,
        chunker=lambda d, t: fixed_size_chunks(
            d, t, size=args.chunk_size, overlap=args.chunk_overlap
        ),
    )
    picked = sample_chunks(
        corpus.iter_chunks(), args.n, seed=args.seed, min_chars=args.min_chunk_chars
    )
    if not picked:
        print("no chunks long enough to synthesise from; lower --min-chunk-chars", file=sys.stderr)
        return 2

    llm = load_object(args.llm) if args.llm else _llm_from_args(
        args.provider, args.model, args.effort, args.base_url, args.api_key_env
    )

    name = getattr(llm, "name", type(llm).__name__)
    print(f"synthesising from {len(picked)} passages across "
          f"{len({c.doc_id for c in picked})} documents with {name}")

    cases, rejections = [], []
    interrupted = False
    stopped: RetraceError | None = None
    try:
        for i, item in enumerate(iter_synthesize(picked, llm, max_overlap=args.max_overlap), 1):
            if isinstance(item, Rejection):
                rejections.append(item)
                mark = "x"
            else:
                cases.append(item)
                mark = "+"
            print(f"\r  {i}/{len(picked)}  kept {len(cases)}  rejected {len(rejections)}  {mark}",
                  end="", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n  interrupted -- saving what was produced")
    except RetraceError as exc:
        # Auth failure, daily quota, persistent server errors: stop, but never
        # throw away candidates that were already paid for.
        stopped = exc
        print(f"\n  stopped: {exc}\n  saving what was produced")
    print()

    out = Path(args.out)
    write_fixture(out, cases)
    rej_path = out.with_name(f"{out.stem}.rejections.jsonl")
    with rej_path.open("w", encoding="utf-8") as handle:
        for r in rejections:
            handle.write(json.dumps(
                {"chunk_id": r.chunk_id, "reason": r.reason, "detail": r.detail,
                 "question": r.question, "evidence": r.evidence},
                ensure_ascii=False,
            ) + "\n")

    counts: dict[str, int] = {}
    for r in rejections:
        counts[r.reason] = counts.get(r.reason, 0) + 1

    print(f"wrote {len(cases)} candidates -> {out}")
    if counts:
        print("rejected before review:")
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<24} {n}")
    usage = getattr(llm, "usage", None)
    if usage is not None and hasattr(usage, "as_dict"):
        u = usage.as_dict()
        print(f"model usage: {u['calls']} calls, {u['input_tokens']} in / "
              f"{u['output_tokens']} out tokens"
              + (f", {u['fallbacks']} served by fallback" if u.get("fallbacks") else "")
              + (f", {u['rate_limit_waits']} rate-limit waits ({u['waited_s']}s)"
                 if u.get("rate_limit_waits") else ""))
    if stopped is not None:
        return 2
    print(f"\nnext: retrace label --candidates {out} --docs {args.docs} --out golden.jsonl")
    return 130 if interrupted else 0


# --------------------------------------------------------------------------
# label
# --------------------------------------------------------------------------


def cmd_label(args: argparse.Namespace) -> int:
    from .corpus import load_documents
    from .labelling import label

    documents = load_documents(args.docs)
    try:
        label(args.candidates, documents, args.out)
    except KeyboardInterrupt:
        print("\ninterrupted -- every decision so far is saved; re-run to resume")
        return 130
    return 0


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> int:
    from .fixtures import fixture_stats, read_fixture

    for path in args.fixtures:
        stats = fixture_stats(read_fixture(path))
        print(f"{path}")
        print(f"  total      {stats['total']}")
        print(f"  trusted    {stats['trusted']}  (human-written or human-verified)")
        for prov, n in sorted(stats["by_provenance"].items()):
            print(f"    {prov:<16} {n}")
        print(f"  with spans {stats['with_spans']}")
        print(f"  documents  {stats['documents']}")
        if stats["total"] and stats["trusted"] < stats["total"]:
            print(f"  note: {stats['total'] - stats['trusted']} unverified cases -- "
                  "do not report metrics on them as ground truth")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _resolve_target(args: argparse.Namespace) -> tuple[Any, Any]:
    """Build (pipeline, corpus) from --demo / --pipeline / --corpus / --docs."""
    from .chunking import fixed_size_chunks
    from .corpus import build_corpus, load_documents

    corpus = None
    if args.docs:
        documents = load_documents(args.docs)
        corpus = build_corpus(
            documents,
            chunker=lambda d, t: fixed_size_chunks(d, t, size=args.chunk_size),
        )

    if args.demo:
        if corpus is None:  # pragma: no cover - _apply_demo_defaults always supplies docs
            raise RetraceError("--demo needs --docs: the naive baseline indexes those documents")
        from .baseline import NaiveRetriever

        return NaiveRetriever(corpus.iter_chunks()), corpus

    if not args.pipeline:
        raise RetraceError("pass --pipeline module:factory() or --demo")

    target = load_object(args.pipeline)
    pipeline = target
    if isinstance(target, tuple) and len(target) == 2:
        pipeline, factory_corpus = target
        corpus = corpus if corpus is not None else factory_corpus
    if args.corpus:
        corpus = load_object(args.corpus)
    return pipeline, corpus


def _build_judge(args: argparse.Namespace):
    if args.judge == "none":
        return None
    if args.judge == "llm":
        from .judge import LLMJudge

        return LLMJudge(_llm_from_args(args.judge_provider, args.judge_model, "low", None, None))
    from .judge import TokenRecallJudge

    return TokenRecallJudge()


def _apply_demo_defaults(args: argparse.Namespace) -> None:
    """Fill run options left unset.

    `--demo` with nothing else runs the bundled sample at the settings the
    README shows (300-character chunks, k=5), so `pip install retrace-rag &&
    retrace run --demo` works from any directory. Anything passed explicitly
    wins; without --demo the general defaults apply (1200 chars, k=10).
    """
    from . import SAMPLE_DIR

    if args.demo:
        args.docs = args.docs or str(SAMPLE_DIR / "documents.json")
        args.fixture = args.fixture or str(SAMPLE_DIR / "golden.jsonl")
        args.chunk_size = args.chunk_size or 300
        args.k = args.k or 5
    else:
        if not args.fixture:
            raise RetraceError("--fixture is required (or use --demo for the bundled sample)")
        args.chunk_size = args.chunk_size or 1200
        args.k = args.k or 10


def cmd_run(args: argparse.Namespace) -> int:
    from .fixtures import read_fixture
    from .runner import evaluate, save_run
    from .summary import print_summary

    _apply_demo_defaults(args)
    cases = read_fixture(args.fixture)
    pipeline, corpus = _resolve_target(args)
    judge = _build_judge(args)

    interactive = sys.stdout.isatty()

    def progress(i: int, n: int, attribution) -> None:
        # A carriage-return progress line is noise in a CI log or a pipe.
        if interactive:
            print(f"\r  {i}/{n}  {attribution.verdict.value:<16}", end="", flush=True)

    print(f"running {len(cases)} cases against {pipeline!r}")
    run = evaluate(pipeline, cases, corpus=corpus, judge=judge, k=args.k, progress=progress)
    run.meta.update({"fixture": str(args.fixture), "k": args.k})
    if interactive:
        print()

    out = save_run(run, args.out)
    print_summary(run)
    print(f"\nrun saved -> {out}")
    if args.report:
        from .report import write_report

        print(f"report    -> {write_report(run, out.with_suffix('.html'))}")

    if args.max_failure_rate is not None:
        failures = sum(1 for a in run.attributions if a.is_failure)
        rate = failures / len(run.attributions) if run.attributions else 0.0
        if rate > args.max_failure_rate:
            print(f"\nFAIL: failure rate {rate:.0%} exceeds --max-failure-rate "
                  f"{args.max_failure_rate:.0%}", file=sys.stderr)
            return EXIT_THRESHOLD
        print(f"\npass: failure rate {rate:.0%} within {args.max_failure_rate:.0%}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report
    from .runner import load_run

    run = load_run(args.run)
    out = Path(args.out) if args.out else Path(args.run).with_suffix(".html")
    print(f"report -> {write_report(run, out)}")
    return 0


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrace",
        description="Find out which stage of your RAG pipeline is actually broken.",
    )
    parser.add_argument("--version", action="version", version=f"retrace {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="propose candidate questions from a document collection")
    p.add_argument("--docs", required=True, help="documents: .json {id: text} or .jsonl {id, text}")
    p.add_argument("--out", required=True, help="candidate fixture to write (.jsonl)")
    p.add_argument("--n", type=int, default=60, help="passages to sample (default 60)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=1200, help="passage size in characters")
    p.add_argument("--chunk-overlap", type=int, default=0)
    p.add_argument("--min-chunk-chars", type=int, default=200)
    p.add_argument("--max-overlap", type=_parse_overlap, default=0.9,
                   help="reject questions copying more than this fraction of their "
                        "content words from their passage (default 0.9; 'none' disables)")
    _add_llm_args(p)
    p.add_argument("--llm", help="use your own model instead: 'module:attr' resolving to an LLM "
                                 "or a callable (system, prompt, schema) -> dict")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("label", help="review candidates and build a trusted golden set")
    p.add_argument("--candidates", required=True)
    p.add_argument("--docs", required=True)
    p.add_argument("--out", required=True, help="golden fixture to write (.jsonl)")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("stats", help="summarise one or more fixtures")
    p.add_argument("fixtures", nargs="+")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("run", help="attribute every failure in a golden set to a pipeline stage")
    p.add_argument("--fixture", help="golden set (.jsonl); --demo defaults to the bundled sample")
    target = p.add_mutually_exclusive_group()
    target.add_argument("--pipeline", help="'module:attr' or 'module:factory()' returning a "
                                           "pipeline, a retriever, or (pipeline, corpus)")
    target.add_argument("--demo", action="store_true",
                        help="diagnose the built-in naive baseline over --docs")
    p.add_argument("--docs", help="the indexed documents, enabling CORPUS_GAP detection")
    p.add_argument("--corpus", help="'module:attr' resolving to a Corpus (overrides --docs)")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="chunk size used to index --docs (default 1200; 300 with --demo)")
    p.add_argument("--k", type=int, default=None,
                   help="top-k for retrieve-only pipelines (default 10; 5 with --demo)")
    p.add_argument("--judge", choices=["recall", "llm", "none"], default="recall",
                   help="answer judge for GENERATION_MISS (default: free token-recall)")
    p.add_argument("--judge-provider", choices=PROVIDER_CHOICES, default="anthropic",
                   help="provider for --judge llm (default anthropic)")
    p.add_argument("--judge-model", default=None,
                   help="model for --judge llm (default: the provider's default)")
    p.add_argument("--out", default="runs/run.json", help="run file to write")
    p.add_argument("--report", action="store_true",
                   help="also write a self-contained HTML report next to the run file")
    p.add_argument("--max-failure-rate", type=_parse_rate, default=None, metavar="RATE",
                   help="for CI: exit with status 3 if more than RATE (0-1) of cases fail")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="render a saved run as a single-file HTML report")
    p.add_argument("run", help="run file written by `retrace run`")
    p.add_argument("--out", help="HTML file to write (default: next to the run file)")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # A gitignored .env next to the command is the easiest place for an API key
    # that should never be typed into a shell history or pasted into a chat.
    from .llm import load_env_file

    load_env_file(".env")
    try:
        return args.func(args)
    except RetraceError as exc:
        print(f"retrace: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
