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

    if args.llm:
        llm = load_object(args.llm)
    else:
        from .llm import AnthropicLLM

        llm = AnthropicLLM(model=args.model, effort=args.effort)

    name = getattr(llm, "name", type(llm).__name__)
    print(f"synthesising from {len(picked)} passages across "
          f"{len({c.doc_id for c in picked})} documents with {name}")

    cases, rejections = [], []
    interrupted = False
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
    print()

    out = Path(args.out)
    write_fixture(out, cases)
    rej_path = out.with_name(f"{out.stem}.rejections.jsonl")
    with rej_path.open("w", encoding="utf-8") as handle:
        for r in rejections:
            handle.write(json.dumps(
                {"chunk_id": r.chunk_id, "reason": r.reason, "detail": r.detail,
                 "question": r.question},
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
              + (f", {u['fallbacks']} served by fallback" if u.get("fallbacks") else ""))
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
    p.add_argument("--max-overlap", type=_parse_overlap, default=0.8,
                   help="reject questions copying more than this fraction of their "
                        "content words from the evidence (default 0.8; 'none' disables)")
    p.add_argument("--model", default="claude-opus-5", help="Anthropic model id")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RetraceError as exc:
        print(f"retrace: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
