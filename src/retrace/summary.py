"""The terminal summary printed at the end of `retrace run`."""

from __future__ import annotations

import sys
from typing import Any, TextIO

from .metrics import compute_metrics, fix_first
from .types import Verdict

_ORDER = (
    Verdict.OK,
    Verdict.CORPUS_GAP,
    Verdict.RETRIEVAL_MISS,
    Verdict.RANKING_MISS,
    Verdict.GENERATION_MISS,
    Verdict.UNSCORABLE,
)


def _encodable(out: TextIO, text: str) -> bool:
    try:
        text.encode(getattr(out, "encoding", None) or "utf-8")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _ms(value: float) -> str:
    return f"{value:.2f} ms" if value < 10 else f"{value:.0f} ms"


def print_summary(run: Any, out: TextIO | None = None) -> None:
    out = out or sys.stdout
    fancy = _encodable(out, "█·→")
    bar_char, dot, arrow = ("█", "·", "→") if fancy else ("#", "-", "->")
    w = out.write

    m = compute_metrics(run)
    trust = f"{m.trusted} trusted" if m.trusted == m.cases else f"only {m.trusted} trusted"
    w(f"\nVERDICTS  {m.cases} cases ({trust})\n")
    for verdict in _ORDER:
        n = m.verdicts[verdict.value]
        if verdict is Verdict.RANKING_MISS and run.tier == 1:
            w(f"  {verdict.value:<16} {'':<24}  not observable (tier 1 pipeline)\n")
            continue
        if verdict is Verdict.GENERATION_MISS and not m.generation_evaluated:
            w(f"  {verdict.value:<16} {'':<24}  not evaluated (no answers or no judge)\n")
            continue
        share = n / m.cases if m.cases else 0.0
        bar = bar_char * round(share * 24)
        w(f"  {verdict.value:<16} {bar:<24} {n:>4}  {share:>4.0%}\n")

    if m.answerable:
        excluded = m.scored - m.answerable
        w(f"\nRETRIEVAL  {m.answerable} answerable")
        w(f" ({excluded} corpus gaps excluded)\n" if excluded else "\n")
        cells = [f"hit@{k} {v:.0%}" for k, v in m.hit_rate.items()]
        if m.mrr is not None:
            cells.append(f"MRR {m.mrr:.2f}")
        w("  " + f"  {dot}  ".join(cells) + "\n")
        if m.latency_p50_ms is not None:
            w(f"  latency p50 {_ms(m.latency_p50_ms)}  {dot}  p95 {_ms(m.latency_p95_ms)}\n")

    recs = fix_first(run)
    if recs:
        w("\nFIX FIRST\n")
        for i, rec in enumerate(recs, start=1):
            w(f"  {i}. {rec.verdict.value} {dot} {rec.count} of "
              f"{sum(r.count for r in recs)} failures ({rec.share:.0%})\n")
            w(f"     {rec.headline}\n")
            if rec.detail:
                w(f"     {rec.detail}\n")
            w(f"     {arrow} {rec.actions[0]}\n")
            w(f"     cases: {', '.join(rec.case_ids)}\n")
    else:
        w("\nNo failures.\n")

    warnings = []
    if run.coverage_warning:
        warnings.append(run.coverage_warning)
    if not run.corpus_checked:
        warnings.append("No corpus supplied: corpus gaps are hidden inside RETRIEVAL_MISS. "
                        "Pass --docs to separate them.")
    if m.trusted < m.cases:
        warnings.append(f"{m.cases - m.trusted} cases are unverified model output -- "
                        "review them with `retrace label` before quoting these numbers.")
    if m.errors:
        warnings.append(f"{m.errors} cases raised inside the pipeline; see the run file.")
    if warnings:
        w("\nWARNINGS\n")
        for warning in warnings:
            w(f"  ! {warning}\n")
    out.flush()
