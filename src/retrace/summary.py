"""The terminal summary printed at the end of `retrace run`."""

from __future__ import annotations

import os
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

# One colour per verdict, the same family the HTML report uses, so a reader who
# has seen one recognises the other. Basic ANSI only: it renders everywhere.
_VERDICT_SGR = {
    Verdict.OK: "32",               # green
    Verdict.CORPUS_GAP: "35",       # magenta
    Verdict.RETRIEVAL_MISS: "33",   # yellow
    Verdict.RANKING_MISS: "34",     # blue
    Verdict.GENERATION_MISS: "31",  # red
    Verdict.UNSCORABLE: "90",       # grey
}


def _encodable(out: TextIO, text: str) -> bool:
    try:
        text.encode(getattr(out, "encoding", None) or "utf-8")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def colour_enabled(out: TextIO) -> bool:
    """Colour only for a real terminal, and never when NO_COLOR is set (no-color.org)."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    return bool(getattr(out, "isatty", None) and out.isatty())


def _ms(value: float) -> str:
    return f"{value:.2f} ms" if value < 10 else f"{value:.0f} ms"


def print_summary(run: Any, out: TextIO | None = None) -> None:
    out = out or sys.stdout
    fancy = _encodable(out, "█·→")
    bar_char, dot, arrow = ("█", "·", "→") if fancy else ("#", "-", "->")
    use_colour = colour_enabled(out)

    def paint(text: str, sgr: str) -> str:
        return f"\x1b[{sgr}m{text}\x1b[0m" if use_colour else text

    w = out.write

    m = compute_metrics(run)
    trust = f"{m.trusted} trusted" if m.trusted == m.cases else f"only {m.trusted} trusted"
    w(f"\n{paint('VERDICTS', '1')}  {m.cases} cases ({trust})\n")
    for verdict in _ORDER:
        sgr = _VERDICT_SGR[verdict]
        n = m.verdicts[verdict.value]
        label = paint(f"{verdict.value:<16}", sgr)
        if verdict is Verdict.RANKING_MISS and run.tier == 1:
            w(f"  {label} {'':<24}  {paint('not observable (tier 1 pipeline)', '2')}\n")
            continue
        if verdict is Verdict.GENERATION_MISS and not m.generation_evaluated:
            w(f"  {label} {'':<24}  {paint('not evaluated (no answers or no judge)', '2')}\n")
            continue
        share = n / m.cases if m.cases else 0.0
        bar = paint(f"{bar_char * round(share * 24):<24}", sgr)
        w(f"  {label} {bar} {n:>4}  {share:>4.0%}\n")

    if m.answerable:
        excluded = m.scored - m.answerable
        w(f"\n{paint('RETRIEVAL', '1')}  {m.answerable} answerable")
        w(f" ({excluded} corpus gaps excluded)\n" if excluded else "\n")
        cells = [f"hit@{k} {v:.0%}" for k, v in m.hit_rate.items()]
        if m.mrr is not None:
            cells.append(f"MRR {m.mrr:.2f}")
        w("  " + f"  {dot}  ".join(cells) + "\n")
        if m.latency_p50_ms is not None:
            w(f"  latency p50 {_ms(m.latency_p50_ms)}  {dot}  p95 {_ms(m.latency_p95_ms)}\n")

    recs = fix_first(run)
    if recs:
        w(f"\n{paint('FIX FIRST', '1')}\n")
        for i, rec in enumerate(recs, start=1):
            sgr = _VERDICT_SGR[rec.verdict]
            w(f"  {paint(f'{i}.', '1')} {paint(rec.verdict.value, sgr)} {dot} {rec.count} of "
              f"{sum(r.count for r in recs)} failures ({rec.share:.0%})\n")
            w(f"     {rec.headline}\n")
            if rec.detail:
                w(f"     {paint(rec.detail, '2')}\n")
            w(f"     {paint(arrow, '36')} {rec.actions[0]}\n")
            w(f"     {paint('cases:', '2')} {', '.join(rec.case_ids)}\n")
    else:
        w(f"\n{paint('No failures.', '32')}\n")

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
        w(f"\n{paint('WARNINGS', '1')}\n")
        for warning in warnings:
            w(f"  {paint('!', '33')} {warning}\n")
    out.flush()
