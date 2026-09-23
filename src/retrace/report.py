"""Render a run as one self-contained HTML file.

The report is the artefact that travels: an engineer runs Retrace and forwards
the file to someone who never will. So it is built to survive that trip:

* One file. No web fonts, no CDN, no external requests -- it renders the same
  from an email attachment, a CI artefact, or a laptop on a plane.
* Verdicts and the fix-first list come before any detail, so a reader who
  stops after one screen still knows what to do on Monday.
* Works without JavaScript. The only script is an optional search filter.
* Every string that came from a user's documents is escaped. The report
  renders third-party text and must never become an injection vector.
* The full run is embedded as JSON, so tooling can read the report directly.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from .attribution import Attribution
from .metrics import Metrics, Recommendation, compute_metrics, fix_first
from .runner import RunResult, run_to_dict
from .types import Chunk, EvalCase, RetrievalResult, Verdict

_VERDICT_ORDER = (
    Verdict.CORPUS_GAP,
    Verdict.RETRIEVAL_MISS,
    Verdict.RANKING_MISS,
    Verdict.GENERATION_MISS,
    Verdict.UNSCORABLE,
    Verdict.OK,
)

_VERDICT_LABEL = {
    Verdict.OK: "OK",
    Verdict.CORPUS_GAP: "Corpus gap",
    Verdict.RETRIEVAL_MISS: "Retrieval miss",
    Verdict.RANKING_MISS: "Ranking miss",
    Verdict.GENERATION_MISS: "Generation miss",
    Verdict.UNSCORABLE: "Unscorable",
}

_VERDICT_BLURB = {
    Verdict.CORPUS_GAP: "The answer is not in the indexed documents.",
    Verdict.RETRIEVAL_MISS: "Indexed, but never retrieved.",
    Verdict.RANKING_MISS: "Retrieved, then cut before the model saw it.",
    Verdict.GENERATION_MISS: "The model had the evidence and answered wrong.",
    Verdict.UNSCORABLE: "Retrace could not decide, and says so.",
    Verdict.OK: "Evidence retrieved and kept in context.",
}


def e(value: Any) -> str:
    """Escape for HTML text and attribute contexts."""
    return html.escape("" if value is None else str(value), quote=True)


def _pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d else "&ndash;"


CSS = """
:root{
  --ground:#F4F6F8; --surface:#FFFFFF; --sunken:#EBEEF2;
  --ink:#141A22; --ink-2:#4A5563; --ink-3:#7B8594;
  --line:#DCE1E7; --line-2:#C6CDD6;
  --accent:#2C5A94; --mark:#FFE9A8; --mark-ink:#3A2E00;
  --v-ok:#2F7D5B; --v-gap:#7A5BA8; --v-ret:#C0601D; --v-rank:#2F6DB5;
  --v-gen:#B7404A; --v-uns:#7A8290;
  --v-ok-bg:#E3F2EA; --v-gap-bg:#EEE8F6; --v-ret-bg:#FBEBDD; --v-rank-bg:#E2ECF8;
  --v-gen-bg:#F8E3E5; --v-uns-bg:#ECEEF1;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"Cascadia Mono","SF Mono",Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0E1318; --surface:#161C24; --sunken:#1D252F;
    --ink:#E3E8EE; --ink-2:#9DA8B5; --ink-3:#6E7987;
    --line:#26303B; --line-2:#34404D;
    --accent:#7FA9E0; --mark:#5A4A12; --mark-ink:#FFF3CC;
    --v-ok:#5CC49A; --v-gap:#B7A1DD; --v-ret:#EFA365; --v-rank:#80B0EC;
    --v-gen:#EC8C94; --v-uns:#9AA3B0;
    --v-ok-bg:#132A21; --v-gap-bg:#221C30; --v-ret-bg:#2E2014; --v-rank-bg:#15223A;
    --v-gen-bg:#301A1D; --v-uns-bg:#1E242C;
  }
}
:root[data-theme="dark"]{
  --ground:#0E1318; --surface:#161C24; --sunken:#1D252F;
  --ink:#E3E8EE; --ink-2:#9DA8B5; --ink-3:#6E7987;
  --line:#26303B; --line-2:#34404D;
  --accent:#7FA9E0; --mark:#5A4A12; --mark-ink:#FFF3CC;
  --v-ok:#5CC49A; --v-gap:#B7A1DD; --v-ret:#EFA365; --v-rank:#80B0EC;
  --v-gen:#EC8C94; --v-uns:#9AA3B0;
  --v-ok-bg:#132A21; --v-gap-bg:#221C30; --v-ret-bg:#2E2014; --v-rank-bg:#15223A;
  --v-gen-bg:#301A1D; --v-uns-bg:#1E242C;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.55 var(--sans)}
.wrap{max-width:1040px;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
h1,h2,h3{margin:0;line-height:1.2;text-wrap:balance}
h1{font-size:24px;font-weight:650;letter-spacing:-.01em}
h2{font-size:17px;font-weight:650;margin-bottom:12px}
p{margin:0}
code,.mono{font-family:var(--mono);font-size:.9em}
.muted{color:var(--ink-2)} .faint{color:var(--ink-3)}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
section{margin-top:36px}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:10px;color:var(--ink-2);font-size:13px}
.meta b{color:var(--ink);font-weight:600}

.headline{margin-top:22px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:20px}
.headline .lede{font-size:18px;font-weight:600;max-width:62ch}
.stack{display:flex;height:14px;border-radius:3px;overflow:hidden;margin-top:16px;background:var(--sunken)}
.stack span{display:block;height:100%}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:10px;margin-top:14px}
.legend div{display:flex;flex-direction:column;gap:2px;padding-left:10px;border-left:3px solid var(--c)}
.legend .n{font:600 20px/1.1 var(--sans);font-variant-numeric:tabular-nums}
.legend .l{font-size:12.5px;color:var(--ink-2)}
.legend .na .n{font-size:13px;font-weight:500;color:var(--ink-3)}

.fix{counter-reset:fix;display:flex;flex-direction:column;gap:12px}
.fix article{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px 18px;
  display:grid;grid-template-columns:auto 1fr;gap:4px 14px}
.fix .num{grid-row:span 4;font:700 22px/1 var(--mono);color:var(--c);padding-top:2px}
.fix h3{font-size:16px}
.fix .share{font:500 12.5px var(--mono);color:var(--ink-3);margin-left:6px}
.fix ol{margin:6px 0 0;padding-left:18px;color:var(--ink-2)}
.fix ol li{margin-bottom:4px;max-width:78ch}
.fix ol li:first-child{color:var(--ink)}
.fix .detail{font-size:13px;color:var(--ink-2)}
.fix .cases{font-size:12.5px;color:var(--ink-3)}
.fix .cases a{color:var(--accent)}

.pill{display:inline-block;font:600 11px/1 var(--mono);letter-spacing:.04em;padding:4px 7px;
  border-radius:3px;color:var(--c);background:var(--bg);white-space:nowrap}

.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 11px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);background:var(--sunken)}
tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums;font-family:var(--mono);text-align:right;white-space:nowrap}
.metrics td:first-child{color:var(--ink-2)}

.warn{border-left:3px solid var(--v-ret);background:var(--v-ret-bg);padding:10px 14px;border-radius:0 4px 4px 0;
  font-size:13.5px;margin-top:8px;max-width:86ch}

.filter{display:flex;gap:10px;align-items:center;margin-bottom:14px}
.filter input{flex:1;max-width:420px;font:inherit;padding:8px 10px;border:1px solid var(--line-2);
  border-radius:4px;background:var(--surface);color:var(--ink)}
.group{margin-top:22px}
.group>h3{display:flex;align-items:center;gap:10px;font-size:15px;margin-bottom:8px}
.group>h3 .blurb{font-weight:400;color:var(--ink-2);font-size:13.5px}
details.case{background:var(--surface);border:1px solid var(--line);border-radius:6px;margin-top:8px}
details.case>summary{cursor:pointer;padding:11px 14px;display:grid;grid-template-columns:1fr auto;gap:12px;
  list-style:none;align-items:baseline}
details.case>summary::-webkit-details-marker{display:none}
details.case>summary:hover{background:var(--sunken)}
details.case[open]>summary{border-bottom:1px solid var(--line)}
.q{font-weight:550}
.cid{font:12px var(--mono);color:var(--ink-3)}
.body{padding:12px 14px 16px;display:flex;flex-direction:column;gap:10px}
.kv{display:grid;grid-template-columns:130px 1fr;gap:4px 14px;font-size:13.5px}
.kv dt{color:var(--ink-3)} .kv dd{margin:0;max-width:80ch}
.notes{margin:0;padding-left:18px;font-size:13px;color:var(--ink-2)}
.trace td.x{font-size:12.5px;line-height:1.5;max-width:560px}
.trace tr.gold td{background:var(--v-ok-bg)}
.trace .tick{font:600 11px var(--mono)}
.trace .in{color:var(--v-ok)} .trace .out{color:var(--ink-3)}
mark{background:var(--mark);color:var(--mark-ink);padding:0 1px;border-radius:2px}
.legs{font:11px var(--mono);color:var(--ink-3)}
footer{margin-top:48px;padding-top:14px;border-top:1px solid var(--line);font-size:12.5px;color:var(--ink-3);
  display:flex;flex-wrap:wrap;gap:8px 18px;justify-content:space-between}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:620px){
  .kv{grid-template-columns:1fr}
  details.case>summary{grid-template-columns:1fr}
}
@media print{
  body{background:#fff}
  details.case{break-inside:avoid}
  details.case:not([open])>summary{border-bottom:none}
  .filter{display:none}
}
"""

_COLOUR_VARS = {
    Verdict.OK: ("--v-ok", "--v-ok-bg"),
    Verdict.CORPUS_GAP: ("--v-gap", "--v-gap-bg"),
    Verdict.RETRIEVAL_MISS: ("--v-ret", "--v-ret-bg"),
    Verdict.RANKING_MISS: ("--v-rank", "--v-rank-bg"),
    Verdict.GENERATION_MISS: ("--v-gen", "--v-gen-bg"),
    Verdict.UNSCORABLE: ("--v-uns", "--v-uns-bg"),
}


def _style(verdict: Verdict) -> str:
    fg, bg = _COLOUR_VARS[verdict]
    return f"--c:var({fg});--bg:var({bg})"


def _pill(verdict: Verdict) -> str:
    return f'<span class="pill" style="{_style(verdict)}">{e(verdict.value)}</span>'


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


_NOUNS = {
    Verdict.RETRIEVAL_MISS: ("retrieval miss", "retrieval misses"),
    Verdict.CORPUS_GAP: ("corpus gap", "corpus gaps"),
    Verdict.RANKING_MISS: ("ranking miss", "ranking misses"),
    Verdict.GENERATION_MISS: ("generation miss", "generation misses"),
}


def _lede(run: RunResult, m: Metrics) -> str:
    counts = {v: m.verdicts[v.value] for v in _NOUNS}
    failures = sum(counts.values())
    if not failures:
        return f"All {m.cases} questions passed."
    parts = [
        f"{n} {_NOUNS[v][0] if n == 1 else _NOUNS[v][1]}"
        for v, n in sorted(counts.items(), key=lambda kv: -kv[1])
        if n
    ]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    text = f"{failures} of {m.cases} questions failed: {joined}."
    gaps = m.verdicts[Verdict.CORPUS_GAP.value]
    if gaps:
        text += (f" {gaps} of those cannot be fixed by any retrieval change "
                 "&mdash; the answer is not indexed.")
    return text


def _headline(run: RunResult, m: Metrics) -> str:
    segs, legend = [], []
    for v in _VERDICT_ORDER:
        n = m.verdicts[v.value]
        not_observed = (v is Verdict.RANKING_MISS and run.tier == 1) or (
            v is Verdict.GENERATION_MISS and not m.generation_evaluated
        )
        if n:
            width = n / m.cases * 100 if m.cases else 0
            segs.append(f'<span style="width:{width:.3f}%;background:var({_COLOUR_VARS[v][0]})" '
                        f'title="{e(v.value)}: {n}"></span>')
        if not_observed:
            reason = "tier 1 pipeline" if v is Verdict.RANKING_MISS else "no answers judged"
            legend.append(f'<div class="na" style="{_style(v)}"><span class="n">not observed</span>'
                          f'<span class="l">{e(_VERDICT_LABEL[v])} &middot; {e(reason)}</span></div>')
        else:
            legend.append(f'<div style="{_style(v)}"><span class="n">{n}</span>'
                          f'<span class="l">{e(_VERDICT_LABEL[v])} &middot; {_pct(n, m.cases)}</span></div>')
    return (
        '<div class="headline">'
        f'<p class="lede">{_lede(run, m)}</p>'
        f'<div class="stack" role="img" aria-label="Verdict breakdown">{"".join(segs)}</div>'
        f'<div class="legend">{"".join(legend)}</div>'
        "</div>"
    )


def _fix_first(recs: list[Recommendation]) -> str:
    if not recs:
        return ""
    total = sum(r.count for r in recs)
    items = []
    for i, r in enumerate(recs, start=1):
        cases = ", ".join(f'<a href="#case-{e(c)}">{e(c)}</a>' for c in r.case_ids)
        more = r.count - len(r.case_ids)
        if more > 0:
            cases += f" and {more} more"
        detail = f'<p class="detail">{e(r.detail)}</p>' if r.detail else ""
        actions = "".join(f"<li>{e(a)}</li>" for a in r.actions)
        items.append(
            f'<article style="{_style(r.verdict)}">'
            f'<span class="num">{i}</span>'
            f"<h3>{e(r.headline)}<span class=\"share\">{r.count} of {total} failures &middot; "
            f"{r.share:.0%}</span></h3>"
            f"{detail}<ol>{actions}</ol>"
            f'<p class="cases">{_pill(r.verdict)} &nbsp;{cases}</p>'
            "</article>"
        )
    return f'<section><h2>Fix first</h2><div class="fix">{"".join(items)}</div></section>'


def _metrics_table(run: RunResult, m: Metrics) -> str:
    if not m.answerable:
        return ""
    ks = list(m.hit_rate)
    head = "".join(f'<th class="num">@{k}</th>' for k in ks)

    def row(label: str, values: dict[int, float]) -> str:
        cells = "".join(
            f'<td class="num">{values[k]:.0%}</td>' if values.get(k) is not None
            else '<td class="num faint">&ndash;</td>'
            for k in ks
        )
        return f"<tr><td>{e(label)}</td>{cells}</tr>"

    rows = row("Hit rate", m.hit_rate) + row("Precision", m.precision)
    if m.recall:
        rows += row("Recall", m.recall)

    extra = []
    if m.mrr is not None:
        extra.append(f"MRR <b class=\"mono\">{m.mrr:.2f}</b>")
    if m.latency_p50_ms is not None:
        extra.append(f"latency p50 <b class=\"mono\">{m.latency_p50_ms:g} ms</b> &middot; "
                     f"p95 <b class=\"mono\">{m.latency_p95_ms:g} ms</b>")
    if m.generation_accuracy is not None:
        extra.append(f"answer accuracy <b class=\"mono\">{m.generation_accuracy:.0%}</b> "
                     f"of {m.generation_evaluated} judged")
    excluded = m.scored - m.answerable
    note = (f" &middot; {excluded} corpus gap{'s' if excluded != 1 else ''} excluded, because "
            "an answer that is not indexed is not a retrieval failure") if excluded else ""
    return (
        "<section><h2>Retrieval</h2>"
        f'<p class="muted" style="margin-bottom:10px">Over {m.answerable} answerable questions{note}.</p>'
        f'<div class="tablewrap"><table class="metrics"><thead><tr><th></th>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        + (f'<p class="meta">{" &nbsp;&middot;&nbsp; ".join(extra)}</p>' if extra else "")
        + "</section>"
    )


def _warnings(run: RunResult, m: Metrics) -> str:
    items = []
    if run.coverage_warning:
        items.append(run.coverage_warning)
    if not run.corpus_checked:
        items.append("No corpus was supplied, so corpus gaps are hidden inside retrieval "
                     "misses. Re-run with the indexed documents to separate them.")
    if run.tier == 1:
        items.append("This pipeline exposes only its final context, so retrieval and ranking "
                     "failures cannot be told apart. Expose the pre-rerank candidate list to "
                     "get the ranking stage.")
    if m.trusted < m.cases:
        items.append(f"{m.cases - m.trusted} of {m.cases} cases are unverified model output. "
                     "Review them before quoting these numbers.")
    if m.errors:
        items.append(f"{m.errors} cases raised an error inside the pipeline and are unscorable.")
    if not items:
        return ""
    body = "".join(f'<p class="warn">{e(t)}</p>' for t in items)
    return f"<section><h2>Before you quote these numbers</h2>{body}</section>"


def _highlight(chunk: Chunk, case: EvalCase) -> str:
    """Chunk text with the gold evidence marked, when offsets allow it."""
    text = chunk.text
    truncated = text.endswith("…")
    body = text[:-1] if truncated else text
    if chunk.span is None or not case.gold_spans:
        return e(text)

    ranges = []
    for gold in case.gold_spans:
        if gold.doc_id != chunk.span.doc_id:
            continue
        lo = max(chunk.span.start, gold.start) - chunk.span.start
        hi = min(chunk.span.end, gold.end) - chunk.span.start
        lo, hi = max(0, lo), min(len(body), hi)
        if hi > lo:
            ranges.append((lo, hi))
    if not ranges:
        return e(text)

    ranges.sort()
    out, pos = [], 0
    for lo, hi in ranges:
        lo = max(lo, pos)
        if hi <= lo:
            continue
        out.append(e(body[pos:lo]))
        out.append(f"<mark>{e(body[lo:hi])}</mark>")
        pos = hi
    out.append(e(body[pos:]))
    if truncated:
        out.append("…")
    return "".join(out)


def _legs(chunk: Chunk) -> str:
    legs = chunk.meta.get("legs") if isinstance(chunk.meta, dict) else None
    if not isinstance(legs, dict) or not legs:
        return ""
    parts = [f"{e(k)} #{e(v)}" if v is not None else f"{e(k)} &ndash;" for k, v in legs.items()]
    return f'<div class="legs">{" &middot; ".join(parts)}</div>'


def _trace(case: EvalCase, a: Attribution, trace: RetrievalResult | None) -> str:
    if trace is None:
        return ""
    from .attribution import gold_matcher

    is_gold = gold_matcher(case)
    context_ids = {c.id for c in trace.context}
    rows = []
    shown = list(trace.candidates)
    # Context chunks injected outside the candidate list still belong in the trace.
    shown += [c for c in trace.context if c.id not in {x.id for x in trace.candidates}]
    for rank, chunk in enumerate(shown, start=1):
        gold = bool(is_gold(chunk))
        in_ctx = chunk.id in context_ids
        tick = '<span class="tick in">in</span>' if in_ctx else '<span class="tick out">cut</span>'
        score = f"{chunk.score:.3f}" if isinstance(chunk.score, (int, float)) else "-"
        gold_cell = '<span class="tick in">gold</span>' if gold else ""
        rows.append(
            f'<tr class="{"gold" if gold else ""}"><td class="num">{rank}</td><td>{tick}</td>'
            f"<td>{gold_cell}</td><td class=\"num\">{e(score)}</td>"
            f'<td><div class="mono">{e(chunk.doc_id)}</div>{_legs(chunk)}</td>'
            f'<td class="x">{_highlight(chunk, case)}</td></tr>'
        )
    if not rows:
        return '<p class="faint">The pipeline returned nothing for this question.</p>'
    return (
        '<div class="tablewrap"><table class="trace"><thead><tr><th class="num">#</th>'
        "<th>Context</th><th>Gold</th><th class=\"num\">Score</th><th>Document</th>"
        f"<th>Text</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _case(case: EvalCase, a: Attribution, trace: RetrievalResult | None) -> str:
    rows = [("Why", e(a.reason))]
    if case.answer:
        rows.append(("Expected", e(case.answer)))
    if trace is not None and trace.answer:
        rows.append(("Answered", e(trace.answer)))
    if a.candidate_rank is not None:
        where = f"candidate rank {a.candidate_rank} of {a.n_candidates}"
        if a.context_rank is not None:
            where += f", context position {a.context_rank} of {a.n_context}"
        rows.append(("Gold found at", e(where)))
    if a.query_gold_overlap is not None and a.verdict is Verdict.RETRIEVAL_MISS:
        kind = "vocabulary mismatch" if a.query_gold_overlap < 0.34 else "findable by keyword"
        rows.append(("Word overlap", e(f"{a.query_gold_overlap:.0%} of question terms in the "
                                        f"evidence ({kind})")))
    gold_desc = ", ".join(f"{s.doc_id} [{s.start}:{s.end}]" for s in case.gold_spans) or ", ".join(
        case.gold_doc_ids)
    rows.append(("Gold evidence", f'<span class="mono">{e(gold_desc)}</span>'))
    if not a.trusted:
        rows.append(("Label", "unverified model output"))
    if a.error:
        rows.append(("Error", f'<span class="mono">{e(a.error)}</span>'))

    kv = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    notes = "".join(f"<li>{e(n)}</li>" for n in a.notes)
    notes_html = f'<ul class="notes">{notes}</ul>' if notes else ""
    lucky = ' <span class="pill" style="--c:var(--v-ok);--bg:var(--v-ok-bg)">LUCKY</span>' if a.lucky else ""
    return (
        f'<details class="case" id="case-{e(case.id)}" data-search="{e(case.question.lower())} {e(case.id)}">'
        f'<summary><span class="q">{e(case.question)}{lucky}</span><span class="cid">{e(case.id)}</span></summary>'
        f'<div class="body"><dl class="kv">{kv}</dl>{notes_html}{_trace(case, a, trace)}</div>'
        "</details>"
    )


def _cases(run: RunResult) -> str:
    by_case = {c.id: c for c in run.cases}
    groups = []
    for v in _VERDICT_ORDER:
        items = [a for a in run.attributions if a.verdict is v]
        if not items:
            continue
        body = "".join(_case(by_case[a.case_id], a, run.traces.get(a.case_id)) for a in items)
        groups.append(
            f'<div class="group"><h3>{_pill(v)} {len(items)}'
            f'<span class="blurb">{e(_VERDICT_BLURB[v])}</span></h3>{body}</div>'
        )
    return (
        "<section><h2>Every question</h2>"
        '<div class="filter"><input id="q" type="search" placeholder="Filter by question or id" '
        'aria-label="Filter questions"><span class="faint" id="shown"></span></div>'
        + "".join(groups)
        + "</section>"
    )


_SCRIPT = """
(function(){
  var input=document.getElementById('q'), shown=document.getElementById('shown');
  if(!input) return;
  var cases=[].slice.call(document.querySelectorAll('details.case'));
  input.addEventListener('input',function(){
    var q=input.value.trim().toLowerCase(), n=0;
    cases.forEach(function(el){
      var hit=!q||el.getAttribute('data-search').indexOf(q)!==-1;
      el.hidden=!hit; if(hit) n++;
    });
    shown.textContent=q?(n+' of '+cases.length):'';
  });
})();
"""


def _json_island(data: Any) -> str:
    """JSON that is safe to place inside a <script> element.

    Escaping only ``</`` is not enough: a ``<!--`` followed by ``<script`` in the
    data pushes the HTML parser into its double-escaped script state and moves
    where the element ends. Encoding every ``<``, ``>`` and ``&`` as a JSON
    unicode escape leaves no markup of any kind, and ``JSON.parse`` restores it.
    """
    text = json.dumps(data, ensure_ascii=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_report(run: RunResult) -> str:
    """The complete HTML document as a string."""
    m = compute_metrics(run)
    recs = fix_first(run)

    meta = [
        f"<span><b>{m.cases}</b> questions</span>",
        f"<span>tier <b>{run.tier}</b></span>",
        f"<span>corpus <b>{'checked' if run.corpus_checked else 'not supplied'}</b></span>",
        f"<span>judge <b>{e(run.judge or 'none')}</b></span>",
    ]
    if run.started_at:
        meta.append(f"<span>{e(run.started_at)}</span>")

    payload = _json_island(run_to_dict(run))

    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>Retrace report &middot; {m.cases} questions</title>"
        f"<style>{CSS}</style></head><body><main class=\"wrap\">"
        '<p class="eyebrow">Retrace report</p>'
        f'<h1 style="margin-top:8px">{e(run.pipeline or "Pipeline")}</h1>'
        f'<div class="meta">{"".join(meta)}</div>'
        + _headline(run, m)
        + _fix_first(recs)
        + _metrics_table(run, m)
        + _warnings(run, m)
        + _cases(run)
        + '<footer><span>Generated by Retrace '
        + e(run.retrace_version)
        + "</span><span>The full run is embedded in this file as JSON "
        "(<code>#retrace-run</code>).</span></footer>"
        f'</main><script type="application/json" id="retrace-run">{payload}</script>'
        f"<script>{_SCRIPT}</script></body></html>\n"
    )


def write_report(run: RunResult, path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(run), encoding="utf-8", newline="\n")
    return path
