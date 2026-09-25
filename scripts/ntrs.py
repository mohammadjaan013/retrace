"""Build the NASA Technical Reports Server (NTRS) demo corpus, reproducibly.

Steps, deliberately separate so the download list can be reviewed first:

    python scripts/ntrs.py manifest   # search + header checks only; writes the list
    python scripts/ntrs.py fetch      # downloads exactly what the manifest lists, then prepares
    python scripts/ntrs.py prepare    # re-derives prepared text from raw, no network

RAW AND PREPARED TEXT
---------------------
`fetch` stores NASA's bytes untouched in raw.jsonl and pins their SHA-256 --
that is provenance. `prepare` derives documents.jsonl from it with a
deterministic, versioned conversion, and pins that text's hash too. Every gold
span is an offset into the *prepared* text, so the conversion is part of the
contract: anyone re-running this script gets byte-identical text and valid
offsets, and a changed conversion is caught by the hash instead of silently
shifting labels.

The conversion exists because 15 of the first 200 documents (all 2021 or
later) came back as full XHTML rather than plain text -- 9-19% of each file was
tags, with PDF metadata in the head. Left in, "p" and "div" become retrieval
terms and synthesis spends model calls on questions about XMP metadata.

WHY NASA'S .txt AND NOT THE PDFs
--------------------------------
Gold labels are character offsets into document text. If Retrace extracted text
from PDFs itself, a different PDF library version on another machine would
produce slightly different text and silently shift every offset -- the public
golden set would stop reproducing. NTRS serves its own extracted full text next
to each PDF: the same bytes for everyone, typically ~1000x smaller than the PDF,
and still realistically messy (OCR noise, broken columns, drifting formats).
`fetch` records a SHA-256 per file so any drift on NASA's side is detected
rather than silently corrupting labels.

WHAT QUALIFIES
--------------
* distribution PUBLIC, with downloadable files;
* copyright determination explicitly permitting public use -- being hosted by
  NASA is not the same as being free to reuse, and about half of all results
  are not marked that way;
* a document type likely to contain prose (papers, memoranda, reports), not
  abstracts, slide decks or whole proceedings volumes;
* a full-text file that actually exists and is long enough to ask questions of.

Standard library only. Requests are rate-limited and identify the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "ntrs-manifest.json"
RAW_OUT = ROOT / "data" / "ntrs" / "raw.jsonl"
DOCS_OUT = ROOT / "data" / "ntrs" / "documents.jsonl"

PREPARE_VERSION = 1
"""Bump whenever `to_plain_text` changes: offsets into prepared text depend on it."""

API = "https://ntrs.nasa.gov"
USER_AGENT = "retrace-rag/0.1 corpus builder (+https://github.com/mohammadjaan013/retrace)"
DELAY_S = 0.4  # between requests: be a polite guest on a public service

TOPICS = {
    "thermal": "spacecraft thermal control",
    "power": "spacecraft power system",
    "attitude": "spacecraft attitude determination and control",
    "propulsion": "spacecraft electric propulsion thruster",
    "comms": "deep space communications link",
}
PER_TOPIC = 40

ALLOWED_COPYRIGHT = {"GOV_PUBLIC_USE_PERMITTED", "PUBLIC_USE_PERMITTED"}
EXCLUDED_TYPES = {"ABSTRACT", "PRESENTATION", "POSTER", "CONFERENCE_PROCEEDINGS", "OTHER"}
MIN_BYTES = 4_000      # below this it is usually an abstract or a failed extraction
MAX_BYTES = 400_000    # above this it is usually a whole volume; skip to keep topics balanced


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(_request(url), timeout=60) as resp:
        return json.load(resp)


def _head_size(url: str) -> int | None:
    """Content-Length of `url`, or None if it does not exist."""
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _fulltext_link(result: dict[str, Any]) -> str | None:
    for dl in result.get("downloads") or []:
        link = (dl.get("links") or {}).get("fulltext")
        if link and not dl.get("draft"):
            return API + link
    return None


def _eligible(result: dict[str, Any]) -> str | None:
    """None when eligible, otherwise the reason it is not."""
    if result.get("distribution") != "PUBLIC":
        return "not public"
    if not result.get("downloadsAvailable"):
        return "no files"
    if (result.get("copyright") or {}).get("determinationType") not in ALLOWED_COPYRIGHT:
        return "not marked public-use"
    if result.get("stiType") in EXCLUDED_TYPES:
        return f"type {result.get('stiType')}"
    if not _fulltext_link(result):
        return "no full text"
    return None


def build_manifest() -> dict[str, Any]:
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    for topic, query in TOPICS.items():
        kept, offset = 0, 0
        print(f"[{topic}] {query}")
        while kept < PER_TOPIC and offset < 1000:
            params = urllib.parse.urlencode({"q": query, "page.size": 100, "page.from": offset})
            page = _get_json(f"{API}/api/citations/search?{params}")["results"]
            time.sleep(DELAY_S)
            if not page:
                break
            offset += len(page)
            for r in page:
                if kept >= PER_TOPIC:
                    break
                doc_id = str(r["id"])
                if doc_id in seen:
                    continue
                reason = _eligible(r)
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                url = _fulltext_link(r)
                size = _head_size(url)
                time.sleep(DELAY_S)
                if size is None:
                    key = "full text missing (404)"
                    skipped[key] = skipped.get(key, 0) + 1
                    continue
                if not MIN_BYTES <= size <= MAX_BYTES:
                    key = "too short" if size < MIN_BYTES else "too long"
                    skipped[key] = skipped.get(key, 0) + 1
                    continue
                seen.add(doc_id)
                pub = (r.get("publications") or [{}])[0].get("publicationDate") or ""
                entries.append({
                    "id": doc_id,
                    "topic": topic,
                    "title": r.get("title", "").strip(),
                    "type": r.get("stiType"),
                    "year": pub[:4] or None,
                    "copyright": r["copyright"]["determinationType"],
                    "url": url,
                    "bytes": size,
                    "sha256": None,
                })
                kept += 1
                print(f"  {kept:>3}/{PER_TOPIC}  {size:>7,} B  {doc_id}  {r.get('title', '')[:60]}")

    return {
        "source": "NASA Technical Reports Server (ntrs.nasa.gov) full-text extractions",
        "built": time.strftime("%Y-%m-%d"),
        "criteria": {
            "copyright": sorted(ALLOWED_COPYRIGHT),
            "excluded_types": sorted(EXCLUDED_TYPES),
            "bytes": [MIN_BYTES, MAX_BYTES],
            "topics": TOPICS,
            "per_topic": PER_TOPIC,
        },
        "skipped": dict(sorted(skipped.items(), key=lambda kv: -kv[1])),
        "documents": entries,
    }


def summarise(manifest: dict[str, Any]) -> None:
    docs = manifest["documents"]
    total = sum(d["bytes"] for d in docs)
    print(f"\n{len(docs)} documents, {total / 1e6:.2f} MB of text in total")
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        by_topic.setdefault(d["topic"], []).append(d)
    for topic, items in by_topic.items():
        size = sum(i["bytes"] for i in items)
        print(f"  {topic:<11} {len(items):>3} docs  {size / 1e6:>5.2f} MB")
    years = sorted(int(d["year"]) for d in docs if d.get("year"))
    if years:
        print(f"  years      {years[0]}-{years[-1]} (median {years[len(years) // 2]})")
    print("  skipped   ", ", ".join(f"{k}: {v}" for k, v in manifest["skipped"].items()))


def _save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomic write: a half-written corpus must never look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def fetch(manifest: dict[str, Any]) -> None:
    """Download NASA's bytes untouched, pin their hashes, then prepare."""
    docs = manifest["documents"]
    records = []
    for i, d in enumerate(docs, 1):
        with urllib.request.urlopen(_request(d["url"]), timeout=60) as resp:
            raw = resp.read()
        time.sleep(DELAY_S)
        digest = hashlib.sha256(raw).hexdigest()
        if d["sha256"] and d["sha256"] != digest:
            print(f"  ! {d['id']}: content changed since the manifest was pinned", file=sys.stderr)
        d["sha256"] = digest
        records.append({"id": d["id"], "text": raw.decode("utf-8", errors="replace")})
        print(f"  {i:>3}/{len(docs)}  {len(raw):>7,} B  {d['id']}")
    _write_jsonl(RAW_OUT, records)
    _save_manifest(manifest)
    print(f"\nwrote {len(records)} raw documents -> {RAW_OUT}")
    prepare(manifest)


# --------------------------------------------------------------------------
# Raw -> prepared text
# --------------------------------------------------------------------------

_BLOCK_TAGS = frozenset(
    ["p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4",
     "h5", "h6", "title", "blockquote", "pre", "section"]
)
_SKIP_TAGS = frozenset({"head", "script", "style"})


class _PlainText(HTMLParser):
    """XHTML to text: drop the head, keep body text, turn blocks into line breaks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS and not self._skip:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS and not self._skip:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def is_markup(text: str) -> bool:
    head = text.lstrip()[:500].lower()
    return head.startswith("<") and ("<html" in head or "<meta" in head or "<?xml" in head)


def to_plain_text(text: str) -> str:
    """Deterministic conversion. Part of the offset contract: see PREPARE_VERSION."""
    if not is_markup(text):
        return text
    parser = _PlainText()
    parser.feed(text)
    parser.close()
    out = "".join(parser.parts)
    out = re.sub(r"[ \t]+\n", "\n", out)       # trailing spaces
    out = re.sub(r"\n{3,}", "\n\n", out)        # runs of blank lines
    return out.strip() + "\n"


def prepare(manifest: dict[str, Any]) -> None:
    """Derive documents.jsonl from raw.jsonl. No network."""
    if not RAW_OUT.exists():
        print(f"no raw corpus at {RAW_OUT}: run `python scripts/ntrs.py fetch`", file=sys.stderr)
        raise SystemExit(2)
    raw = {}
    with RAW_OUT.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                raw[record["id"]] = record["text"]

    records, converted = [], 0
    for d in manifest["documents"]:
        text = raw[d["id"]]
        markup = is_markup(text)
        prepared = to_plain_text(text)
        converted += markup
        d["format"] = "xhtml" if markup else "text"
        d["text_sha256"] = hashlib.sha256(prepared.encode("utf-8")).hexdigest()
        d["chars"] = len(prepared)
        records.append({"id": d["id"], "text": prepared, "title": d["title"], "topic": d["topic"]})

    manifest["prepare_version"] = PREPARE_VERSION
    _write_jsonl(DOCS_OUT, records)
    _save_manifest(manifest)
    print(f"prepared {len(records)} documents ({converted} converted from XHTML) -> {DOCS_OUT}")
    print(f"pinned prepared-text hashes (prepare v{PREPARE_VERSION}) -> {MANIFEST}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["manifest", "fetch", "prepare", "summary"])
    args = parser.parse_args(argv)

    if args.step == "manifest":
        manifest = build_manifest()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        summarise(manifest)
        print(f"\nmanifest -> {MANIFEST}\nreview it, then: python scripts/ntrs.py fetch")
        return 0

    if not MANIFEST.exists():
        print("no manifest yet: run `python scripts/ntrs.py manifest` first", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.step == "summary":
        summarise(manifest)
        return 0
    if args.step == "prepare":
        prepare(manifest)
        return 0
    fetch(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
