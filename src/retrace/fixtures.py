"""Reading and writing golden sets.

Format is JSONL: one `EvalCase` per line, schema version carried **per record**
rather than once per file. That costs a few bytes a line and buys three things:
a fixture can be migrated case by case, a half-migrated file still reads, and
two fixtures at different versions can be concatenated without ceremony.

Writes are atomic. A golden set represents hours of human labelling, and a
truncated fixture that still parses is the kind of silent corruption that
destroys trust in a measurement tool.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .errors import FixtureError, SchemaVersionError
from .types import (
    FIXTURE_SCHEMA_VERSION,
    SUPPORTED_FIXTURE_VERSIONS,
    EvalCase,
    Provenance,
    Span,
)


def case_to_dict(case: EvalCase) -> dict[str, Any]:
    """Serialise a case. Empty collections are omitted to keep fixtures readable."""
    out: dict[str, Any] = {
        "v": case.schema_version,
        "id": case.id,
        "question": case.question,
        "provenance": case.provenance.value,
    }
    if case.answer is not None:
        out["answer"] = case.answer
    if case.gold_spans:
        out["gold_spans"] = [
            {"doc_id": s.doc_id, "start": s.start, "end": s.end} for s in case.gold_spans
        ]
    if case.gold_doc_ids:
        out["gold_doc_ids"] = list(case.gold_doc_ids)
    if case.gold_chunk_ids:
        out["gold_chunk_ids"] = list(case.gold_chunk_ids)
    if case.source_chunk_id is not None:
        out["source_chunk_id"] = case.source_chunk_id
    if case.labels:
        out["labels"] = case.labels
    return out


def case_from_dict(data: dict[str, Any]) -> EvalCase:
    """Rebuild a case, validating the schema version first."""
    version = data.get("v", FIXTURE_SCHEMA_VERSION)
    if version not in SUPPORTED_FIXTURE_VERSIONS:
        raise SchemaVersionError(found=version, supported=SUPPORTED_FIXTURE_VERSIONS)

    try:
        spans = tuple(
            Span(doc_id=s["doc_id"], start=int(s["start"]), end=int(s["end"]))
            for s in data.get("gold_spans", ())
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FixtureError(f"case {data.get('id')!r} has a malformed gold_span: {exc}") from exc

    raw_provenance = data.get("provenance", Provenance.MODEL.value)
    try:
        provenance = Provenance(raw_provenance)
    except ValueError as exc:
        valid = ", ".join(p.value for p in Provenance)
        raise FixtureError(
            f"case {data.get('id')!r} has unknown provenance {raw_provenance!r}; "
            f"expected one of: {valid}"
        ) from exc

    try:
        return EvalCase(
            id=data["id"],
            question=data["question"],
            answer=data.get("answer"),
            gold_spans=spans,
            gold_doc_ids=tuple(data.get("gold_doc_ids", ())),
            gold_chunk_ids=tuple(data.get("gold_chunk_ids", ())),
            provenance=provenance,
            source_chunk_id=data.get("source_chunk_id"),
            labels=dict(data.get("labels", {})),
            schema_version=version,
        )
    except KeyError as exc:
        raise FixtureError(f"fixture record is missing required field {exc}") from exc
    except ValueError as exc:
        raise FixtureError(str(exc)) from exc


def iter_fixture(path: str | os.PathLike[str]) -> Iterator[EvalCase]:
    """Stream cases from a JSONL fixture.

    Blank lines are skipped. Lines beginning with ``#`` are treated as comments,
    because a hand-edited golden set is a thing people genuinely do and losing
    their annotations to a parse error is needless.
    """
    path = Path(path)
    if not path.exists():
        raise FixtureError(f"fixture not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise FixtureError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise FixtureError(
                    f"{path}:{lineno}: expected a JSON object, "
                    f"got {type(record).__name__}"
                )
            try:
                yield case_from_dict(record)
            except FixtureError as exc:
                raise FixtureError(f"{path}:{lineno}: {exc}") from exc


def read_fixture(path: str | os.PathLike[str]) -> list[EvalCase]:
    """Read a whole fixture, checking for duplicate ids."""
    cases = list(iter_fixture(path))
    seen: dict[str, int] = {}
    for position, case in enumerate(cases):
        if case.id in seen:
            raise FixtureError(
                f"{path}: duplicate case id {case.id!r} at positions "
                f"{seen[case.id]} and {position}"
            )
        seen[case.id] = position
    return cases


def write_fixture(path: str | os.PathLike[str], cases: Iterable[EvalCase]) -> int:
    """Write cases atomically. Returns how many were written.

    Writes to a temporary file in the destination directory, flushes, fsyncs,
    then renames. Either the previous fixture survives intact or the new one
    replaces it whole -- there is no state in which a partially written golden
    set is silently loaded as if complete.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # SIM115: the temp file deliberately outlives its opening expression. It is
    # closed by the `with` below, then renamed over the destination -- a context
    # manager around the whole thing would delete it before the rename.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    count = 0
    try:
        with handle:
            for case in cases:
                handle.write(json.dumps(case_to_dict(case), ensure_ascii=False))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return count


def fixture_stats(cases: Iterable[EvalCase]) -> dict[str, Any]:
    """Summary a reviewer can eyeball before trusting a golden set.

    `trusted` is reported separately from the total everywhere it appears: a set
    of 200 unverified model-generated cases is not a 200-case golden set, and
    conflating the two is the most common way eval numbers turn out to be
    fiction.
    """
    cases = list(cases)
    by_provenance: dict[str, int] = {}
    for case in cases:
        by_provenance[case.provenance.value] = by_provenance.get(case.provenance.value, 0) + 1

    return {
        "total": len(cases),
        "trusted": sum(1 for c in cases if c.is_trusted),
        "by_provenance": by_provenance,
        "with_spans": sum(1 for c in cases if c.gold_spans),
        "with_answer": sum(1 for c in cases if c.answer),
        "documents": len({d for c in cases for d in c.implied_doc_ids}),
    }
