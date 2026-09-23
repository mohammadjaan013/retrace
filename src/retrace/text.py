"""Shared text primitives: tokenisation and stopwords.

Deliberately crude and dependency-free. Everything built on these is either a
deliberately naive baseline or a cheap first-pass signal that the report labels
as such -- never a claim of linguistic precision.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were be been
    being it its this that these those what which who whom whose when where why how
    do does did done can could should would will shall may might must has have had
    not no nor so than then there their them they we you your our i me my he she his
    her us into onto over under about above below between during through per via
    any some each every all both either neither such same other own very too just
    also only more most much many few less least one two
    """.split()  # noqa: SIM905 - a word list reads better as prose
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that is not a letter or digit. That is all."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """Distinct tokens that carry meaning: no stopwords, no single characters."""
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1}


def fold_plural(token: str) -> str:
    """Strip a regular English plural: trackers -> tracker, batteries -> battery.

    Used only where Retrace *measures* vocabulary overlap, never inside the naive
    retriever -- that retriever's lack of any normalisation is one of the
    weaknesses it exists to exhibit. Deliberately conservative: words ending in
    -ss, -us or -is are left alone (class, bus, analysis).
    """
    if len(token) <= 3 or token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s"):
        return token[:-1]
    return token


def content_terms(text: str) -> set[str]:
    """`content_tokens` with plurals folded -- the unit for overlap measurements."""
    return {fold_plural(t) for t in content_tokens(text)}


def term_set(text: str) -> set[str]:
    """Every token with plurals folded, stopwords kept (the side being searched)."""
    return {fold_plural(t) for t in tokenize(text)}
