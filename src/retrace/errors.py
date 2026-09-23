"""Exception hierarchy.

One base class so callers can ``except RetraceError`` and catch everything this
library raises deliberately, without also swallowing genuine bugs.
"""

from __future__ import annotations


class RetraceError(Exception):
    """Base class for every error Retrace raises on purpose."""


class FixtureError(RetraceError):
    """A fixture file is malformed, or a record fails validation."""


class SchemaVersionError(FixtureError):
    """A fixture record carries a schema version this build cannot read."""

    def __init__(self, found: int, supported: tuple[int, ...]) -> None:
        self.found = found
        self.supported = supported
        super().__init__(
            f"fixture record declares schema version {found}; "
            f"this build reads {sorted(supported)}. "
            "Migrate the fixture, or pin an older Retrace."
        )


class CorpusError(RetraceError):
    """A corpus is inconsistent, or a lookup was made against the wrong one."""


class LLMError(RetraceError):
    """A language-model call failed in a way the caller should handle per item."""


class LLMRefusalError(LLMError):
    """The model declined the request (and any configured fallback declined too)."""


class LLMOutputError(LLMError):
    """The model answered, but not in a shape Retrace can use.

    Covers truncation at max_tokens and output that is not the requested JSON.
    """
