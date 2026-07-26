"""Data structures shared by every stage of the pipeline.

The important property here is that :class:`TextBlock` is produced both by the
PDF text layer and by OCR.  Everything downstream of extraction works on these
blocks alone and cannot tell which of the two produced them, so the matching and
scoring code never needs a text-layer branch and an OCR branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

BBox = tuple[float, float, float, float]
"""Axis-aligned box as ``(x0, y0, x1, y1)`` in PDF points, origin top-left."""


class Source(StrEnum):
    """Where the text of a page came from."""

    TEXT_LAYER = "text_layer"
    """Extracted from the PDF's own text layer -- characters are exact."""

    OCR = "ocr"
    """Recognised from a rasterised image -- characters may be wrong."""


class MatchKind(StrEnum):
    """How a keyword was matched against the page text."""

    EXACT = "exact"
    SUBSEQUENCE = "subsequence"
    FUZZY = "fuzzy"
    REGEX = "regex"


class Scope(StrEnum):
    """Which part of the page a rule is allowed to match in."""

    WHOLE = "whole"
    TOP_QUARTER = "top_quarter"
    """The top 25% of the page, where document titles almost always sit."""


class Verdict(StrEnum):
    """Final classification of a document."""

    INVOICE = "invoice"
    NEEDS_REVIEW = "needs_review"
    OTHER = "other"


VERDICT_LABELS: dict[Verdict, str] = {
    Verdict.INVOICE: "Match",
    Verdict.NEEDS_REVIEW: "Review",
    Verdict.OTHER: "Other",
}
"""The single source of truth for how a verdict is named to a person.

The enum values (``invoice`` etc.) are internal and appear in the machine-facing
JSON. These labels are what a human sees -- in the GUI's result, in the progress
window, and as the destination folder names -- so all three read the same and a
run tuned for another document type is never labelled "invoice". Defined here,
in the one module both the core and the front ends already import, so the three
places cannot drift apart.
"""


@dataclass(frozen=True)
class TextBlock:
    """A run of text with a position on the page."""

    text: str
    bbox: BBox
    confidence: float | None = None
    """OCR confidence in 0..1, or ``None`` for text-layer blocks (always exact)."""


@dataclass(frozen=True)
class Page:
    """One extracted page."""

    number: int
    """1-based page number."""

    width: float
    height: float
    blocks: list[TextBlock]
    source: Source

    def top_quarter_cutoff(self) -> float:
        """Y coordinate below which a block is no longer "near the top"."""
        return self.height * 0.25


@dataclass(frozen=True)
class Match:
    """A single occurrence of a pattern inside the normalized page text."""

    start: int
    """Start index into the *normalized* text."""

    end: int
    """End index (exclusive) into the *normalized* text."""

    matched_text: str
    """The text as it actually appears, so a reviewer can see ``請求澤書``
    rather than just being told that ``請求書`` matched."""

    kind: MatchKind
    distance: int = 0
    """Edit distance for fuzzy matches, 0 otherwise."""


@dataclass(frozen=True)
class Hit:
    """A rule that fired, together with what it contributed to the score."""

    rule_id: str
    pattern: str
    weight: float
    """The rule's weight; negative for exclusion keywords."""

    match: Match
    blocks: list[int] = field(default_factory=list)
    """Indices of the page blocks the match spans, for highlighting."""


@dataclass(frozen=True)
class ScoreResult:
    """The outcome of scoring one document, including why."""

    score: float
    verdict: Verdict
    hits: list[Hit]
    """Every rule that fired, in rule order. This is the audit trail: it is what
    the debug GUI renders and what gets written to the JSONL log."""

    source: Source
    page_number: int

    @property
    def positive_hits(self) -> list[Hit]:
        return [h for h in self.hits if h.weight > 0]

    @property
    def negative_hits(self) -> list[Hit]:
        return [h for h in self.hits if h.weight < 0]

