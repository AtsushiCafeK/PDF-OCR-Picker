"""Text normalization, applied as a ladder of independently switchable steps.

Invoices defeat naive keyword matching in two different ways, and the two need
different remedies:

* ``請　求　書`` -- the characters are correct but padded apart for layout.
  This is deterministic and belongs here, in normalization.
* ``請求澤書`` -- OCR inserted a character that is not on the page at all.
  No amount of normalization fixes that; see :mod:`pdf_ocr.core.matcher`.

Every step records where each output character came from, so a match found in
the normalized text can still be pointed back at the pixels it came from. That
mapping is what lets the debug GUI highlight the matched region on the page
image, which is the whole reason the tool is worth building.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from pdf_ocr.core.types import TextBlock

BLOCK_SEPARATOR = "\n"
"""Inserted between blocks so that, with whitespace stripping off, text from two
different blocks cannot accidentally form a match."""

LAYOUT_NOISE = frozenset(
    "|｜/／\\＼:：;；,，、。.．·・*＊#＃~～^＾_＿"
    "-‐‑–—―−ｰ"
    "─━│┃┌┐└┘├┤┬┴┼╌╍═║╔╗╚╝"
    "「」『』（）()［］[]｛｝{}【】〔〕〈〉《》"
    "▪▫■□▲△▼▽●○◆◇★☆※"
)
"""Characters that carry layout rather than meaning: table rules, bullets,
brackets and separators. Deliberately excludes ``ー`` (U+30FC, the katakana
prolonged sound mark), which looks like a dash but is part of words such as
``データ`` -- removing it would corrupt real Japanese text."""


@dataclass(frozen=True)
class NormalizeOptions:
    """Which rungs of the ladder to apply.

    Each flag is exposed as a toggle in the debug GUI so the effect of every
    step can be seen in isolation against a real document.
    """

    nfkc: bool = True
    """Unicode NFKC: full-width ``ＩＮＶＯＩＣＥ`` to ASCII, ``㈱`` to ``(株)``,
    half-width kana to full-width. Practically always wanted."""

    strip_whitespace: bool = True
    """Drop every space, tab and newline. This is what defeats ``請　求　書``
    and also joins characters that OCR reported as separate blocks."""

    strip_symbols: bool = False
    """Drop :data:`LAYOUT_NOISE`. Helps against ``請|求|書`` and table rules
    bleeding into the text, at the cost of merging unrelated fragments. Off by
    default because it is the most destructive step."""

    lowercase: bool = True
    """Fold case so one ``invoice`` rule covers ``INVOICE`` and ``Invoice``."""


@dataclass(frozen=True)
class NormalizedText:
    """Normalized text plus the trail back to where it came from."""

    text: str
    """The normalized text, which is what patterns are matched against."""

    origin: tuple[int, ...]
    """``origin[i]`` is the index into :attr:`raw` that ``text[i]`` came from.
    Always the same length as :attr:`text`."""

    raw: str
    """Block texts joined by :data:`BLOCK_SEPARATOR`, before normalization."""

    block_of: tuple[int, ...]
    """``block_of[j]`` is the index of the block that ``raw[j]`` belongs to, or
    ``-1`` for the separators between blocks."""

    def raw_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a span of normalized text back to a span of :attr:`raw`."""
        if start >= end:
            return (0, 0)
        origins = self.origin[start:end]
        return (min(origins), max(origins) + 1)

    def raw_slice(self, start: int, end: int) -> str:
        """The original, un-normalized text behind a normalized span.

        This is what makes a hit legible to a human: the rule says ``請求書``
        but this returns the ``請 求 澤 書`` that was actually on the page.
        """
        lo, hi = self.raw_span(start, end)
        return self.raw[lo:hi]

    def blocks_for(self, start: int, end: int) -> list[int]:
        """Indices of the blocks a normalized span touches, for highlighting."""
        lo, hi = self.raw_span(start, end)
        seen: list[int] = []
        for block_index in self.block_of[lo:hi]:
            if block_index >= 0 and block_index not in seen:
                seen.append(block_index)
        return seen


HALFWIDTH_SOUND_MARKS = frozenset("ﾞﾟ")
"""Halfwidth voiced (``ﾞ``) and semi-voiced (``ﾟ``) sound marks.

Unlike their full-width counterparts these have a combining class of zero, so
:func:`unicodedata.combining` does not identify them as marks even though they
behave as ones: ``ﾌ`` followed by ``ﾟ`` is the single character ``プ``.
"""


def _apply_nfkc(text: str, origin: list[int]) -> tuple[str, list[int]]:
    """NFKC-normalize while preserving the origin mapping.

    Normalizing character by character would be simpler but would fail to
    recombine a base character with the mark that follows it -- ``か`` + U+3099
    would survive as two characters instead of becoming ``が``, and half-width
    ``ﾌﾟ`` would become ``フ`` plus a stray mark instead of ``プ``. So each base
    character is normalized together with its trailing marks, and the whole
    cluster is attributed to the base character's position.
    """
    out: list[str] = []
    out_origin: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        j = i + 1
        while j < n and (
            unicodedata.combining(text[j]) or text[j] in HALFWIDTH_SOUND_MARKS
        ):
            j += 1
        for char in unicodedata.normalize("NFKC", text[i:j]):
            out.append(char)
            out_origin.append(origin[i])
        i = j
    return "".join(out), out_origin


def _drop(
    text: str, origin: list[int], predicate
) -> tuple[str, list[int]]:
    """Remove characters matching ``predicate``, keeping the origin mapping."""
    out: list[str] = []
    out_origin: list[int] = []
    for char, source_index in zip(text, origin, strict=True):
        if not predicate(char):
            out.append(char)
            out_origin.append(source_index)
    return "".join(out), out_origin


def _apply_lowercase(text: str, origin: list[int]) -> tuple[str, list[int]]:
    """Lowercase, allowing for the rare character that lowercases to several."""
    out: list[str] = []
    out_origin: list[int] = []
    for char, source_index in zip(text, origin, strict=True):
        for lowered in char.lower():
            out.append(lowered)
            out_origin.append(source_index)
    return "".join(out), out_origin


def normalize_text(
    raw: str,
    options: NormalizeOptions | None = None,
    block_of: tuple[int, ...] | None = None,
) -> NormalizedText:
    """Normalize a raw string. Used directly by tests, which need no PDF."""
    options = options or NormalizeOptions()
    text = raw
    origin = list(range(len(raw)))

    if options.nfkc:
        text, origin = _apply_nfkc(text, origin)
    if options.strip_whitespace:
        text, origin = _drop(text, origin, str.isspace)
    if options.strip_symbols:
        text, origin = _drop(text, origin, LAYOUT_NOISE.__contains__)
    if options.lowercase:
        text, origin = _apply_lowercase(text, origin)

    return NormalizedText(
        text=text,
        origin=tuple(origin),
        raw=raw,
        block_of=block_of if block_of is not None else (-1,) * len(raw),
    )


def normalize_blocks(
    blocks: list[TextBlock], options: NormalizeOptions | None = None
) -> NormalizedText:
    """Join extracted blocks into one document string and normalize it.

    Blocks are joined rather than matched individually because OCR routinely
    splits a title into separate detections -- ``請``, ``求``, ``書`` can arrive
    as three blocks. Joining them and then stripping whitespace reunites the
    title. The cost is that the end of one line becomes adjacent to the start of
    the next, which can create text that is not really on the page; the window
    limit in subsequence matching is what keeps that from producing false hits.
    """
    parts: list[str] = []
    block_of: list[int] = []
    for index, block in enumerate(blocks):
        if index > 0:
            parts.append(BLOCK_SEPARATOR)
            block_of.extend([-1] * len(BLOCK_SEPARATOR))
        parts.append(block.text)
        block_of.extend([index] * len(block.text))
    return normalize_text("".join(parts), options, tuple(block_of))
