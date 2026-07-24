"""Strategies for finding a keyword in page text that OCR may have mangled.

Normalization handles characters that are correct but spaced out. This module
handles characters that are simply wrong: ``請求澤書`` for ``請求書``, where the
recogniser invented a character, or ``請氷書``, where it substituted one.

The strategies form a ladder from strict to permissive. Which rung to use is not
a fixed choice -- it depends on where the text came from. Text lifted from a
PDF's own text layer is exact, so anything looser than :func:`find_exact` there
only invites false positives. Text from OCR needs the slack.
"""

from __future__ import annotations

import re

from rapidfuzz.distance import Levenshtein

from pdf_ocr.core.types import Match, MatchKind

MIN_FUZZY_PATTERN_LENGTH = 4
"""Shortest pattern that may be matched by edit distance.

At distance 1, a two-character pattern matches any string sharing one character
with it -- ``税込`` would fire on ``税抜``, inverting the meaning. Short, common
keywords like ``様`` or ``税`` are exactly the ones that appear on every kind of
document, so letting them match loosely is the fastest way to a false positive.
"""

DEFAULT_SUBSEQUENCE_SLACK = 3
"""How many stray characters a subsequence match may absorb by default.

The span limit is what separates a real hit from a coincidence: without it,
``請``, ``求`` and ``書`` scattered across an entire page would count as the
title ``請求書``.
"""


def find_exact(text: str, pattern: str) -> list[Match]:
    """Every non-overlapping literal occurrence of ``pattern``."""
    if not pattern:
        return []
    matches: list[Match] = []
    start = text.find(pattern)
    while start != -1:
        end = start + len(pattern)
        matches.append(
            Match(
                start=start,
                end=end,
                matched_text=text[start:end],
                kind=MatchKind.EXACT,
            )
        )
        start = text.find(pattern, end)
    return matches


def find_subsequence(
    text: str, pattern: str, window: int | None = None
) -> list[Match]:
    """Occurrences where ``pattern``'s characters appear in order and close together.

    This is the rung that handles inserted characters. ``請求書`` is found inside
    ``請求澤書`` because the three characters still appear in sequence within a
    short span. It cannot recover a substituted or dropped character -- for that
    see :func:`find_fuzzy`.

    ``window`` caps the total span a match may occupy. It defaults to the
    pattern length plus :data:`DEFAULT_SUBSEQUENCE_SLACK`.
    """
    if not pattern:
        return []
    limit = window if window is not None else len(pattern) + DEFAULT_SUBSEQUENCE_SLACK
    if limit < len(pattern):
        return []

    matches: list[Match] = []
    text_length = len(text)
    start = 0
    while start < text_length:
        if text[start] != pattern[0]:
            start += 1
            continue

        # Greedily consume the rest of the pattern without leaving the window.
        next_needed = 1
        position = start + 1
        horizon = min(text_length, start + limit)
        while position < horizon and next_needed < len(pattern):
            if text[position] == pattern[next_needed]:
                next_needed += 1
            position += 1

        if next_needed == len(pattern):
            end = position
            span = text[start:end]
            matches.append(
                Match(
                    start=start,
                    end=end,
                    matched_text=span,
                    # An unpadded span is an ordinary literal match; reporting it
                    # as such keeps the debug view honest about which documents
                    # actually needed the fuzzy machinery.
                    kind=(
                        MatchKind.EXACT
                        if len(span) == len(pattern)
                        else MatchKind.SUBSEQUENCE
                    ),
                )
            )
            start = end
        else:
            start += 1
    return matches


def find_fuzzy(text: str, pattern: str, max_distance: int = 1) -> list[Match]:
    """Occurrences within ``max_distance`` edits of ``pattern``.

    The most permissive rung: it absorbs substitutions and deletions as well as
    insertions, so ``請氷書`` still matches ``請求書``. That power is also why
    :data:`MIN_FUZZY_PATTERN_LENGTH` exists.
    """
    if not pattern:
        return []
    if len(pattern) < MIN_FUZZY_PATTERN_LENGTH:
        raise ValueError(
            f"fuzzy matching needs a pattern of at least "
            f"{MIN_FUZZY_PATTERN_LENGTH} characters; {pattern!r} is too short "
            f"and would match unrelated text"
        )
    if max_distance < 1:
        return find_exact(text, pattern)

    pattern_length = len(pattern)
    shortest = max(1, pattern_length - max_distance)
    longest = pattern_length + max_distance

    matches: list[Match] = []
    text_length = len(text)
    start = 0
    while start < text_length:
        best: tuple[int, int] | None = None  # (distance, end)
        for length in range(shortest, longest + 1):
            end = start + length
            if end > text_length:
                break
            distance = Levenshtein.distance(
                pattern, text[start:end], score_cutoff=max_distance
            )
            if distance <= max_distance and (best is None or distance < best[0]):
                best = (distance, end)
                if distance == 0:
                    break

        if best is None:
            start += 1
            continue

        distance, end = best
        matches.append(
            Match(
                start=start,
                end=end,
                matched_text=text[start:end],
                kind=MatchKind.EXACT if distance == 0 else MatchKind.FUZZY,
                distance=distance,
            )
        )
        start = end
    return matches


def find_regex(text: str, pattern: str) -> list[Match]:
    """Every non-overlapping regular-expression match.

    Used for structured strings that no keyword list can enumerate, such as the
    ``T`` + 13 digits registration number introduced by the Japanese qualified
    invoice system.
    """
    matches: list[Match] = []
    for found in re.finditer(pattern, text):
        matches.append(
            Match(
                start=found.start(),
                end=found.end(),
                matched_text=found.group(0),
                kind=MatchKind.REGEX,
            )
        )
    return matches


def find_matches(
    text: str,
    pattern: str,
    kind: MatchKind,
    *,
    window: int | None = None,
    max_distance: int = 1,
) -> list[Match]:
    """Dispatch to the strategy named by ``kind``."""
    if kind is MatchKind.EXACT:
        return find_exact(text, pattern)
    if kind is MatchKind.SUBSEQUENCE:
        return find_subsequence(text, pattern, window)
    if kind is MatchKind.FUZZY:
        return find_fuzzy(text, pattern, max_distance)
    if kind is MatchKind.REGEX:
        return find_regex(text, pattern)
    raise ValueError(f"unknown match kind: {kind!r}")
