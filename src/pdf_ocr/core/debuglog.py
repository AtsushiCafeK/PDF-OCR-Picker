"""A human-readable debug log of how a batch was classified.

Distinct from the JSONL audit log, which is written for a machine. This one is
written for a person who wants to know why the classifier decided what it did
and, more to the point, where the rules are falling short. So each document
records not only its score but the reasoning behind it: every rule that fired
and what it matched, the positive rules that did *not* fire (the likeliest place
an improvement hides), and the recognised text those rules actually saw.

The last of those is what makes a low-resolution scan debuggable at all. When a
title scores nothing, the recognised text shows whether the word was misread
(``請氷書`` -- a rule problem) or never captured (an OCR problem), which are
fixed in completely different places.

Kept free of any GUI dependency so it can be tested on synthetic results.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from pdf_ocr.core.normalize import NormalizedText
from pdf_ocr.core.score import RuleSet
from pdf_ocr.core.types import ScoreResult, Source, Verdict

WIDTH = 80
RULE = "=" * WIDTH
THIN = "-" * WIDTH


def _wrap(text: str, prefix: str = "  | ") -> str:
    """Indent and wrap a blob of text so a long OCR string stays readable."""
    if not text:
        return f"{prefix}(empty)"
    lines: list[str] = []
    for source_line in text.splitlines() or [text]:
        wrapped = textwrap.wrap(source_line, width=WIDTH - len(prefix)) or [""]
        lines.extend(prefix + piece for piece in wrapped)
    return "\n".join(lines)


class DebugLog:
    """Append-only, human-readable log for one classification run.

    Opened fresh per run -- truncating rather than appending -- because it lives
    beside the sorted copies, which are themselves rebuilt each run. A log that
    outlived the copies it describes would only mislead.
    """

    def __init__(
        self, path: Path, labels: Mapping[Verdict, str] | None = None
    ) -> None:
        self.path = path
        # Fall back to the raw verdict names if the caller has no display labels.
        self.labels = labels or {v: v.value for v in Verdict}
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        self._index = 0

    def _write(self, text: str) -> None:
        self._handle.write(text + "\n")
        self._handle.flush()

    def header(self, rules: RuleSet, source: Path | None = None) -> None:
        normalize = [
            name
            for name, on in (
                ("nfkc", rules.normalize.nfkc),
                ("strip_whitespace", rules.normalize.strip_whitespace),
                ("strip_symbols", rules.normalize.strip_symbols),
                ("lowercase", rules.normalize.lowercase),
            )
            if on
        ]
        self._write(RULE)
        self._write(" PDF classification debug log")
        self._write(f" started      : {datetime.now():%Y-%m-%d %H:%M:%S}")
        if source is not None:
            self._write(f" input        : {source}")
        # source_path only exists on some RuleSet versions; fall back cleanly.
        rules_path = getattr(rules, "source_path", None) or "(built-in)"
        self._write(f" rules        : {rules_path}")
        self._write(
            f" thresholds   : {self.labels[Verdict.INVOICE]} >= {rules.thresholds.high:g}"
            f"   {self.labels[Verdict.NEEDS_REVIEW]} >= {rules.thresholds.low:g}"
        )
        self._write(f" normalize    : {' '.join(normalize) or '(none)'}")
        self._write(RULE)

    def document(
        self,
        path: Path,
        result: ScoreResult,
        normalized: NormalizedText,
        rules: RuleSet,
        *,
        elapsed: float | None = None,
        destination: Path | None = None,
    ) -> None:
        self._index += 1
        source = "OCR" if result.source is Source.OCR else "text layer"
        timing = f"   elapsed {elapsed:.1f}s" if elapsed is not None else ""

        self._write("")
        self._write(THIN)
        self._write(f"[{self._index}] {path.name}")
        self._write(THIN)
        self._write(
            f"  result     : {self.labels[result.verdict]}   score {result.score:g}"
            f"      ({self.labels[Verdict.INVOICE]}>={rules.thresholds.high:g}"
            f"  {self.labels[Verdict.NEEDS_REVIEW]}>={rules.thresholds.low:g})"
        )
        blocks = len({index for index in normalized.block_of if index >= 0})
        self._write(
            f"  read from  : {source}   blocks {blocks}"
            f"   chars {len(normalized.text)}{timing}"
        )
        if destination is not None:
            self._write(f"  copied to  : {destination.parent.name}\\{destination.name}")

        # The scoring process: every rule that fired, and what it matched.
        self._write("")
        self._write("  scored:")
        if result.hits:
            for hit in result.hits:
                distance = f" d={hit.match.distance}" if hit.match.distance else ""
                self._write(
                    f"     {hit.weight:>+5.0f}  {hit.rule_id:<26}"
                    f' "{hit.pattern}" -> "{hit.match.matched_text}"'
                    f" ({hit.match.kind.value}{distance})"
                )
            self._write(f"      {'-' * 5}")
        self._write(f"      total {result.score:g}")

        # Where an improvement most likely hides: a keyword that should have
        # counted and did not. Pairs with the recognised text below -- the two
        # together say whether the word was misread or missing.
        fired = {hit.rule_id for hit in result.hits}
        missed = [
            rule.id for rule in rules.rules if rule.weight > 0 and rule.id not in fired
        ]
        if missed:
            self._write("")
            self._write("  positive rules that did NOT fire (possible misses):")
            self._write(_wrap(", ".join(missed), prefix="      "))

        self._write("")
        self._write("  recognised text (normalized -- what the rules matched against):")
        self._write(_wrap(normalized.text))
        self._write("")
        self._write("  raw text (as extracted, before normalization):")
        self._write(_wrap(normalized.raw))

    def failure(self, path: Path, error: str) -> None:
        self._index += 1
        self._write("")
        self._write(THIN)
        self._write(f"[{self._index}] {path.name}")
        self._write(THIN)
        self._write(f"  ERROR: {error}")

    def footer(self, counts: Mapping[Verdict, int], stopped: bool = False) -> None:
        tally = "   ".join(
            f"{self.labels[verdict]} {counts.get(verdict, 0)}" for verdict in Verdict
        )
        self._write("")
        self._write(RULE)
        note = "   (STOPPED early)" if stopped else ""
        self._write(f" finished     : {datetime.now():%Y-%m-%d %H:%M:%S}{note}")
        self._write(f" classified   : {self._index} document(s)")
        self._write(f" outcome      : {tally}")
        self._write(RULE)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> DebugLog:
        return self

    def __exit__(self, *exception) -> None:
        self.close()
