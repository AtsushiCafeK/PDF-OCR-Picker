"""Weighted keyword scoring, and the three-way verdict it produces.

Invoice layouts vary without limit, which makes extracting *fields* from them
hard. Deciding whether a document *is* an invoice is a much smaller problem:
regardless of layout, the vocabulary is stable. ``請求書``, ``請求金額``,
``振込先``, ``お支払期限`` and a registration number appear on essentially every
one, and ``見積`` or ``納品書`` appear on essentially none. So this module never
looks at table structure -- only at which words are present, plus a weak signal
for whether a title sits near the top of the page.

The output is deliberately three-way. Misfiling a non-invoice into the invoice
folder is invisible to the person who later opens that folder; a document parked
in a review folder is not. The costs are asymmetric, so uncertain documents are
escalated rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pdf_ocr.core.matcher import MIN_FUZZY_PATTERN_LENGTH, find_matches
from pdf_ocr.core.normalize import (
    NormalizedText,
    NormalizeOptions,
    normalize_blocks,
    normalize_text,
)
from pdf_ocr.core.types import Hit, MatchKind, Page, Scope, ScoreResult, Source, Verdict


class RuleError(ValueError):
    """Raised when a rules file cannot be understood.

    Carries every problem found rather than only the first, because these files
    are edited by hand in the debug GUI and fixing one error at a time is
    needlessly slow.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid rules:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class Rule:
    """One keyword and what its presence is worth."""

    id: str
    pattern: str
    weight: float
    """Positive for evidence of an invoice, negative for evidence against."""

    kind: MatchKind = MatchKind.SUBSEQUENCE
    scope: Scope = Scope.WHOLE
    window: int | None = None
    max_distance: int = 1
    note: str = ""


@dataclass(frozen=True)
class Thresholds:
    """Score boundaries between the three verdicts."""

    high: float = 70.0
    low: float = 40.0


@dataclass(frozen=True)
class RuleSet:
    """A complete, self-contained scoring configuration.

    Kept in a YAML file beside the executable rather than compiled in, so a
    keyword can be added for a supplier whose invoices are being missed without
    rebuilding and redistributing anything.
    """

    rules: list[Rule]
    thresholds: Thresholds = field(default_factory=Thresholds)
    normalize: NormalizeOptions = field(default_factory=NormalizeOptions)
    fuzzy_on_text_layer: bool = False
    """Whether to allow loose matching against a PDF's own text layer.

    Off by default. Text-layer characters are exact, so loosening the match
    there cannot rescue a missed keyword -- it can only invent one.
    """

    @classmethod
    def from_dict(cls, data: dict) -> RuleSet:
        problems: list[str] = []
        rules: list[Rule] = []
        seen_ids: set[str] = set()

        raw_rules = data.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise RuleError(["'rules' must be a non-empty list"])

        for position, raw in enumerate(raw_rules):
            if not isinstance(raw, dict):
                problems.append(f"rule #{position}: expected a mapping")
                continue

            rule_id = str(raw.get("id") or f"rule_{position}")
            if rule_id in seen_ids:
                problems.append(f"{rule_id}: duplicate rule id")
            seen_ids.add(rule_id)

            pattern = raw.get("pattern") or raw.get("regex")
            if not pattern:
                problems.append(f"{rule_id}: needs a 'pattern' or a 'regex'")
                continue

            kind_name = raw.get("match") or ("regex" if raw.get("regex") else "subsequence")
            try:
                kind = MatchKind(kind_name)
            except ValueError:
                problems.append(
                    f"{rule_id}: unknown match kind {kind_name!r}; expected one of "
                    + ", ".join(k.value for k in MatchKind)
                )
                continue

            try:
                scope = Scope(raw.get("scope", "whole"))
            except ValueError:
                problems.append(f"{rule_id}: unknown scope {raw.get('scope')!r}")
                continue

            if "weight" not in raw:
                problems.append(f"{rule_id}: needs a 'weight'")
                continue

            if kind is MatchKind.FUZZY and len(str(pattern)) < MIN_FUZZY_PATTERN_LENGTH:
                problems.append(
                    f"{rule_id}: fuzzy matching needs at least "
                    f"{MIN_FUZZY_PATTERN_LENGTH} characters, but {pattern!r} is shorter; "
                    f"use 'subsequence' instead"
                )
                continue

            rules.append(
                Rule(
                    id=rule_id,
                    pattern=str(pattern),
                    weight=float(raw["weight"]),
                    kind=kind,
                    scope=scope,
                    window=raw.get("window"),
                    max_distance=int(raw.get("max_distance", 1)),
                    note=str(raw.get("note", "")),
                )
            )

        if problems:
            raise RuleError(problems)

        raw_thresholds = data.get("thresholds") or {}
        thresholds = Thresholds(
            high=float(raw_thresholds.get("high", 70.0)),
            low=float(raw_thresholds.get("low", 40.0)),
        )
        if thresholds.low > thresholds.high:
            raise RuleError(
                [f"thresholds: low ({thresholds.low}) is above high ({thresholds.high})"]
            )

        raw_normalize = data.get("normalize") or {}
        defaults = NormalizeOptions()
        normalize_options = NormalizeOptions(
            nfkc=bool(raw_normalize.get("nfkc", defaults.nfkc)),
            strip_whitespace=bool(
                raw_normalize.get("strip_whitespace", defaults.strip_whitespace)
            ),
            strip_symbols=bool(raw_normalize.get("strip_symbols", defaults.strip_symbols)),
            lowercase=bool(raw_normalize.get("lowercase", defaults.lowercase)),
        )

        return cls(
            rules=rules,
            thresholds=thresholds,
            normalize=normalize_options,
            fuzzy_on_text_layer=bool(data.get("fuzzy_on_text_layer", False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> RuleSet:
        """Read a rules file from disk."""
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise RuleError([f"{path}: expected a YAML mapping at the top level"])
        return cls.from_dict(data)


def verdict_for(score: float, thresholds: Thresholds) -> Verdict:
    """Turn a score into a verdict."""
    if score >= thresholds.high:
        return Verdict.INVOICE
    if score >= thresholds.low:
        return Verdict.NEEDS_REVIEW
    return Verdict.OTHER


def _effective_kind(rule: Rule, source: Source, allow_fuzzy_on_text_layer: bool) -> MatchKind:
    """Tighten a rule's match strategy when the text is known to be exact."""
    if rule.kind is MatchKind.REGEX or allow_fuzzy_on_text_layer:
        return rule.kind
    if source is Source.TEXT_LAYER and rule.kind in (
        MatchKind.SUBSEQUENCE,
        MatchKind.FUZZY,
    ):
        return MatchKind.EXACT
    return rule.kind


def _in_scope(rule: Rule, block_indices: list[int], page: Page) -> bool:
    """Whether a match's position satisfies the rule's scope."""
    if rule.scope is Scope.WHOLE:
        return True
    if not block_indices:
        # No positional information survived, so a position-restricted rule
        # cannot be shown to apply. Declining to fire is the safe direction.
        return False
    cutoff = page.top_quarter_cutoff()
    return any(page.blocks[index].bbox[1] < cutoff for index in block_indices)


def score_page(
    page: Page, ruleset: RuleSet, normalized: NormalizedText | None = None
) -> ScoreResult:
    """Score one page against a rule set.

    ``normalized`` may be passed in when it has already been computed -- the
    debug GUI reuses it so that moving a threshold slider re-scores instantly
    instead of re-running OCR.
    """
    if normalized is None:
        normalized = normalize_blocks(page.blocks, ruleset.normalize)

    hits: list[Hit] = []
    total = 0.0

    for rule in ruleset.rules:
        kind = _effective_kind(rule, page.source, ruleset.fuzzy_on_text_layer)

        # Patterns must go through the same normalization as the text, or an
        # 'Invoice' rule would never match text that has been lowercased.
        # Regexes are exempt: normalizing one would corrupt its syntax, so they
        # are written to be case-insensitive with an inline (?i) flag instead.
        pattern = (
            rule.pattern
            if kind is MatchKind.REGEX
            else normalize_text(rule.pattern, ruleset.normalize).text
        )
        if not pattern:
            continue

        matches = find_matches(
            normalized.text,
            pattern,
            kind,
            window=rule.window,
            max_distance=rule.max_distance,
        )

        for match in matches:
            blocks = normalized.blocks_for(match.start, match.end)
            if not _in_scope(rule, blocks, page):
                continue
            # A keyword counts once. Repeating "請求書" in a footer says no more
            # about the document than printing it once does.
            hits.append(
                Hit(
                    rule_id=rule.id,
                    pattern=rule.pattern,
                    weight=rule.weight,
                    match=match,
                    blocks=blocks,
                )
            )
            total += rule.weight
            break

    return ScoreResult(
        score=total,
        verdict=verdict_for(total, ruleset.thresholds),
        hits=hits,
        source=page.source,
        page_number=page.number,
    )
