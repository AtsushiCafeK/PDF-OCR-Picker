"""Tests for the match strategies, especially against OCR damage."""

from __future__ import annotations

import pytest

from pdf_ocr.core.matcher import (
    MIN_FUZZY_PATTERN_LENGTH,
    find_exact,
    find_fuzzy,
    find_matches,
    find_regex,
    find_subsequence,
)
from pdf_ocr.core.types import MatchKind


class TestExact:
    def test_finds_every_occurrence(self):
        matches = find_exact("請求書と請求書", "請求書")
        assert [m.start for m in matches] == [0, 4]

    def test_reports_nothing_when_absent(self):
        assert find_exact("御見積書", "請求書") == []

    def test_occurrences_do_not_overlap(self):
        assert len(find_exact("aaa", "aa")) == 1


class TestSubsequence:
    def test_survives_a_character_inserted_by_ocr(self):
        """The 請求澤書 case, which normalization cannot touch."""
        matches = find_subsequence("御請求澤書", "請求書")
        assert len(matches) == 1
        assert matches[0].matched_text == "請求澤書"
        assert matches[0].kind is MatchKind.SUBSEQUENCE

    def test_a_clean_match_is_reported_as_exact(self):
        """Otherwise the debug view cannot tell which documents needed slack."""
        matches = find_subsequence("御請求書", "請求書")
        assert matches[0].kind is MatchKind.EXACT
        assert matches[0].matched_text == "請求書"

    def test_characters_scattered_across_a_page_do_not_count(self):
        """Without a span limit, any page containing 請, 求 and 書 would match."""
        text = "請" + "あ" * 40 + "求" + "あ" * 40 + "書"
        assert find_subsequence(text, "請求書") == []

    def test_the_window_sets_how_much_noise_is_tolerated(self):
        text = "請XX求XX書"  # span of 9
        assert find_subsequence(text, "請求書", window=6) == []
        assert len(find_subsequence(text, "請求書", window=9)) == 1

    def test_order_matters(self):
        assert find_subsequence("書求請", "請求書") == []


class TestFuzzy:
    def test_survives_a_substituted_character(self):
        """Subsequence matching cannot do this; only edit distance can."""
        matches = find_fuzzy("請求全額", "請求金額", max_distance=1)
        assert len(matches) == 1
        assert matches[0].matched_text == "請求全額"
        assert matches[0].kind is MatchKind.FUZZY
        assert matches[0].distance == 1

    def test_survives_a_dropped_character(self):
        assert len(find_fuzzy("請求額", "請求金額", max_distance=1)) == 1

    def test_a_clean_match_is_reported_as_exact(self):
        matches = find_fuzzy("請求金額", "請求金額", max_distance=1)
        assert matches[0].kind is MatchKind.EXACT
        assert matches[0].distance == 0

    def test_respects_the_distance_limit(self):
        assert find_fuzzy("請求全店", "請求金額", max_distance=1) == []

    def test_refuses_patterns_too_short_to_match_safely(self):
        """At distance 1 a short pattern matches almost anything.

        税込 and 税抜 are one edit apart and mean opposite things, so allowing
        this would not be a loose match but a wrong one.
        """
        with pytest.raises(ValueError, match="at least"):
            find_fuzzy("税抜", "税込", max_distance=1)

    def test_the_minimum_is_what_the_rule_loader_enforces(self):
        assert MIN_FUZZY_PATTERN_LENGTH >= 3


class TestRegex:
    def test_finds_a_registration_number(self):
        matches = find_regex("登録番号t1234567890123", r"(?i)(?<![0-9a-z])t\d{13}(?!\d)")
        assert len(matches) == 1
        assert matches[0].matched_text == "t1234567890123"

    def test_rejects_a_number_of_the_wrong_length(self):
        pattern = r"(?i)(?<![0-9a-z])t\d{13}(?!\d)"
        assert find_regex("t12345678901234", pattern) == []
        assert find_regex("t123456789012", pattern) == []


class TestDispatch:
    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (MatchKind.EXACT, 0),
            (MatchKind.SUBSEQUENCE, 1),
            (MatchKind.FUZZY, 1),
        ],
    )
    def test_strategies_differ_in_what_they_tolerate(self, kind, expected):
        """The same damaged text, matched at three levels of strictness."""
        found = find_matches("請求澤金額", "請求金額", kind, window=6, max_distance=1)
        assert len(found) == expected

    def test_rejects_an_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown match kind"):
            find_matches("請求書", "請求書", "nonsense")  # type: ignore[arg-type]
