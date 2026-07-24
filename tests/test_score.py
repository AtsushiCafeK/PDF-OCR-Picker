"""Tests for rule loading and for the scoring decisions the rules encode."""

from __future__ import annotations

import pytest

from pdf_ocr.core.score import RuleError, RuleSet, Thresholds, score_page, verdict_for
from pdf_ocr.core.types import Source, Verdict
from tests.conftest import make_page

MINIMAL_RULES = {
    "thresholds": {"high": 70, "low": 40},
    "rules": [{"id": "title", "pattern": "請求書", "weight": 50, "match": "subsequence"}],
}


class TestClassification:
    def test_an_invoice_is_recognised(self, rules, invoice_lines):
        result = score_page(make_page(invoice_lines), rules)
        assert result.verdict is Verdict.INVOICE

    def test_a_quotation_is_not(self, rules, quotation_lines):
        """Quotations share most of an invoice's vocabulary, so the exclusion
        keywords are doing the work here, not the absence of positive ones."""
        result = score_page(make_page(quotation_lines), rules)
        assert result.verdict is Verdict.OTHER

    def test_an_empty_page_is_not_an_invoice(self, rules):
        result = score_page(make_page([]), rules)
        assert result.score == 0
        assert result.verdict is Verdict.OTHER

    def test_layout_does_not_matter(self, rules, invoice_lines):
        """The premise of the whole design: the same words in a different
        arrangement are the same document."""
        title, *rest = invoice_lines
        reordered = [title] + [
            (text, 300.0 + index * 40.0)
            for index, (text, _) in enumerate(reversed(rest))
        ]
        result = score_page(make_page(reordered), rules)
        assert result.verdict is Verdict.INVOICE


class TestOcrDamage:
    def test_a_title_mangled_by_ocr_still_scores(self, rules):
        """請求澤書 is the case that motivated subsequence matching."""
        result = score_page(make_page([("請求澤書", 50.0)]), rules)
        assert {hit.rule_id for hit in result.hits} == {
            "title_seikyusho",
            "title_seikyusho_at_top",
        }

    def test_a_spaced_title_still_scores(self, rules):
        """請　求　書 is handled a step earlier, by normalization."""
        result = score_page(make_page([("請　求　書", 50.0)]), rules)
        assert any(hit.rule_id == "title_seikyusho" for hit in result.hits)

    def test_the_hit_records_what_was_actually_printed(self, rules):
        """A reviewer tuning thresholds needs to see 請求澤書, not just 請求書."""
        result = score_page(make_page([("請求澤書", 50.0)]), rules)
        assert result.hits[0].match.matched_text == "請求澤書"
        assert result.hits[0].pattern == "請求書"


class TestTextLayerIsTrusted:
    def test_loose_matching_is_disabled_for_text_layer_pages(self, rules):
        """Text-layer characters are exact. Loosening the match there cannot
        recover a keyword that is missing -- it can only invent one."""
        page = make_page([("請求澤書", 50.0)], source=Source.TEXT_LAYER)
        assert score_page(page, rules).hits == []

    def test_an_undamaged_title_still_matches_in_a_text_layer(self, rules):
        page = make_page([("請求書", 50.0)], source=Source.TEXT_LAYER)
        assert score_page(page, rules).score > 0

    def test_the_tightening_can_be_switched_off(self, rules):
        page = make_page([("請求澤書", 50.0)], source=Source.TEXT_LAYER)
        permissive = RuleSet(
            rules=rules.rules,
            thresholds=rules.thresholds,
            normalize=rules.normalize,
            fuzzy_on_text_layer=True,
        )
        assert score_page(page, permissive).hits != []


class TestScope:
    def test_a_title_at_the_top_earns_the_position_bonus(self, rules):
        result = score_page(make_page([("請求書", 50.0)]), rules)
        assert {hit.rule_id for hit in result.hits} == {
            "title_seikyusho",
            "title_seikyusho_at_top",
        }

    def test_the_same_word_lower_down_does_not(self, rules):
        result = score_page(make_page([("請求書", 600.0)]), rules)
        assert {hit.rule_id for hit in result.hits} == {"title_seikyusho"}


class TestPatternNormalization:
    def test_an_ascii_rule_matches_full_width_text(self, rules):
        """Patterns go through the same normalization as the page text, so an
        'invoice' rule has to survive ＩＮＶＯＩＣＥ on the page."""
        result = score_page(make_page([("ＩＮＶＯＩＣＥ", 50.0)]), rules)
        assert any(hit.rule_id == "title_invoice" for hit in result.hits)

    def test_a_regex_rule_matches_a_registration_number(self, rules):
        result = score_page(make_page([("登録番号 T1234567890123", 300.0)]), rules)
        assert any(hit.rule_id == "touroku_bangou" for hit in result.hits)


class TestKeywordCountedOnce:
    def test_repeating_a_keyword_adds_nothing(self, rules):
        """A footer that repeats the title says no more than printing it once."""
        once = score_page(make_page([("請求書", 50.0)]), rules)
        twice = score_page(
            make_page([("請求書", 50.0), ("請求書", 600.0)]), rules
        )
        assert once.score == twice.score


class TestEmailPrintouts:
    """A printed email forwarding an invoice carries every keyword the
    classifier looks for, and is not an invoice. Filing it as one puts a
    duplicate in the accounting folder that reconciles against nothing."""

    def _email(self) -> list[tuple[str, float]]:
        return [
            ("差出人 田中太郎", 50.0),
            ("送信日時 2026年7月24日 10:32", 70.0),
            ("件名 【ご請求】請求書送付のご案内 (Invoice)", 110.0),
            ("添付ファイル 請求書_202607.pdf", 130.0),
            ("いつもお世話になっております。", 200.0),
            ("ご請求金額は 110,000円 でございます。", 260.0),
            ("お支払期限は 2026年8月31日 です。", 290.0),
        ]

    def test_it_is_not_filed_as_an_invoice(self, rules):
        assert score_page(make_page(self._email()), rules).verdict is not Verdict.INVOICE

    def test_the_mail_header_is_what_rejects_it(self, rules):
        """Not the absence of invoice vocabulary -- all of it is present."""
        result = score_page(make_page(self._email()), rules)
        fired = {hit.rule_id for hit in result.hits}
        assert "title_seikyusho" in fired
        assert "amount_seikyu_kingaku" in fired
        assert {"exclude_mail_from", "exclude_mail_sent"} <= fired

    def test_an_english_mail_header_rejects_it_too(self, rules):
        page = make_page(
            [
                ("From: John Smith", 50.0),
                ("Sent: Friday, 24 July 2026", 70.0),
                ("Subject: Invoice INV-2026-0118", 110.0),
                ("Attachments: INV-2026-0118.pdf", 130.0),
                ("Payment is due by 31 August 2026.", 260.0),
            ]
        )
        assert score_page(page, rules).verdict is Verdict.OTHER

    def test_a_weak_mail_signal_escalates_rather_than_files(self, rules):
        """If OCR loses part of the header, the score should drift towards
        review, never towards filing it as an invoice."""
        partial = [line for line in self._email() if "差出人" not in line[0]]
        assert score_page(make_page(partial), rules).verdict is not Verdict.INVOICE

    def test_a_real_invoice_is_untouched_by_the_mail_rules(self, rules, invoice_lines):
        result = score_page(make_page(invoice_lines), rules)
        assert not any(hit.rule_id.startswith("exclude_mail") for hit in result.hits)


class TestVerdictBoundaries:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (70.0, Verdict.INVOICE),
            (69.9, Verdict.NEEDS_REVIEW),
            (40.0, Verdict.NEEDS_REVIEW),
            (39.9, Verdict.OTHER),
        ],
    )
    def test_thresholds_are_inclusive_at_the_lower_edge(self, score, expected):
        assert verdict_for(score, Thresholds(high=70, low=40)) is expected

    def test_uncertain_documents_are_escalated_rather_than_guessed(self, rules):
        """A document that is genuinely both -- 納品書兼請求書 is a real form --
        must land in review rather than being silently discarded."""
        page = make_page(
            [
                ("納品書兼請求書", 50.0),
                ("請求金額 ¥110,000", 300.0),
                ("振込先 みずほ銀行", 450.0),
            ]
        )
        assert score_page(page, rules).verdict is Verdict.NEEDS_REVIEW


class TestRuleLoading:
    def test_the_shipped_rules_load(self, rules):
        assert rules.rules
        assert rules.thresholds.low <= rules.thresholds.high

    def test_a_rule_without_a_weight_is_rejected(self):
        with pytest.raises(RuleError, match="weight"):
            RuleSet.from_dict({"rules": [{"id": "x", "pattern": "請求書"}]})

    def test_a_rule_without_a_pattern_is_rejected(self):
        with pytest.raises(RuleError, match="pattern"):
            RuleSet.from_dict({"rules": [{"id": "x", "weight": 10}]})

    def test_duplicate_rule_ids_are_rejected(self):
        with pytest.raises(RuleError, match="duplicate"):
            RuleSet.from_dict(
                {
                    "rules": [
                        {"id": "x", "pattern": "請求書", "weight": 10},
                        {"id": "x", "pattern": "納品書", "weight": -10},
                    ]
                }
            )

    def test_a_short_fuzzy_pattern_is_rejected_at_load_time(self):
        """Better to refuse the rule than to let it match unrelated documents."""
        with pytest.raises(RuleError, match="fuzzy"):
            RuleSet.from_dict(
                {"rules": [{"id": "x", "pattern": "税込", "weight": 5, "match": "fuzzy"}]}
            )

    def test_inverted_thresholds_are_rejected(self):
        with pytest.raises(RuleError, match="thresholds"):
            RuleSet.from_dict({**MINIMAL_RULES, "thresholds": {"high": 10, "low": 50}})

    def test_every_problem_is_reported_at_once(self):
        """These files are hand-edited; fixing one error per run is too slow."""
        with pytest.raises(RuleError) as caught:
            RuleSet.from_dict(
                {"rules": [{"id": "a", "pattern": "請求書"}, {"id": "b", "weight": 1}]}
            )
        assert len(caught.value.problems) == 2

    def test_normalization_settings_come_from_the_file(self):
        ruleset = RuleSet.from_dict({**MINIMAL_RULES, "normalize": {"strip_symbols": True}})
        assert ruleset.normalize.strip_symbols is True
        assert ruleset.normalize.nfkc is True
