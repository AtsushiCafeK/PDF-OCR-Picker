"""Tests for the human-readable debug log.

The value of this log is that a person can read a run and find where the rules
fall short, so the tests assert the things that make that possible: the score
breakdown, the keywords that should have fired but did not, and the recognised
text the rules actually saw.
"""

from __future__ import annotations

from pdf_ocr import DEFAULT_RULES_PATH
from pdf_ocr.core.debuglog import DebugLog
from pdf_ocr.core.normalize import normalize_blocks
from pdf_ocr.core.score import RuleSet, score_page
from pdf_ocr.core.types import Verdict
from tests.conftest import make_page

LABELS = {Verdict.INVOICE: "Match", Verdict.NEEDS_REVIEW: "Review", Verdict.OTHER: "Other"}


def scored(lines, rules):
    page = make_page(lines)
    normalized = normalize_blocks(page.blocks, rules.normalize)
    return page, normalized, score_page(page, rules, normalized)


def write_run(path, lines, rules, **document_kwargs) -> str:
    _, normalized, result = scored(lines, rules)
    with DebugLog(path, LABELS) as log:
        log.header(rules)
        log.document(path.parent / "doc.pdf", result, normalized, rules, **document_kwargs)
        log.footer({result.verdict: 1})
    return path.read_text(encoding="utf-8")


class TestScoringProcess:
    def test_the_hit_breakdown_is_written(self, tmp_path, rules):
        text = write_run(tmp_path / "log.txt", [("請求書", 50.0)], rules)
        assert "title_seikyusho" in text
        assert "+30" in text
        assert "total" in text

    def test_what_a_rule_matched_is_shown_not_just_that_it_did(self, tmp_path, rules):
        """The whole point on a scan: 請求書 matched the mangled 請求澤書."""
        text = write_run(tmp_path / "log.txt", [("請求澤書", 50.0)], rules)
        assert "請求澤書" in text
        assert "subsequence" in text

    def test_the_verdict_uses_the_display_labels(self, tmp_path, rules):
        """The outcome line reads 'Match', not the internal 'invoice' -- so a
        run tuned for another document type is not labelled as invoices. (Rule
        ids like title_invoice may still appear; those are identifiers.)"""
        text = write_run(tmp_path / "log.txt", [("請求書", 50.0)], rules)
        result_line = next(line for line in text.splitlines() if "result" in line)
        assert "Match" in result_line
        assert "invoice" not in result_line


class TestImprovementPoints:
    def test_positive_rules_that_did_not_fire_are_listed(self, tmp_path, rules):
        """This is where an improvement usually hides: a keyword that should
        have counted and did not."""
        text = write_run(tmp_path / "log.txt", [("請求書", 50.0)], rules)
        assert "did NOT fire" in text
        # A clean title page has no amount line, so this rule cannot have fired.
        assert "amount_seikyu_kingaku" in text

    def test_recognised_text_is_recorded(self, tmp_path, rules):
        """So a missed keyword can be traced to a misread rather than guessed at."""
        text = write_run(
            tmp_path / "log.txt", [("請求書", 50.0), ("振込先 みずほ銀行", 400.0)], rules
        )
        assert "recognised text" in text
        assert "みずほ銀行" in text


class TestStructure:
    def test_a_run_is_written_fresh_each_time(self, tmp_path, rules):
        """The log lives beside copies that are rebuilt each run; a log that
        outlived them would mislead."""
        path = tmp_path / "log.txt"
        write_run(path, [("請求書", 50.0)], rules)
        write_run(path, [("御見積書", 50.0)], rules)
        second = path.read_text(encoding="utf-8")
        # A fresh file: one header, and no trace of the first run's document.
        assert second.count("PDF classification debug log") == 1
        assert "請求書" not in second

    def test_the_footer_tallies_the_outcome(self, tmp_path, rules):
        text = write_run(tmp_path / "log.txt", [("請求書", 50.0)], rules)
        assert "finished" in text
        assert "outcome" in text

    def test_a_failure_is_recorded(self, tmp_path, rules):
        path = tmp_path / "log.txt"
        with DebugLog(path, LABELS) as log:
            log.header(rules)
            log.failure(tmp_path / "broken.pdf", "is password protected")
            log.footer({})
        text = path.read_text(encoding="utf-8")
        assert "broken.pdf" in text
        assert "password protected" in text

    def test_the_destination_is_noted_when_copied(self, tmp_path, rules):
        text = write_run(
            tmp_path / "log.txt",
            [("請求書", 50.0)],
            rules,
            destination=tmp_path / "請求書" / "a.pdf",
        )
        assert "copied to" in text
        assert "請求書" in text

    def test_the_shipped_rules_feed_the_header(self, tmp_path):
        """The header prints the thresholds and normalization in force; make
        sure the real file feeds through without error."""
        rules = RuleSet.load(DEFAULT_RULES_PATH)
        path = tmp_path / "log.txt"
        with DebugLog(path, LABELS) as log:
            log.header(rules, source=tmp_path / "in")
        text = path.read_text(encoding="utf-8")
        assert "thresholds" in text
        assert f"Match >= {rules.thresholds.high:g}" in text
        assert str(tmp_path / "in") in text
