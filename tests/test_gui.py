"""Tests for the debug GUI's logic, run against an offscreen Qt platform.

Only the parts that can be got wrong silently are covered: the rules table has
to round-trip through the same loader the command-line tool uses, or the GUI
would be tuning something subtly different from what ships. Painting and mouse
handling are left to be judged by looking at them.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTableWidgetItem  # noqa: E402

from pdf_ocr import DEFAULT_RULES_PATH  # noqa: E402
from pdf_ocr.gui import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application):
    window = MainWindow(DEFAULT_RULES_PATH)
    yield window
    window.close()


class TestConstruction:
    def test_the_window_builds(self, window):
        assert window.windowTitle()

    def test_every_shipped_rule_reaches_the_table(self, window):
        assert window.rules_table.rowCount() == len(window.ruleset.rules)

    def test_the_controls_start_from_the_rules_file(self, window):
        assert window.high_slider.value() == int(window.ruleset.thresholds.high)
        assert window.low_slider.value() == int(window.ruleset.thresholds.low)
        assert (
            window.normalize_checks["strip_whitespace"].isChecked()
            is window.ruleset.normalize.strip_whitespace
        )


class TestRulesRoundTrip:
    def test_the_table_reloads_into_an_equivalent_rule_set(self, window):
        """The table is the tuning surface; if it cannot reproduce the file, the
        GUI is tuning something other than what ships."""
        from pdf_ocr.core.score import RuleSet

        rebuilt = RuleSet.from_dict(window._rules_from_table())
        assert len(rebuilt.rules) == len(window.ruleset.rules)
        for original, copy in zip(window.ruleset.rules, rebuilt.rules, strict=True):
            assert copy.id == original.id
            assert copy.pattern == original.pattern
            assert copy.weight == original.weight
            assert copy.kind is original.kind
            assert copy.scope is original.scope
            assert copy.window == original.window

    def test_a_regex_rule_survives_the_round_trip(self, window):
        """Regexes must go back under their own key, or the loader would put the
        pattern through normalization and corrupt its syntax."""
        entries = window._rules_from_table()["rules"]
        registration = next(e for e in entries if e["id"] == "touroku_bangou")
        assert "regex" in registration
        assert "pattern" not in registration


class TestEditing:
    def test_changing_a_weight_takes_effect(self, window):
        row = next(
            index
            for index in range(window.rules_table.rowCount())
            if window.rules_table.item(index, 0).text() == "title_seikyusho"
        )
        window.rules_table.setItem(row, 2, QTableWidgetItem("99"))
        applied = next(r for r in window.ruleset.rules if r.id == "title_seikyusho")
        assert applied.weight == 99

    def test_an_invalid_edit_leaves_the_previous_rules_in_force(self, window):
        """Half-finished edits are normal while typing; scoring with them is not."""
        before = list(window.ruleset.rules)
        window.rules_table.setItem(0, 2, QTableWidgetItem("not-a-number"))
        assert window.ruleset.rules == before

    def test_an_invalid_edit_is_reported(self, window):
        window.rules_table.setItem(0, 3, QTableWidgetItem("nonsense-match-kind"))
        assert "nonsense-match-kind" in window.statusBar().currentMessage()


class TestControlsFeedScoring:
    def test_threshold_sliders_reach_the_rule_set(self, window):
        window.high_slider.setValue(123)
        window.low_slider.setValue(45)
        ruleset = window._current_ruleset()
        assert ruleset.thresholds.high == 123
        assert ruleset.thresholds.low == 45

    def test_normalization_toggles_reach_the_rule_set(self, window):
        window.normalize_checks["strip_symbols"].setChecked(True)
        window.normalize_checks["lowercase"].setChecked(False)
        options = window._current_ruleset().normalize
        assert options.strip_symbols is True
        assert options.lowercase is False

    def test_rescoring_without_a_document_is_harmless(self, window):
        """Every control fires this on change, including before anything is open."""
        window._rescore()
        assert window.result is None
