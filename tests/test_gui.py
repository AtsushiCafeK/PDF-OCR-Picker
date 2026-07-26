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

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QMessageBox,
    QTableWidgetItem,
)

from pdf_ocr import DEFAULT_RULES_PATH  # noqa: E402
from pdf_ocr.core.types import Verdict  # noqa: E402
from pdf_ocr.gui import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def _shared_window(application, tmp_path_factory):
    """One window for the whole module.

    Creating and destroying a QMainWindow per test churns Qt's offscreen
    platform hard enough that, on Windows, it eventually faults with an access
    violation during an unrelated test. Building it once and resetting its state
    between tests removes that churn -- and the reset below is what keeps the
    tests independent despite the sharing.

    The config path is a throwaway, so a test that remembers a folder writes to
    a temp file rather than the package's config.yaml.
    """
    config_path = tmp_path_factory.mktemp("gui") / "config.yaml"
    window = MainWindow(DEFAULT_RULES_PATH, config_path)
    yield window
    window.close()
    window.deleteLater()
    application.processEvents()


@pytest.fixture
def window(_shared_window):
    from pdf_ocr.core.config import Config

    w = _shared_window
    # Return the shared window to a clean state, so each test starts as if it
    # had a fresh one.
    w.config = Config(path=w.config.path)
    if w.config.path.exists():
        w.config.path.unlink()
    w._close_debug_log()
    w._batch_running = False
    w._pending_batch = []
    w._stop_requested = False
    w.stop_button.setEnabled(False)
    w.sort_check.setChecked(False)
    w.sort_directory = None
    w.force_ocr_check.setChecked(False)
    w.document = w.result = w.normalized = None
    w._current_path = None
    w._last_error = None
    w.file_list.clear()
    w._reload_rules()  # resets the ruleset, the table, the sliders and the toggles
    return w


class TestProgressWindow:
    """The window that makes a headless batch visible: n/total and the tally."""

    def _window(self, application, total):
        from pdf_ocr.progress import ProgressWindow

        return ProgressWindow(total)

    def test_it_shows_the_running_tally(self, application):
        w = self._window(application, 32)
        try:
            w.update_progress(5, 32, {"invoice": 3, "needs_review": 1, "other": 1})
            assert w.count_label.text() == "5/32"
            assert w.verdict_labels[Verdict.INVOICE].text() == "Match - 3"
            assert w.verdict_labels[Verdict.NEEDS_REVIEW].text() == "Review - 1"
            assert w.verdict_labels[Verdict.OTHER].text() == "Other - 1"
        finally:
            w._done = True
            w.close()
            w.deleteLater()

    def test_done_shows_completion(self, application):
        w = self._window(application, 2)
        try:
            w.mark_done({"processed": 2, "stopped": False, "verdicts": {"invoice": 2}})
            assert w.status_label.text() == "完了"
            assert w.count_label.text() == "2/2"
            assert w.verdict_labels[Verdict.INVOICE].text() == "Match - 2"
        finally:
            w.close()
            w.deleteLater()

    def test_a_stopped_run_says_so(self, application):
        w = self._window(application, 5)
        try:
            w.mark_done({"processed": 2, "stopped": True, "verdicts": {}})
            assert "停止" in w.status_label.text()
        finally:
            w.close()
            w.deleteLater()

    def test_closing_mid_run_requests_stop_and_stays_open(self, application):
        from PySide6.QtGui import QCloseEvent

        w = self._window(application, 5)
        try:
            requested = []
            w.stop_requested.connect(lambda: requested.append(True))
            event = QCloseEvent()
            w.closeEvent(event)  # not done yet
            assert requested == [True]
            assert not event.isAccepted()  # window stays until the worker returns
        finally:
            w._done = True
            w.close()
            w.deleteLater()


class TestNeutralLabels:
    """The window names the top outcome for what it is -- a match -- not
    'invoice', so someone tuning the rules for receipts or quotations is not
    told their receipts are invoices."""

    def test_the_result_reads_neutrally(self, window):
        from pdf_ocr.gui import VERDICT_LABELS

        assert VERDICT_LABELS[Verdict.INVOICE] == "Match"
        assert "invoice" not in {v.lower() for v in VERDICT_LABELS.values()}

    def test_every_verdict_has_a_label(self, window):
        from pdf_ocr.gui import VERDICT_LABELS

        assert set(VERDICT_LABELS) == set(Verdict)


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


class TestOpenAndReload:
    """You can load a saved rules file, and reload always returns to the
    default -- so tuning that is not saved is cheap to abandon."""

    def _write_rules(self, tmp_path, high, low):
        path = tmp_path / "tuned.yaml"
        path.write_text(
            f"thresholds: {{high: {high}, low: {low}}}\n"
            "rules:\n"
            "  - {id: t, pattern: 請求書, weight: 40, match: subsequence, window: 6}\n",
            encoding="utf-8",
        )
        return path

    def test_open_applies_the_loaded_thresholds(self, window, tmp_path, monkeypatch):
        loaded = self._write_rules(tmp_path, high=88, low=33)
        monkeypatch.setattr(
            "pdf_ocr.gui.QFileDialog.getOpenFileName", lambda *a, **k: (str(loaded), "")
        )
        window._open_rules_file()
        assert window.high_slider.value() == 88
        assert window.low_slider.value() == 33
        assert len(window.ruleset.rules) == 1

    def test_open_does_not_change_the_default(self, window, tmp_path, monkeypatch):
        """The point the user asked for: reload still goes back to the default,
        so opening a file is easy to undo."""
        default_high = window.high_slider.value()
        loaded = self._write_rules(tmp_path, high=88, low=33)
        monkeypatch.setattr(
            "pdf_ocr.gui.QFileDialog.getOpenFileName", lambda *a, **k: (str(loaded), "")
        )
        window._open_rules_file()
        assert window.high_slider.value() == 88

        window._reload_rules()
        assert window.high_slider.value() == default_high

    def test_cancelling_open_changes_nothing(self, window, monkeypatch):
        before = window.high_slider.value()
        monkeypatch.setattr(
            "pdf_ocr.gui.QFileDialog.getOpenFileName", lambda *a, **k: ("", "")
        )
        window._open_rules_file()
        assert window.high_slider.value() == before

    def test_a_broken_file_is_reported_not_applied(self, window, tmp_path, monkeypatch):
        before = list(window.ruleset.rules)
        broken = tmp_path / "broken.yaml"
        broken.write_text("rules: [{id: x, weight: 1}]", encoding="utf-8")  # no pattern
        monkeypatch.setattr(
            "pdf_ocr.gui.QFileDialog.getOpenFileName", lambda *a, **k: (str(broken), "")
        )
        warned = []
        monkeypatch.setattr(
            "pdf_ocr.gui.QMessageBox.warning", lambda *a, **k: warned.append(a)
        )
        window._open_rules_file()
        assert warned
        assert window.ruleset.rules == before


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


class TestSortedCopy:
    """Copying a classified batch into verdict folders, so a large run can be
    reviewed as piles of documents rather than a column of numbers."""

    def test_it_starts_switched_off_with_no_destination(self, window):
        assert window.sort_check.isChecked() is False
        assert window.sort_directory is None

    def test_classify_all_refuses_without_a_destination(self, window, monkeypatch):
        """Better to say so than to classify a whole folder and copy nothing."""
        monkeypatch.setattr(
            "pdf_ocr.gui.QMessageBox.information", lambda *a, **k: None
        )
        window.sort_check.setChecked(True)
        window._classify_all()
        assert window._pending_batch == []

    def test_a_destination_inside_the_source_is_refused(self, window, tmp_path, monkeypatch):
        """Copies landing in the input folder would be reclassified next run,
        breeding a new generation of duplicates each time."""
        source = tmp_path / "in"
        source.mkdir()
        (source / "a.pdf").write_bytes(b"pdf")
        window._populate([source / "a.pdf"])

        warned = []
        monkeypatch.setattr(
            "pdf_ocr.gui.QMessageBox.warning", lambda *a, **k: warned.append(a)
        )
        monkeypatch.setattr(
            "pdf_ocr.gui.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(source / "sorted"),
        )
        window._choose_sort_directory()
        assert warned
        assert window.sort_directory is None

    def test_a_destination_outside_the_source_is_accepted(self, window, tmp_path, monkeypatch):
        source = tmp_path / "in"
        source.mkdir()
        (source / "a.pdf").write_bytes(b"pdf")
        window._populate([source / "a.pdf"])

        monkeypatch.setattr(
            "pdf_ocr.gui.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(tmp_path / "out"),
        )
        window._choose_sort_directory()
        assert window.sort_directory == tmp_path / "out"
        assert window.sort_check.isChecked()

    def test_an_empty_destination_needs_no_confirmation(self, window, tmp_path):
        window.sort_directory = tmp_path / "out"
        assert window._prepare_sort_directory() is True

    def test_a_previous_run_is_cleared_once_confirmed(self, window, tmp_path, monkeypatch):
        from pdf_ocr.core.mover import DEFAULT_FOLDER_NAMES

        folder = tmp_path / "out" / DEFAULT_FOLDER_NAMES[Verdict.INVOICE]
        folder.mkdir(parents=True)
        (folder / "stale.pdf").write_bytes(b"pdf")

        window.sort_directory = tmp_path / "out"
        monkeypatch.setattr(
            "pdf_ocr.gui.QMessageBox.question",
            lambda *a, **k: QMessageBox.StandardButton.Yes,
        )
        assert window._prepare_sort_directory() is True
        assert not (folder / "stale.pdf").exists()

    def test_the_last_document_of_a_batch_is_not_skipped(self, window, tmp_path):
        """The completion that empties the queue is still a completion. Handling
        it only when work remains drops the final document -- which cost a
        missing copy before this was fixed."""
        from PySide6.QtGui import QPixmap

        from pdf_ocr.core.score import RuleSet, score_page
        from pdf_ocr.gui import LoadedDocument
        from tests.conftest import make_page

        source = tmp_path / "in"
        source.mkdir()
        only = source / "a.pdf"
        only.write_bytes(b"pdf")

        window._populate([only])
        window.sort_directory = tmp_path / "out"
        window.sort_check.setChecked(True)

        rules = RuleSet.load(DEFAULT_RULES_PATH)
        page = make_page([("請求書", 50.0), ("ご請求金額 110,000", 300.0)])
        window.document = LoadedDocument(
            path=only, page=page, image=QPixmap(), scale=1.0, elapsed=0.0
        )
        window.result = score_page(page, rules)

        # Stand where the worker leaves off on the final document: queue empty,
        # batch still nominally running.
        window._pending_batch = []
        window._batch_running = True
        window._on_thread_finished()

        copied = list((tmp_path / "out").rglob("*.pdf"))
        assert len(copied) == 1
        assert copied[0].name == "a.pdf"
        assert only.exists()

    def test_declining_the_confirmation_deletes_nothing(self, window, tmp_path, monkeypatch):
        from pdf_ocr.core.mover import DEFAULT_FOLDER_NAMES

        folder = tmp_path / "out" / DEFAULT_FOLDER_NAMES[Verdict.INVOICE]
        folder.mkdir(parents=True)
        (folder / "stale.pdf").write_bytes(b"pdf")

        window.sort_directory = tmp_path / "out"
        monkeypatch.setattr(
            "pdf_ocr.gui.QMessageBox.question",
            lambda *a, **k: QMessageBox.StandardButton.Cancel,
        )
        assert window._prepare_sort_directory() is False
        assert (folder / "stale.pdf").exists()


def _finish_one(window, path, lines):
    """Put the window where the worker leaves off after one document.

    Used with the same seam as test_the_last_document: set the state a completed
    load would leave, then call _on_thread_finished directly, so nothing has to
    spawn a real OCR thread against a fake PDF.
    """
    from PySide6.QtGui import QPixmap

    from pdf_ocr.core.normalize import normalize_blocks
    from pdf_ocr.core.score import RuleSet, score_page
    from pdf_ocr.gui import LoadedDocument
    from tests.conftest import make_page

    page = make_page(lines)
    window.document = LoadedDocument(
        path=path, page=page, image=QPixmap(), scale=1.0, elapsed=0.3
    )
    window.result = score_page(page, RuleSet.load(DEFAULT_RULES_PATH))
    window.normalized = normalize_blocks(page.blocks, window._current_ruleset().normalize)
    window._current_path = path


class TestDebugLog:
    """The run leaves a readable log.txt in the destination, so a large batch
    can be debugged after the window is closed."""

    def test_a_log_is_written_beside_the_copies(self, window, tmp_path):
        only = tmp_path / "in" / "a.pdf"
        only.parent.mkdir()
        only.write_bytes(b"pdf")
        window._populate([only])
        window.sort_directory = tmp_path / "out"
        window.sort_check.setChecked(True)
        assert window._prepare_sort_directory() is True

        _finish_one(window, only, [("請求書", 50.0), ("振込先 みずほ", 400.0)])
        window._pending_batch = []
        window._batch_running = True
        window._on_thread_finished()

        text = (tmp_path / "out" / "log.txt").read_text(encoding="utf-8")
        assert "a.pdf" in text
        assert "title_seikyusho" in text  # the scoring process
        assert "みずほ" in text  # the recognised text
        assert "outcome" in text  # the footer, so the log was closed

    def test_no_log_without_sorted_copy(self, window):
        """The log lives in the destination; with sorted copy off there is
        nowhere for it to go, so none is opened."""
        assert window._debug_log is None


class TestRemembersFolders:
    """The folders a person picks are remembered, so they are the defaults next
    time -- and the command line, sharing the config, inherits them."""

    def test_choosing_a_destination_is_remembered(self, window, tmp_path, monkeypatch):
        out = tmp_path / "sorted"
        monkeypatch.setattr(
            "pdf_ocr.gui.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(out),
        )
        window._choose_sort_directory()
        assert window.config.output_dir == out
        # Written to disk, so the command line and the next launch see it.
        from pdf_ocr.core.config import Config

        assert Config.load(window.config.path).output_dir == out

    def test_opening_a_folder_is_remembered(self, window, tmp_path, monkeypatch):
        inbox = tmp_path / "in"
        inbox.mkdir()
        (inbox / "a.pdf").write_bytes(b"pdf")
        monkeypatch.setattr(
            "pdf_ocr.gui.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(inbox),
        )
        window._open_folder()
        assert window.config.input_dir == inbox

    def test_a_configured_input_folder_is_listed_at_startup(self, tmp_path):
        """Listed, not classified: selecting a row would start OCR, which a
        person opening the window has not asked for."""
        from pdf_ocr.core.config import Config
        from pdf_ocr.gui import MainWindow

        inbox = tmp_path / "in"
        inbox.mkdir()
        (inbox / "a.pdf").write_bytes(b"pdf")
        (inbox / "b.pdf").write_bytes(b"pdf")
        config_path = tmp_path / "config.yaml"
        Config(path=config_path, input_dir=inbox).save()

        w = MainWindow(DEFAULT_RULES_PATH, config_path)
        try:
            assert w.file_list.count() == 2
            assert w.file_list.currentRow() == -1  # nothing selected -> no OCR
        finally:
            w.close()
            w.deleteLater()


class TestRunFromSelected:
    """Classify from the selected document to the end -- to resume after a Stop
    or re-check from a document after changing a rule."""

    def _populate(self, window, tmp_path, count):
        source = tmp_path / "in"
        source.mkdir()
        for i in range(count):
            (source / f"{i}.pdf").write_bytes(b"pdf")
        window._populate(sorted(source.glob("*.pdf")))
        return source

    def test_it_queues_from_the_selected_row_to_the_end(self, window, tmp_path):
        self._populate(window, tmp_path, 5)
        window.file_list.setCurrentRow(2)
        # Prevent the real load from spawning an OCR thread on a fake PDF.
        started = []
        window._load = lambda path: started.append(path)  # type: ignore[assignment]
        window._classify_from_selected()
        # Row 2 was popped and handed to _load; rows 3 and 4 remain queued.
        assert window._pending_batch == [3, 4]
        assert window._batch_running is True

    def test_it_needs_a_selection(self, window, tmp_path):
        self._populate(window, tmp_path, 3)
        window.file_list.setCurrentRow(-1)
        window._classify_from_selected()
        assert window._batch_running is False
        assert "Select" in window.statusBar().currentMessage()

    def test_the_button_disables_during_a_run(self, window, tmp_path):
        self._populate(window, tmp_path, 3)
        window.file_list.setCurrentRow(0)
        window._load = lambda path: None  # type: ignore[assignment]
        window._classify_from_selected()
        # _load ran with inputs disabled; run_from is toggled with the others.
        window.setEnabled_inputs(False)
        assert window.run_from_button.isEnabled() is False


class TestStop:
    def test_stop_starts_disabled(self, window):
        assert window.stop_button.isEnabled() is False

    def test_stop_drains_the_queue_and_ends_after_the_current_file(self, window, tmp_path):
        only = tmp_path / "in" / "a.pdf"
        only.parent.mkdir()
        only.write_bytes(b"pdf")
        window._populate([only])

        # Stand mid-batch, as _classify_all would leave things with work queued.
        window._pending_batch = [1, 2]
        window._batch_running = True
        window.stop_button.setEnabled(True)

        window._stop_batch()
        assert window._stop_requested is True
        assert window._pending_batch == []
        assert window.stop_button.isEnabled() is False

        # The document in flight completing must end the run, not advance.
        _finish_one(window, only, [("請求書", 50.0)])
        window._on_thread_finished()
        assert window._batch_running is False

    def test_a_stopped_run_is_reported_as_stopped_in_the_log(self, window, tmp_path):
        only = tmp_path / "in" / "a.pdf"
        only.parent.mkdir()
        only.write_bytes(b"pdf")
        window._populate([only])
        window.sort_directory = tmp_path / "out"
        window.sort_check.setChecked(True)
        window._prepare_sort_directory()

        window._batch_running = True
        window._pending_batch = [1]
        window._stop_batch()
        _finish_one(window, only, [("請求書", 50.0)])
        window._on_thread_finished()
        assert "STOPPED" in (tmp_path / "out" / "log.txt").read_text(encoding="utf-8")


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
