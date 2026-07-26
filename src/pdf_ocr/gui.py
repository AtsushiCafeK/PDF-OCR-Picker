"""Debug GUI for tuning the classifier against real documents.

This is a development tool, not part of what ships. It exists because the rules
cannot be tuned from a score alone: seeing that a document scored 38 tells you
nothing, whereas seeing that ``請求書`` matched ``請求澤書`` by subsequence in a
block near the top of the page tells you exactly which weight to move.

The expensive step -- OCR -- runs once per document, on a worker thread, and its
result is kept. Everything the tuning loop touches (normalization toggles,
thresholds, rule weights) re-scores from that stored result, so those controls
respond immediately rather than re-running the recogniser.

Run it with::

    poetry run python -m pdf_ocr.gui
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PySide6.QtCore import QObject, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pdf_ocr import resolve_config_path, resolve_rules_path
from pdf_ocr.core.config import Config
from pdf_ocr.core.debuglog import DebugLog
from pdf_ocr.core.extract import DEFAULT_DPI, ExtractionError, extract_page
from pdf_ocr.core.mover import (
    DEFAULT_FOLDER_NAMES,
    Routing,
    clear_sorted_output,
    copy_file,
    count_sorted_output,
)
from pdf_ocr.core.normalize import NormalizedText, NormalizeOptions, normalize_blocks
from pdf_ocr.core.ocr.easy import EasyOcrEngine
from pdf_ocr.core.score import RuleError, RuleSet, Thresholds, score_page
from pdf_ocr.core.types import VERDICT_LABELS, Page, ScoreResult, Source, Verdict

logger = logging.getLogger(__name__)

DISPLAY_DPI = 150
"""Resolution for the on-screen page image. Independent of the OCR DPI: the
recogniser wants detail, the screen wants responsiveness."""

VERDICT_COLORS = {
    Verdict.INVOICE: "#1f7a3d",
    Verdict.NEEDS_REVIEW: "#a8710a",
    Verdict.OTHER: "#8a2b2b",
}

BOX_NEUTRAL = QColor(90, 140, 210, 110)
BOX_POSITIVE = QColor(30, 140, 70, 230)
BOX_NEGATIVE = QColor(190, 50, 50, 230)
BOX_FOCUS = QColor(230, 140, 0, 255)

RULE_COLUMNS = ("id", "pattern", "weight", "match", "scope", "window", "max_distance")


# --------------------------------------------------------------------------
# Background work
# --------------------------------------------------------------------------


@dataclass
class LoadedDocument:
    """Everything one document contributes to the view."""

    path: Path
    page: Page
    image: QPixmap
    scale: float
    """Display pixels per PDF point, for placing boxes over the image."""

    elapsed: float


class ExtractWorker(QObject):
    """Extracts one document off the UI thread.

    OCR takes seconds per page on a CPU. Doing it inline would freeze the window
    for the whole of it, which in a tool meant for repeated comparison is the
    difference between usable and not.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, engine, force_ocr: bool, dpi: int) -> None:
        super().__init__()
        self.path = path
        self.engine = engine
        self.force_ocr = force_ocr
        self.dpi = dpi

    @Slot()
    def run(self) -> None:
        started = time.perf_counter()
        try:
            with pymupdf.open(self.path) as document:
                if document.needs_pass:
                    raise ExtractionError(f"{self.path.name}: is password protected")
                if document.page_count == 0:
                    raise ExtractionError(f"{self.path.name}: has no pages")

                pdf_page = document[0]
                page = extract_page(
                    pdf_page, self.engine, dpi=self.dpi, force_ocr=self.force_ocr
                )

                pixmap_data = pdf_page.get_pixmap(dpi=DISPLAY_DPI, alpha=False)
                image = QPixmap()
                image.loadFromData(pixmap_data.tobytes("png"), "PNG")
                scale = pixmap_data.width / pdf_page.rect.width
        except Exception as error:
            self.failed.emit(f"{self.path.name}: {error}")
            return

        self.finished.emit(
            LoadedDocument(
                path=self.path,
                page=page,
                image=image,
                scale=scale,
                elapsed=time.perf_counter() - started,
            )
        )


# --------------------------------------------------------------------------
# Page view
# --------------------------------------------------------------------------


class PageView(QGraphicsView):
    """The page image with text boxes drawn over it.

    Zoom and pan matter more here than they might appear to. Judging whether the
    recogniser read a title correctly means looking at the characters, and at
    fit-to-window scale a 10pt line is a few pixels tall.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QBrush(QColor("#3a3a3a")))
        self._boxes: list[QGraphicsRectItem] = []

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def clear(self) -> None:
        self._scene.clear()
        self._boxes = []

    def show_document(
        self,
        document: LoadedDocument,
        result: ScoreResult,
    ) -> None:
        """Draw the page and outline every block, colouring the ones that scored."""
        self.clear()
        self._scene.addPixmap(document.image)
        self._scene.setSceneRect(QRectF(document.image.rect()))

        positive: set[int] = set()
        negative: set[int] = set()
        for hit in result.hits:
            (positive if hit.weight >= 0 else negative).update(hit.blocks)

        scale = document.scale
        for index, block in enumerate(document.page.blocks):
            x0, y0, x1, y1 = (value * scale for value in block.bbox)
            if index in negative:
                color, width = BOX_NEGATIVE, 2.0
            elif index in positive:
                color, width = BOX_POSITIVE, 2.0
            else:
                color, width = BOX_NEUTRAL, 1.0
            item = self._scene.addRect(
                x0, y0, x1 - x0, y1 - y0, QPen(color, width), QBrush(Qt.BrushStyle.NoBrush)
            )
            self._boxes.append(item)

        self.fit()

    def focus_blocks(self, indices: list[int]) -> None:
        """Emphasise the blocks behind one hit and bring them into view."""
        for index, item in enumerate(self._boxes):
            if index in indices:
                item.setPen(QPen(BOX_FOCUS, 3.0))
                item.setZValue(1.0)
            else:
                item.setZValue(0.0)
        if indices:
            self.ensureVisible(self._boxes[indices[0]], 80, 80)

    def fit(self) -> None:
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# --------------------------------------------------------------------------
# Logging bridge
# --------------------------------------------------------------------------


class LogBridge(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    """Routes log records to a widget, safely across threads via a signal."""

    def __init__(self, bridge: LogBridge) -> None:
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        self.bridge.message.emit(self.format(record))


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self, rules_path: Path, config_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("PDF invoice classifier -- debug")
        self.resize(1500, 950)

        self.rules_path = rules_path
        self.ruleset = RuleSet.load(rules_path)
        # The installation's remembered folders, shared with the command line so
        # a person sets them once and Power Automate inherits them.
        self.config = Config.load(config_path or resolve_config_path())
        self.engine = EasyOcrEngine()

        self.document: LoadedDocument | None = None
        self.normalized: NormalizedText | None = None
        self.result: ScoreResult | None = None
        self._thread: QThread | None = None
        self._worker: ExtractWorker | None = None
        self._suspend_rescore = False
        self._pending_batch: list[int] = []
        self._batch_running = False
        self._stop_requested = False
        self._current_path: Path | None = None
        self._last_error: str | None = None
        self._debug_log: DebugLog | None = None
        # Start from the remembered destination, if any, so a person who set it
        # last time does not have to choose it again.
        self.sort_directory: Path | None = self.config.output_dir
        self._sorted_counts: Counter[Verdict] = Counter()

        self.page_view = PageView()
        self.setCentralWidget(self.page_view)

        self._build_controls()
        self._build_bottom_panel()
        self._build_menu()
        self._install_logging()

        self._load_rules_into_table()
        self._update_sort_label()
        if self.config.input_dir and self.config.input_dir.is_dir():
            # List the remembered folder, but do not auto-select -- selecting a
            # row would start OCR on it, and a person opening the window has not
            # asked for that yet.
            self._populate(sorted(self.config.input_dir.glob("*.pdf")), select_first=False)
            self.statusBar().showMessage(
                f"{self.file_list.count()} document(s) from {self.config.input_dir}"
            )
        else:
            self.statusBar().showMessage("Open a PDF or a folder to begin.")

    # -- construction ------------------------------------------------------

    def _build_controls(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        layout.addWidget(QLabel("Documents"))
        layout.addWidget(self.file_list, 1)

        run_row = QHBoxLayout()
        self.classify_all_button = QPushButton("Classify all")
        self.classify_all_button.clicked.connect(self._classify_all)
        run_row.addWidget(self.classify_all_button, 1)

        self.run_from_button = QPushButton("Run from selected")
        self.run_from_button.setToolTip(
            "Classify from the selected document to the end of the list, rather "
            "than the whole folder. After a Stop, select where to pick up and "
            "resume here; after changing a rule, re-run from the document you "
            "were looking at without paying for the ones before it again."
        )
        self.run_from_button.clicked.connect(self._classify_from_selected)
        run_row.addWidget(self.run_from_button, 1)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "Stop the run. OCR cannot be interrupted mid-page, so the document "
            "being read now finishes; nothing after it is started."
        )
        self.stop_button.clicked.connect(self._stop_batch)
        run_row.addWidget(self.stop_button)
        layout.addLayout(run_row)

        # -- sorted preview
        sort_box = QGroupBox("Sorted copy")
        sort_layout = QVBoxLayout(sort_box)
        self.sort_check = QCheckBox("Copy into verdict folders while classifying")
        self.sort_check.setToolTip(
            "During 'Classify all', copy each document into "
            + " / ".join(DEFAULT_FOLDER_NAMES.values())
            + " under the chosen folder, so a large batch can be reviewed the "
            "way it will actually look. Copies, never moves -- the source folder "
            "is left untouched."
        )
        self.sort_check.stateChanged.connect(lambda _state: self._update_sort_label())
        sort_layout.addWidget(self.sort_check)

        choose_row = QHBoxLayout()
        self.sort_dir_button = QPushButton("Destination...")
        self.sort_dir_button.clicked.connect(self._choose_sort_directory)
        choose_row.addWidget(self.sort_dir_button)
        choose_row.addStretch(1)
        sort_layout.addLayout(choose_row)

        self.sort_dir_label = QLabel("no destination chosen")
        self.sort_dir_label.setWordWrap(True)
        self.sort_dir_label.setStyleSheet("color: grey;")
        sort_layout.addWidget(self.sort_dir_label)
        layout.addWidget(sort_box)

        # -- verdict readout
        self.verdict_label = QLabel("--")
        self.verdict_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.score_label = QLabel("")
        self.source_label = QLabel("")
        verdict_box = QGroupBox("Result")
        verdict_layout = QVBoxLayout(verdict_box)
        verdict_layout.addWidget(self.verdict_label)
        verdict_layout.addWidget(self.score_label)
        verdict_layout.addWidget(self.source_label)
        layout.addWidget(verdict_box)

        # -- normalization ladder
        normalize_box = QGroupBox("Normalization")
        normalize_layout = QVBoxLayout(normalize_box)
        self.normalize_checks: dict[str, QCheckBox] = {}
        for field, label in (
            ("nfkc", "NFKC  (ＩＮＶＯＩＣＥ, ㈱, half-width kana)"),
            ("strip_whitespace", "Strip whitespace  (請　求　書)"),
            ("strip_symbols", "Strip symbols  (請|求|書)"),
            ("lowercase", "Lowercase"),
        ):
            check = QCheckBox(label)
            check.setChecked(getattr(self.ruleset.normalize, field))
            check.stateChanged.connect(lambda _state: self._rescore())
            normalize_layout.addWidget(check)
            self.normalize_checks[field] = check
        layout.addWidget(normalize_box)

        # -- matching
        matching_box = QGroupBox("Matching")
        matching_layout = QVBoxLayout(matching_box)
        self.fuzzy_text_layer_check = QCheckBox("Allow loose matching on text layers")
        self.fuzzy_text_layer_check.setChecked(self.ruleset.fuzzy_on_text_layer)
        self.fuzzy_text_layer_check.setToolTip(
            "Text-layer characters are already exact, so loosening the match there "
            "cannot recover a missing keyword -- only invent one."
        )
        self.fuzzy_text_layer_check.stateChanged.connect(lambda _state: self._rescore())
        matching_layout.addWidget(self.fuzzy_text_layer_check)

        self.force_ocr_check = QCheckBox("Force OCR (ignore the text layer)")
        self.force_ocr_check.setToolTip(
            "Re-reads the page with the recogniser even when it has a text layer, "
            "so the two can be compared. Costs seconds per page."
        )
        self.force_ocr_check.stateChanged.connect(lambda _state: self._reload_current())
        matching_layout.addWidget(self.force_ocr_check)
        layout.addWidget(matching_box)

        # -- thresholds
        threshold_box = QGroupBox("Thresholds")
        threshold_layout = QFormLayout(threshold_box)
        self.high_slider, self.high_value = self._threshold_row(
            int(self.ruleset.thresholds.high)
        )
        self.low_slider, self.low_value = self._threshold_row(
            int(self.ruleset.thresholds.low)
        )
        threshold_layout.addRow(
            "invoice at or above", self._pair(self.high_slider, self.high_value)
        )
        threshold_layout.addRow(
            "review at or above", self._pair(self.low_slider, self.low_value)
        )
        layout.addWidget(threshold_box)

        dock = QDockWidget("Controls", self)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _threshold_row(self, initial: int) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-50, 250)
        slider.setValue(initial)
        value = QLabel(str(initial))
        value.setMinimumWidth(36)
        slider.valueChanged.connect(lambda v, label=value: label.setText(str(v)))
        slider.valueChanged.connect(lambda _v: self._rescore())
        return slider, value

    @staticmethod
    def _pair(slider: QSlider, label: QLabel) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(slider)
        row.addWidget(label)
        return holder

    def _build_bottom_panel(self) -> None:
        tabs = QTabWidget()

        # -- hits
        self.hits_table = QTableWidget(0, 7)
        self.hits_table.setHorizontalHeaderLabels(
            ["weight", "rule", "pattern", "matched text", "kind", "dist", "scope"]
        )
        self.hits_table.horizontalHeader().setStretchLastSection(True)
        self.hits_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hits_table.itemSelectionChanged.connect(self._on_hit_selected)
        tabs.addTab(self.hits_table, "Hits")

        # -- text
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.raw_text = QPlainTextEdit()
        self.raw_text.setReadOnly(True)
        self.normalized_text = QPlainTextEdit()
        self.normalized_text.setReadOnly(True)
        panes = ((self.raw_text, "As extracted"), (self.normalized_text, "Normalized"))
        for widget, caption in panes:
            holder = QWidget()
            column = QVBoxLayout(holder)
            column.setContentsMargins(0, 0, 0, 0)
            column.addWidget(QLabel(caption))
            column.addWidget(widget)
            splitter.addWidget(holder)
        tabs.addTab(splitter, "Text")

        # -- rules
        rules_panel = QWidget()
        rules_layout = QVBoxLayout(rules_panel)
        self.rules_table = QTableWidget(0, len(RULE_COLUMNS))
        self.rules_table.setHorizontalHeaderLabels(list(RULE_COLUMNS))
        self.rules_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.rules_table.itemChanged.connect(self._on_rules_edited)
        rules_layout.addWidget(self.rules_table)

        buttons = QHBoxLayout()
        for caption, slot in (
            ("Add rule", self._add_rule),
            ("Remove rule", self._remove_rule),
            ("Open...", self._open_rules_file),
            ("Reload from file", self._reload_rules),
            ("Save as...", self._save_rules_as),
        ):
            button = QPushButton(caption)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        rules_layout.addLayout(buttons)
        tabs.addTab(rules_panel, "Rules")

        # -- log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        tabs.addTab(self.log_view, "Log")

        dock = QDockWidget("Detail", self)
        dock.setWidget(tabs)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setMinimumHeight(280)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for caption, shortcut, slot in (
            ("Open PDF...", "Ctrl+O", self._open_file),
            ("Open folder...", "Ctrl+Shift+O", self._open_folder),
        ):
            action = QAction(caption, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)

        view_menu = self.menuBar().addMenu("&View")
        fit_action = QAction("Fit page", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.page_view.fit)
        view_menu.addAction(fit_action)

    def _install_logging(self) -> None:
        bridge = LogBridge(self)
        bridge.message.connect(self.log_view.appendPlainText)
        handler = QtLogHandler(bridge)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    # -- file handling -----------------------------------------------------

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF files (*.pdf)")
        if path:
            self._populate([Path(path)])

    def _open_folder(self) -> None:
        start = str(self.config.input_dir) if self.config.input_dir else ""
        directory = QFileDialog.getExistingDirectory(self, "Open folder", start)
        if directory:
            self._remember_input_dir(Path(directory))
            self._populate(sorted(Path(directory).glob("*.pdf")))

    def _remember_input_dir(self, directory: Path) -> None:
        """Persist the chosen input folder, so it is the default next time."""
        self.config.input_dir = directory
        self._save_config()

    def _save_config(self) -> None:
        try:
            self.config.save()
        except OSError as error:
            # Remembering folders is a convenience; failing to persist should
            # not interrupt the work in front of the user.
            logger.warning("could not save config: %s", error)

    def _populate(self, paths: list[Path], select_first: bool = True) -> None:
        self.file_list.clear()
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.file_list.addItem(item)
        if paths and select_first:
            self.file_list.setCurrentRow(0)
        elif not paths:
            self.statusBar().showMessage("No PDFs found in that folder.")

    def _on_file_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is not None:
            self._load(current.data(Qt.ItemDataRole.UserRole))

    def _reload_current(self) -> None:
        item = self.file_list.currentItem()
        if item is not None:
            self._load(item.data(Qt.ItemDataRole.UserRole))

    def _load(self, path: Path) -> None:
        if self._thread is not None:
            self.statusBar().showMessage("Still reading the previous document...")
            return

        self.statusBar().showMessage(f"Reading {path.name}...")
        self.setEnabled_inputs(False)
        self._current_path = path

        thread = QThread(self)
        worker = ExtractWorker(
            path, self.engine, self.force_ocr_check.isChecked(), DEFAULT_DPI
        )
        # The worker has no Qt parent -- it belongs to the thread, not to this
        # window -- so without a Python reference it is collected the moment
        # this method returns and the job never runs.
        self._worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_loaded)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        thread.start()

    def setEnabled_inputs(self, enabled: bool) -> None:
        self.file_list.setEnabled(enabled)
        self.force_ocr_check.setEnabled(enabled)
        self.classify_all_button.setEnabled(enabled)
        self.run_from_button.setEnabled(enabled)

    @Slot()
    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self.setEnabled_inputs(True)

        if not self._batch_running:
            return

        # Deal with the document that just finished before starting the next.
        # This has to happen even when the queue is now empty, or the final
        # document of a batch would never be labelled or copied -- it is the one
        # whose completion empties the queue.
        item = self.file_list.currentItem()
        if item is not None and self.result is not None and self.document is not None:
            self._annotate(item, self.result)
            destination = self._sort_current(self.result) if self.sort_check.isChecked() else None
            if self._debug_log is not None and self.normalized is not None:
                self._debug_log.document(
                    self.document.path,
                    self.result,
                    self.normalized,
                    self._current_ruleset(),
                    elapsed=self.document.elapsed,
                    destination=destination,
                )
        elif self._debug_log is not None and self._current_path is not None:
            # The load failed; record that, so a run's log accounts for every
            # file rather than silently dropping the ones that could not be read.
            self._debug_log.failure(self._current_path, self._last_error or "unreadable")

        if self._stop_requested:
            self._finish_batch(stopped=True)
        elif self._pending_batch:
            self._advance_batch()
        else:
            self._finish_batch()

    @Slot(object)
    def _on_loaded(self, document: LoadedDocument) -> None:
        self.document = document
        source = "text layer" if document.page.source is Source.TEXT_LAYER else "OCR"
        logger.info(
            "%s: %d blocks from the %s in %.1fs",
            document.path.name,
            len(document.page.blocks),
            source,
            document.elapsed,
        )
        self._rescore()

    @Slot(str)
    def _on_load_failed(self, message: str) -> None:
        # Clear the result too, not just the document: otherwise a failed load
        # in the middle of a batch would leave the previous file's result in
        # place, and the batch step would label and copy the failed file with
        # the wrong verdict.
        self.document = None
        self.result = None
        self._last_error = message
        self.page_view.clear()
        logger.error("%s", message)
        self.statusBar().showMessage(message)

    # -- scoring -----------------------------------------------------------

    def _current_ruleset(self) -> RuleSet:
        """The rule set as the controls currently describe it."""
        return RuleSet(
            rules=self.ruleset.rules,
            thresholds=Thresholds(
                high=float(self.high_slider.value()), low=float(self.low_slider.value())
            ),
            normalize=NormalizeOptions(
                **{
                    field: check.isChecked()
                    for field, check in self.normalize_checks.items()
                }
            ),
            fuzzy_on_text_layer=self.fuzzy_text_layer_check.isChecked(),
        )

    @Slot()
    def _rescore(self) -> None:
        """Re-run scoring on the stored extraction. Never touches OCR."""
        if self._suspend_rescore or self.document is None:
            return

        ruleset = self._current_ruleset()
        page: Page = self.document.page
        self.normalized = normalize_blocks(page.blocks, ruleset.normalize)
        self.result = score_page(page, ruleset, self.normalized)

        self._show_result(self.result)
        self.page_view.show_document(self.document, self.result)
        self._fill_hits(self.result)
        self.raw_text.setPlainText(self.normalized.raw)
        self.normalized_text.setPlainText(self.normalized.text)
        self.statusBar().showMessage(
            f"{self.document.path.name} -- {len(page.blocks)} blocks, "
            f"{len(self.normalized.text)} characters after normalization"
        )

    def _show_result(self, result: ScoreResult) -> None:
        self.verdict_label.setText(VERDICT_LABELS[result.verdict])
        self.verdict_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {VERDICT_COLORS[result.verdict]};"
        )
        positive = sum(hit.weight for hit in result.positive_hits)
        negative = sum(hit.weight for hit in result.negative_hits)
        self.score_label.setText(f"score {result.score:.0f}   (+{positive:.0f} / {negative:.0f})")
        source = "text layer" if result.source is Source.TEXT_LAYER else "OCR"
        self.source_label.setText(f"read from the {source}")

    def _fill_hits(self, result: ScoreResult) -> None:
        self.hits_table.setRowCount(len(result.hits))
        for row, hit in enumerate(result.hits):
            rule = next((r for r in self.ruleset.rules if r.id == hit.rule_id), None)
            cells = [
                f"{hit.weight:+.0f}",
                hit.rule_id,
                hit.pattern,
                hit.match.matched_text,
                hit.match.kind.value,
                str(hit.match.distance) if hit.match.distance else "",
                rule.scope.value if rule else "",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if hit.weight < 0:
                    item.setForeground(QBrush(QColor(VERDICT_COLORS[Verdict.OTHER])))
                self.hits_table.setItem(row, column, item)
        self.hits_table.resizeColumnsToContents()

    def _on_hit_selected(self) -> None:
        rows = {index.row() for index in self.hits_table.selectedIndexes()}
        if not rows or self.result is None:
            return
        hit = self.result.hits[min(rows)]
        self.page_view.focus_blocks(hit.blocks)

    # -- rules -------------------------------------------------------------

    def _load_rules_into_table(self) -> None:
        self._suspend_rescore = True
        self.rules_table.setRowCount(len(self.ruleset.rules))
        for row, rule in enumerate(self.ruleset.rules):
            values = [
                rule.id,
                rule.pattern,
                f"{rule.weight:g}",
                rule.kind.value,
                rule.scope.value,
                "" if rule.window is None else str(rule.window),
                str(rule.max_distance),
            ]
            for column, text in enumerate(values):
                self.rules_table.setItem(row, column, QTableWidgetItem(text))
        self.rules_table.resizeColumnsToContents()
        self._suspend_rescore = False

    def _rules_from_table(self) -> dict:
        rules: list[dict] = []
        for row in range(self.rules_table.rowCount()):
            cell = {
                name: (self.rules_table.item(row, column).text().strip()
                       if self.rules_table.item(row, column) else "")
                for column, name in enumerate(RULE_COLUMNS)
            }
            entry: dict = {
                "id": cell["id"],
                "weight": cell["weight"],
                "match": cell["match"],
                "scope": cell["scope"],
            }
            # A regex lives in the same column as a literal, but has to be handed
            # to the loader under its own key so it is not normalized.
            entry["regex" if cell["match"] == "regex" else "pattern"] = cell["pattern"]
            if cell["window"]:
                entry["window"] = int(cell["window"])
            if cell["max_distance"]:
                entry["max_distance"] = int(cell["max_distance"])
            rules.append(entry)
        return {"rules": rules}

    def _on_rules_edited(self) -> None:
        if self._suspend_rescore:
            return
        try:
            candidate = RuleSet.from_dict(self._rules_from_table())
        except (RuleError, ValueError) as error:
            # Keep the previous rules in force rather than scoring with a
            # half-edited set; every problem is listed at once.
            self.statusBar().showMessage(str(error).replace("\n", "  "))
            logger.warning("rules not applied: %s", error)
            return
        self.ruleset = RuleSet(
            rules=candidate.rules,
            thresholds=self.ruleset.thresholds,
            normalize=self.ruleset.normalize,
            fuzzy_on_text_layer=self.ruleset.fuzzy_on_text_layer,
        )
        self._rescore()

    def _add_rule(self) -> None:
        self._suspend_rescore = True
        row = self.rules_table.rowCount()
        self.rules_table.insertRow(row)
        for column, value in enumerate(
            [f"rule_{row}", "請求書", "10", "subsequence", "whole", "", "1"]
        ):
            self.rules_table.setItem(row, column, QTableWidgetItem(value))
        self._suspend_rescore = False
        self._on_rules_edited()

    def _remove_rule(self) -> None:
        rows = sorted({index.row() for index in self.rules_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._suspend_rescore = True
        for row in rows:
            self.rules_table.removeRow(row)
        self._suspend_rescore = False
        self._on_rules_edited()

    def _apply_loaded_ruleset(self, ruleset: RuleSet) -> None:
        """Push a freshly loaded rule set into the table, the controls and the
        score, all at once."""
        self.ruleset = ruleset
        self._load_rules_into_table()
        self._sync_controls_from_rules()
        self._rescore()

    def _open_rules_file(self) -> None:
        """Load a rules file the user picks -- typically one saved earlier with
        'Save as...'.

        Deliberately does NOT change what 'the default' is: self.rules_path is
        left pointing at the file the window started with, so 'Reload from file'
        still returns there. That keeps tuning cheap to abandon -- open a saved
        set to try it, reload to get back to the default -- which is the whole
        reason unsaved edits are allowed to reset rather than persist.
        """
        start = str(self.rules_path.parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a rules file", start, "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            ruleset = RuleSet.load(Path(path))
        except (RuleError, OSError) as error:
            QMessageBox.warning(self, "Rules", str(error))
            return
        self._apply_loaded_ruleset(ruleset)
        self.statusBar().showMessage(
            f"Loaded {path}   ('Reload from file' still returns to the default)"
        )

    def _reload_rules(self) -> None:
        try:
            ruleset = RuleSet.load(self.rules_path)
        except (RuleError, OSError) as error:
            QMessageBox.warning(self, "Rules", str(error))
            return
        self._apply_loaded_ruleset(ruleset)

    def _sync_controls_from_rules(self) -> None:
        self._suspend_rescore = True
        for field, check in self.normalize_checks.items():
            check.setChecked(getattr(self.ruleset.normalize, field))
        self.fuzzy_text_layer_check.setChecked(self.ruleset.fuzzy_on_text_layer)
        self.high_slider.setValue(int(self.ruleset.thresholds.high))
        self.low_slider.setValue(int(self.ruleset.thresholds.low))
        self._suspend_rescore = False

    def _save_rules_as(self) -> None:
        """Write the tuned rules to a new file.

        Deliberately 'save as' rather than 'save': the shipped rules.yaml is
        heavily commented, and those comments explain why each weight is what it
        is. Rewriting the file from the table would silently discard them.
        """
        import yaml

        suggested = str(self.rules_path.with_name("rules.tuned.yaml"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save rules as", suggested, "YAML (*.yaml *.yml)"
        )
        if not path:
            return

        ruleset = self._current_ruleset()
        data = {
            "thresholds": {
                "high": ruleset.thresholds.high,
                "low": ruleset.thresholds.low,
            },
            "normalize": {
                "nfkc": ruleset.normalize.nfkc,
                "strip_whitespace": ruleset.normalize.strip_whitespace,
                "strip_symbols": ruleset.normalize.strip_symbols,
                "lowercase": ruleset.normalize.lowercase,
            },
            "fuzzy_on_text_layer": ruleset.fuzzy_on_text_layer,
            **self._rules_from_table(),
        }
        Path(path).write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        logger.info("wrote %s (comments from the original are not carried over)", path)
        self.statusBar().showMessage(f"Saved {path}")

    # -- sorted preview ----------------------------------------------------

    def _choose_sort_directory(self) -> None:
        start = str(self.config.output_dir) if self.config.output_dir else ""
        directory = QFileDialog.getExistingDirectory(
            self, "Where to copy the sorted documents", start
        )
        if not directory:
            return

        chosen = Path(directory)
        source = self._source_directory()
        if source is not None and (chosen == source or source in chosen.parents):
            # Copies landing inside the folder being classified would be picked
            # up as input the next time it is opened, and each run would breed
            # another generation of duplicates.
            QMessageBox.warning(
                self,
                "Sorted copy",
                f"{chosen}\n\nis inside the folder being classified. Choose a "
                f"destination outside it, or the copies will be classified "
                f"again on the next run.",
            )
            return

        self.sort_directory = chosen
        self.config.output_dir = chosen
        self._save_config()
        self.sort_check.setChecked(True)
        self._update_sort_label()

    def _source_directory(self) -> Path | None:
        """The folder the listed documents came from, if they share one."""
        item = self.file_list.item(0)
        if item is None:
            return None
        return Path(item.data(Qt.ItemDataRole.UserRole)).parent

    def _update_sort_label(self) -> None:
        if self.sort_directory is None:
            self.sort_dir_label.setText("no destination chosen")
            return
        existing = count_sorted_output(self.sort_directory, DEFAULT_FOLDER_NAMES.values())
        suffix = f"  ({existing} already sorted there)" if existing else ""
        self.sort_dir_label.setText(f"{self.sort_directory}{suffix}")

    def _prepare_sort_directory(self) -> bool:
        """Ask before emptying the destination. Returns whether to go ahead.

        A preview gets re-run after every rule change, and a document whose
        verdict changed would otherwise be left in both its old folder and its
        new one -- which would make the thing being inspected misleading. So the
        previous run is cleared first, and since that deletes files, it is
        confirmed rather than assumed.
        """
        if self.sort_directory is None:
            QMessageBox.information(
                self, "Sorted copy", "Choose a destination folder first."
            )
            return False

        existing = count_sorted_output(self.sort_directory, DEFAULT_FOLDER_NAMES.values())
        if existing:
            answer = QMessageBox.question(
                self,
                "Sorted copy",
                f"{self.sort_directory}\n\n"
                f"already holds {existing} sorted PDF(s) from an earlier run.\n"
                f"Delete them and sort again?\n\n"
                f"Only PDFs directly inside "
                f"{', '.join(DEFAULT_FOLDER_NAMES.values())} are removed. "
                f"The documents being classified are not touched.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer is not QMessageBox.StandardButton.Yes:
                return False
            removed = clear_sorted_output(
                self.sort_directory, DEFAULT_FOLDER_NAMES.values()
            )
            logger.info("cleared %d file(s) from %s", removed, self.sort_directory)

        self._sorted_counts = Counter()

        # A run-long debug log, written beside the copies it describes. It lives
        # at the destination root, so clearing the verdict subfolders above never
        # touches it; opening in write mode gives each run a fresh one.
        self._close_debug_log()
        try:
            self._debug_log = DebugLog(self.sort_directory / "log.txt", VERDICT_LABELS)
            self._debug_log.header(self._current_ruleset(), self._source_directory())
        except OSError as error:
            # A log that cannot be written is not a reason to abandon the run.
            logger.error("could not open the debug log: %s", error)
            self._debug_log = None
        return True

    def _sort_current(self, result: ScoreResult) -> Path | None:
        """Copy the document just classified into its verdict folder."""
        if self.sort_directory is None or self.document is None:
            return None
        routing = Routing(self.sort_directory)
        try:
            destination = copy_file(
                self.document.path, routing.directory_for(result.verdict)
            )
        except OSError as error:
            # One unreadable file should not abandon the rest of the batch.
            logger.error("could not copy %s: %s", self.document.path.name, error)
            return None
        self._sorted_counts[result.verdict] += 1
        logger.info("copied %s -> %s", self.document.path.name, destination.parent.name)
        return destination

    def _close_debug_log(self) -> None:
        if self._debug_log is not None:
            self._debug_log.close()
            self._debug_log = None

    # -- batch -------------------------------------------------------------

    def _classify_all(self) -> None:
        """Walk the whole list, so the make-up of a real folder becomes visible.

        Worth the wait on a folder of scans: it answers how many documents even
        have a text layer, and where the score distribution actually splits --
        which is how a threshold gets chosen on evidence rather than by feel.

        With 'Sorted copy' enabled each document is also copied into its verdict
        folder as it goes, which on a real batch is the difference between
        reading a column of numbers and seeing the piles the rules would
        actually produce.
        """
        self._start_batch(range(self.file_list.count()))

    def _classify_from_selected(self) -> None:
        """Classify from the selected document to the end, not the whole folder.

        The tuning loop this serves: watch a run, Stop when a document scores
        wrong, add or adjust a rule, then resume from that document -- without
        paying the OCR cost of everything before it a second time.
        """
        start = self.file_list.currentRow()
        if start < 0:
            self.statusBar().showMessage("Select a document to start from first.")
            return
        self._start_batch(range(start, self.file_list.count()))

    def _start_batch(self, rows) -> None:
        """Run a classification over the given rows of the list."""
        rows = list(rows)
        if not rows:
            return
        # Sorted copy, when on, is prepared once per run: the destination is
        # cleared and a fresh log opened. A 'Run from selected' therefore also
        # starts a clean copy of that partial run rather than merging into an
        # earlier one -- simpler to reason about than a half-updated folder.
        if self.sort_check.isChecked() and not self._prepare_sort_directory():
            return
        self._pending_batch = rows
        self._stop_requested = False
        self._batch_running = True
        self.stop_button.setEnabled(True)
        self._advance_batch()

    def _stop_batch(self) -> None:
        """Ask the run to stop. It ends once the current document is done.

        OCR of a page cannot be interrupted cleanly, so rather than kill the
        worker mid-read -- which risks a half-written copy or a wedged thread --
        the request simply drains the queue and lets the document in flight
        finish.
        """
        if not self._batch_running:
            return
        self._stop_requested = True
        self._pending_batch = []
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Stopping after the current document...")

    def _finish_batch(self, stopped: bool = False) -> None:
        self._batch_running = False
        self._stop_requested = False
        self.stop_button.setEnabled(False)
        if self._debug_log is not None:
            self._debug_log.footer(self._sorted_counts, stopped=stopped)
            self._close_debug_log()
        prefix = "Stopped" if stopped else "Finished"
        self.statusBar().showMessage(self._batch_summary(prefix))
        self._update_sort_label()

    def _advance_batch(self) -> None:
        if not self._pending_batch:
            self._finish_batch()
            return
        row = self._pending_batch.pop(0)
        item = self.file_list.item(row)
        if item is None:
            self._advance_batch()
            return
        # Drive the load directly rather than through the selection signal,
        # which would not fire if the row were already current and would stall
        # the run.
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(row)
        self.file_list.blockSignals(False)
        self._load(item.data(Qt.ItemDataRole.UserRole))

    def _batch_summary(self, prefix: str = "Finished") -> str:
        if not self._sorted_counts:
            return f"{prefix} classifying the folder."
        tally = "  ".join(
            f"{DEFAULT_FOLDER_NAMES[verdict]} {self._sorted_counts[verdict]}"
            for verdict in Verdict
            if self._sorted_counts[verdict]
        )
        return f"{prefix}. Copied into {self.sort_directory}:  {tally}   (log.txt written)"

    def _annotate(self, item: QListWidgetItem, result: ScoreResult) -> None:
        name = item.data(Qt.ItemDataRole.UserRole).name
        item.setText(f"{name}   [{VERDICT_LABELS[result.verdict]} {result.score:.0f}]")
        item.setForeground(QBrush(QColor(VERDICT_COLORS[result.verdict])))


def main(rules_path: Path | None = None, config_path: Path | None = None) -> int:
    # A windowed build has no console, so sys.stderr is None and the default
    # stream handler would fail on its first record. The GUI shows its own log
    # pane regardless, which is where these end up.
    if sys.stderr is None:
        logging.getLogger().addHandler(logging.NullHandler())
    else:
        logging.basicConfig(level=logging.INFO)

    application = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(rules_path or resolve_rules_path(), config_path)
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
