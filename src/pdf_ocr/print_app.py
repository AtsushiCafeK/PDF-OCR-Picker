"""A standalone PDF print utility -- a separate window, unrelated to the sorter.

It shares nothing with the classifier: no rules, no OCR, no folders. You add
files or folders, choose a printer, and print; a running job can be stopped.

Printing renders each PDF page with PyMuPDF and paints it onto a QPrinter, so any
PDF prints -- scanned or digital -- with no separate PDF reader installed. The
job runs on the GUI thread, pumping events between pages so the Stop button stays
live; QPrinter is finicky off the main thread, and printing a page is quick
enough that this stays responsive.

Runs as its own executable (pdf-print.exe) or ``pdf-sorter print``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import pymupdf
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage, QPageLayout, QPainter, QTransform
from PySide6.QtPrintSupport import (
    QPrintDialog,
    QPrinter,
    QPrinterInfo,
    QPrintPreviewDialog,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

PRINT_DPI = 200
"""Resolution to rasterise pages at. Enough for text to print cleanly without
making each page image needlessly large."""


# --------------------------------------------------------------------------
# Orientation and page-range logic -- pure functions, so they can be tested
# without a printer, a document or a window.
# --------------------------------------------------------------------------


def is_landscape(rect: pymupdf.Rect) -> bool:
    """Whether a page is wider than it is tall."""
    return rect.width > rect.height


def needs_rotation(page_is_landscape: bool, paper_is_landscape: bool) -> bool:
    """Whether the page image must turn a quarter to match the paper's shape.

    This is the safety net that stops a sheet being wasted: even if the printer
    driver ignores a per-page orientation change, rotating the image makes a
    landscape page fill a portrait sheet (and the reverse) instead of printing
    small and centred with half the paper blank.
    """
    return page_is_landscape != paper_is_landscape


def clamp_page_range(first: int, last: int, page_count: int) -> list[int]:
    """0-based page indices for a 1-based inclusive range, clamped to the document.

    A ``first`` or ``last`` of 0 means that end is open. Applied per document, so
    a range from the print dialog is clamped to each file's own length rather
    than failing on a file shorter than the range.
    """
    start = first if first and first > 0 else 1
    end = last if last and last > 0 else page_count
    start = max(1, start)
    end = min(page_count, end)
    if end < start:
        return []
    return list(range(start - 1, end))


def _orientation_for(page: pymupdf.Page) -> QPageLayout.Orientation:
    if is_landscape(page.rect):
        return QPageLayout.Orientation.Landscape
    return QPageLayout.Orientation.Portrait


def _first_page_orientation(
    files: list[Path], page_range: tuple[int, int] | None
) -> QPageLayout.Orientation | None:
    """The orientation of the very first page that will actually be printed.

    The first page's orientation has to be set before the print job begins --
    unlike later pages, which are set just before ``newPage()`` -- so both the
    print and preview paths need to know it up front.
    """
    for path in files:
        with pymupdf.open(path) as document:
            if page_range is None:
                indices: Iterable[int] = range(document.page_count)
            else:
                indices = clamp_page_range(page_range[0], page_range[1], document.page_count)
            for page_index in indices:
                return _orientation_for(document[page_index])
    return None


# --------------------------------------------------------------------------
# Job model -- kept free of the window so it can be tested without a printer
# --------------------------------------------------------------------------


def collect_pdfs(paths: Iterable[Path], recursive: bool = False) -> list[Path]:
    """Expand a mix of files and folders into a de-duplicated list of PDFs.

    Order is preserved and duplicates are dropped, so adding a folder and then
    one of its files does not queue that file twice.
    """
    result: list[Path] = []
    seen: set[Path] = set()

    def add(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(candidate)

    for path in paths:
        path = Path(path)
        if path.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            for pdf in sorted(path.glob(pattern)):
                add(pdf)
        elif path.is_file() and path.suffix.lower() == ".pdf":
            add(path)
    return result


def local_paths(mime) -> list[Path]:
    """Local filesystem paths from a drag-and-drop payload.

    A drop carries URLs; the ones that name a local file or folder convert to a
    path, and anything else (a web URL dragged from a browser, say) yields an
    empty string and is dropped. Folders are kept as-is -- ``collect_pdfs``, via
    ``_add_paths``, expands them into their PDFs.
    """
    paths: list[Path] = []
    for url in mime.urls():
        local = url.toLocalFile()
        if local:
            paths.append(Path(local))
    return paths


def run_print_job(
    files: list[Path],
    print_one: Callable[[Path], bool],
    *,
    on_progress: Callable[[int, int, Path], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[int, int, bool]:
    """Print each file via ``print_one``, returning ``(printed, total, stopped)``.

    ``print_one(path)`` returns True when the file printed fully and False when
    it was stopped partway. ``should_stop`` is polled before each file so a
    stop takes effect between documents even when one is mid-print.
    """
    printed = 0
    total = len(files)
    stopped = False
    for index, path in enumerate(files, start=1):
        if should_stop is not None and should_stop():
            stopped = True
            break
        if on_progress is not None:
            on_progress(index, total, path)
        if print_one(path):
            printed += 1
        else:
            stopped = True
            break
    return printed, total, stopped


def print_pdf(
    path: Path,
    printer: QPrinter,
    *,
    copies: int = 1,
    prefer_pdf_orientation: bool = True,
    page_range: tuple[int, int] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Print one PDF onto ``printer``. Returns False if it was stopped partway.

    Stopping is checked before each page: the pages already sent print, the rest
    do not. A page in flight to the spooler cannot be recalled.

    When ``prefer_pdf_orientation`` is on, each page decides the paper's
    orientation from its own shape -- so a landscape page prints on a landscape
    sheet -- and the rendered image is turned a quarter where the sheet still
    does not match, so a page always fills the paper instead of wasting it. This
    is the whole point of the tool over a generic print path: because it rasters
    and paints each page itself, the PDF's geometry wins over the driver's.

    ``page_range`` is a 1-based inclusive ``(first, last)``, clamped to this
    document; ``None`` prints every page.
    """
    with pymupdf.open(path) as document:
        printer.setDocName(path.name)
        if copies > 1:
            printer.setCopyCount(copies)

        if page_range is None:
            indices = list(range(document.page_count))
        else:
            indices = clamp_page_range(page_range[0], page_range[1], document.page_count)
        if not indices:
            return True  # nothing selected in this file; not a stop

        if prefer_pdf_orientation:
            printer.setPageOrientation(_orientation_for(document[indices[0]]))

        painter = QPainter()
        if not painter.begin(printer):
            raise RuntimeError(f"could not start a print job on {printer.printerName()!r}")

        stopped = False
        try:
            for position, page_index in enumerate(indices):
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if position > 0:
                    if prefer_pdf_orientation:
                        printer.setPageOrientation(_orientation_for(document[page_index]))
                    printer.newPage()
                _draw_page(painter, document[page_index], fit_orientation=prefer_pdf_orientation)
        finally:
            painter.end()
    return not stopped


def _draw_page(
    painter: QPainter, page: pymupdf.Page, *, fit_orientation: bool = False
) -> None:
    """Render a page and paint it centred and scaled to fit the printable area.

    With ``fit_orientation`` on, a page whose shape disagrees with the paper is
    turned a quarter first, so it fills the sheet rather than printing small.
    """
    pixmap = page.get_pixmap(dpi=PRINT_DPI, alpha=False)
    image = QImage(
        pixmap.samples, pixmap.width, pixmap.height, pixmap.stride,
        QImage.Format.Format_RGB888,
    ).copy()  # copy: the pixmap buffer must not outlive this call otherwise

    area = painter.viewport()
    if fit_orientation and needs_rotation(
        image.width() > image.height(), area.width() > area.height()
    ):
        image = image.transformed(
            QTransform().rotate(90), Qt.TransformationMode.SmoothTransformation
        )

    scaled = image.scaled(
        area.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    x = area.x() + (area.width() - scaled.width()) // 2
    y = area.y() + (area.height() - scaled.height()) // 2
    painter.drawImage(x, y, scaled)


def available_printer_names() -> list[str]:
    return [info.printerName() for info in QPrinterInfo.availablePrinters()]


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------


class PrintWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PDF印刷")
        self.resize(560, 520)
        self.setAcceptDrops(True)  # drop files or folders anywhere on the window
        self._stop = False
        self._printing = False

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("印刷するファイル"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.file_list, 1)

        add_row = QHBoxLayout()
        for caption, slot in (
            ("ファイルを追加…", self._add_files),
            ("フォルダを追加…", self._add_folder),
            ("削除", self._remove_selected),
            ("クリア", self._clear),
        ):
            button = QPushButton(caption)
            button.clicked.connect(slot)
            add_row.addWidget(button)
        layout.addLayout(add_row)

        self.recurse_check = QCheckBox("フォルダ追加時にサブフォルダも含める")
        layout.addWidget(self.recurse_check)

        # The settings the print dialog adjusts live on one long-lived QPrinter,
        # so what you set in "Paper setup..." is what both Preview and Print use.
        self._printer = QPrinter(QPrinter.PrinterMode.HighResolution)

        # On by default: the reason the tool exists is to stop wasted sheets when
        # a PDF's orientation and the printer's disagree.
        self.prefer_orientation_check = QCheckBox(
            "PDFの向きを優先する（縦・横を自動で合わせ、用紙いっぱいに印刷）"
        )
        self.prefer_orientation_check.setChecked(True)
        layout.addWidget(self.prefer_orientation_check)

        printer_row = QHBoxLayout()
        printer_row.addWidget(QLabel("プリンター"))
        self.printer_combo = QComboBox()
        printer_row.addWidget(self.printer_combo, 1)
        self.page_setup_button = QPushButton("用紙設定…")
        self.page_setup_button.setToolTip(
            "用紙サイズ・両面・ページ範囲などを設定します。"
            "このプリンターが対応している項目だけが表示されます。"
        )
        self.page_setup_button.clicked.connect(self._page_setup)
        printer_row.addWidget(self.page_setup_button)
        printer_row.addWidget(QLabel("部数"))
        self.copies_spin = QSpinBox()
        self.copies_spin.setRange(1, 99)
        printer_row.addWidget(self.copies_spin)
        layout.addLayout(printer_row)

        run_row = QHBoxLayout()
        self.preview_button = QPushButton("プレビュー")
        self.preview_button.setToolTip("印刷前に、各ページが用紙にどう載るかを確認します。")
        self.preview_button.clicked.connect(self._preview)
        run_row.addWidget(self.preview_button)
        self.print_button = QPushButton("印刷")
        self.print_button.clicked.connect(self._print)
        run_row.addWidget(self.print_button, 1)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "印刷を停止します。すでにプリンターへ送られたページは出ることが"
            "あります。それ以降は送信しません。"
        )
        self.stop_button.clicked.connect(self._request_stop)
        run_row.addWidget(self.stop_button)
        layout.addLayout(run_row)

        self._load_printers()
        # Wire the printer/copies controls to the shared QPrinter after loading,
        # so populating the combo does not fire a change into a half-built state.
        self.printer_combo.currentTextChanged.connect(self._on_printer_changed)
        self.copies_spin.valueChanged.connect(self._on_copies_changed)
        self._on_printer_changed(self.printer_combo.currentText())
        self._build_menu()
        self.statusBar().showMessage("ファイルかフォルダを追加し、プリンターを選んで「印刷」します。")

    # -- construction ------------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        for caption, shortcut, slot in (
            ("ファイルを追加…", "Ctrl+O", self._add_files),
            ("フォルダを追加…", "Ctrl+Shift+O", self._add_folder),
        ):
            action = QAction(caption, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)

    def _load_printers(self) -> None:
        names = available_printer_names()
        self.printer_combo.addItems(names)
        default = QPrinterInfo.defaultPrinter()
        if not default.isNull():
            index = self.printer_combo.findText(default.printerName())
            if index >= 0:
                self.printer_combo.setCurrentIndex(index)
        if not names:
            self.print_button.setEnabled(False)
            self.statusBar().showMessage("このPCにプリンターが見つかりません。")

    # -- printer settings --------------------------------------------------

    def _on_printer_changed(self, name: str) -> None:
        if name:
            self._printer.setPrinterName(name)

    def _on_copies_changed(self, count: int) -> None:
        self._printer.setCopyCount(max(1, count))

    def _page_setup(self) -> None:
        """Open the OS print dialog to set paper size, duplex, range, etc.

        The native dialog only offers what the chosen printer supports, so there
        is nothing to validate here; we just reflect its result back into the
        combo and the copies box.
        """
        dialog = QPrintDialog(self._printer, self)
        dialog.setWindowTitle("用紙設定")
        if not dialog.exec():
            return
        index = self.printer_combo.findText(self._printer.printerName())
        if index >= 0:
            self.printer_combo.setCurrentIndex(index)
        self.copies_spin.setValue(max(1, self._printer.copyCount()))
        self.statusBar().showMessage("用紙設定を更新しました。")

    def _current_page_range(self) -> tuple[int, int] | None:
        """The page range chosen in the print dialog, or None for every page."""
        if self._printer.printRange() == QPrinter.PrintRange.PageRange:
            return (self._printer.fromPage(), self._printer.toPage())
        return None

    # -- file list ---------------------------------------------------------

    def _add_paths(self, paths: list[Path]) -> None:
        existing = {
            self.file_list.item(i).data(Qt.ItemDataRole.UserRole).resolve()
            for i in range(self.file_list.count())
        }
        added = 0
        for path in collect_pdfs(paths, self.recurse_check.isChecked()):
            if path.resolve() in existing:
                continue
            existing.add(path.resolve())
            item = QListWidgetItem(str(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.file_list.addItem(item)
            added += 1
        self.statusBar().showMessage(f"{self.file_list.count()} 件をキューに追加しました。")
        if added == 0 and paths:
            self.statusBar().showMessage("追加する新しいPDFはありません。")

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "PDFを追加", "", "PDFファイル (*.pdf)")
        if files:
            self._add_paths([Path(f) for f in files])

    def _add_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "PDFのフォルダを追加")
        if directory:
            self._add_paths([Path(directory)])

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _clear(self) -> None:
        self.file_list.clear()
        self.statusBar().showMessage("キューをクリアしました。")

    def _queued_files(self) -> list[Path]:
        return [
            self.file_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.file_list.count())
        ]

    # -- drag and drop -----------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        # Accept the drag only when it carries file/folder URLs, so the cursor
        # shows "can drop here" over this window and nowhere misleading.
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = local_paths(event.mimeData())
        if paths:
            self._add_paths(paths)
            event.acceptProposedAction()

    # -- printing ----------------------------------------------------------

    def _request_stop(self) -> None:
        self._stop = True
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("停止中…")

    def _check_stop(self) -> bool:
        # Pump events so a Stop click is seen, then report whether it happened.
        QApplication.processEvents()
        return self._stop

    def _preview(self) -> None:
        files = self._queued_files()
        if not files:
            QMessageBox.information(self, "プレビュー", "先にファイルを追加してください。")
            return
        preview = QPrintPreviewDialog(self._printer, self)
        preview.setWindowTitle("印刷プレビュー")
        preview.paintRequested.connect(self._render_preview)
        preview.exec()
        preview.paintRequested.disconnect(self._render_preview)

    def _render_preview(self, printer: QPrinter) -> None:
        """Paint every queued page into the preview, exactly as printing would.

        Shares the orientation logic with the real print path so the preview is
        honest: what you see here is what the paper will show.
        """
        prefer = self.prefer_orientation_check.isChecked()
        page_range = self._current_page_range()
        files = self._queued_files()

        if prefer:
            first_orientation = _first_page_orientation(files, page_range)
            if first_orientation is not None:
                printer.setPageOrientation(first_orientation)

        painter = QPainter()
        if not painter.begin(printer):
            return
        try:
            first = True
            for path in files:
                with pymupdf.open(path) as document:
                    if page_range is None:
                        indices: Iterable[int] = range(document.page_count)
                    else:
                        indices = clamp_page_range(
                            page_range[0], page_range[1], document.page_count
                        )
                    for page_index in indices:
                        page = document[page_index]
                        if prefer:
                            printer.setPageOrientation(_orientation_for(page))
                        if not first:
                            printer.newPage()
                        first = False
                        _draw_page(painter, page, fit_orientation=prefer)
        finally:
            painter.end()

    def _print(self) -> None:
        files = self._queued_files()
        if not files:
            QMessageBox.information(self, "印刷", "先にファイルを追加してください。")
            return
        if self.printer_combo.count() == 0:
            QMessageBox.information(self, "印刷", "利用できるプリンターがありません。")
            return

        prefer = self.prefer_orientation_check.isChecked()
        page_range = self._current_page_range()
        copies = self.copies_spin.value()
        self._stop = False
        self._set_printing(True)

        def print_one(path: Path) -> bool:
            try:
                return print_pdf(
                    path,
                    self._printer,
                    copies=copies,
                    prefer_pdf_orientation=prefer,
                    page_range=page_range,
                    should_stop=self._check_stop,
                )
            except Exception as error:  # a bad PDF should not abandon the queue
                logger.error("could not print %s: %s", path.name, error)
                self.statusBar().showMessage(f"スキップ: {path.name}（{error}）")
                return True  # treat as handled; move on to the next file

        printed, total, stopped = run_print_job(
            files,
            print_one,
            on_progress=lambda done, count, path: self.statusBar().showMessage(
                f"印刷中 {done}/{count}: {path.name}"
            ),
            should_stop=self._check_stop,
        )

        self._set_printing(False)
        prefix = "停止しました" if stopped else "完了しました"
        self.statusBar().showMessage(f"{prefix}。{printed}/{total} 件を印刷しました。")

    def _set_printing(self, printing: bool) -> None:
        self._printing = printing
        self.print_button.setEnabled(not printing)
        self.preview_button.setEnabled(not printing)
        self.page_setup_button.setEnabled(not printing)
        self.stop_button.setEnabled(printing)
        self.file_list.setEnabled(not printing)


def main() -> int:
    if sys.stderr is None:
        logging.getLogger().addHandler(logging.NullHandler())
    else:
        logging.basicConfig(level=logging.INFO)

    application = QApplication.instance() or QApplication(sys.argv)
    window = PrintWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
