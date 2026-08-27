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
from PySide6.QtGui import QAction, QImage, QPainter
from PySide6.QtPrintSupport import QPrinter, QPrinterInfo
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
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Print one PDF onto ``printer``. Returns False if it was stopped partway.

    Stopping is checked before each page: the pages already sent print, the rest
    do not. A page in flight to the spooler cannot be recalled.
    """
    with pymupdf.open(path) as document:
        printer.setDocName(path.name)
        if copies > 1:
            printer.setCopyCount(copies)

        painter = QPainter()
        if not painter.begin(printer):
            raise RuntimeError(f"could not start a print job on {printer.printerName()!r}")

        stopped = False
        try:
            for page_index in range(document.page_count):
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if page_index > 0:
                    printer.newPage()
                _draw_page(painter, document[page_index])
        finally:
            painter.end()
    return not stopped


def _draw_page(painter: QPainter, page: pymupdf.Page) -> None:
    """Render a page and paint it centred and scaled to fit the printable area."""
    pixmap = page.get_pixmap(dpi=PRINT_DPI, alpha=False)
    image = QImage(
        pixmap.samples, pixmap.width, pixmap.height, pixmap.stride,
        QImage.Format.Format_RGB888,
    ).copy()  # copy: the pixmap buffer must not outlive this call otherwise

    area = painter.viewport()
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


def _printer_for(name: str) -> QPrinter:
    for info in QPrinterInfo.availablePrinters():
        if info.printerName() == name:
            return QPrinter(info, QPrinter.PrinterMode.HighResolution)
    return QPrinter(QPrinter.PrinterMode.HighResolution)


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------


class PrintWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PDF print")
        self.resize(560, 520)
        self._stop = False
        self._printing = False

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("Documents to print"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.file_list, 1)

        add_row = QHBoxLayout()
        for caption, slot in (
            ("Add files...", self._add_files),
            ("Add folder...", self._add_folder),
            ("Remove", self._remove_selected),
            ("Clear", self._clear),
        ):
            button = QPushButton(caption)
            button.clicked.connect(slot)
            add_row.addWidget(button)
        layout.addLayout(add_row)

        self.recurse_check = QCheckBox("Include subfolders when adding a folder")
        layout.addWidget(self.recurse_check)

        printer_row = QHBoxLayout()
        printer_row.addWidget(QLabel("Printer"))
        self.printer_combo = QComboBox()
        printer_row.addWidget(self.printer_combo, 1)
        printer_row.addWidget(QLabel("Copies"))
        self.copies_spin = QSpinBox()
        self.copies_spin.setRange(1, 99)
        printer_row.addWidget(self.copies_spin)
        layout.addLayout(printer_row)

        run_row = QHBoxLayout()
        self.print_button = QPushButton("Print")
        self.print_button.clicked.connect(self._print)
        run_row.addWidget(self.print_button, 1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "Stop printing. Pages already sent to the printer may still come "
            "out; nothing after the current point is sent."
        )
        self.stop_button.clicked.connect(self._request_stop)
        run_row.addWidget(self.stop_button)
        layout.addLayout(run_row)

        self._load_printers()
        self._build_menu()
        self.statusBar().showMessage("Add files or a folder, choose a printer, then Print.")

    # -- construction ------------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for caption, shortcut, slot in (
            ("Add files...", "Ctrl+O", self._add_files),
            ("Add folder...", "Ctrl+Shift+O", self._add_folder),
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
            self.statusBar().showMessage("No printers found on this PC.")

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
        self.statusBar().showMessage(f"{self.file_list.count()} document(s) queued.")
        if added == 0 and paths:
            self.statusBar().showMessage("No new PDFs to add.")

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add PDFs", "", "PDF files (*.pdf)")
        if files:
            self._add_paths([Path(f) for f in files])

    def _add_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Add a folder of PDFs")
        if directory:
            self._add_paths([Path(directory)])

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _clear(self) -> None:
        self.file_list.clear()
        self.statusBar().showMessage("Queue cleared.")

    def _queued_files(self) -> list[Path]:
        return [
            self.file_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.file_list.count())
        ]

    # -- printing ----------------------------------------------------------

    def _request_stop(self) -> None:
        self._stop = True
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Stopping...")

    def _check_stop(self) -> bool:
        # Pump events so a Stop click is seen, then report whether it happened.
        QApplication.processEvents()
        return self._stop

    def _print(self) -> None:
        files = self._queued_files()
        if not files:
            QMessageBox.information(self, "Print", "Add some documents first.")
            return
        if self.printer_combo.count() == 0:
            QMessageBox.information(self, "Print", "No printer is available.")
            return

        printer = _printer_for(self.printer_combo.currentText())
        copies = self.copies_spin.value()
        self._stop = False
        self._set_printing(True)

        def print_one(path: Path) -> bool:
            try:
                return print_pdf(path, printer, copies=copies, should_stop=self._check_stop)
            except Exception as error:  # a bad PDF should not abandon the queue
                logger.error("could not print %s: %s", path.name, error)
                self.statusBar().showMessage(f"Skipped {path.name}: {error}")
                return True  # treat as handled; move on to the next file

        printed, total, stopped = run_print_job(
            files,
            print_one,
            on_progress=lambda done, count, path: self.statusBar().showMessage(
                f"Printing {done}/{count}: {path.name}"
            ),
            should_stop=self._check_stop,
        )

        self._set_printing(False)
        prefix = "Stopped" if stopped else "Finished"
        self.statusBar().showMessage(f"{prefix}. Printed {printed} of {total}.")

    def _set_printing(self, printing: bool) -> None:
        self._printing = printing
        self.print_button.setEnabled(not printing)
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
