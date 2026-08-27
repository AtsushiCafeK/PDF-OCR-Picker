"""Tests for the standalone print utility.

The printing itself is exercised by printing to a PDF file -- QPrinter's PDF
output -- so the render-and-paint path is covered without a physical printer.
The window and job model are checked directly.
"""

from __future__ import annotations

import os

import pymupdf
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtPrintSupport import QPrinter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pdf_ocr.print_app import (  # noqa: E402
    PrintWindow,
    collect_pdfs,
    print_pdf,
    run_print_job,
)
from tools.sample_pdfs import SAMPLES, build  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def sample_pdf(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    sample = next(s for s in SAMPLES if s.name == name)
    path = directory / f"{name}.pdf"
    doc = build(sample)
    doc.save(path)
    doc.close()
    return path


def multipage_pdf(directory, filename, *names):
    directory.mkdir(parents=True, exist_ok=True)
    merged = pymupdf.open()
    for name in names:
        with pymupdf.open(sample_pdf(directory / "_parts", name)) as part:
            merged.insert_pdf(part)
    path = directory / filename
    merged.save(path)
    merged.close()
    return path


class TestCollectPdfs:
    def test_files_are_kept_in_order(self, tmp_path):
        a = sample_pdf(tmp_path, "invoice_01_centered")
        b = sample_pdf(tmp_path, "other_01_quotation")
        assert collect_pdfs([a, b]) == [a, b]

    def test_a_folder_expands_to_its_pdfs(self, tmp_path):
        folder = tmp_path / "in"
        sample_pdf(folder, "invoice_01_centered")
        sample_pdf(folder, "other_01_quotation")
        assert len(collect_pdfs([folder])) == 2

    def test_subfolders_only_when_recursive(self, tmp_path):
        folder = tmp_path / "in"
        sample_pdf(folder, "invoice_01_centered")
        sample_pdf(folder / "sub", "other_01_quotation")
        assert len(collect_pdfs([folder])) == 1
        assert len(collect_pdfs([folder], recursive=True)) == 2

    def test_non_pdfs_are_ignored(self, tmp_path):
        (tmp_path / "note.txt").write_text("x", encoding="utf-8")
        assert collect_pdfs([tmp_path / "note.txt"]) == []

    def test_duplicates_are_dropped(self, tmp_path):
        folder = tmp_path / "in"
        a = sample_pdf(folder, "invoice_01_centered")
        # The folder and one of its files: the file must not queue twice.
        assert collect_pdfs([folder, a]) == [a]


class TestRunPrintJob:
    def test_every_file_is_printed(self):
        printed = []
        result = run_print_job(
            [object(), object(), object()], lambda p: printed.append(p) or True
        )
        assert result == (3, 3, False)
        assert len(printed) == 3

    def test_stop_before_the_next_file(self):
        seen = []

        def should_stop():
            return len(seen) >= 1  # stop once one file has been handed over

        printed, total, stopped = run_print_job(
            [object(), object(), object()],
            lambda p: seen.append(p) or True,
            should_stop=should_stop,
        )
        assert printed == 1
        assert stopped is True

    def test_a_file_stopped_partway_ends_the_run(self):
        printed, total, stopped = run_print_job(
            [object(), object()], lambda p: False  # print_one reports it was stopped
        )
        assert printed == 0
        assert stopped is True


class TestPrintPdf:
    """Printing to a PDF file exercises the real render-and-paint path."""

    def _printer(self, out):
        printer = QPrinter()
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(str(out))
        return printer

    def test_a_one_page_document_prints_one_page(self, app, tmp_path):
        src = sample_pdf(tmp_path, "invoice_01_centered")
        out = tmp_path / "out.pdf"
        assert print_pdf(src, self._printer(out)) is True
        with pymupdf.open(out) as printed:
            assert printed.page_count == 1

    def test_every_page_of_a_multipage_document_prints(self, app, tmp_path):
        src = multipage_pdf(tmp_path, "doc.pdf", "invoice_01_centered", "other_01_quotation")
        out = tmp_path / "out.pdf"
        assert print_pdf(src, self._printer(out)) is True
        with pymupdf.open(out) as printed:
            assert printed.page_count == 2

    def test_stopping_partway_prints_only_the_pages_sent(self, app, tmp_path):
        src = multipage_pdf(tmp_path, "doc.pdf", "invoice_01_centered", "other_01_quotation")
        out = tmp_path / "out.pdf"
        calls = {"n": 0}

        def should_stop():
            calls["n"] += 1
            return calls["n"] > 1  # allow page 1, stop before page 2

        assert print_pdf(src, self._printer(out), should_stop=should_stop) is False
        with pymupdf.open(out) as printed:
            assert printed.page_count == 1


class TestWindow:
    @pytest.fixture
    def window(self, app):
        window = PrintWindow()
        yield window
        window.close()
        window.deleteLater()
        app.processEvents()

    def test_it_builds(self, window):
        assert window.windowTitle() == "PDF print"

    def test_adding_files_queues_them(self, window, tmp_path):
        a = sample_pdf(tmp_path, "invoice_01_centered")
        b = sample_pdf(tmp_path, "other_01_quotation")
        window._add_paths([a, b])
        assert window.file_list.count() == 2
        assert window._queued_files() == [a, b]

    def test_a_file_is_not_queued_twice(self, window, tmp_path):
        a = sample_pdf(tmp_path, "invoice_01_centered")
        window._add_paths([a])
        window._add_paths([a])
        assert window.file_list.count() == 1

    def test_clear_empties_the_queue(self, window, tmp_path):
        window._add_paths([sample_pdf(tmp_path, "invoice_01_centered")])
        window._clear()
        assert window.file_list.count() == 0

    def test_stop_starts_disabled(self, window):
        assert window.stop_button.isEnabled() is False

    def test_requesting_stop_sets_the_flag(self, window):
        window._request_stop()
        assert window._check_stop() is True
