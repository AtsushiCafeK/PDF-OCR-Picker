"""A small window that shows a batch run's progress.

The command-line tool is otherwise invisible: started from Power Automate or a
double-click, there is no sign of how far along it is. ``batch --progress`` opens
this window so a person can watch -- n of total, and the running tally of what
each document was judged to be.

The classification itself is unchanged; this only observes it. The work runs on
a thread so the window stays responsive, and the same run_batch loop drives both
this and the headless path, so what the window reports is exactly what gets
filed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from pdf_ocr.core.ocr.easy import EasyOcrEngine
from pdf_ocr.core.score import RuleSet
from pdf_ocr.core.types import VERDICT_LABELS, Verdict


class _Worker(QObject):
    """Runs the batch on a thread and reports each file as it finishes."""

    progressed = Signal(int, int, dict)  # processed, total, counts
    finished = Signal(dict)  # summary

    def __init__(self, run) -> None:
        super().__init__()
        self._run = run
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    @Slot()
    def run(self) -> None:
        summary = self._run(
            on_file=lambda done, total, counts: self.progressed.emit(
                done, total, dict(counts)
            ),
            should_stop=lambda: self._stop,
        )
        self.finished.emit(summary)


class ProgressWindow(QWidget):
    """Status line, an n/total counter, and a per-verdict tally."""

    stop_requested = Signal()

    def __init__(self, total: int) -> None:
        super().__init__()
        self.setWindowTitle("PDF sorter")
        self.setMinimumWidth(280)
        self._total = total
        self._done = False

        layout = QVBoxLayout(self)
        self.status_label = QLabel("実行中")
        self.status_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.count_label = QLabel(f"0/{total}")
        self.count_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        layout.addWidget(self.status_label)
        layout.addWidget(self.count_label)

        self.verdict_labels: dict[Verdict, QLabel] = {}
        for verdict in Verdict:
            label = QLabel(f"{VERDICT_LABELS[verdict]} - 0")
            layout.addWidget(label)
            self.verdict_labels[verdict] = label

    @Slot(int, int, dict)
    def update_progress(self, processed: int, total: int, counts: dict) -> None:
        self.count_label.setText(f"{processed}/{total}")
        for verdict, label in self.verdict_labels.items():
            label.setText(f"{VERDICT_LABELS[verdict]} - {counts.get(verdict.value, 0)}")

    def mark_done(self, summary: dict) -> None:
        self._done = True
        self.status_label.setText("停止しました" if summary.get("stopped") else "完了")
        self.count_label.setText(f"{summary.get('processed', 0)}/{self._total}")
        verdicts = summary.get("verdicts", {})
        for verdict, label in self.verdict_labels.items():
            label.setText(f"{VERDICT_LABELS[verdict]} - {verdicts.get(verdict.value, 0)}")

    def closeEvent(self, event) -> None:
        # While the run is going, a close is a request to stop, not an instant
        # kill: the document being read still finishes, so no half-written copy
        # is left behind. The window stays until the worker reports back.
        if not self._done:
            self.status_label.setText("停止中... (現在の1件が終わったら停止)")
            self.stop_requested.emit()
            event.ignore()
        else:
            event.accept()


def run_batch_with_progress(
    paths: list[Path],
    engine: EasyOcrEngine | None,
    rules: RuleSet,
    arguments: argparse.Namespace,
    *,
    directory: Path,
    move_to: Path | None,
    out: Path | None,
) -> dict:
    """Run a batch behind a progress window, returning the same summary as the
    headless path once the window is closed."""
    from pdf_ocr.cli import run_batch

    def run(on_file, should_stop):
        return run_batch(
            paths, engine, rules, arguments,
            directory=directory, move_to=move_to, out=out,
            on_file=on_file, should_stop=should_stop,
        )

    application = QApplication.instance() or QApplication([])
    # Completion should stay on screen -- the whole point is that a person sees
    # the run finish -- so the app does not quit when the window merely reports
    # done; it quits when the person closes the window.
    application.setQuitOnLastWindowClosed(True)

    window = ProgressWindow(len(paths))
    holder: dict = {"summary": _empty_summary(directory, len(paths), out)}

    thread = QThread()
    worker = _Worker(run)
    worker.moveToThread(thread)
    window.stop_requested.connect(worker.request_stop)
    thread.started.connect(worker.run)

    def on_finished(summary: dict) -> None:
        holder["summary"] = summary
        window.mark_done(summary)
        thread.quit()
        # A close mid-run was a stop request; now that the run has ended, honour
        # it by closing for real rather than leaving a "停止中" window hanging.
        if summary.get("stopped"):
            window.close()

    worker.progressed.connect(window.update_progress)
    worker.finished.connect(on_finished)
    thread.finished.connect(worker.deleteLater)

    window.show()
    thread.start()
    application.exec()

    if thread.isRunning():
        thread.quit()
        thread.wait()
    return holder["summary"]


def _empty_summary(directory: Path, total: int, out: Path | None) -> dict:
    """A summary to fall back on if the window is closed before anything ran."""
    return {
        "directory": str(directory),
        "files": total,
        "processed": 0,
        "stopped": True,
        "elapsed": 0.0,
        "dry_run": False,
        "verdicts": {v.value: 0 for v in Verdict},
        "errors": 0,
        "read_from": {},
        "log": str(out) if out else None,
    }
