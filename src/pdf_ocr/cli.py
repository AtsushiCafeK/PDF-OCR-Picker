"""Command-line entry point -- the thing that becomes the executable.

Shaped by how Power Automate calls a program: it can read an exit code, and it
can read stdout. So a single-file classification reports its verdict both ways,
and stdout carries nothing but JSON. Anything diagnostic goes to stderr, which
is also why the OCR engine's progress bars are switched off.

``batch`` exists because of what OCR costs. Loading the recognition models takes
seconds, and a flow that invokes the executable once per file pays that for
every file. Handing it a folder pays it once.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from collections import Counter
from enum import IntEnum
from pathlib import Path

from pdf_ocr import __version__, resolve_config_path, resolve_rules_path
from pdf_ocr.core.config import Config
from pdf_ocr.core.extract import (
    DEFAULT_DPI,
    MIN_TEXT_LAYER_CHARS,
    ExtractionError,
    extract_first_page,
)
from pdf_ocr.core.mover import DEFAULT_FOLDER_NAMES, AuditLog, Routing, move_file
from pdf_ocr.core.ocr.easy import EasyOcrEngine
from pdf_ocr.core.score import RuleError, RuleSet, score_page
from pdf_ocr.core.types import Source, Verdict

logger = logging.getLogger("pdf_ocr")


class ExitCode(IntEnum):
    """Process exit codes.

    A flow can branch on these without parsing anything, which in Power Automate
    is markedly less work than reading JSON back out of stdout.
    """

    INVOICE = 0
    NOT_INVOICE = 1
    NEEDS_REVIEW = 2
    ERROR = 9


VERDICT_EXIT_CODES = {
    Verdict.INVOICE: ExitCode.INVOICE,
    Verdict.OTHER: ExitCode.NOT_INVOICE,
    Verdict.NEEDS_REVIEW: ExitCode.NEEDS_REVIEW,
}


def emit(payload: dict, indent: int | None = None) -> None:
    """Write one JSON object to stdout, in ASCII.

    Japanese is escaped as \\uXXXX rather than written as UTF-8. It looks worse
    to a human, but stdout is a machine interface here: on a Japanese Windows
    console the code page is cp932, and UTF-8 bytes sent through it arrive as
    mojibake -- 請求書 becomes unrecoverable noise, taking the matched text with
    it. Escapes survive any code page and every JSON parser decodes them back.
    The JSONL log written by --out stays readable UTF-8, because that file is
    for people.
    """
    print(json.dumps(payload, ensure_ascii=True, indent=indent))


def configure_streams() -> None:
    """Make the output streams tolerate characters the console cannot show.

    Without this a supplier name in a filename is enough to end a batch run with
    a UnicodeEncodeError from the logging call, rather than a classified folder.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # Redirected to something that is not a real text stream: nothing to
        # configure, and not a reason to refuse to run.
        with contextlib.suppress(ValueError, AttributeError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


def owns_console() -> bool:
    """Whether this process created the console window it is writing to.

    Distinguishes a double-click in Explorer, where the window is destroyed the
    moment the process ends, from a shell invocation, where whatever was printed
    stays on screen. Only in the first case is pausing a kindness rather than a
    hang.
    """
    if sys.platform != "win32" or not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        import ctypes

        buffer = (ctypes.c_uint * 2)()
        # A console shared with a parent shell lists at least two processes.
        return ctypes.windll.kernel32.GetConsoleProcessList(buffer, 2) == 1
    except Exception:
        return False


GUI_EXECUTABLE_MARKER = "gui"


def launched_as_gui() -> bool:
    """Whether this process is the windowed build.

    Both builds run this same module -- one bundle rather than two, because a
    second one would duplicate PyTorch and the OCR models. They are told apart
    by the name of the executable, which is set in the spec file.
    """
    if not getattr(sys, "frozen", False):
        return False
    return GUI_EXECUTABLE_MARKER in Path(sys.executable).stem.lower()


def load_config(arguments: argparse.Namespace) -> Config:
    """The installation's default folders, so an explicit argument is optional.

    A configured default is what lets a Power Automate step be just
    ``pdf-sorter.exe batch``; an explicit argument always wins over it.
    """
    return Config.load(resolve_config_path(getattr(arguments, "config", None)))


def build_engine(arguments: argparse.Namespace) -> EasyOcrEngine | None:
    """One engine per process, or none at all if OCR is switched off."""
    if arguments.no_ocr:
        return None
    return EasyOcrEngine(
        gpu=arguments.gpu,
        model_dir=arguments.model_dir,
        verbose=False,
    )


def classify_one(path: Path, engine, rules: RuleSet, arguments: argparse.Namespace):
    """Extract and score one file. Returns ``(result, elapsed)``."""
    started = time.perf_counter()
    page = extract_first_page(
        path,
        engine,
        dpi=arguments.dpi,
        min_chars=arguments.min_text_chars,
        force_ocr=arguments.force_ocr,
    )
    result = score_page(page, rules)
    return result, time.perf_counter() - started


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def command_classify(arguments: argparse.Namespace) -> int:
    """Classify one file, reporting on stdout and through the exit code."""
    rules = RuleSet.load(resolve_rules_path(arguments.rules))
    config = load_config(arguments)
    move_to = arguments.move_to or config.output_dir
    log = arguments.log or config.log
    engine = build_engine(arguments)

    try:
        result, elapsed = classify_one(arguments.path, engine, rules, arguments)
    except ExtractionError as error:
        emit({"source": str(arguments.path), "error": str(error)})
        logger.error("%s", error)
        return ExitCode.ERROR

    destination = None
    if move_to is not None:
        routing = Routing(move_to)
        destination = move_file(
            arguments.path, routing.directory_for(result.verdict), arguments.dry_run
        )

    with AuditLog(log) as audit:
        entry = audit.record(
            arguments.path,
            result,
            destination=destination,
            dry_run=arguments.dry_run,
            elapsed=elapsed,
        )

    emit(entry)
    return VERDICT_EXIT_CODES[result.verdict]


def command_batch(arguments: argparse.Namespace) -> int:
    """Classify every PDF in a folder, paying the model load once."""
    rules = RuleSet.load(resolve_rules_path(arguments.rules))
    config = load_config(arguments)
    directory = arguments.directory or config.input_dir
    if directory is None:
        emit({"error": "no folder given and no input_dir configured"})
        logger.error("give a folder, or set input_dir in config.yaml")
        return ExitCode.ERROR
    move_to = arguments.move_to or config.output_dir
    out = arguments.out or config.log

    engine = build_engine(arguments)
    paths = sorted(directory.rglob("*.pdf") if arguments.recursive
                   else directory.glob("*.pdf"))
    if not paths:
        logger.warning("no PDFs found in %s", directory)

    if getattr(arguments, "progress", False):
        # Imported lazily: only a --progress run needs Qt, and the headless path
        # (the one Power Automate uses) must still work where PySide6 is absent.
        from pdf_ocr.progress import run_batch_with_progress

        summary = run_batch_with_progress(
            paths, engine, rules, arguments, directory=directory, move_to=move_to, out=out
        )
    else:
        summary = run_batch(
            paths, engine, rules, arguments, directory=directory, move_to=move_to, out=out
        )

    emit(summary)
    return ExitCode.ERROR if summary["errors"] else ExitCode.INVOICE


def run_batch(
    paths: list[Path],
    engine: EasyOcrEngine | None,
    rules: RuleSet,
    arguments: argparse.Namespace,
    *,
    directory: Path,
    move_to: Path | None,
    out: Path | None,
    on_file=None,
    should_stop=None,
) -> dict:
    """Classify every path, filing and logging as it goes, and return a summary.

    Factored out of the command so both the headless run and the progress-window
    run share exactly one classification loop -- the window is only a view onto
    this. ``on_file(processed, total, counts)`` is called after each document so
    a caller can show progress; ``should_stop()`` is polled before each so a
    caller can cancel, and a document already being read still finishes.
    """
    routing = Routing(move_to) if move_to else None
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    processed = 0
    stopped = False

    with AuditLog(out) as audit:
        for path in paths:
            if should_stop is not None and should_stop():
                stopped = True
                break
            try:
                result, elapsed = classify_one(path, engine, rules, arguments)
            except ExtractionError as error:
                counts["error"] += 1
                logger.error("%s", error)
                audit.record(path, None, error=str(error), dry_run=arguments.dry_run)
                processed += 1
                if on_file is not None:
                    on_file(processed, len(paths), counts)
                continue

            destination = None
            if routing is not None:
                destination = move_file(
                    path, routing.directory_for(result.verdict), arguments.dry_run
                )

            counts[result.verdict.value] += 1
            counts[f"read:{result.source.value}"] += 1
            audit.record(
                path,
                result,
                destination=destination,
                dry_run=arguments.dry_run,
                elapsed=elapsed,
            )
            logger.info(
                "%s -> %s (%.0f) in %.1fs", path.name, result.verdict.value,
                result.score, elapsed,
            )
            processed += 1
            if on_file is not None:
                on_file(processed, len(paths), counts)

    return {
        "directory": str(directory),
        "files": len(paths),
        "processed": processed,
        "stopped": stopped,
        "elapsed": round(time.perf_counter() - started, 2),
        "dry_run": arguments.dry_run,
        "verdicts": {verdict.value: counts[verdict.value] for verdict in Verdict},
        "errors": counts["error"],
        "read_from": {
            source.value: counts[f"read:{source.value}"] for source in Source
        },
        "log": str(out) if out else None,
    }


def command_gui(arguments: argparse.Namespace) -> int:
    """Open the tuning GUI.

    A subcommand rather than a second executable: a separate bundle would carry
    its own copy of PyTorch and the OCR models, roughly doubling what has to be
    distributed. Sharing one bundle costs only Qt.
    """
    # Imported here, not at module scope, so a batch run does not pay for
    # loading Qt -- and so the command-line tool still works on a machine where
    # PySide6 is missing.
    from pdf_ocr.gui import main as gui_main

    return gui_main(arguments.rules, arguments.config)


def command_diag(arguments: argparse.Namespace) -> int:
    """Report what a folder is made of, without moving anything.

    Answers the question the whole design has been assuming an answer to: how
    many of these documents carry a text layer, and therefore how much of a run
    is going to cost seconds of OCR rather than milliseconds.
    """
    rules = RuleSet.load(resolve_rules_path(arguments.rules))
    config = load_config(arguments)
    directory = arguments.directory or config.input_dir
    if directory is None:
        emit({"error": "no folder given and no input_dir configured"})
        logger.error("give a folder, or set input_dir in config.yaml")
        return ExitCode.ERROR
    engine = build_engine(arguments)

    paths = sorted(directory.rglob("*.pdf") if arguments.recursive
                   else directory.glob("*.pdf"))

    rows: list[dict] = []
    for path in paths:
        try:
            result, elapsed = classify_one(path, engine, rules, arguments)
        except ExtractionError as error:
            rows.append({"file": path.name, "error": str(error)})
            continue
        rows.append(
            {
                "file": path.name,
                "read_from": result.source.value,
                "score": result.score,
                "verdict": result.verdict.value,
                "elapsed": round(elapsed, 2),
                "top_hits": [hit.rule_id for hit in result.positive_hits[:3]],
            }
        )

    scored = [row for row in rows if "score" in row]
    text_layer = sum(1 for row in scored if row["read_from"] == Source.TEXT_LAYER.value)
    emit(
        {
            "directory": str(directory),
            "files": len(paths),
            "with_text_layer": text_layer,
            "needing_ocr": len(scored) - text_layer,
            "errors": len(rows) - len(scored),
            "scores": sorted(row["score"] for row in scored),
            "files_detail": rows,
        },
        indent=2 if arguments.pretty else None,
    )
    return ExitCode.INVOICE


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rules", type=Path, help="path to a rules.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        help="path to a config.yaml of default folders (default: beside the exe)",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=MIN_TEXT_LAYER_CHARS,
        help="below this many characters, a text layer is treated as absent",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="ignore the text layer even when it is usable",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="never run OCR; scanned pages come back empty",
    )
    parser.add_argument("--gpu", action="store_true", help="use CUDA if available")
    parser.add_argument(
        "--model-dir", type=Path, help="where the OCR models live (default: beside the exe)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")


def add_move_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--move-to",
        type=Path,
        help=(
            "root folder to file documents under; without it nothing is moved. "
            "Subfolders: " + ", ".join(DEFAULT_FOLDER_NAMES.values())
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the moves that would be made without making them",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-sorter",
        description="Classify PDFs as invoices and file them.",
        # Shown to anyone who double-clicks the executable, so it has to answer
        # "what is this and how do I use it" rather than just list flags.
        epilog=(
            "Examples:\n"
            "  pdf-sorter batch C:\\in --out result.jsonl --dry-run\n"
            "      classify a folder and report what it would do, moving nothing\n"
            "\n"
            "  pdf-sorter batch C:\\in --move-to C:\\sorted --out result.jsonl\n"
            "      the same, but actually file the documents\n"
            "\n"
            "  pdf-sorter classify C:\\in\\one.pdf\n"
            "      a single file; the exit code is the verdict\n"
            "      0 invoice   1 not an invoice   2 needs review   9 error\n"
            "\n"
            "  pdf-sorter diag C:\\in\n"
            "      report what a folder is made of; moves nothing\n"
            "\n"
            "  pdf-sorter gui\n"
            "      open the tuning window: see why each document scored what it\n"
            "      did, adjust keywords, and preview how a folder would sort\n"
            "\n"
            "Prefer 'batch' over calling 'classify' in a loop: loading the OCR\n"
            "models takes several seconds, and batch pays that once per run\n"
            "rather than once per file.\n"
            "\n"
            "Keywords and thresholds come from rules.yaml beside this executable;\n"
            "edit it to handle a supplier whose invoices are being missed.\n"
            "\n"
            "For a window instead of a command line, run 'pdf-sorter gui'.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser(
        "classify", help="classify one file; the verdict is also the exit code"
    )
    classify.add_argument("path", type=Path)
    classify.add_argument("--log", type=Path, help="append a JSONL audit record here")
    add_move_arguments(classify)
    add_shared_arguments(classify)
    classify.set_defaults(function=command_classify)

    batch = subparsers.add_parser(
        "batch", help="classify a folder, loading the OCR models only once"
    )
    batch.add_argument(
        "directory",
        type=Path,
        nargs="?",
        help="folder to classify (default: input_dir from config.yaml)",
    )
    batch.add_argument("--out", type=Path, help="write one JSONL record per file here")
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument(
        "--progress",
        action="store_true",
        help="show a window with live progress (n/total and the running tally)",
    )
    add_move_arguments(batch)
    add_shared_arguments(batch)
    batch.set_defaults(function=command_batch)

    diag = subparsers.add_parser(
        "diag", help="report what a folder is made of; moves nothing"
    )
    diag.add_argument(
        "directory",
        type=Path,
        nargs="?",
        help="folder to inspect (default: input_dir from config.yaml)",
    )
    diag.add_argument("--recursive", action="store_true")
    diag.add_argument("--pretty", action="store_true")
    diag.set_defaults(dry_run=True, move_to=None)
    add_shared_arguments(diag)
    diag.set_defaults(function=command_diag)

    gui = subparsers.add_parser("gui", help="open the tuning window")
    gui.add_argument("--rules", type=Path, help="path to a rules.yaml")
    gui.add_argument("--config", type=Path, help="path to a config.yaml of default folders")
    gui.add_argument("-v", "--verbose", action="store_true")
    gui.set_defaults(function=command_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_streams()
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv

    if not argv:
        # The windowed build is the same program under a different name, and
        # the people it is for open it by double-clicking. Printing help at
        # them would be doubly useless: it is not what they want, and there is
        # no console to print it to.
        if launched_as_gui():
            argv = ["gui"]
        else:
            # What a double-click on the console build produces. Argparse's
            # usage error is correct and completely useless here, because the
            # window carrying it closes with the process; the reasonable reading
            # of "no arguments" is that someone wants to know what this does.
            parser.print_help()
            if owns_console():
                print("\nPress Enter to close this window.")
                with contextlib.suppress(EOFError, KeyboardInterrupt):
                    input()
            return ExitCode.ERROR

    arguments = parser.parse_args(argv)

    # stderr, always: stdout is reserved for the JSON that the caller parses.
    # A windowed build has neither, and its log goes to the GUI's own pane.
    if sys.stderr is not None:
        logging.basicConfig(
            stream=sys.stderr,
            level=logging.INFO if arguments.verbose else logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
        )

    try:
        return int(arguments.function(arguments))
    except RuleError as error:
        logger.error("%s", error)
        return ExitCode.ERROR
    except (OSError, ExtractionError) as error:
        logger.error("%s", error)
        return ExitCode.ERROR


if __name__ == "__main__":
    sys.exit(main())

