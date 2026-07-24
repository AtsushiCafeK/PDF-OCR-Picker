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

from pdf_ocr import DEFAULT_RULES_PATH, __version__
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


def resolve_rules_path(override: Path | None = None) -> Path:
    """Find the rules file, preferring one the operator can edit.

    A copy sitting beside the executable wins over the bundled default, so a
    supplier whose invoices are being missed can be handled by adding a keyword
    on the machine where the problem is, without a rebuild and a redeploy.
    """
    if override is not None:
        return override
    if getattr(sys, "frozen", False):
        beside = Path(sys.executable).parent / "rules.yaml"
        if beside.exists():
            return beside
    return DEFAULT_RULES_PATH


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
    engine = build_engine(arguments)

    try:
        result, elapsed = classify_one(arguments.path, engine, rules, arguments)
    except ExtractionError as error:
        emit({"source": str(arguments.path), "error": str(error)})
        logger.error("%s", error)
        return ExitCode.ERROR

    destination = None
    if arguments.move_to is not None:
        routing = Routing(arguments.move_to)
        destination = move_file(
            arguments.path, routing.directory_for(result.verdict), arguments.dry_run
        )

    with AuditLog(arguments.log) as audit:
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
    engine = build_engine(arguments)
    routing = Routing(arguments.move_to) if arguments.move_to else None

    paths = sorted(arguments.directory.rglob("*.pdf") if arguments.recursive
                   else arguments.directory.glob("*.pdf"))
    if not paths:
        logger.warning("no PDFs found in %s", arguments.directory)

    counts: Counter[str] = Counter()
    started = time.perf_counter()

    with AuditLog(arguments.out) as audit:
        for path in paths:
            try:
                result, elapsed = classify_one(path, engine, rules, arguments)
            except ExtractionError as error:
                counts["error"] += 1
                logger.error("%s", error)
                audit.record(path, None, error=str(error), dry_run=arguments.dry_run)
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

    summary = {
        "directory": str(arguments.directory),
        "files": len(paths),
        "elapsed": round(time.perf_counter() - started, 2),
        "dry_run": arguments.dry_run,
        "verdicts": {
            verdict.value: counts[verdict.value] for verdict in Verdict
        },
        "errors": counts["error"],
        "read_from": {
            source.value: counts[f"read:{source.value}"] for source in Source
        },
        "log": str(arguments.out) if arguments.out else None,
    }
    emit(summary)
    return ExitCode.ERROR if counts["error"] else ExitCode.INVOICE


def command_diag(arguments: argparse.Namespace) -> int:
    """Report what a folder is made of, without moving anything.

    Answers the question the whole design has been assuming an answer to: how
    many of these documents carry a text layer, and therefore how much of a run
    is going to cost seconds of OCR rather than milliseconds.
    """
    rules = RuleSet.load(resolve_rules_path(arguments.rules))
    engine = build_engine(arguments)

    paths = sorted(arguments.directory.rglob("*.pdf") if arguments.recursive
                   else arguments.directory.glob("*.pdf"))

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
            "directory": str(arguments.directory),
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
            "Prefer 'batch' over calling 'classify' in a loop: loading the OCR\n"
            "models takes several seconds, and batch pays that once per run\n"
            "rather than once per file.\n"
            "\n"
            "Keywords and thresholds come from rules.yaml beside this executable;\n"
            "edit it to handle a supplier whose invoices are being missed.\n"
            "\n"
            "This is the command-line tool. The tuning GUI is a separate\n"
            "development program and is not part of this executable.\n"
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
    batch.add_argument("directory", type=Path)
    batch.add_argument("--out", type=Path, help="write one JSONL record per file here")
    batch.add_argument("--recursive", action="store_true")
    add_move_arguments(batch)
    add_shared_arguments(batch)
    batch.set_defaults(function=command_batch)

    diag = subparsers.add_parser(
        "diag", help="report what a folder is made of; moves nothing"
    )
    diag.add_argument("directory", type=Path)
    diag.add_argument("--recursive", action="store_true")
    diag.add_argument("--pretty", action="store_true")
    diag.set_defaults(dry_run=True, move_to=None)
    add_shared_arguments(diag)
    diag.set_defaults(function=command_diag)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_streams()
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv

    if not argv:
        # What a double-click produces. Argparse's usage error is correct and
        # completely useless here, because the window carrying it closes with
        # the process; the reasonable reading of "no arguments" is that someone
        # wants to know what this program does.
        parser.print_help()
        if owns_console():
            print("\nPress Enter to close this window.")
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input()
        return ExitCode.ERROR

    arguments = parser.parse_args(argv)

    # stderr, always: stdout is reserved for the JSON that the caller parses.
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

