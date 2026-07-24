"""Run the classifier over the sample corpus and report where it is wrong.

This is the feedback loop the rules are tuned in: change a weight or a threshold,
re-run, and see which layouts moved. Because every sample declares the verdict it
should receive, the output is a scoreboard rather than a wall of numbers.

Requires the corpus to exist -- generate it with ``python -m tools.sample_pdfs``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from pdf_ocr import DEFAULT_RULES_PATH
from pdf_ocr.core.extract import extract_first_page
from pdf_ocr.core.ocr.easy import EasyOcrEngine
from pdf_ocr.core.score import RuleSet, score_page
from pdf_ocr.core.types import Source, Verdict
from tools.sample_pdfs import DEFAULT_OUTPUT_DIR, SAMPLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-r", "--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="skip samples that need OCR, for a fast text-layer-only run",
    )
    parser.add_argument("--hits", action="store_true", help="show every rule that fired")
    parser.add_argument("-v", "--verbose", action="store_true")
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    rules = RuleSet.load(arguments.rules)
    engine = None if arguments.no_ocr else EasyOcrEngine()

    print(f"{'sample':<36} {'src':<5} {'diff':<6} {'score':>6}  {'verdict':<13} expected")
    print("-" * 92)

    correct = 0
    considered = 0
    failures: list[tuple[str, Verdict, Verdict, str]] = []

    for sample in SAMPLES:
        path = arguments.dir / f"{sample.name}.pdf"
        if not path.exists():
            print(f"{sample.name:<36} MISSING -- run 'python -m tools.sample_pdfs'")
            continue
        if arguments.no_ocr and sample.scan is not None:
            continue

        started = time.perf_counter()
        page = extract_first_page(path, engine)
        result = score_page(page, rules)
        elapsed = time.perf_counter() - started

        considered += 1
        ok = result.verdict is sample.expected
        correct += ok
        if not ok:
            failures.append((sample.name, result.verdict, sample.expected, sample.note))

        source = "text" if page.source is Source.TEXT_LAYER else "ocr"
        mark = " " if ok else "X"
        print(
            f"{mark}{sample.name:<35} {source:<5} {sample.difficulty:<6}"
            f" {result.score:>6.0f}  {result.verdict.value:<13} {sample.expected.value:<13}"
            f" {elapsed:>5.1f}s"
        )

        if arguments.hits:
            for hit in result.hits:
                print(
                    f"      {hit.weight:>+5.0f}  {hit.rule_id:<24}"
                    f" {hit.pattern!r} -> {hit.match.matched_text!r}"
                    f" ({hit.match.kind.value})"
                )

    print("-" * 92)
    print(f"{correct}/{considered} correct")

    if failures:
        print("\nmisclassified:")
        for name, got, expected, note in failures:
            print(f"  {name}: got {got.value}, expected {expected.value}")
            print(f"      {note}")


if __name__ == "__main__":
    main()
