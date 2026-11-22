"""Generate PDFs for exercising the print tool's orientation handling.

Each page is labelled with its intended orientation and size, and carries a
"TOP" marker along its top edge, so a misprint -- shrunk, rotated or upside
down -- is obvious at a glance. Point pdf-print at the output folder, turn
"Match each page's orientation to the PDF" on, and print (to Microsoft Print to
PDF is fine) to check the result.

Usage:
    poetry run python -m tools.make_print_samples [OUTPUT_DIR]

OUTPUT_DIR defaults to ./print_samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

# Page sizes in points (1 pt = 1/72 inch).
A4_PORTRAIT = (595, 842)
A4_LANDSCAPE = (842, 595)
A3_LANDSCAPE = (1191, 842)

# filename -> list of (size, label) pages.
SAMPLES: dict[str, list[tuple[tuple[int, int], str]]] = {
    "portrait_only.pdf": [(A4_PORTRAIT, f"Page {n}") for n in range(1, 4)],
    "landscape_only.pdf": [(A4_LANDSCAPE, f"Page {n}") for n in range(1, 4)],
    # The important one: orientation changes page to page.
    "mixed_orientation.pdf": [
        (A4_PORTRAIT, "Page 1"),
        (A4_LANDSCAPE, "Page 2"),
        (A4_PORTRAIT, "Page 3"),
        (A4_LANDSCAPE, "Page 4"),
    ],
    "a3_landscape.pdf": [(A3_LANDSCAPE, f"Page {n}") for n in range(1, 3)],
    "mixed_sizes.pdf": [
        (A4_PORTRAIT, "Page 1 (A4)"),
        (A4_LANDSCAPE, "Page 2 (A4)"),
        (A3_LANDSCAPE, "Page 3 (A3)"),
    ],
}


def _add_page(document: pymupdf.Document, size: tuple[int, int], label: str) -> None:
    width, height = size
    page = document.new_page(width=width, height=height)
    orientation = "LANDSCAPE" if width > height else "PORTRAIT"

    margin = 24
    border = pymupdf.Rect(margin, margin, width - margin, height - margin)
    page.draw_rect(border, width=2)

    # A marker along the top edge, so a rotated or upside-down print is obvious.
    page.draw_line((margin, margin + 40), (width - margin, margin + 40), width=1)
    page.insert_textbox(
        pymupdf.Rect(margin, margin + 6, width - margin, margin + 38),
        "TOP EDGE",
        fontsize=18,
        align=pymupdf.TEXT_ALIGN_CENTER,
    )

    page.insert_textbox(
        border,
        f"{label}\n{orientation}\n{width} x {height} pt",
        fontsize=44,
        align=pymupdf.TEXT_ALIGN_CENTER,
    )


def build_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, pages in SAMPLES.items():
        document = pymupdf.open()
        for size, label in pages:
            _add_page(document, size, label)
        path = out_dir / filename
        document.save(path)
        document.close()
        written.append(path)
    return written


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("print_samples")
    out_dir = out_dir.resolve()
    for path in build_all(out_dir):
        print("wrote", path)
    print(f"\n{len(SAMPLES)} files in {out_dir}")


if __name__ == "__main__":
    main()
