"""Tests for choosing between the text layer and OCR, and for what comes back.

The OCR engine itself is stubbed almost everywhere. What matters here is the
routing decision and the coordinate conversion, and neither needs a real
recogniser -- which keeps these tests running in milliseconds rather than the
seconds per page that EasyOCR costs on a CPU.
"""

from __future__ import annotations

import pymupdf
import pytest

from pdf_ocr.core.extract import (
    MIN_TEXT_LAYER_CHARS,
    ExtractionError,
    extract_first_page,
    extract_page,
    has_usable_text_layer,
    render,
    text_layer_blocks,
)
from pdf_ocr.core.types import Source, TextBlock
from tools.sample_pdfs import SAMPLES, build

PIXELS_PER_POINT = 300 / 72


class FakeEngine:
    """Returns fixed blocks in pixel coordinates, and records that it was used."""

    name = "fake"

    def __init__(self, blocks: list[TextBlock] | None = None) -> None:
        self.blocks = blocks or []
        self.calls = 0

    def read(self, image) -> list[TextBlock]:
        self.calls += 1
        self.image_shape = image.shape
        return self.blocks


def sample_pdf(tmp_path, name: str):
    """Render one catalogue sample to disk."""
    sample = next(s for s in SAMPLES if s.name == name)
    path = tmp_path / f"{name}.pdf"
    document = build(sample)
    document.save(path)
    document.close()
    return path


class TestTextLayerDetection:
    def test_a_generated_invoice_carries_a_text_layer(self, tmp_path):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        with pymupdf.open(path) as document:
            blocks = text_layer_blocks(document[0])
        assert has_usable_text_layer(blocks)
        assert any("請求書" in block.text for block in blocks)

    def test_a_scanned_page_carries_none(self, tmp_path):
        """Flattening to an image is what forces the OCR path."""
        path = sample_pdf(tmp_path, "scan_01_centered")
        with pymupdf.open(path) as document:
            blocks = text_layer_blocks(document[0])
        assert not has_usable_text_layer(blocks)

    def test_a_few_stray_characters_do_not_count_as_a_text_layer(self):
        """A scanning appliance's header must not pass for real content."""
        blocks = [TextBlock(text="Scanned by MFP", bbox=(0, 0, 100, 10))]
        assert not has_usable_text_layer(blocks)

    def test_the_threshold_is_what_decides(self):
        blocks = [TextBlock(text="あ" * MIN_TEXT_LAYER_CHARS, bbox=(0, 0, 100, 10))]
        assert has_usable_text_layer(blocks)


class TestRouting:
    def test_ocr_is_skipped_when_the_text_layer_serves(self, tmp_path):
        """Running OCR on a digital PDF would be slower and less accurate."""
        path = sample_pdf(tmp_path, "invoice_01_centered")
        engine = FakeEngine()
        page = extract_first_page(path, engine)
        assert page.source is Source.TEXT_LAYER
        assert engine.calls == 0

    def test_ocr_runs_when_the_text_layer_is_missing(self, tmp_path):
        path = sample_pdf(tmp_path, "scan_01_centered")
        engine = FakeEngine([TextBlock(text="請求書", bbox=(0, 0, 100, 40))])
        page = extract_first_page(path, engine)
        assert page.source is Source.OCR
        assert engine.calls == 1

    def test_ocr_can_be_forced_for_comparison(self, tmp_path):
        """The debug GUI needs to show what the recogniser makes of a page that
        does have a text layer, so the two can be compared side by side."""
        path = sample_pdf(tmp_path, "invoice_01_centered")
        engine = FakeEngine([TextBlock(text="請求書", bbox=(0, 0, 100, 40))])
        page = extract_first_page(path, engine, force_ocr=True)
        assert page.source is Source.OCR
        assert engine.calls == 1

    def test_a_scan_without_an_engine_yields_an_empty_page(self, tmp_path):
        """Not an error: a text-layer-only run is a legitimate mode."""
        path = sample_pdf(tmp_path, "scan_01_centered")
        page = extract_first_page(path, engine=None)
        assert page.source is Source.OCR
        assert page.blocks == []


class TestCoordinates:
    def test_ocr_boxes_are_converted_from_pixels_to_points(self, tmp_path):
        """Without this the top-quarter rules would compare pixels against a
        cutoff measured in points, and would effectively never fire."""
        path = sample_pdf(tmp_path, "scan_01_centered")
        engine = FakeEngine(
            [
                TextBlock(
                    text="請求書",
                    bbox=(
                        100 * PIXELS_PER_POINT,
                        50 * PIXELS_PER_POINT,
                        200 * PIXELS_PER_POINT,
                        70 * PIXELS_PER_POINT,
                    ),
                )
            ]
        )
        page = extract_first_page(path, engine, dpi=300)
        x0, y0, x1, y1 = page.blocks[0].bbox
        assert x0 == pytest.approx(100, abs=1.0)
        assert y0 == pytest.approx(50, abs=1.0)
        assert x1 == pytest.approx(200, abs=1.0)
        assert y1 == pytest.approx(70, abs=1.0)

    def test_a_converted_box_lands_inside_the_top_quarter(self, tmp_path):
        """The end-to-end consequence of the conversion being right."""
        path = sample_pdf(tmp_path, "scan_01_centered")
        engine = FakeEngine(
            [TextBlock(text="請求書", bbox=(0, 40 * PIXELS_PER_POINT, 400, 300))]
        )
        page = extract_first_page(path, engine, dpi=300)
        assert page.blocks[0].bbox[1] < page.top_quarter_cutoff()

    def test_page_size_is_reported_in_points(self, tmp_path):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        page = extract_first_page(path)
        assert page.width == pytest.approx(595.0, abs=1.0)
        assert page.height == pytest.approx(842.0, abs=1.0)

    def test_a_landscape_page_reports_its_true_orientation(self, tmp_path):
        path = sample_pdf(tmp_path, "invoice_08_landscape")
        page = extract_first_page(path)
        assert page.width > page.height


class TestRendering:
    def test_rendering_produces_an_rgb_image_and_a_scale(self, tmp_path):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        with pymupdf.open(path) as document:
            image, scale = render(document[0], dpi=150)
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert scale == pytest.approx(150 / 72, rel=0.01)

    def test_resolution_follows_the_requested_dpi(self, tmp_path):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        with pymupdf.open(path) as document:
            low, _ = render(document[0], dpi=100)
            high, _ = render(document[0], dpi=200)
        assert high.shape[0] > low.shape[0]


class TestFailures:
    def test_a_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(ExtractionError, match="cannot be opened"):
            extract_first_page(tmp_path / "nope.pdf")

    def test_a_file_that_is_not_a_pdf_is_reported_clearly(self, tmp_path):
        path = tmp_path / "not-a.pdf"
        path.write_bytes(b"this is not a PDF")
        with pytest.raises(ExtractionError, match="cannot be opened"):
            extract_first_page(path)

    def test_a_password_protected_pdf_is_reported_clearly(self, tmp_path):
        """These turn up in practice and must not look like an empty document."""
        path = tmp_path / "locked.pdf"
        document = pymupdf.open()
        document.new_page()
        document.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )
        document.close()
        with pytest.raises(ExtractionError, match="password"):
            extract_first_page(path)

    # extract_first_page also guards against a PDF with zero pages. That case
    # is left untested because PyMuPDF refuses to write such a file, so the
    # fixture cannot be built with the tools to hand.


class TestPageNumbering:
    def test_pages_are_numbered_from_one(self, tmp_path):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        assert extract_first_page(path).number == 1

    def test_extract_page_accepts_a_page_directly(self, tmp_path):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        with pymupdf.open(path) as document:
            page = extract_page(document[0])
        assert page.number == 1
        assert page.source is Source.TEXT_LAYER
