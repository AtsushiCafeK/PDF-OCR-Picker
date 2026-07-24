"""Helpers for building pages without needing a PDF or an OCR engine.

The classification core only ever sees :class:`TextBlock` objects, so the whole
pipeline can be exercised from synthetic text. That keeps the tests fast and,
more usefully, keeps them runnable before any real invoice samples exist.
"""

from __future__ import annotations

import pytest

from pdf_ocr import DEFAULT_RULES_PATH
from pdf_ocr.core.score import RuleSet
from pdf_ocr.core.types import Page, Source, TextBlock

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 800.0
"""Chosen so the top-quarter cutoff lands on a round 200.0."""

LINE_HEIGHT = 20.0


def block(text: str, y: float) -> TextBlock:
    """A block of text whose top edge sits at ``y``."""
    return TextBlock(text=text, bbox=(50.0, y, 500.0, y + LINE_HEIGHT))


def make_page(
    lines: list[tuple[str, float]],
    source: Source = Source.OCR,
    height: float = PAGE_HEIGHT,
) -> Page:
    """Build a page from ``(text, y)`` pairs."""
    return Page(
        number=1,
        width=PAGE_WIDTH,
        height=height,
        blocks=[block(text, y) for text, y in lines],
        source=source,
    )


@pytest.fixture
def rules() -> RuleSet:
    """The rule set that actually ships, not a test-only stand-in.

    Tuning these rules changes behaviour in production, so the tests assert
    against the real file to catch a change that breaks classification.
    """
    return RuleSet.load(DEFAULT_RULES_PATH)


@pytest.fixture
def invoice_lines() -> list[tuple[str, float]]:
    """A plausible invoice: title at the top, amounts and payment details below."""
    return [
        ("請求書", 50.0),
        ("株式会社サンプル 御中", 120.0),
        ("請求金額 ¥110,000", 300.0),
        ("消費税 ¥10,000", 340.0),
        ("お支払期限 2026年8月31日", 400.0),
        ("振込先 みずほ銀行 渋谷支店", 450.0),
        ("登録番号 T1234567890123", 500.0),
    ]


@pytest.fixture
def quotation_lines() -> list[tuple[str, float]]:
    """A quotation, which shares much of an invoice's vocabulary but is not one."""
    return [
        ("御見積書", 50.0),
        ("株式会社サンプル 御中", 120.0),
        ("御見積金額 ¥110,000", 300.0),
        ("消費税 ¥10,000", 340.0),
        ("有効期限 2026年8月31日", 400.0),
    ]
