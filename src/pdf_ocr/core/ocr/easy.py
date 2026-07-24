"""EasyOCR engine.

Two decisions here are driven by the fact that this ends up inside a PyInstaller
executable that gets copied between PCs:

* The recognition models are loaded from a directory next to the executable and
  downloading is disabled once frozen. EasyOCR otherwise fetches ~100MB from the
  internet on first use, which fails on an offline machine or behind a corporate
  proxy -- and fails at the worst possible moment, in production, on someone
  else's PC.
* Constructing the ``Reader`` is the expensive part, so it is built lazily and
  reused. A batch of 100 files must not pay for it 100 times.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pdf_ocr.core.types import TextBlock

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ("ja", "en")
"""Japanese plus English.

EasyOCR only allows languages that share a recognition model to be combined, so
Japanese and Korean cannot both be loaded into one reader. Japanese is the right
half of that trade: the Japanese model still reads Latin characters, and Korean
invoices print the English word "Invoice", so they remain classifiable. The
Hangul body text comes out as nonsense, but nothing downstream reads it.
"""

MODEL_DIR_ENV = "PDF_OCR_MODEL_DIR"


def default_model_dir() -> Path:
    """Where to look for the recognition models.

    Beside the executable once frozen, so the whole thing stays copyable as a
    single folder; under the project during development.
    """
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "models"
    return Path(__file__).resolve().parents[4] / "models"


class EasyOcrEngine:
    """Recognises page images with EasyOCR."""

    name = "easyocr"

    def __init__(
        self,
        languages: tuple[str, ...] = DEFAULT_LANGUAGES,
        gpu: bool = False,
        model_dir: Path | None = None,
        download_enabled: bool | None = None,
        verbose: bool = False,
    ) -> None:
        self.languages = languages
        self.gpu = gpu
        self.model_dir = model_dir or default_model_dir()
        # EasyOCR draws progress bars on stdout. The command-line tool writes
        # its JSON result there for Power Automate to parse, so anything else
        # arriving on that stream corrupts the output.
        self.verbose = verbose
        # Allowed during development so a fresh checkout works without a manual
        # download step, refused once frozen so a deployed copy can never depend
        # on network access it may not have.
        self.download_enabled = (
            download_enabled
            if download_enabled is not None
            else not getattr(sys, "frozen", False)
        )
        self._reader = None

    @property
    def reader(self):
        """The underlying EasyOCR reader, built on first use."""
        if self._reader is None:
            import easyocr  # imported lazily; it pulls in torch

            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "loading EasyOCR models for %s from %s",
                "+".join(self.languages),
                self.model_dir,
            )
            self._reader = easyocr.Reader(
                list(self.languages),
                gpu=self.gpu,
                model_storage_directory=str(self.model_dir),
                download_enabled=self.download_enabled,
                verbose=self.verbose,
            )
        return self._reader

    def read(self, image: np.ndarray) -> list[TextBlock]:
        """Recognise an RGB image, returning blocks in pixel coordinates."""
        blocks: list[TextBlock] = []
        for polygon, text, confidence in self.reader.readtext(image):
            if not text.strip():
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            # EasyOCR returns a quadrilateral, which can be tilted on a scanned
            # page. Only the vertical position matters downstream -- for the
            # "is this near the top" test -- so the enclosing box is enough.
            blocks.append(
                TextBlock(
                    text=text,
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=float(confidence),
                )
            )
        return blocks
