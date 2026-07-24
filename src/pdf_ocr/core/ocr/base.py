"""The interface every OCR engine implements.

Engines return blocks in **image pixel coordinates**. Converting those to PDF
points is :mod:`pdf_ocr.core.extract`'s job, because only it knows the DPI the
page was rendered at. Keeping the conversion out of the engines means a new
engine only has to know how to read an image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from pdf_ocr.core.types import TextBlock


@runtime_checkable
class OcrEngine(Protocol):
    """Reads text out of a rasterised page.

    Implementations are expected to be expensive to construct and cheap to call:
    loading recognition models takes seconds, so one instance is built per
    process and reused across every file in a batch.
    """

    name: str

    def read(self, image: np.ndarray) -> list[TextBlock]:
        """Recognise text in an RGB image, with boxes in pixel coordinates."""
        ...
