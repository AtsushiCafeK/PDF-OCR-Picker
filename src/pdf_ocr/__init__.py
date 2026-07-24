"""Classify PDFs as invoices and file them accordingly."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).parent

DEFAULT_RULES_PATH = PACKAGE_ROOT / "rules.yaml"
"""The rules shipped with the package.

The command-line tool prefers a rules.yaml sitting beside the executable, so a
deployed copy can be tuned without a rebuild; this is the fallback.
"""
