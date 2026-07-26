"""Machine-local defaults for where documents come from and go to.

Separate from rules.yaml on purpose. rules.yaml is *what an invoice looks like*
-- the same on every machine, and committed. This is *where this particular
installation reads and files* -- different on every machine, and never
committed. Keeping them apart means a site can set its folders once without
touching the tuned rules, and a tuned rules.yaml can be copied between machines
without dragging one site's paths along.

Both front ends read it: the GUI remembers the folders a person picked, and the
command line falls back to them, so a Power Automate step can be just
``pdf-sorter.exe batch`` with the folders already known.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_FIELDS = ("input_dir", "output_dir", "log")


@dataclass
class Config:
    """Default input/output locations for one installation."""

    input_dir: Path | None = None
    """Folder to classify when none is given on the command line."""

    output_dir: Path | None = None
    """Root to file documents under (the CLI's --move-to, the GUI's sorted-copy
    destination)."""

    log: Path | None = None
    """Where to append the JSONL audit log (the CLI's --out)."""

    path: Path | None = None
    """Where this was loaded from and will be saved to. Not written into the
    file itself."""

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Read a config file. A missing file is not an error -- it just means
        nothing has been configured yet, so every default is ``None``."""
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            # A malformed config should not take the tool down; treat it as
            # empty and let the caller fall back to explicit arguments.
            return cls(path=path)
        values = {
            field: (Path(data[field]) if data.get(field) else None) for field in _FIELDS
        }
        return cls(path=path, **values)

    def save(self) -> None:
        """Write the set fields back, so a folder chosen once is remembered.

        Only non-empty fields are written, and comments in an existing file are
        not preserved -- this file is generated, not hand-authored, unlike
        rules.yaml.
        """
        if self.path is None:
            raise ValueError("Config has no path to save to")
        data = {
            field: str(value)
            for field in _FIELDS
            if (value := getattr(self, field)) is not None
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
