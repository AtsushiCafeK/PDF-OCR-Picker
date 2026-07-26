"""Filing documents into folders, and keeping a record of having done so.

Moving files is where an otherwise reversible tool becomes destructive, so the
rules here are conservative: never overwrite, never delete, and write down
enough about every move that it can be undone by reading the log backwards.

Dry-run is not an afterthought. Until the thresholds have been tuned against a
real folder, the honest way to run this is to have it say what it *would* do.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pdf_ocr.core.types import VERDICT_LABELS, ScoreResult, Verdict

logger = logging.getLogger(__name__)

DEFAULT_FOLDER_NAMES: dict[Verdict, str] = dict(VERDICT_LABELS)
"""Destination folder per verdict. Identical to the labels shown in the GUI and
the progress window, so a document filed under "Match" is the same word the
window used to describe it -- one vocabulary across the result, the folders and
the progress display."""

MAX_COLLISION_ATTEMPTS = 1000


@dataclass(frozen=True)
class Routing:
    """Where each verdict sends a document."""

    root: Path
    names: dict[Verdict, str] = field(default_factory=lambda: dict(DEFAULT_FOLDER_NAMES))

    def directory_for(self, verdict: Verdict) -> Path:
        return self.root / self.names[verdict]


def unique_destination(directory: Path, name: str) -> Path:
    """A path in ``directory`` that does not already exist.

    Two suppliers both sending ``invoice.pdf`` is ordinary, and silently
    overwriting the first with the second would destroy a document nobody knows
    is missing. Collisions get a numeric suffix instead.
    """
    candidate = directory / name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for attempt in range(2, MAX_COLLISION_ATTEMPTS):
        candidate = directory / f"{stem} ({attempt}){suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"{directory / name}: still colliding after {MAX_COLLISION_ATTEMPTS} attempts"
    )


def move_file(source: Path, directory: Path, dry_run: bool = False) -> Path:
    """Move ``source`` into ``directory``, returning where it went.

    In dry-run mode nothing is created or moved, but the returned path is the
    one that would have been used -- including any collision suffix -- so a
    rehearsal reports what the real run will actually do.
    """
    if dry_run:
        # Resolved against the directory as it stands; good enough to review,
        # and it avoids creating folders during a rehearsal.
        return unique_destination(directory, source.name)

    directory.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(directory, source.name)
    if source.drive == destination.drive:
        # Atomic within a volume: the file is never in both places or neither.
        source.replace(destination)
    else:
        shutil.move(str(source), str(destination))
    return destination


def copy_file(source: Path, directory: Path) -> Path:
    """Copy ``source`` into ``directory``, returning where it went.

    The non-destructive counterpart of :func:`move_file`, for previewing how a
    folder would be sorted. Tuning is an iterative business -- change a weight,
    look again -- and that is only possible if the input survives the run.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(directory, source.name)
    shutil.copy2(source, destination)
    return destination


def clear_sorted_output(root: Path, names: Iterable[str]) -> int:
    """Delete the PDFs previously sorted into ``root``, returning how many.

    Needed because a preview is re-run after every rule change, and a document
    that moves from one verdict to another would otherwise be left sitting in
    both folders -- turning the very thing being inspected into a lie.

    Deliberately narrow: only files directly inside the named verdict folders,
    only ``.pdf``, and the folders themselves are left in place. Nothing else
    under ``root`` is touched, and the source folder is never involved.
    """
    removed = 0
    for name in names:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.glob("*.pdf"):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def count_sorted_output(root: Path, names: Iterable[str]) -> int:
    """How many PDFs :func:`clear_sorted_output` would delete."""
    return sum(
        1
        for name in names
        if (root / name).is_dir()
        for path in (root / name).glob("*.pdf")
        if path.is_file()
    )


class AuditLog:
    """Append-only record of what was decided and what was done about it.

    Serves three purposes at once: it is the evidence for reversing a move, the
    input for tuning thresholds against real documents, and the result file
    Power Automate reads instead of parsing console output.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    def record(
        self,
        source: Path,
        result: ScoreResult | None,
        destination: Path | None = None,
        dry_run: bool = False,
        elapsed: float | None = None,
        error: str | None = None,
    ) -> dict:
        """Write one entry and return it, so the caller can also print it."""
        entry: dict = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": str(source),
            "dry_run": dry_run,
        }
        if error is not None:
            entry["error"] = error
        if elapsed is not None:
            entry["elapsed"] = round(elapsed, 3)
        if destination is not None:
            entry["destination"] = str(destination)
        if result is not None:
            entry.update(
                {
                    "verdict": result.verdict.value,
                    "score": result.score,
                    "text_source": result.source.value,
                    "page": result.page_number,
                    # The hits are what make a decision reviewable months later,
                    # when nobody remembers what the weights were that day.
                    "hits": [
                        {
                            "rule": hit.rule_id,
                            "pattern": hit.pattern,
                            "matched": hit.match.matched_text,
                            "kind": hit.match.kind.value,
                            "weight": hit.weight,
                        }
                        for hit in result.hits
                    ],
                }
            )

        if self._handle is not None:
            self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._handle.flush()
        return entry

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, *exception) -> None:
        self.close()
