"""Tests for filing and for the audit record.

Weighted towards the destructive edges. Everything else in this project can be
re-run; a move that overwrote a document cannot be undone by running it again.
"""

from __future__ import annotations

import json

import pytest

from pdf_ocr.core.mover import (
    DEFAULT_FOLDER_NAMES,
    AuditLog,
    Routing,
    move_file,
    unique_destination,
)
from pdf_ocr.core.score import score_page
from pdf_ocr.core.types import Verdict
from tests.conftest import make_page


def a_file(directory, name: str, content: bytes = b"pdf") -> object:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


class TestRouting:
    def test_each_verdict_gets_its_own_folder(self, tmp_path):
        routing = Routing(tmp_path)
        directories = {routing.directory_for(verdict) for verdict in Verdict}
        assert len(directories) == len(Verdict)

    def test_the_folders_needing_a_human_sort_to_the_top(self):
        """An operator should see the review pile without going looking for it."""
        assert DEFAULT_FOLDER_NAMES[Verdict.NEEDS_REVIEW].startswith("_")
        assert DEFAULT_FOLDER_NAMES[Verdict.OTHER].startswith("_")

    def test_folder_names_can_be_overridden(self, tmp_path):
        routing = Routing(tmp_path, names=dict.fromkeys(Verdict, "flat"))
        assert routing.directory_for(Verdict.INVOICE) == tmp_path / "flat"


class TestCollisions:
    def test_a_free_name_is_used_as_is(self, tmp_path):
        assert unique_destination(tmp_path, "a.pdf") == tmp_path / "a.pdf"

    def test_a_taken_name_gets_a_suffix(self, tmp_path):
        a_file(tmp_path, "a.pdf")
        assert unique_destination(tmp_path, "a.pdf") == tmp_path / "a (2).pdf"

    def test_suffixes_keep_counting(self, tmp_path):
        a_file(tmp_path, "a.pdf")
        a_file(tmp_path, "a (2).pdf")
        assert unique_destination(tmp_path, "a.pdf") == tmp_path / "a (3).pdf"

    def test_two_suppliers_sending_the_same_filename_both_survive(self, tmp_path):
        """invoice.pdf from two companies is ordinary; losing one is not."""
        first = a_file(tmp_path / "in-a", "invoice.pdf", b"first")
        second = a_file(tmp_path / "in-b", "invoice.pdf", b"second")
        destination = tmp_path / "out"

        move_file(first, destination)
        move_file(second, destination)

        contents = sorted(path.read_bytes() for path in destination.iterdir())
        assert contents == [b"first", b"second"]


class TestMoving:
    def test_the_file_ends_up_in_the_destination(self, tmp_path):
        source = a_file(tmp_path / "in", "a.pdf")
        moved = move_file(source, tmp_path / "out")
        assert moved.exists()
        assert not source.exists()
        assert moved.parent == tmp_path / "out"

    def test_the_destination_folder_is_created(self, tmp_path):
        source = a_file(tmp_path / "in", "a.pdf")
        move_file(source, tmp_path / "out" / "請求書")
        assert (tmp_path / "out" / "請求書").is_dir()

    def test_content_is_preserved(self, tmp_path):
        source = a_file(tmp_path / "in", "a.pdf", b"the original bytes")
        assert move_file(source, tmp_path / "out").read_bytes() == b"the original bytes"


class TestDryRun:
    def test_nothing_is_moved(self, tmp_path):
        source = a_file(tmp_path / "in", "a.pdf")
        move_file(source, tmp_path / "out", dry_run=True)
        assert source.exists()

    def test_nothing_is_created(self, tmp_path):
        """A rehearsal that leaves empty folders behind is not a rehearsal."""
        source = a_file(tmp_path / "in", "a.pdf")
        move_file(source, tmp_path / "out", dry_run=True)
        assert not (tmp_path / "out").exists()

    def test_the_reported_path_is_the_one_the_real_run_would_use(self, tmp_path):
        source = a_file(tmp_path / "in", "a.pdf")
        a_file(tmp_path / "out", "a.pdf")
        assert move_file(source, tmp_path / "out", dry_run=True).name == "a (2).pdf"


class TestAuditLog:
    def test_an_entry_is_written_per_file(self, tmp_path, rules):
        result = score_page(make_page([("請求書", 50.0)]), rules)
        path = tmp_path / "log.jsonl"
        with AuditLog(path) as audit:
            audit.record(tmp_path / "a.pdf", result)
            audit.record(tmp_path / "b.pdf", result)
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_the_entry_records_enough_to_reverse_the_move(self, tmp_path, rules):
        result = score_page(make_page([("請求書", 50.0)]), rules)
        with AuditLog(tmp_path / "log.jsonl") as audit:
            entry = audit.record(
                tmp_path / "in" / "a.pdf", result, destination=tmp_path / "out" / "a.pdf"
            )
        assert entry["source"].endswith("a.pdf")
        assert entry["destination"].endswith("a.pdf")

    def test_the_entry_records_why_the_verdict_was_reached(self, tmp_path, rules):
        """Months later, nobody remembers what the weights were that day."""
        result = score_page(make_page([("請求澤書", 50.0)]), rules)
        with AuditLog(tmp_path / "log.jsonl") as audit:
            entry = audit.record(tmp_path / "a.pdf", result)
        assert entry["hits"][0]["pattern"] == "請求書"
        assert entry["hits"][0]["matched"] == "請求澤書"
        assert entry["hits"][0]["kind"] == "subsequence"

    def test_japanese_is_written_readably(self, tmp_path, rules):
        """A log full of \\u8acb\\u6c42\\u66f8 is not one anybody will read."""
        result = score_page(make_page([("請求書", 50.0)]), rules)
        path = tmp_path / "log.jsonl"
        with AuditLog(path) as audit:
            audit.record(tmp_path / "a.pdf", result)
        assert "請求書" in path.read_text(encoding="utf-8")

    def test_entries_are_valid_json_lines(self, tmp_path, rules):
        result = score_page(make_page([("請求書", 50.0)]), rules)
        path = tmp_path / "log.jsonl"
        with AuditLog(path) as audit:
            audit.record(tmp_path / "a.pdf", result)
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            assert json.loads(line)["verdict"]

    def test_a_failure_is_recorded_too(self, tmp_path):
        with AuditLog(tmp_path / "log.jsonl") as audit:
            entry = audit.record(tmp_path / "a.pdf", None, error="is password protected")
        assert entry["error"] == "is password protected"
        assert "verdict" not in entry

    def test_appending_does_not_truncate_an_earlier_run(self, tmp_path, rules):
        result = score_page(make_page([("請求書", 50.0)]), rules)
        path = tmp_path / "log.jsonl"
        for _ in range(2):
            with AuditLog(path) as audit:
                audit.record(tmp_path / "a.pdf", result)
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_no_path_means_no_file(self, tmp_path, rules):
        """Logging is optional; a classify call that only wants the exit code
        should not be forced to leave a file behind."""
        result = score_page(make_page([("請求書", 50.0)]), rules)
        with AuditLog(None) as audit:
            assert audit.record(tmp_path / "a.pdf", result)["verdict"]
        assert list(tmp_path.iterdir()) == []


class TestGuards:
    def test_collisions_do_not_loop_forever(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pdf_ocr.core.mover.MAX_COLLISION_ATTEMPTS", 3)
        for name in ("a.pdf", "a (2).pdf"):
            a_file(tmp_path, name)
        with pytest.raises(FileExistsError):
            unique_destination(tmp_path, "a.pdf")
