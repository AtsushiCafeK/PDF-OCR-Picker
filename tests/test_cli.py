"""Tests for the command-line contract Power Automate depends on.

Two properties matter more than the rest and are asserted repeatedly: stdout
carries nothing but JSON, and the exit code alone is enough to branch on. A flow
that has to parse console output to find out what happened is a flow that breaks
the first time a warning is printed.

OCR is switched off throughout via ``--no-ocr``; the routing between text layer
and recogniser is covered in test_extract.py.
"""

from __future__ import annotations

import json

import pytest

from pdf_ocr.cli import ExitCode, main
from pdf_ocr.core.mover import DEFAULT_FOLDER_NAMES
from pdf_ocr.core.types import Verdict
from tools.sample_pdfs import SAMPLES, build


def sample_pdf(directory, name: str):
    sample = next(s for s in SAMPLES if s.name == name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pdf"
    document = build(sample)
    document.save(path)
    document.close()
    return path


@pytest.fixture
def inbox(tmp_path):
    """A folder of text-layer documents: one of each verdict."""
    directory = tmp_path / "in"
    sample_pdf(directory, "invoice_01_centered")
    sample_pdf(directory, "invoice_09_korean")
    sample_pdf(directory, "other_01_quotation")
    return directory


def stdout_json(capsys):
    captured = capsys.readouterr().out.strip()
    return json.loads(captured)


class TestClassifyExitCodes:
    @pytest.mark.parametrize(
        ("sample", "expected"),
        [
            ("invoice_01_centered", ExitCode.INVOICE),
            ("invoice_09_korean", ExitCode.NEEDS_REVIEW),
            ("other_01_quotation", ExitCode.NOT_INVOICE),
        ],
    )
    def test_the_verdict_is_the_exit_code(self, tmp_path, sample, expected):
        """So a flow can branch without parsing anything at all."""
        path = sample_pdf(tmp_path, sample)
        assert main(["classify", str(path), "--no-ocr"]) == expected

    def test_an_unreadable_file_exits_with_the_error_code(self, tmp_path, capsys):
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"not a pdf")
        assert main(["classify", str(path), "--no-ocr"]) == ExitCode.ERROR
        assert "error" in stdout_json(capsys)

    def test_a_missing_rules_file_exits_with_the_error_code(self, tmp_path):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        code = main(
            ["classify", str(path), "--no-ocr", "--rules", str(tmp_path / "none.yaml")]
        )
        assert code == ExitCode.ERROR


class TestStdoutIsMachineReadable:
    def test_classify_prints_one_json_object(self, tmp_path, capsys):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        main(["classify", str(path), "--no-ocr"])
        entry = stdout_json(capsys)
        assert entry["verdict"] == Verdict.INVOICE.value
        assert entry["score"] > 0

    def test_the_reasoning_travels_with_the_result(self, tmp_path, capsys):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        main(["classify", str(path), "--no-ocr"])
        hits = stdout_json(capsys)["hits"]
        assert any(hit["rule"] == "title_seikyusho" for hit in hits)

    def test_logging_stays_off_stdout(self, tmp_path, capsys):
        """Anything but JSON on this stream breaks the caller's parse."""
        path = sample_pdf(tmp_path, "invoice_01_centered")
        main(["classify", str(path), "--no-ocr", "--verbose"])
        captured = capsys.readouterr()
        json.loads(captured.out.strip())

    def test_stdout_is_pure_ascii(self, tmp_path, capsys):
        """A Japanese Windows console is cp932. UTF-8 bytes sent through it
        arrive as mojibake, and the matched text is lost with them; escapes
        survive any code page."""
        path = sample_pdf(tmp_path, "invoice_01_centered")
        main(["classify", str(path), "--no-ocr"])
        out = capsys.readouterr().out
        out.encode("ascii")

    def test_the_escaped_japanese_still_decodes(self, tmp_path, capsys):
        path = sample_pdf(tmp_path, "invoice_01_centered")
        main(["classify", str(path), "--no-ocr"])
        hits = stdout_json(capsys)["hits"]
        assert any(hit["pattern"] == "請求書" for hit in hits)

    def test_batch_prints_one_json_summary(self, inbox, capsys):
        main(["batch", str(inbox), "--no-ocr"])
        summary = stdout_json(capsys)
        assert summary["files"] == 3
        assert summary["verdicts"][Verdict.INVOICE.value] == 1
        assert summary["verdicts"][Verdict.OTHER.value] == 1


class TestFiling:
    def test_nothing_moves_without_move_to(self, inbox, tmp_path):
        main(["batch", str(inbox), "--no-ocr"])
        assert len(list(inbox.glob("*.pdf"))) == 3

    def test_each_verdict_lands_in_its_own_folder(self, inbox, tmp_path):
        out = tmp_path / "out"
        main(["batch", str(inbox), "--no-ocr", "--move-to", str(out)])
        assert (out / DEFAULT_FOLDER_NAMES[Verdict.INVOICE]).exists()
        assert (out / DEFAULT_FOLDER_NAMES[Verdict.NEEDS_REVIEW]).exists()
        assert (out / DEFAULT_FOLDER_NAMES[Verdict.OTHER]).exists()
        assert list(inbox.glob("*.pdf")) == []

    def test_a_dry_run_moves_nothing(self, inbox, tmp_path):
        """The honest way to run this before the thresholds are tuned."""
        out = tmp_path / "out"
        main(["batch", str(inbox), "--no-ocr", "--move-to", str(out), "--dry-run"])
        assert len(list(inbox.glob("*.pdf"))) == 3
        assert not out.exists()

    def test_a_dry_run_still_reports_where_files_would_go(self, inbox, tmp_path):
        out = tmp_path / "out"
        log = tmp_path / "plan.jsonl"
        main(
            ["batch", str(inbox), "--no-ocr", "--move-to", str(out),
             "--dry-run", "--out", str(log)]
        )
        entries = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
        assert all(entry["dry_run"] for entry in entries)
        assert all("destination" in entry for entry in entries)


class TestBatchLog:
    def test_one_record_per_file(self, inbox, tmp_path):
        log = tmp_path / "result.jsonl"
        main(["batch", str(inbox), "--no-ocr", "--out", str(log)])
        assert len(log.read_text("utf-8").strip().splitlines()) == 3

    def test_a_failure_does_not_stop_the_run(self, inbox, tmp_path):
        """One corrupt file in a folder of a hundred must not lose the other 99."""
        (inbox / "broken.pdf").write_bytes(b"not a pdf")
        log = tmp_path / "result.jsonl"
        code = main(["batch", str(inbox), "--no-ocr", "--out", str(log)])
        entries = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
        assert len(entries) == 4
        assert sum("error" in entry for entry in entries) == 1
        assert code == ExitCode.ERROR

    def test_an_empty_folder_is_not_an_error(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main(["batch", str(empty), "--no-ocr"]) == ExitCode.INVOICE
        assert stdout_json(capsys)["files"] == 0


class TestNoArguments:
    def test_it_shows_help_instead_of_a_usage_error(self, capsys):
        """This is what double-clicking the executable does. Argparse's usage
        error is correct and useless: the window carrying it closes with the
        process, so the user sees a flash and nothing else."""
        assert main([]) == ExitCode.ERROR
        out = capsys.readouterr().out
        assert "Examples:" in out
        assert "batch" in out

    def test_the_help_explains_the_exit_codes(self, capsys):
        """The exit code is the interface a flow branches on."""
        main([])
        out = capsys.readouterr().out
        assert "0 invoice" in out
        assert "9 error" in out

    def test_the_help_says_the_gui_is_not_in_here(self, capsys):
        main([])
        assert "not part of this executable" in capsys.readouterr().out

    def test_it_does_not_block_when_not_launched_from_explorer(self, capsys):
        """Pausing is only ever right when the window is about to vanish; doing
        it in a script or a flow would hang the run."""
        main([])
        assert "Press Enter" not in capsys.readouterr().out


class TestDiag:
    def test_it_reports_the_make_up_of_a_folder(self, inbox, capsys):
        """The question the design has been assuming an answer to."""
        main(["diag", str(inbox), "--no-ocr"])
        report = stdout_json(capsys)
        assert report["files"] == 3
        assert report["with_text_layer"] == 3
        assert report["needing_ocr"] == 0

    def test_it_reports_the_score_distribution(self, inbox, capsys):
        """A threshold chosen from this is chosen on evidence."""
        main(["diag", str(inbox), "--no-ocr"])
        scores = stdout_json(capsys)["scores"]
        assert len(scores) == 3
        assert scores == sorted(scores)

    def test_it_moves_nothing(self, inbox):
        main(["diag", str(inbox), "--no-ocr"])
        assert len(list(inbox.glob("*.pdf"))) == 3


class TestRulesResolution:
    def test_an_explicit_rules_file_is_honoured(self, tmp_path, capsys):
        """Editing rules.yaml beside the exe is how a missed supplier gets fixed
        without a rebuild, so the override has to actually take effect."""
        path = sample_pdf(tmp_path, "other_01_quotation")
        rules = tmp_path / "custom.yaml"
        rules.write_text(
            "thresholds: {high: 1, low: 0}\n"
            "rules:\n"
            "  - {id: anything, pattern: 御中, weight: 10, match: exact}\n",
            encoding="utf-8",
        )
        code = main(["classify", str(path), "--no-ocr", "--rules", str(rules)])
        assert code == ExitCode.INVOICE
        assert stdout_json(capsys)["score"] == 10
