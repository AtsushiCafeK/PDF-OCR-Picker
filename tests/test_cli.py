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

    def test_the_help_points_at_the_gui(self, capsys):
        """Someone who double-clicks this is often looking for a window."""
        main([])
        out = capsys.readouterr().out
        assert "gui" in out

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


class TestGuiSubcommand:
    """The GUI ships inside the same executable. A second bundle would carry
    its own PyTorch and OCR models -- some 550 MB duplicated to gain nothing."""

    def test_it_dispatches_to_the_gui(self, monkeypatch):
        opened = []
        monkeypatch.setattr(
            "pdf_ocr.gui.main", lambda rules=None, config=None: opened.append(rules) or 0
        )
        assert main(["gui"]) == 0
        assert opened == [None]

    def test_an_explicit_rules_file_is_passed_through(self, tmp_path, monkeypatch):
        opened = []
        monkeypatch.setattr(
            "pdf_ocr.gui.main", lambda rules=None, config=None: opened.append(rules) or 0
        )
        rules = tmp_path / "custom.yaml"
        main(["gui", "--rules", str(rules)])
        assert opened == [rules]

    def test_the_windowed_build_is_recognised_by_its_name(self, monkeypatch):
        """Both executables run this module; the filename is what tells them
        apart, so a double-click on the windowed one opens the window."""
        from pdf_ocr.cli import launched_as_gui

        monkeypatch.setattr("pdf_ocr.cli.sys.frozen", True, raising=False)
        monkeypatch.setattr(
            "pdf_ocr.cli.sys.executable", r"C:\app\pdf-sorter-gui.exe", raising=False
        )
        assert launched_as_gui() is True

        monkeypatch.setattr(
            "pdf_ocr.cli.sys.executable", r"C:\app\pdf-sorter.exe", raising=False
        )
        assert launched_as_gui() is False

    def test_running_from_source_is_never_the_windowed_build(self, monkeypatch):
        """Otherwise a checkout in a folder named 'gui' would behave oddly."""
        from pdf_ocr.cli import launched_as_gui

        monkeypatch.delattr("pdf_ocr.cli.sys.frozen", raising=False)
        assert launched_as_gui() is False


class TestRunBatch:
    """The loop both the headless run and the progress window share."""

    def _args(self, dry_run=False):
        from argparse import Namespace

        from pdf_ocr.core.extract import DEFAULT_DPI, MIN_TEXT_LAYER_CHARS

        return Namespace(
            dpi=DEFAULT_DPI,
            min_text_chars=MIN_TEXT_LAYER_CHARS,
            force_ocr=False,
            dry_run=dry_run,
        )

    def _rules(self):
        from pdf_ocr import DEFAULT_RULES_PATH
        from pdf_ocr.core.score import RuleSet

        return RuleSet.load(DEFAULT_RULES_PATH)

    def test_on_file_is_called_once_per_document(self, tmp_path):
        from pdf_ocr.cli import run_batch

        inbox = tmp_path / "in"
        sample_pdf(inbox, "invoice_01_centered")
        sample_pdf(inbox, "other_01_quotation")
        paths = sorted(inbox.glob("*.pdf"))
        seen = []
        summary = run_batch(
            paths, None, self._rules(), self._args(),
            directory=inbox, move_to=None, out=None,
            on_file=lambda done, total, counts: seen.append((done, total)),
        )
        assert seen == [(1, 2), (2, 2)]
        assert summary["processed"] == 2
        assert summary["stopped"] is False

    def test_the_tally_grows_as_it_goes(self, tmp_path):
        from pdf_ocr.cli import run_batch

        inbox = tmp_path / "in"
        sample_pdf(inbox, "invoice_01_centered")
        paths = sorted(inbox.glob("*.pdf"))
        snapshots = []
        run_batch(
            paths, None, self._rules(), self._args(),
            directory=inbox, move_to=None, out=None,
            on_file=lambda done, total, counts: snapshots.append(dict(counts)),
        )
        assert snapshots[-1]["invoice"] == 1

    def test_should_stop_halts_before_the_next_file(self, tmp_path):
        """A document already being read still finishes; nothing after starts."""
        from pdf_ocr.cli import run_batch

        inbox = tmp_path / "in"
        for name in ("invoice_01_centered", "other_01_quotation", "invoice_09_korean"):
            sample_pdf(inbox, name)
        paths = sorted(inbox.glob("*.pdf"))
        done_count = {"n": 0}
        summary = run_batch(
            paths, None, self._rules(), self._args(),
            directory=inbox, move_to=None, out=None,
            on_file=lambda done, total, counts: done_count.__setitem__("n", done),
            should_stop=lambda: done_count["n"] >= 1,
        )
        assert summary["processed"] == 1
        assert summary["stopped"] is True


class TestConfigDefaults:
    """A configured folder lets a Power Automate step be just
    'pdf-sorter.exe batch', with the folders already known."""

    def _config(self, tmp_path, **fields):
        lines = [f"{k}: {str(v)}" for k, v in fields.items()]
        path = tmp_path / "config.yaml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_batch_uses_the_configured_input_folder(self, tmp_path, capsys):
        inbox = tmp_path / "in"
        sample_pdf(inbox, "invoice_01_centered")
        config = self._config(tmp_path, input_dir=inbox)
        code = main(["batch", "--no-ocr", "--config", str(config)])
        assert code == ExitCode.INVOICE
        assert stdout_json(capsys)["files"] == 1

    def test_batch_uses_the_configured_destination(self, tmp_path):
        inbox = tmp_path / "in"
        sample_pdf(inbox, "invoice_01_centered")
        out = tmp_path / "sorted"
        config = self._config(tmp_path, input_dir=inbox, output_dir=out)
        main(["batch", "--no-ocr", "--config", str(config)])
        assert (out / DEFAULT_FOLDER_NAMES[Verdict.INVOICE]).exists()
        assert list(inbox.glob("*.pdf")) == []

    def test_an_explicit_folder_overrides_the_config(self, tmp_path, capsys):
        configured = tmp_path / "configured"
        sample_pdf(configured, "invoice_01_centered")
        explicit = tmp_path / "explicit"
        sample_pdf(explicit, "invoice_09_korean")
        sample_pdf(explicit, "other_01_quotation")
        config = self._config(tmp_path, input_dir=configured)
        main(["batch", str(explicit), "--no-ocr", "--config", str(config)])
        assert stdout_json(capsys)["files"] == 2  # from explicit, not configured

    def test_batch_without_a_folder_or_config_reports_the_gap(self, tmp_path, capsys):
        empty_config = tmp_path / "config.yaml"
        code = main(["batch", "--no-ocr", "--config", str(empty_config)])
        assert code == ExitCode.ERROR
        assert "error" in stdout_json(capsys)

    def test_the_configured_log_is_written(self, tmp_path):
        inbox = tmp_path / "in"
        sample_pdf(inbox, "invoice_01_centered")
        log = tmp_path / "audit.jsonl"
        config = self._config(tmp_path, input_dir=inbox, log=log)
        main(["batch", "--no-ocr", "--config", str(config)])
        assert log.exists()


class TestRulesResolution:
    def test_both_front_ends_look_in_the_same_place(self):
        """If the GUI and the sorter disagreed about where rules live, tuning
        in the GUI would silently fail to change what the sorter does -- the one
        thing the GUI exists to prevent."""
        import pdf_ocr.cli as cli_module
        import pdf_ocr.gui as gui_module

        assert cli_module.resolve_rules_path is gui_module.resolve_rules_path

    def test_a_copy_beside_the_executable_wins(self, tmp_path, monkeypatch):
        """How a deployed copy gets tuned without a rebuild."""
        from pdf_ocr import DEFAULT_RULES_PATH, resolve_rules_path

        beside = tmp_path / "rules.yaml"
        beside.write_text("rules: []", encoding="utf-8")
        monkeypatch.setattr("pdf_ocr.sys.frozen", True, raising=False)
        monkeypatch.setattr(
            "pdf_ocr.sys.executable", str(tmp_path / "pdf-sorter.exe"), raising=False
        )
        assert resolve_rules_path() == beside

        beside.unlink()
        assert resolve_rules_path() == DEFAULT_RULES_PATH

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
