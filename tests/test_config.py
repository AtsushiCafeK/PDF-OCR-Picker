"""Tests for the machine-local default-folders config."""

from __future__ import annotations

from pathlib import Path

from pdf_ocr.core.config import Config


class TestLoading:
    def test_a_missing_file_is_empty_not_an_error(self, tmp_path):
        config = Config.load(tmp_path / "none.yaml")
        assert config.input_dir is None
        assert config.output_dir is None
        assert config.log is None
        assert config.path == tmp_path / "none.yaml"

    def test_it_reads_the_folders(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "input_dir: C:\\in\noutput_dir: C:\\sorted\nlog: C:\\logs\\r.jsonl\n",
            encoding="utf-8",
        )
        config = Config.load(tmp_path / "config.yaml")
        assert config.input_dir == Path("C:\\in")
        assert config.output_dir == Path("C:\\sorted")
        assert config.log == Path("C:\\logs\\r.jsonl")

    def test_a_partial_file_leaves_the_rest_none(self, tmp_path):
        (tmp_path / "config.yaml").write_text("input_dir: C:\\in\n", encoding="utf-8")
        config = Config.load(tmp_path / "config.yaml")
        assert config.input_dir == Path("C:\\in")
        assert config.output_dir is None

    def test_a_malformed_file_is_treated_as_empty(self, tmp_path):
        """A broken config must not take the tool down -- fall back to args."""
        (tmp_path / "config.yaml").write_text("just a string", encoding="utf-8")
        assert Config.load(tmp_path / "config.yaml").input_dir is None


class TestSaving:
    def test_a_folder_survives_a_round_trip(self, tmp_path):
        config = Config(path=tmp_path / "config.yaml")
        config.input_dir = Path("C:\\in")
        config.save()
        assert Config.load(tmp_path / "config.yaml").input_dir == Path("C:\\in")

    def test_only_set_fields_are_written(self, tmp_path):
        config = Config(path=tmp_path / "config.yaml", output_dir=Path("C:\\out"))
        config.save()
        text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
        assert "output_dir" in text
        assert "input_dir" not in text

    def test_saving_creates_the_parent(self, tmp_path):
        config = Config(path=tmp_path / "nested" / "config.yaml", log=Path("C:\\r.jsonl"))
        config.save()
        assert (tmp_path / "nested" / "config.yaml").exists()

    def test_updating_overwrites_the_previous_value(self, tmp_path):
        path = tmp_path / "config.yaml"
        Config(path=path, input_dir=Path("C:\\old")).save()
        Config(path=path, input_dir=Path("C:\\new")).save()
        assert Config.load(path).input_dir == Path("C:\\new")
