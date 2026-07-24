"""Tests for the normalization ladder and its origin mapping."""

from __future__ import annotations

import unicodedata

from pdf_ocr.core.normalize import (
    NormalizeOptions,
    normalize_blocks,
    normalize_text,
)
from tests.conftest import block

ALL_OFF = NormalizeOptions(
    nfkc=False, strip_whitespace=False, strip_symbols=False, lowercase=False
)


class TestWhitespace:
    def test_ideographic_spaces_in_a_title_are_removed(self):
        """The 請　求　書 case: correct characters, padded apart for layout."""
        assert normalize_text("請　求　書").text == "請求書"

    def test_ascii_spaces_and_newlines_are_removed(self):
        assert normalize_text("請 求\n書\t").text == "請求書"

    def test_spacing_survives_when_the_step_is_off(self):
        """The toggle has to actually change something, or it is not a control."""
        options = NormalizeOptions(strip_whitespace=False)
        assert normalize_text("請　求　書", options).text == "請 求 書"


class TestNfkc:
    def test_full_width_latin_folds_to_ascii(self):
        assert normalize_text("ＩＮＶＯＩＣＥ").text == "invoice"

    def test_composed_forms_are_expanded(self):
        assert normalize_text("㈱サンプル").text == "(株)サンプル"

    def test_half_width_kana_folds_to_full_width(self):
        assert normalize_text("ｻﾝﾌﾟﾙ").text == "サンプル"

    def test_decomposed_dakuten_is_recombined(self):
        """A base character and its combining mark must fold back together.

        Normalizing character by character would leave these as two characters,
        so 請求書が would never match a rule written with the composed form.
        """
        decomposed = unicodedata.normalize("NFD", "銀行")
        assert normalize_text(decomposed).text == "銀行"

    def test_composition_is_skipped_when_the_step_is_off(self):
        assert normalize_text("ＩＮＶＯＩＣＥ", ALL_OFF).text == "ＩＮＶＯＩＣＥ"


class TestSymbols:
    def test_layout_noise_is_removed_when_enabled(self):
        options = NormalizeOptions(strip_symbols=True)
        assert normalize_text("請|求|書", options).text == "請求書"

    def test_layout_noise_is_kept_by_default(self):
        assert normalize_text("請|求|書").text == "請|求|書"

    def test_prolonged_sound_mark_is_never_treated_as_a_dash(self):
        """Removing ー would corrupt ordinary katakana such as データ."""
        options = NormalizeOptions(strip_symbols=True)
        assert normalize_text("データ", options).text == "データ"


class TestOriginMapping:
    def test_raw_slice_recovers_the_text_as_printed(self):
        """A rule says 請求書; a reviewer needs to see what was on the page."""
        normalized = normalize_text("請　求　書")
        assert normalized.text == "請求書"
        assert normalized.raw_slice(0, 3) == "請　求　書"

    def test_origin_is_the_same_length_as_the_normalized_text(self):
        normalized = normalize_text("ＩＮＶＯＩＣＥ　No.5")
        assert len(normalized.origin) == len(normalized.text)

    def test_origin_indices_stay_inside_the_raw_text(self):
        normalized = normalize_text("㈱ ｻﾝﾌﾟﾙ　御中")
        assert all(0 <= index < len(normalized.raw) for index in normalized.origin)


class TestBlocks:
    def test_blocks_split_by_ocr_are_rejoined(self):
        """OCR routinely reports a title as one detection per character."""
        blocks = [block("請", 50.0), block("求", 50.0), block("書", 50.0)]
        assert normalize_blocks(blocks).text == "請求書"

    def test_blocks_stay_separated_without_whitespace_stripping(self):
        """The separator is what stops two blocks forming a phantom match."""
        blocks = [block("請", 50.0), block("求", 50.0), block("書", 50.0)]
        options = NormalizeOptions(strip_whitespace=False)
        assert "請求書" not in normalize_blocks(blocks, options).text

    def test_a_match_can_be_traced_back_to_the_blocks_it_spans(self):
        blocks = [block("請", 50.0), block("求", 50.0), block("書", 50.0)]
        normalized = normalize_blocks(blocks)
        assert normalized.blocks_for(0, 3) == [0, 1, 2]

    def test_a_match_inside_one_block_reports_only_that_block(self):
        blocks = [block("株式会社サンプル", 50.0), block("請求書", 120.0)]
        normalized = normalize_blocks(blocks)
        start = normalized.text.index("請求書")
        assert normalized.blocks_for(start, start + 3) == [1]
