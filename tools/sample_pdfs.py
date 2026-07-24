"""Generate synthetic invoices for testing, in deliberately awkward layouts.

Real invoices cannot be committed -- they contain supplier names, bank accounts
and amounts -- so the repository holds this generator instead of the PDFs. Run
it to produce a corpus locally; ``.gitignore`` keeps the output out of git.

The layouts are chosen to attack the classifier rather than flatter it. Anyone
can match ``請求書`` printed in 24pt at the top of a clean page. These include a
title padded with ideographic spaces, a title pushed below the top quarter by a
letterhead, a title rotated onto its side, an invoice that never uses the word
at all, a form that is genuinely both a delivery note and an invoice, and
Korean invoices whose only readable clue is the English word.

Each sample carries the verdict it *should* receive, so the corpus doubles as a
scoreboard: ``python -m tools.evaluate`` reports which layouts the current rules
get wrong.
"""

from __future__ import annotations

import argparse
import io
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

from pdf_ocr.core.types import Verdict

JP = "japan"
KO = "korea"
EN = "helv"

A4 = (595.0, 842.0)
BLACK = (0.0, 0.0, 0.0)
GREY = (0.45, 0.45, 0.45)
WHITE = (1.0, 1.0, 1.0)

DEFAULT_OUTPUT_DIR = Path("data/samples")


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------


def write(
    page: pymupdf.Page,
    x: float,
    y: float,
    text: str,
    size: float = 10.0,
    font: str = JP,
    rotate: int = 0,
    color: tuple[float, float, float] = BLACK,
) -> None:
    """Draw text with its baseline at ``(x, y)``."""
    page.insert_text(
        pymupdf.Point(x, y), text, fontsize=size, fontname=font, rotate=rotate, color=color
    )


def centered(
    page: pymupdf.Page, y: float, text: str, size: float = 10.0, font: str = JP
) -> None:
    """Draw text centred on the page width."""
    width = pymupdf.get_text_length(text, fontname=font, fontsize=size)
    write(page, (page.rect.width - width) / 2, y, text, size, font)


def rule(page: pymupdf.Page, x0: float, y: float, x1: float, width: float = 0.6) -> None:
    page.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x1, y), color=BLACK, width=width)


def box(page: pymupdf.Page, x0: float, y0: float, x1: float, y1: float) -> None:
    page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=BLACK, width=0.6)


def filled_box(
    page: pymupdf.Page,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[float, float, float] = BLACK,
) -> None:
    """A solid rectangle, for a graphic logo that reverses text out of it."""
    page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=color, fill=color)


def item_table(
    page: pymupdf.Page,
    x: float,
    y: float,
    rows: list[tuple[str, str, str, str]],
    width: float = 480.0,
    row_height: float = 18.0,
) -> float:
    """A ruled line-item table. Returns the y of the bottom edge."""
    columns = [0.0, width * 0.52, width * 0.66, width * 0.80, width]
    headers = ("品目", "数量", "単価", "金額")

    box(page, x, y, x + width, y + row_height)
    for index, header in enumerate(headers):
        write(page, x + columns[index] + 4, y + 13, header, 9)

    cursor = y + row_height
    for row in rows:
        box(page, x, cursor, x + width, cursor + row_height)
        for index, cell in enumerate(row):
            write(page, x + columns[index] + 4, cursor + 13, cell, 9)
        cursor += row_height

    for column in columns[1:-1]:
        page.draw_line(
            pymupdf.Point(x + column, y),
            pymupdf.Point(x + column, cursor),
            color=BLACK,
            width=0.6,
        )
    return cursor


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------


def _standard_items() -> list[tuple[str, str, str, str]]:
    return [
        ("Webサイト保守運用費", "1", "80,000", "80,000"),
        ("サーバ利用料 2026年7月分", "1", "12,000", "12,000"),
        ("ドメイン更新費", "2", "4,000", "8,000"),
    ]


def centered_title(page: pymupdf.Page) -> None:
    """The easy case, included as a control."""
    centered(page, 90, "請求書", 24)
    write(page, 60, 140, "株式会社サンプル商事　御中", 11)
    rule(page, 60, 148, 300)
    write(page, 380, 130, "発行日 2026年7月24日", 9)
    write(page, 380, 145, "請求書番号 INV-2026-0731", 9)
    write(page, 380, 160, "株式会社テストシステムズ", 10)
    write(page, 380, 175, "登録番号 T1234567890123", 9)

    write(page, 60, 210, "下記の通りご請求申し上げます。", 10)
    write(page, 60, 245, "ご請求金額", 12)
    write(page, 160, 245, "¥110,000-", 16)
    rule(page, 60, 252, 300, 1.2)

    bottom = item_table(page, 60, 280, _standard_items())
    write(page, 400, bottom + 20, "小計", 10)
    write(page, 480, bottom + 20, "100,000", 10)
    write(page, 400, bottom + 38, "消費税(10%)", 10)
    write(page, 480, bottom + 38, "10,000", 10)
    write(page, 400, bottom + 58, "合計金額", 11)
    write(page, 480, bottom + 58, "110,000", 11)

    write(page, 60, bottom + 110, "お支払期限　2026年8月31日", 10)
    write(page, 60, bottom + 130, "お振込先　みずほ銀行 渋谷支店 普通 1234567", 10)
    write(page, 60, bottom + 150, "口座名義　カ)テストシステムズ", 10)


def spaced_title(page: pymupdf.Page) -> None:
    """Title padded with ideographic spaces -- the 請　求　書 case."""
    write(page, 60, 100, "請　求　書", 22)
    write(page, 380, 90, "No. 2026-0724", 9, EN)
    write(page, 380, 105, "2026年7月24日", 9)
    write(page, 380, 125, "有限会社サンプル工業", 10)
    write(page, 380, 140, "東京都渋谷区1-2-3", 9)
    write(page, 380, 155, "T9876543210987", 9, EN)

    write(page, 60, 160, "サンプル物流株式会社　御中", 11)
    write(page, 60, 200, "ご 請 求 金 額", 12)
    write(page, 200, 200, "¥ 2 6 4 , 0 0 0 -", 15)

    bottom = item_table(
        page,
        60,
        240,
        [
            ("運送費 2026年6月分", "1", "200,000", "200,000"),
            ("燃油サーチャージ", "1", "40,000", "40,000"),
        ],
    )
    write(page, 400, bottom + 24, "消費税", 10)
    write(page, 480, bottom + 24, "24,000", 10)
    write(page, 60, bottom + 70, "支払期日：2026年8月20日", 10)
    write(page, 60, bottom + 90, "振込先：三菱UFJ銀行 新宿支店 当座 7654321", 10)


def english_header(page: pymupdf.Page) -> None:
    """Latin title over Japanese body, as bilingual templates do."""
    centered(page, 80, "INVOICE", 26, EN)
    centered(page, 100, "請求書", 12)
    rule(page, 60, 115, 535, 1.0)

    write(page, 60, 145, "Bill To:", 9, EN)
    write(page, 60, 162, "Global Sample Inc.", 10, EN)
    write(page, 60, 177, "グローバルサンプル株式会社　御中", 10)

    write(page, 360, 145, "Invoice No.", 9, EN)
    write(page, 450, 145, "GS-2026-118", 9, EN)
    write(page, 360, 160, "Issue Date", 9, EN)
    write(page, 450, 160, "2026-07-24", 9, EN)
    write(page, 360, 175, "Due Date", 9, EN)
    write(page, 450, 175, "2026-08-31", 9, EN)

    bottom = item_table(
        page,
        60,
        220,
        [
            ("Consulting service / コンサルティング", "1", "300,000", "300,000"),
            ("Travel expense / 旅費交通費", "1", "45,000", "45,000"),
        ],
    )
    write(page, 380, bottom + 22, "Subtotal / 小計", 9, EN)
    write(page, 500, bottom + 22, "345,000", 9, EN)
    write(page, 380, bottom + 38, "Tax 10% / 消費税", 9, EN)
    write(page, 500, bottom + 38, "34,500", 9, EN)
    write(page, 380, bottom + 58, "Total / 請求金額", 10, EN)
    write(page, 500, bottom + 58, "379,500", 10, EN)

    write(page, 60, bottom + 100, "Remittance / 振込先", 9, EN)
    write(page, 60, bottom + 118, "SMBC Shibuya Branch  Ordinary  2233445", 9, EN)
    write(page, 60, bottom + 136, "登録番号 T5555666677778", 9)


def title_below_the_fold(page: pymupdf.Page) -> None:
    """A letterhead pushes the title out of the top quarter.

    The position bonus must not fire here, and the document must still be
    classified on its vocabulary alone.
    """
    box(page, 50, 50, 545, 195)
    write(page, 70, 80, "株式会社サンプルホールディングス", 14)
    write(page, 70, 100, "SAMPLE HOLDINGS CO., LTD.", 9, EN)
    write(page, 70, 125, "〒150-0001 東京都渋谷区神宮前0-0-0 サンプルビル10F", 9)
    write(page, 70, 143, "TEL 03-0000-0000 / FAX 03-0000-0001", 9, EN)
    write(page, 70, 161, "適格請求書発行事業者 登録番号 T1112223334445", 9)
    write(page, 70, 182, "経理部 請求担当", 9)

    write(page, 60, 240, "請求書", 20)
    write(page, 380, 240, "2026年7月24日発行", 9)
    write(page, 60, 275, "サンプル製作所　御中", 11)

    bottom = item_table(page, 60, 310, _standard_items())
    write(page, 60, bottom + 30, "ご請求金額（税込）　¥110,000", 13)
    write(page, 60, bottom + 60, "お支払期限　2026年8月31日", 10)
    write(page, 60, bottom + 80, "お振込先　りそな銀行 渋谷支店 普通 9988776", 10)


def no_title_word(page: pymupdf.Page) -> None:
    """Never prints 請求書 anywhere. The hardest positive case.

    Templates like this exist -- the header just says the company name and the
    body gets straight to the amount. Everything hangs on the payment and amount
    vocabulary, which is exactly what the design claims can carry a decision.
    """
    write(page, 60, 90, "サンプルクリエイティブ", 18)
    write(page, 60, 112, "creative works & design", 9, EN)
    rule(page, 60, 125, 535, 1.5)

    write(page, 60, 160, "サンプルストア株式会社　御中", 11)
    write(page, 60, 185, "毎度お引き立ていただきありがとうございます。", 9)
    write(page, 60, 200, "下記の通りご案内申し上げます。", 9)

    write(page, 330, 160, "2026年7月24日", 9)
    write(page, 330, 178, "No. SC-0724", 9, EN)
    write(page, 330, 196, "登録番号 T2223334445556", 9)

    write(page, 60, 245, "ご請求金額", 13)
    write(page, 170, 245, "¥ 550,000 -", 17)
    rule(page, 60, 253, 320, 1.5)

    bottom = item_table(
        page,
        60,
        290,
        [
            ("ロゴデザイン一式", "1", "300,000", "300,000"),
            ("名刺・封筒デザイン", "1", "150,000", "150,000"),
            ("修正対応", "1", "50,000", "50,000"),
        ],
    )
    write(page, 400, bottom + 22, "消費税", 10)
    write(page, 480, bottom + 22, "50,000", 10)

    write(page, 60, bottom + 70, "お支払期限", 10)
    write(page, 160, bottom + 70, "2026年8月末日", 10)
    write(page, 60, bottom + 90, "お振込先", 10)
    write(page, 160, bottom + 90, "GMOあおぞらネット銀行 法人第一営業部 普通 1122334", 10)
    write(page, 160, bottom + 108, "口座番号 1122334 口座名義 サンプルクリエイティブ", 9)


def rotated_title(page: pymupdf.Page) -> None:
    """Title set sideways down the left margin."""
    write(page, 40, 620, "請求書", 26, rotate=90)
    page.draw_line(
        pymupdf.Point(70, 120), pymupdf.Point(70, 640), color=BLACK, width=1.2
    )

    write(page, 100, 110, "株式会社サンプルテクノロジー　御中", 12)
    write(page, 400, 110, "2026年7月24日", 9)
    write(page, 400, 128, "登録番号 T7778889990001", 9)

    write(page, 100, 165, "ご請求金額 ¥1,320,000-", 14)
    bottom = item_table(
        page,
        100,
        200,
        [
            ("システム開発 一次請負分", "1", "1,000,000", "1,000,000"),
            ("保守サポート 年間", "1", "200,000", "200,000"),
        ],
        width=440,
    )
    write(page, 380, bottom + 22, "消費税(10%)", 10)
    write(page, 470, bottom + 22, "120,000", 10)
    write(page, 100, bottom + 70, "支払期限 2026年9月15日", 10)
    write(page, 100, bottom + 90, "振込先 ゆうちょ銀行 〇一八支店 普通 5544332", 10)


def dense_table(page: pymupdf.Page) -> None:
    """Small title lost among many line items and boilerplate."""
    write(page, 60, 70, "請求明細書", 13)
    write(page, 450, 70, "2026年7月24日", 8)
    write(page, 60, 88, "サンプル商会株式会社　御中", 9)
    write(page, 450, 88, "発行 株式会社サンプルパーツ", 8)
    write(page, 450, 102, "T3334445556667", 8, EN)

    rows = [
        (f"部品コード SP-{1000 + index}　補修用パーツ", str(index % 5 + 1), "3,200", "3,200")
        for index in range(18)
    ]
    bottom = item_table(page, 60, 120, rows, row_height=15.0)

    write(page, 380, bottom + 18, "小計", 8)
    write(page, 490, bottom + 18, "182,400", 8)
    write(page, 380, bottom + 32, "消費税", 8)
    write(page, 490, bottom + 32, "18,240", 8)
    write(page, 380, bottom + 48, "請求金額", 9)
    write(page, 490, bottom + 48, "200,640", 9)

    write(
        page,
        60,
        bottom + 80,
        "お支払期限 2026年8月31日／振込先 千葉銀行 船橋支店 普通 3344556",
        8,
    )
    write(
        page,
        60,
        bottom + 96,
        "※ 本書は電子帳簿保存法に対応した適格請求書です。"
        "振込手数料は貴社負担にてお願いいたします。",
        7,
        color=GREY,
    )


def landscape_layout(page: pymupdf.Page) -> None:
    """Landscape A4, title in the top-right instead of the top-left."""
    write(page, 600, 80, "請 求 書", 20)
    write(page, 60, 80, "サンプルロジスティクス株式会社　御中", 12)
    write(page, 60, 105, "2026年7月分 運送料金のご請求", 10)
    write(page, 600, 105, "2026年7月24日", 9)
    write(page, 600, 122, "T4445556667778", 9, EN)

    bottom = item_table(
        page,
        60,
        150,
        [
            ("7月1日〜7月10日 配送分", "42", "1,200", "50,400"),
            ("7月11日〜7月20日 配送分", "38", "1,200", "45,600"),
            ("7月21日〜7月31日 配送分", "45", "1,200", "54,000"),
        ],
        width=700.0,
    )
    write(page, 600, bottom + 22, "消費税", 10)
    write(page, 700, bottom + 22, "15,000", 10)
    write(page, 600, bottom + 40, "ご請求金額", 11)
    write(page, 700, bottom + 40, "165,000", 11)
    write(page, 60, bottom + 70, "お支払期限 2026年8月末日", 10)
    write(page, 60, bottom + 90, "振込先 静岡銀行 浜松支店 普通 6677889", 10)


def logo_dominates_title(page: pymupdf.Page) -> None:
    """A large company wordmark, with 請求書 set much smaller.

    On a real invoice the biggest thing on the page is very often the sender's
    brand, not the word 請求書. A reader looking for "the title" as "the largest
    text" would land on the logo. The classifier does not rank by size -- it
    only asks whether the keyword is present -- so this should not trouble it,
    and this sample is here to keep that true.
    """
    write(page, 60, 100, "SAMPLE TECH", 40, EN)
    write(page, 62, 122, "サンプルテクノロジー株式会社", 12)
    rule(page, 60, 135, 360, 1.2)

    # The actual title, a third of the logo's size, tucked into the corner.
    box(page, 430, 78, 535, 108)
    write(page, 452, 100, "請求書", 14)

    write(page, 60, 175, "株式会社サンプル商事　御中", 11)
    write(page, 430, 128, "2026年7月24日", 9)
    write(page, 430, 143, "No. ST-2026-0442", 9, EN)
    write(page, 430, 158, "登録番号 T1234567890123", 9)

    write(page, 60, 215, "ご請求金額", 12)
    write(page, 160, 215, "¥110,000-", 16)
    rule(page, 60, 223, 300, 1.2)

    bottom = item_table(page, 60, 250, _standard_items())
    write(page, 400, bottom + 22, "消費税(10%)", 10)
    write(page, 480, bottom + 22, "10,000", 10)
    write(page, 60, bottom + 70, "お支払期限　2026年8月31日", 10)
    write(page, 60, bottom + 90, "お振込先　みずほ銀行 渋谷支店 普通 1234567", 10)


def graphic_logo_reversed(page: pymupdf.Page) -> None:
    """The logo is a filled block with the brand reversed out of it, and 請求書
    appears only once, small and below the fold.

    The hardest form of the same idea: the largest, highest-contrast mark on the
    page is a graphic, the sole occurrence of the title is a small label further
    down, and the decision therefore cannot lean on the title being prominent.
    It rests on the amount and payment vocabulary instead -- exactly what the
    design claims can carry a document on its own.
    """
    filled_box(page, 60, 70, 250, 120)
    write(page, 78, 104, "NIMBUS", 30, EN, color=WHITE)
    write(page, 260, 95, "Nimbus Solutions K.K.", 11, EN)
    write(page, 260, 110, "ニンバスソリューションズ株式会社", 9)

    write(page, 400, 80, "2026年7月24日", 9)
    write(page, 400, 95, "No. NS-2026-7781", 9, EN)
    write(page, 400, 110, "登録番号 T2223334445556", 9)

    write(page, 60, 160, "株式会社サンプル電子　御中", 11)
    write(page, 60, 185, "毎度格別のお引き立てを賜り厚く御礼申し上げます。", 9)

    write(page, 60, 225, "ご請求金額", 12)
    write(page, 170, 225, "¥ 594,000 -", 17)
    rule(page, 60, 233, 320, 1.4)

    bottom = item_table(
        page,
        60,
        265,
        [
            ("クラウド利用料 2026年7月分", "1", "480,000", "480,000"),
            ("初期構築サポート", "1", "60,000", "60,000"),
        ],
    )
    write(page, 400, bottom + 22, "消費税", 10)
    write(page, 480, bottom + 22, "54,000", 10)

    # The one and only 請求書, a small label beneath the table.
    write(page, 60, bottom + 60, "請求書", 11)
    write(page, 60, bottom + 82, "お支払期限　2026年8月31日", 10)
    write(page, 60, bottom + 102, "お振込先　三井住友銀行 渋谷支店 普通 2233445", 10)


def korean_invoice(page: pymupdf.Page) -> None:
    """Hangul throughout, with the English word as the only readable clue.

    The reader is loaded with Japanese, so the Hangul comes back as noise. This
    is the sample that decides whether the design's bet -- that ``Invoice``
    carries these documents on its own -- actually holds.
    """
    centered(page, 90, "INVOICE", 24, EN)
    centered(page, 115, "청구서", 16, KO)
    rule(page, 60, 130, 535, 1.0)

    write(page, 60, 165, "주식회사 샘플코리아 귀중", 12, KO)
    write(page, 380, 160, "발행일 2026-07-24", 10, KO)
    write(page, 380, 178, "Invoice No. KR-2026-0088", 9, EN)
    write(page, 380, 196, "샘플테크놀로지 주식회사", 10, KO)

    write(page, 60, 235, "청구 금액", 13, KO)
    write(page, 170, 235, "KRW 5,500,000", 16, EN)
    rule(page, 60, 243, 330, 1.4)

    bottom = item_table(
        page,
        60,
        280,
        [
            ("소프트웨어 라이선스", "1", "4,000,000", "4,000,000"),
            ("기술 지원 서비스", "1", "1,000,000", "1,000,000"),
        ],
    )
    write(page, 380, bottom + 22, "부가세 10%", 10, KO)
    write(page, 490, bottom + 22, "500,000", 10, EN)
    write(page, 60, bottom + 70, "지급 기한 2026-08-31", 10, KO)
    write(page, 60, bottom + 90, "입금 계좌 KEB하나은행 123-456-789012", 10, KO)


def delivery_note_and_invoice(page: pymupdf.Page) -> None:
    """納品書兼請求書 -- genuinely both. Belongs in review, not in a bin."""
    centered(page, 90, "納品書 兼 請求書", 20)
    write(page, 60, 140, "株式会社サンプル物産　御中", 11)
    write(page, 380, 130, "2026年7月24日", 9)
    write(page, 380, 148, "伝票番号 DN-2026-4410", 9)
    write(page, 380, 166, "登録番号 T6667778889990", 9)

    write(page, 60, 185, "下記の通り納品いたしました。あわせてご請求申し上げます。", 9)

    bottom = item_table(
        page,
        60,
        215,
        [
            ("業務用洗剤 20L", "6", "8,500", "51,000"),
            ("清掃用モップ 交換ヘッド", "12", "1,200", "14,400"),
        ],
    )
    write(page, 380, bottom + 22, "消費税", 10)
    write(page, 490, bottom + 22, "6,540", 10)
    write(page, 380, bottom + 40, "請求金額", 11)
    write(page, 490, bottom + 40, "71,940", 11)
    write(page, 60, bottom + 80, "お支払期限 2026年8月31日", 10)
    write(page, 60, bottom + 100, "振込先 常陽銀行 水戸支店 普通 4455667", 10)


def quotation(page: pymupdf.Page) -> None:
    """A quotation, which shares most of an invoice's vocabulary."""
    centered(page, 90, "御 見 積 書", 22)
    write(page, 60, 140, "株式会社サンプル建設　御中", 11)
    write(page, 380, 130, "2026年7月24日", 9)
    write(page, 380, 148, "見積番号 EST-2026-0512", 9)
    write(page, 380, 166, "株式会社サンプル設計", 10)

    write(page, 60, 200, "御見積金額", 12)
    write(page, 170, 200, "¥880,000-", 16)
    rule(page, 60, 208, 320, 1.2)

    bottom = item_table(
        page,
        60,
        245,
        [
            ("基本設計業務", "1", "500,000", "500,000"),
            ("実施設計業務", "1", "300,000", "300,000"),
        ],
    )
    write(page, 380, bottom + 22, "消費税", 10)
    write(page, 490, bottom + 22, "80,000", 10)
    write(page, 60, bottom + 70, "見積有効期限　発行日より30日間", 10)
    write(page, 60, bottom + 90, "納期　ご発注後60日", 10)


def delivery_note(page: pymupdf.Page) -> None:
    centered(page, 90, "納 品 書", 22)
    write(page, 60, 140, "サンプル流通株式会社　御中", 11)
    write(page, 380, 130, "2026年7月24日", 9)
    write(page, 380, 148, "納品番号 DL-2026-7781", 9)
    write(page, 60, 180, "下記の通り納品いたしました。", 10)

    bottom = item_table(
        page,
        60,
        215,
        [
            ("A4コピー用紙 5000枚", "10", "2,400", "24,000"),
            ("トナーカートリッジ", "4", "9,800", "39,200"),
        ],
    )
    write(page, 380, bottom + 22, "合計", 10)
    write(page, 490, bottom + 22, "63,200", 10)
    write(page, 60, bottom + 70, "検収のうえ受領書のご返送をお願いいたします。", 9)


def receipt(page: pymupdf.Page) -> None:
    centered(page, 100, "領 収 書", 24)
    write(page, 60, 165, "サンプル商店　様", 13)
    rule(page, 60, 172, 330)
    write(page, 60, 215, "金 額", 12)
    write(page, 160, 215, "¥33,000-", 18)
    rule(page, 60, 223, 330, 1.4)
    write(page, 60, 265, "但し　書籍代として", 11)
    write(page, 60, 295, "上記正に領収いたしました。", 11)
    write(page, 380, 340, "2026年7月24日", 10)
    write(page, 380, 365, "株式会社サンプル書房", 11)
    write(page, 380, 385, "登録番号 T8889990001112", 9)


def email_printout(page: pymupdf.Page) -> None:
    """An email that forwards an invoice, printed to PDF.

    The nastiest false positive available. Every keyword the classifier looks
    for is present -- 請求書 in the subject line at the very top of the page,
    ご請求金額, お支払期限, the English word -- and yet the document is not an
    invoice, it is a covering note. Filing it as one puts a duplicate in the
    accounting folder that reconciles against nothing.

    What separates it is not the invoice vocabulary but the mail header, which
    no invoice ever carries.
    """
    write(page, 60, 70, "差出人:", 9)
    write(page, 150, 70, "田中 太郎 <tanaka@sample-corp.co.jp>", 9)
    write(page, 60, 88, "送信日時:", 9)
    write(page, 150, 88, "2026年7月24日 10:32", 9)
    write(page, 60, 106, "宛先:", 9)
    write(page, 150, 106, "経理部 <keiri@example.co.jp>", 9)
    write(page, 60, 124, "CC:", 9)
    write(page, 150, 124, "営業部 <eigyo@example.co.jp>", 9)
    write(page, 60, 142, "件名:", 9)
    write(page, 150, 142, "【ご請求】2026年7月分 請求書送付のご案内 (Invoice)", 9)
    write(page, 60, 160, "添付ファイル:", 9)
    write(page, 150, 160, "請求書_202607.pdf (128 KB)", 9)
    rule(page, 60, 175, 535, 0.8)

    body = [
        "株式会社サンプル",
        "経理ご担当者様",
        "",
        "いつもお世話になっております。",
        "サンプルコーポレーションの田中でございます。",
        "",
        "2026年7月分の請求書を添付ファイルにてお送りいたします。",
        "ご請求金額は 110,000円（税込）でございます。",
        "お支払期限は 2026年8月31日 となっておりますので、",
        "ご確認のうえお手続きいただけますと幸いです。",
        "",
        "なお、振込先は前回より変更ございません。",
        "",
        "ご不明な点がございましたらお気軽にお問い合わせください。",
        "何卒よろしくお願い申し上げます。",
    ]
    for index, line in enumerate(body):
        write(page, 60, 205 + index * 18, line, 10)

    rule(page, 60, 505, 250, 0.5)
    for index, line in enumerate(
        [
            "サンプルコーポレーション株式会社",
            "営業本部　田中 太郎",
            "TEL 03-0000-0000 / FAX 03-0000-0001",
            "tanaka@sample-corp.co.jp",
        ]
    ):
        write(page, 60, 525 + index * 16, line, 9, color=GREY)


def email_printout_english(page: pymupdf.Page) -> None:
    """The same trap in English, where 'Invoice' is the subject line."""
    write(page, 60, 70, "From:", 9, EN)
    write(page, 150, 70, "John Smith <j.smith@sample-global.com>", 9, EN)
    write(page, 60, 88, "Sent:", 9, EN)
    write(page, 150, 88, "Friday, 24 July 2026 10:32", 9, EN)
    write(page, 60, 106, "To:", 9, EN)
    write(page, 150, 106, "Accounts Payable <ap@example.co.jp>", 9, EN)
    write(page, 60, 124, "Subject:", 9, EN)
    write(page, 150, 124, "Invoice INV-2026-0118 for July 2026", 9, EN)
    write(page, 60, 142, "Attachments:", 9, EN)
    write(page, 150, 142, "INV-2026-0118.pdf (96 KB)", 9, EN)
    rule(page, 60, 158, 535, 0.8)

    body = [
        "Dear Accounts Payable team,",
        "",
        "Please find attached our invoice INV-2026-0118 covering services",
        "provided in July 2026.",
        "",
        "The total amount is USD 3,795.00 and payment is due by 31 August 2026.",
        "Our bank details are unchanged from the previous invoice.",
        "",
        "Please let me know if you need anything further.",
        "",
        "Kind regards,",
        "John Smith",
        "Sample Global Inc.",
    ]
    for index, line in enumerate(body):
        write(page, 60, 190 + index * 18, line, 10, EN)


def purchase_order(page: pymupdf.Page) -> None:
    centered(page, 90, "注 文 書", 22)
    write(page, 60, 140, "株式会社サンプル電機　御中", 11)
    write(page, 380, 130, "2026年7月24日", 9)
    write(page, 380, 148, "注文番号 PO-2026-3390", 9)
    write(page, 60, 180, "下記の通り発注いたします。", 10)

    bottom = item_table(
        page,
        60,
        215,
        [
            ("制御基板 CTL-220", "50", "12,000", "600,000"),
            ("電源ユニット PSU-45", "20", "18,000", "360,000"),
        ],
    )
    write(page, 380, bottom + 22, "消費税", 10)
    write(page, 490, bottom + 22, "96,000", 10)
    write(page, 60, bottom + 70, "納期　2026年9月30日", 10)
    write(page, 60, bottom + 90, "納品場所　サンプル電機 本社倉庫", 10)


# --------------------------------------------------------------------------
# Sample catalogue
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """One document design, independent of how it is later rendered."""

    name: str
    draw: Callable[[pymupdf.Page], None]
    expected: Verdict
    difficulty: str
    note: str
    landscape: bool = False


@dataclass(frozen=True)
class Sample:
    """One test document: a layout, rendered either digitally or as a scan."""

    name: str
    draw: Callable[[pymupdf.Page], None]
    expected: Verdict
    difficulty: str
    note: str
    landscape: bool = False
    scan: dict | None = field(default=None)
    """When set, the page is flattened to an image so no text layer survives and
    OCR is forced. Keys: ``angle``, ``noise``, ``blur``, ``dpi``."""


LAYOUTS: list[Layout] = [
    Layout(
        "invoice_01_centered",
        centered_title,
        Verdict.INVOICE,
        "easy",
        "Control case: centred title, every keyword present.",
    ),
    Layout(
        "invoice_02_spaced_title",
        spaced_title,
        Verdict.INVOICE,
        "medium",
        "請　求　書 padded with ideographic spaces; amounts spaced too.",
    ),
    Layout(
        "invoice_03_english_header",
        english_header,
        Verdict.INVOICE,
        "easy",
        "Bilingual template led by the Latin word.",
    ),
    Layout(
        "invoice_04_title_below_fold",
        title_below_the_fold,
        Verdict.INVOICE,
        "medium",
        "Letterhead pushes the title out of the top quarter; no position bonus.",
    ),
    Layout(
        "invoice_05_no_title_word",
        no_title_word,
        Verdict.INVOICE,
        "hard",
        "Never prints 請求書. Rests entirely on amount and payment vocabulary.",
    ),
    Layout(
        "invoice_06_rotated_title",
        rotated_title,
        Verdict.INVOICE,
        "hard",
        "Title set sideways down the left margin.",
    ),
    Layout(
        "invoice_07_dense_table",
        dense_table,
        Verdict.INVOICE,
        "medium",
        "Small 請求明細書 title buried under eighteen line items.",
    ),
    Layout(
        "invoice_08_landscape",
        landscape_layout,
        Verdict.INVOICE,
        "medium",
        "Landscape page with the title in the top-right.",
        landscape=True,
    ),
    Layout(
        "invoice_09_korean",
        korean_invoice,
        Verdict.NEEDS_REVIEW,
        "hard",
        "Hangul body; only the English word is readable to a ja+en reader.",
    ),
    Layout(
        "invoice_10_delivery_and_invoice",
        delivery_note_and_invoice,
        Verdict.NEEDS_REVIEW,
        "hard",
        "納品書兼請求書 -- genuinely both, so review is the correct answer.",
    ),
    Layout(
        "invoice_11_logo_dominates_title",
        logo_dominates_title,
        Verdict.INVOICE,
        "medium",
        "A large company wordmark dwarfs the 請求書 title. The biggest text on "
        "the page is the brand, not the title -- which the classifier ignores, "
        "because it scores keyword presence rather than size.",
    ),
    Layout(
        "invoice_12_graphic_logo_reversed",
        graphic_logo_reversed,
        Verdict.INVOICE,
        "hard",
        "The dominant mark is a filled logo block, and the sole 請求書 is a "
        "small label below the fold -- so the decision rests on the amount and "
        "payment vocabulary, not on a prominent title.",
    ),
    Layout(
        "other_01_quotation",
        quotation,
        Verdict.OTHER,
        "medium",
        "Shares most invoice vocabulary; only 見積 separates it.",
    ),
    Layout(
        "other_02_delivery_note",
        delivery_note,
        Verdict.OTHER,
        "easy",
        "Delivery note with amounts but no request for payment.",
    ),
    Layout(
        "other_03_receipt",
        receipt,
        Verdict.OTHER,
        "medium",
        "Receipt: has an amount and a registration number, but is not a claim.",
    ),
    Layout(
        "other_04_purchase_order",
        purchase_order,
        Verdict.OTHER,
        "easy",
        "Purchase order, i.e. the mirror image of an invoice.",
    ),
    Layout(
        "other_05_email_printout",
        email_printout,
        Verdict.OTHER,
        "hard",
        "Email forwarding an invoice: 請求書 in the subject at the top of the "
        "page, plus ご請求金額 and お支払期限 in the body. Only the mail header "
        "says it is not the invoice itself.",
    ),
    Layout(
        "other_06_email_english",
        email_printout_english,
        Verdict.OTHER,
        "hard",
        "The same trap in English, with Invoice as the subject line.",
    ),
]

SCAN_PROFILES: list[dict] = [
    {"angle": 0.6, "noise": 7.0, "blur": 0.4},
    {"angle": -1.1, "noise": 10.0, "blur": 0.6},
    {"angle": 0.9, "noise": 9.0, "blur": 0.5},
    {"angle": -0.4, "noise": 5.0, "blur": 0.3},
    {"angle": 1.3, "noise": 11.0, "blur": 0.7},
    {"angle": -0.8, "noise": 8.0, "blur": 0.45},
    {"angle": 0.3, "noise": 13.0, "blur": 0.8},
]
"""Scanner characteristics, cycled across the layouts so no two scans are
degraded identically. Real scanners differ in how they feed paper and how much
their sensors hiss, and a corpus where every page is skewed by the same 0.6
degrees would prove less than it appears to."""

HARDER = {"easy": "medium", "medium": "hard", "hard": "hard"}


def _scanned(layout: Layout, profile: dict) -> Sample:
    """The same layout as a photocopy: no text layer, so OCR has to carry it."""
    return Sample(
        name=f"scan_{layout.name}",
        draw=layout.draw,
        expected=layout.expected,
        difficulty=HARDER[layout.difficulty],
        note=f"Scanned. {layout.note}",
        landscape=layout.landscape,
        scan=profile,
    )


# Every layout is produced twice: once as a digital PDF carrying a text layer,
# and once flattened to an image. The pair matters more than either half. A
# difference between them is never the layout -- it is what the recogniser did
# to it, which is the only way to see the OCR path's failures separately from
# the rules'.
SAMPLES: list[Sample] = [
    Sample(
        name=layout.name,
        draw=layout.draw,
        expected=layout.expected,
        difficulty=layout.difficulty,
        note=layout.note,
        landscape=layout.landscape,
    )
    for layout in LAYOUTS
] + [
    _scanned(layout, SCAN_PROFILES[index % len(SCAN_PROFILES)])
    for index, layout in enumerate(LAYOUTS)
]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _flatten_to_scan(
    document: pymupdf.Document,
    angle: float = 0.0,
    noise: float = 0.0,
    blur: float = 0.0,
    dpi: int = 200,
    seed: int = 0,
) -> pymupdf.Document:
    """Re-render a document as page images, the way a scanner would.

    The result has no text layer at all, which is the point: it forces the OCR
    path. The skew, noise and softening imitate what a sheet-fed office scanner
    does to a page, so the recogniser is not being handed a clean render.
    """
    generator = np.random.default_rng(seed)
    scanned = pymupdf.open()

    for page in document:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

        if angle:
            image = image.rotate(
                angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
            )
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        if noise:
            array = np.asarray(image, dtype=np.float32)
            array += generator.normal(0.0, noise, array.shape)
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

        buffer = io.BytesIO()
        # JPEG rather than PNG so the compression artefacts are present too.
        image.save(buffer, format="JPEG", quality=80)

        new_page = scanned.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=buffer.getvalue())

    return scanned


def build(sample: Sample) -> pymupdf.Document:
    """Render one sample to an in-memory document."""
    width, height = A4
    if sample.landscape:
        width, height = height, width

    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    sample.draw(page)

    if sample.scan is None:
        return document

    scan_options = dict(sample.scan)
    scanned = _flatten_to_scan(
        document,
        angle=scan_options.get("angle", 0.0),
        noise=scan_options.get("noise", 0.0),
        blur=scan_options.get("blur", 0.0),
        dpi=scan_options.get("dpi", 200),
        seed=abs(hash(sample.name)) % (2**32),
    )
    document.close()
    return scanned


def write_all(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Generate every sample into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for sample in SAMPLES:
        path = output_dir / f"{sample.name}.pdf"
        document = build(sample)
        document.save(path)
        document.close()
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"where to write the PDFs (default: {DEFAULT_OUTPUT_DIR})",
    )
    arguments = parser.parse_args()

    random.seed(0)
    written = write_all(arguments.output)

    print(f"wrote {len(written)} samples to {arguments.output}")
    for sample, path in zip(SAMPLES, written, strict=True):
        kind = "scan " if sample.scan else "text "
        print(
            f"  {kind} {sample.difficulty:<6} {sample.expected.value:<13}"
            f" {path.name:<36} {sample.note}"
        )


if __name__ == "__main__":
    main()
