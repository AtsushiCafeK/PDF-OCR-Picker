# PyInstaller build definition. Committed deliberately -- this file *is* the
# build configuration, and .gitignore carries an exception for it.
#
# Two layouts, measured on this project rather than assumed:
#
#             on disk   startup   one scanned page
#   onedir     816 MB      0.5s        16.7s
#   onefile    353 MB      7.3s        21.3s
#
# onefile is compressed and so is less than half the size, but it unpacks itself
# into a temporary directory on every single launch, and this bundle is
# dominated by PyTorch. Seven seconds of that is paid per invocation, before any
# work happens. Which layout is right therefore depends entirely on how the flow
# calls the tool: once per folder via `batch`, onefile costs seven seconds a run
# and saves 460 MB; once per file via `classify`, it costs seven seconds a file.
#
# onedir is the default because it is the one that cannot go badly wrong.
#
# Build with `python -m tools.build_exe [--onefile]`, which also puts rules.yaml
# and the OCR models beside the executable afterwards.

import os

from PyInstaller.utils.hooks import collect_data_files

ONEFILE = os.environ.get("PDF_SORTER_ONEFILE") == "1"

# Whole packages the classifier never imports. The debug GUI is a development
# tool and is not shipped, so PySide6 -- several hundred megabytes of it --
# stays out.
#
# Submodules of torch are deliberately NOT excluded, however tempting the size
# is. Excluding torch.distributed builds without complaint and then dies on the
# target machine, because torch.utils.data.dataloader imports it during a plain
# `import torch`. PyTorch's internal imports are dense enough that pruning them
# trades a smaller bundle for a failure that only appears in production.
EXCLUDES = [
    "PySide6",
    "shiboken6",
    "matplotlib",
    "tkinter",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "_pytest",
]

datas = [
    # Also copied beside the executable by the build script; this bundled copy
    # is the fallback when that one has been deleted.
    ("src/pdf_ocr/rules.yaml", "pdf_ocr"),
]
datas += collect_data_files("easyocr")

analysis = Analysis(
    ["src/pdf_ocr/cli.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=["pdf_ocr.core.ocr.easy"],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(analysis.pure)

# console=True in both layouts. The Power Automate contract is a JSON object on
# stdout, and a windowed build has no stdout to write it to -- --noconsole would
# not merely hide a window, it would remove the interface. A console that
# flashes is suppressed on the caller's side instead, by running the process
# hidden, or avoided entirely by reading the --out JSONL file.
COMMON = dict(
    name="pdf-sorter",
    debug=False,
    strip=False,
    # UPX is left off: it mangles some of the PyTorch DLLs, and the failure
    # shows up at run time on the target machine rather than here at build time.
    upx=False,
    console=True,
)

if ONEFILE:
    exe = EXE(
        pyz, analysis.scripts, analysis.binaries, analysis.datas, [], **COMMON
    )
else:
    exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, **COMMON)
    collection = COLLECT(
        exe, analysis.binaries, analysis.datas, strip=False, upx=False, name="pdf-sorter"
    )
