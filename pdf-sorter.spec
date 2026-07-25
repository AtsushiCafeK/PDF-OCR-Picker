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

# Whole packages nothing here imports.
#
# Submodules of torch are deliberately NOT excluded, however tempting the size
# is. Excluding torch.distributed builds without complaint and then dies on the
# target machine, because torch.utils.data.dataloader imports it during a plain
# `import torch`. PyTorch's internal imports are dense enough that pruning them
# trades a smaller bundle for a failure that only appears in production.
EXCLUDES = [
    "matplotlib",
    "tkinter",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "_pytest",
]

# Qt subsystems the GUI never touches. Unlike torch, these are safe to drop:
# they are optional modules with no import path from QtWidgets, and PySide6 as
# installed is 634 MB of which the window here uses only Core, Gui and Widgets.
# WebEngineCore alone is 195 MB of browser that nothing opens.
EXCLUDES += [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtWebView",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtBluetooth",
    "PySide6.QtPositioning",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
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

COMMON = dict(
    debug=False,
    strip=False,
    # UPX is left off: it mangles some of the PyTorch DLLs, and the failure
    # shows up at run time on the target machine rather than here at build time.
    upx=False,
)

# Two executables over one set of dependencies.
#
# pdf-sorter.exe keeps its console because the Power Automate contract is a JSON
# object on stdout, and a windowed build has no stdout to write it to --
# --noconsole would not hide an interface, it would remove one.
#
# pdf-sorter-gui.exe is windowed, for the people who open the tuning window by
# double-clicking it and have no use for a console behind it. It runs the same
# module and recognises itself by its filename.
#
# They are separate EXE objects sharing one COLLECT rather than two bundles,
# because a second bundle would carry its own PyTorch and its own OCR models --
# some 550 MB duplicated to gain nothing.
if ONEFILE:
    # One self-contained file cannot also be a second self-contained file
    # without doubling in size, so the onefile layout ships the console build
    # only; `pdf-sorter.exe gui` still opens the window.
    console_exe = EXE(
        pyz,
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        [],
        name="pdf-sorter",
        console=True,
        **COMMON,
    )
else:
    console_exe = EXE(
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name="pdf-sorter",
        console=True,
        **COMMON,
    )
    gui_exe = EXE(
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name="pdf-sorter-gui",
        console=False,
        **COMMON,
    )
    collection = COLLECT(
        console_exe,
        gui_exe,
        analysis.binaries,
        analysis.datas,
        strip=False,
        upx=False,
        name="pdf-sorter",
    )
