"""Build the distributable folder and report what it weighs.

Produces ``dist/pdf-sorter/`` containing the executable, the recognition models
and an editable rules.yaml. The whole directory is the unit of distribution:
copy it to another PC and it runs, with no Python and no network access.

Downloading the OCR models is refused once frozen, precisely so that a deployed
copy cannot silently depend on the internet. That makes putting them here part
of the build rather than something the first run does.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = PROJECT_ROOT / "pdf-sorter.spec"
DIST_ROOT = PROJECT_ROOT / "dist"
RULES = PROJECT_ROOT / "src" / "pdf_ocr" / "rules.yaml"
MODELS = PROJECT_ROOT / "models"


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def megabytes(size: int) -> str:
    return f"{size / 1024 / 1024:,.0f} MB"


def largest(path: Path, count: int = 12) -> list[tuple[str, int]]:
    """The biggest things in the bundle, which is where trimming has to start."""
    sizes: dict[str, int] = {}
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(path)
        # Group by top-level entry inside _internal, where the bulk lives.
        parts = relative.parts
        key = "/".join(parts[:2]) if parts[0] == "_internal" else parts[0]
        sizes[key] = sizes.get(key, 0) + item.stat().st_size
    return sorted(sizes.items(), key=lambda pair: pair[1], reverse=True)[:count]


def ensure_models() -> None:
    if MODELS.is_dir() and any(MODELS.glob("*.pth")):
        return
    sys.exit(
        f"no OCR models in {MODELS}.\n"
        "Run the classifier once against a scanned PDF first -- during "
        "development downloading is allowed, and the models land there:\n"
        "  poetry run python -m tools.evaluate"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onefile",
        action="store_true",
        help=(
            "build a single compressed executable: less than half the size, but "
            "it unpacks itself on every launch and costs about seven seconds a "
            "run. Sensible when the flow calls 'batch' once per folder."
        ),
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="build without bundling the models, to measure the code alone",
    )
    parser.add_argument("--clean", action="store_true", help="discard cached analysis")
    arguments = parser.parse_args()

    if not arguments.skip_models:
        ensure_models()

    environment = dict(os.environ)
    environment["PDF_SORTER_ONEFILE"] = "1" if arguments.onefile else "0"

    command = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm"]
    if arguments.clean:
        command.append("--clean")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=environment)

    # onefile leaves a single executable in dist/; onedir leaves a folder. Either
    # way the models and rules sit beside the executable, which is what
    # default_model_dir() and resolve_rules_path() look for at run time.
    beside = DIST_ROOT if arguments.onefile else DIST_ROOT / "pdf-sorter"
    code_size = directory_size(beside)

    shutil.copy2(RULES, beside / "rules.yaml")
    if not arguments.skip_models:
        shutil.copytree(MODELS, beside / "models", dirs_exist_ok=True)

    total = directory_size(beside)
    print(f"\n{beside}")
    print(f"  code and dependencies  {megabytes(code_size)}")
    print(f"  models                 {megabytes(total - code_size)}")
    print(f"  total                  {megabytes(total)}")
    print("\nlargest components:")
    for name, size in largest(beside):
        print(f"  {megabytes(size):>10}  {name}")


if __name__ == "__main__":
    main()
