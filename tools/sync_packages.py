"""Copies an exported model (tools/export_onnx.py) and the shared word list into the Python package.

  python tools/sync_packages.py export/

Run it after every export; then bump the version in packages/python/pyproject.toml and
packages/python/src/egyptian_arabizi_franco/__init__.py, and add the release to CHANGELOG.md."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("encoder.onnx", "decoder.onnx", "vocab.json")
TARGETS = (ROOT / "packages/python/src/egyptian_arabizi_franco",)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "export")
    missing = [f for f in FILES if not (src / f).exists()]
    if missing:
        sys.exit(f"missing in {src}: {', '.join(missing)}")
    for target in TARGETS:
        (target / "models").mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copy2(src / f, target / "models" / f)
        shutil.copy2(ROOT / "packages/shared/english_words.txt", target / "english_words.txt")
        print("updated", target.relative_to(ROOT))


if __name__ == "__main__":
    main()
