"""Losslessly recompress the documentation PNGs (run after scripts/generate_gallery.py).

Usage: python scripts/optimize_assets.py [directory ...]     # default: docs/assets
Requires ``pyoxipng`` (``pip install pyoxipng``, included in the ``docs`` extra). The decoded pixels
are unchanged; only the PNG encoding is made smaller, and every file is verified before it is replaced.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


def main(arguments: list[str]) -> int:
    try:
        import oxipng
    except ImportError:
        print("pyoxipng is not installed (pip install pyoxipng); nothing to do.")
        return 1

    directories = [Path(a) for a in arguments] or [PROJECT_ROOT / "docs" / "assets"]
    before = after = 0
    for directory in directories:
        for path in sorted(directory.glob("*.png")):
            original = path.read_bytes()
            optimized = oxipng.optimize_from_memory(original, level=4)
            before += len(original)
            if len(optimized) < len(original) and np.array_equal(pixels(original), pixels(optimized)):
                path.write_bytes(optimized)
                after += len(optimized)
            else:
                after += len(original)
    saved = before - after
    print(f"{before / 1e6:.2f} MB -> {after / 1e6:.2f} MB ({saved / max(before, 1):.0%} saved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
