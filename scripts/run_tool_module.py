from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import site
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
USER_SITE = Path(site.getusersitepackages())


def _preload_scientific_stack() -> None:
    # Keep compiled scientific packages pinned to the stable base environment
    # before exposing the user's site-packages for pure-Python tooling.
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import xarray  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Python module with project-aware path setup.")
    parser.add_argument("module", help="Module name, for example mkdocs or streamlit.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the module.")
    args = parser.parse_args()

    _preload_scientific_stack()

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if USER_SITE.exists():
        sys.path.insert(0, str(USER_SITE))

    sys.argv = [args.module, *args.args]
    runpy.run_module(args.module, run_name="__main__")


if __name__ == "__main__":
    main()
