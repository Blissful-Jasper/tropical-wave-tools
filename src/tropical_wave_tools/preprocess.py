"""Deprecated alias of :mod:`tropical_wave_tools.preprocessing`.

``preprocess`` and ``preprocessing`` were merged; import from ``tropical_wave_tools.preprocessing``.
"""

from __future__ import annotations

import warnings

from tropical_wave_tools.preprocessing import *  # noqa: F403
from tropical_wave_tools.preprocessing import __all__  # noqa: F401

warnings.warn(
    "`tropical_wave_tools.preprocess` is deprecated; import from `tropical_wave_tools.preprocessing`.",
    DeprecationWarning,
    stacklevel=2,
)
