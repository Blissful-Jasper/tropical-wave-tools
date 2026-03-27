"""Custom exceptions for the package."""


class TropicalWaveToolsError(Exception):
    """Base exception for the package."""


class InvalidDataArrayError(TropicalWaveToolsError):
    """Raised when input data dimensions or coordinates are incompatible."""


class UnknownWaveError(TropicalWaveToolsError):
    """Raised when an unsupported wave name is requested."""

