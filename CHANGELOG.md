# Changelog

## Unreleased

### Fixed
- `WaveFilter._kf_filter` returned a scrambled field when the longitude axis
  contained a cyclic point (`lon[0] == lon[-1] mod 360`): the zonal FFT was
  inverted on the wrong length. The cyclic column is now restored explicitly.
- `netCDF4` was missing from the declared dependencies although NetCDF I/O is
  core functionality.

### Changed
- `WaveFilter._kf_filter` now reproduces NCL `kf_filter` to float32 precision (relative RMS
  <= 1e-7 on five bands x two inputs): split-cosine-bell time taper instead of Tukey, and
  period cut-offs rounded to the nearest frequency index instead of truncated. Before, the
  band-edge bin differed (up to 34% RMS for MJO on real OLR) and taper leakage by ~1%.
- Removed the `numpy<2` cap (tests pass on NumPy 2.x).
- Cleaned all `ruff` findings; CI now runs lint, a Python 3.9/3.11/3.12 test
  matrix with coverage, and a documentation build on every pull request.

### Added
- Analytic plane-wave tests for the filter and WK-spectrum sign convention.
- `validation/` tooling to cross-check `kf_filter` against NCL output.
- Redesigned Streamlit app and dark-mode aware documentation theme.
