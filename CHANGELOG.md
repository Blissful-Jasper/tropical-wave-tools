# Changelog

## Unreleased

### Fixed
- `WaveFilter._kf_filter` returned a scrambled field when the longitude axis
  contained a cyclic point (`lon[0] == lon[-1] mod 360`): the zonal FFT was
  inverted on the wrong length. The cyclic column is now restored explicitly.
- `netCDF4` was missing from the declared dependencies although NetCDF I/O is
  core functionality.

### Changed
- **Behaviour change:** `SpectralConfig.window_skip_days` now follows NCL `wkSpaceTime`
  (`nDaySkip`): the gap between windows, negative = overlap, so starts are
  `window_size_days + window_skip_days` apart. The default changes from `30` (a 30-day step)
  to `-60` (Wheeler-Kiladis 1999: 96-day windows overlapping by 60 days); the old behaviour is
  `window_skip_days=-66`. The CLI `--skip-days` default and the Streamlit control follow.
- `analyze_wk_spectrum` now reproduces NCL `wkSpaceTime`: split-cosine-bell window taper,
  one 1-2-1 frequency pass on the raw symmetric/antisymmetric spectra (zero-frequency and
  Nyquist rows excluded), a frequency axis that includes the Nyquist bin, and background
  smoothing that leaves out the zero-frequency row. With `nDaySkip=30` every field agrees to
  <= 1.6e-6; with `-60` the median agrees to ~1e-6 and one low-frequency bin still differs.
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
