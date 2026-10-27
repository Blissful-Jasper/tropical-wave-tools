# Changelog

## Unreleased

### Fixed
- Monthly-RMS significance (`compute_monthly_climatology_and_significance` and its longitude-mean
  variant, now one shared implementation): the annual mean used for the anomalies included partial
  first/last years, shifting the t-test (p = 0.028 vs 0.016 for a record starting in July). Only
  complete 12-month years enter the annual mean now; the climatology still uses every month.
- Atlas latitude selections (`equatorial_mean`, `wave_longitude_projection`, Hovmoller and Case 05/10
  bands, spectral band) used `sel(lat=slice(lo, hi))`, which is empty on a *descending* latitude axis
  (the usual ERA5 layout): every equatorial mean became NaN, event detection silently returned zero
  events, and antisymmetric waves fell back to an equatorial mean. A new `_lat_range` handles both
  orders.
- `detect_wave_events` / `detect_point_events`: `min_distance_days` was passed to `find_peaks` as a
  number of *samples*, so for 6-hourly or hourly data events a few days apart were not merged. It is
  now converted with the time step (daily data is unchanged). An all-NaN reference series now warns
  instead of silently returning no events. The two functions share one implementation.
- `phase.lag_composite` crashed when `lon_ref` was not exactly a grid longitude (float equality) or
  when exactly one reference time qualified. `lon_ref` now snaps to the nearest grid longitude (error
  if farther than one grid step) and any number of events is handled.
- `phase.phase_composite` used `n_bins` bins over `2*pi + width`, so bins were wider than `2*pi/n_bins`
  and the `+-pi` phase was split over two half-populated end bins. Bins are now equal-width and
  wrap-aware (centres shift accordingly).
- `phase.find_local_extrema` flagged every sample of a flat run as an extremum (a constant series was
  all "peaks"); a plateau is now one extremum.
- Peak amplitudes were stored as float32, limiting the phase at crests to ~1e-4 rad; now float64.
- `phase.remove_clm` was named and labelled "lowpass_filtered_anomaly" but only removes the time mean
  (its `fs`/`cutoff`/`order` arguments were unused); the output is now named `anomaly` and the
  docstring says so.
- `pearson_correlation` / `linear_regression` computed means and standard deviations over
  different samples when the inputs had different missing values, so the correlation could leave
  [-1, 1] (1.014 in a test) and slopes were biased. Both now use pairwise-complete samples. This
  also feeds the Atlas Case 10 lagged regressions.
- `EOFAnalyzer` (horizontal EOF): a single masked grid point (e.g. a land mask) removed *every*
  time step and silently returned zero modes. Missing grid points are now dropped instead.
- `EOFAnalyzer` (vertical EOF): `degrees_of_freedom` took the lag-1 autocorrelation along the stacked
  (time, lat, lon) axis, i.e. between neighbouring longitudes (86 387 "independent" samples for
  600 days). It is now the temporal effective sample size (39 where the true value is ~38), which
  feeds `eigenvalue_errors`.
- `relative_vorticity` / `horizontal_divergence` omitted the spherical metric terms
  (`+u tan(phi)/a`, `-v tan(phi)/a`): vorticity of solid-body rotation was half its true value at
  25N. They are now included and the periodic longitude seam is handled with a wrapped column.
- With MetPy installed, CF wind units such as `m s-1` crashed the diagnostics (the unit string was
  parsed as 1/millisecond). CF spellings are now accepted; unreadable units raise a clear error.
- `WaveFilter._kf_filter` returned a scrambled field when the longitude axis
  contained a cyclic point (`lon[0] == lon[-1] mod 360`): the zonal FFT was
  inverted on the wrong length. The cyclic column is now restored explicitly.
- `netCDF4` was missing from the declared dependencies although NetCDF I/O is
  core functionality.

### Changed
- Taper length is now `round(N * fraction) // 2` points per end (NCL rule fixed by the 382-sample
  golden file: 9 points for N=382 at 5%, 5 points for N=96 at 10%). With it all 21 filter
  comparisons (7 bands x 3 inputs, including the `ig0`/`ig1` branches) agree with NCL to ~1e-7.
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
- Case 10 outputs (`shading`, `contours`) now carry `long_name`, `predictor`, `units` ("per standard
  deviation") and `lag_convention` attributes. The predictor is the *filtered OLR* at the base point
  (+1 sigma = suppressed convection) while the shading is the convective proxy (-OLR), so the shading
  is negative at the base point at lag 0 and the contours are positive.
- Checks for variance fractions (known ratios, hand-computed area weighting incl. longitude ranges that
  wrap through 0), the monthly t-test (against scipy), and the Case 10 pipeline (against
  `scipy.stats.linregress` on the active samples, propagation direction and speed).
- `lagged_composite` returns the number of events behind each lag as the `n_events` coordinate
  (events whose shifted index leaves the record are dropped per lag, so the set shrinks at large lags).
- Event-detection checks: known sinusoids, sub-daily sampling, descending latitude, NaN gaps, composite
  means, Hovmoller window, activity mask, projection conventions, and an end-to-end wave-packet test
  (detect events, composite, recover the 15 m/s propagation within 10%).
- `composite_kw_phase(..., to_mm_per_day=None, strong_threshold=1.5)`: the unit factor (previously
  hard-coded to 86400 except for model name "GPCP") and the phase-speed threshold are now arguments
  with the historical defaults. `optimize_peak_detection` logs timings at DEBUG instead of printing.
- Phase checks against scipy, known phase, analytic bin means and a wave packet of known speed
  (`Cp_ave` recovers 10, 15 and 20 m/s within 0.5%).
- `linear_trend(..., adjust_for_autocorrelation=True)`: effective-sample-size correction for the
  slope uncertainty. Without it, AR(1) noise with rho=0.9 gives ~70% false positives at the nominal
  5% level (5% -> 8% with the option).
- `tests/test_reference_checks.py`: statistics, EOF, Matsuno branches and cross-spectrum phase
  checked against scipy, an eigen-decomposition, the dispersion cubic and plane waves of known phase.
- Analytic plane-wave tests for the filter and WK-spectrum sign convention.
- `validation/` tooling to cross-check `kf_filter` against NCL output.
- Redesigned Streamlit app and dark-mode aware documentation theme.
