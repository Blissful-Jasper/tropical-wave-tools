# Changelog

## Unreleased

### Fixed
- `io.normalize_longitude` (and so `standardize_data`) left a duplicated column when the file stored a
  cyclic point (0 and 360, or -180 and 180): after wrapping, longitude 0 appeared twice, which defeats
  the cyclic-point detection in `_kf_filter`. Duplicated longitudes are now dropped. A longitude of
  `-1e-14` also wrapped to exactly `360.0` (`-1e-14 % 360`); it now maps to `0.0`.
- `valid_range` was compared with the *packed* values converted to physical units, but CF defines it in
  packed units. For packed data (`scale_factor` / `add_offset`) the bounds are now converted before
  masking (a test file kept all 120 points instead of the 27 inside the range).
- `io.infer_variable_name` returned `crs` (or another scalar/auxiliary variable) when it came first; it
  now picks the non-bounds variable that covers the most of time / lat / lon.
- Saving attributes the netCDF format cannot hold (`None`, tuples containing `None` such as an unset
  equivalent depth, booleans, dictionaries) raised `TypeError`; this also broke
  `filter-wave --method legacy` for TD and MJO. Attributes are sanitized: `None` is dropped, booleans
  become 0/1, dictionaries become JSON, tuples with `None` become float arrays with NaN.
- `stats.linear_trend` and the minimum event separation of `detect_wave_events` crashed on non-standard
  calendars (`noleap`, `360_day`, i.e. cftime time axes from CMIP models). New helpers
  `io.time_axis_in_days` / `io.time_deltas_in_days` handle datetime64 and cftime axes alike.
- `workflows.compute_spatial_metrics` averaged each field over its own valid points, so with missing
  data `mean_bias` was not the difference of the two reported means. Both now use the jointly valid
  points.
- `CCKWFilter` silently returned all zeros for a wave band that cannot be resolved at the sampling
  rate (EIG, WIG, IG with daily data). It now emits a `RuntimeWarning` and records a `note` attribute;
  `method="legacy"` clips such a band at the Nyquist frequency instead.
- `atlas.regress_field_onto_pcs` took the PC mean over all times but the field mean over the times the
  field exists, so missing days biased the slope (0.5% with 23% missing days, more with clustered gaps).
  It now uses the days valid in both (same fix as `stats.linear_regression`).
- `cross_spectrum_analysis.preprocess_data_with_mask`: a float mask with NaN (`1` / `NaN`, the common
  land-sea format) was ignored because `NaN.astype(bool)` is `True`. NaN and 0 now mean "masked out".
- `calculate_cross_spectrum` silently paired latitudes by array position, so an asymmetric grid
  (e.g. -5..15) was accepted and the "symmetric" component was meaningless. xarray input is now
  transposed to (time, lat, lon), sorted to ascending latitude and must be symmetric about the equator.
  The antisymmetric component is therefore always `0.5 * (south - north)` whatever the input order.
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
- `compute_wave_eof` fixes the sign of every mode (largest-magnitude loading positive; EOF and PC flip
  together). The sign was whatever the SVD routine returned, so figures could mirror between machines;
  patterns and explained variance are unchanged, but individual published figures may flip once.
- **Repository layout:** `plotting.py` is now the package `plotting/` (`_style`, `_geo`, `_wind`,
  `spectrum`, `hovmoller`, `maps`, `eof`, `climatology`) and `atlas.py` the package `atlas/` (`_config`,
  `_paths`, `monthly`, `events`, `case10`, `wave_eof`, `workflow`). The split was generated from the AST
  (every definition is AST-identical to the old one, apart from `_PROJECT_ROOT`, which now lives one
  directory deeper); all old names remain importable from `tropical_wave_tools.plotting` and
  `tropical_wave_tools.atlas`, and representative figures render pixel-identically.
- `preprocess.py` was merged into `preprocessing.py`; `tropical_wave_tools.preprocess` remains as a
  deprecated alias that emits a `DeprecationWarning`.
- Documentation PNGs were recompressed losslessly (15.4 MB -> 10.7 MB, identical pixels) with the new
  `scripts/optimize_assets.py`. The repository history still contains the old files.
- Build requirement raised to `setuptools>=77`: `license = "MIT"` / `license-files` (PEP 639) need it,
  while `>=61` was declared. CI now builds the wheel and checks that the new subpackages are inside.
- **Phase-speed estimator:** `composite_kw_phase` now estimates `Cp_ave` by following the single
  strongest crest of the lag-longitude composite from `lon_ref` (`speed_method="ridge"`, new
  `ridge_window_deg`), instead of one straight line through every point above the threshold. The
  old fit failed whenever the composite held several crests (a wavenumber-3 wave gave -1.5 m/s for a
  true +15 m/s) and when the ridge crossed 0/360 within the lag range (22 m/s read as 11.4 m/s). For a
  single localised ridge both agree to ~1%. The old behaviour stays available as
  `speed_method="threshold_fit"`. New public helpers: `trace_composite_ridge`, `estimate_phase_speed`.
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
- `prepare-sample-data --output PATH`: the command could only overwrite the packaged sample file.
- Checks for `io`, `workflows` and the command-line entry points (`tests/test_io_and_workflows.py`):
  cyclic longitudes (and filtering a grid that stores one), packed `valid_range`, variable inference,
  attribute round trips, noleap / 360_day calendars, selection on descending latitude and wrapping
  longitude ranges, spatial metrics against numpy, the WK workflow against a direct call, the
  unresolvable-band warning, and each CLI subcommand end to end.
- `compare_filter_spatial_fields` documents that `legacy` and `cckw` share one implementation, so
  RMSE 0 / correlation 1 is not an independent check (see `docs/validation.md` for the NCL comparison).
- `calculate_cross_spectrum(..., dof_per_segment=2.667)` (also in `compute_cross_spectrum_for_experiments`
  and `analyze_cross_spectrum`): the degrees of freedom behind the coherence significance were a fixed
  `2.667 * nseg`. With 96-day segments overlapping by 65 days and independent noise, the nominal 5%
  false-alarm rate was 9% when all latitudes are perfectly coherent and 0.2% when they are independent,
  so the constant is a convention to calibrate for your data; this is now documented.
- Checks for the wave EOFs (known symmetric/antisymmetric modes, weighting, regression) and the
  cross-spectrum workflow (masks, seasonal-cycle removal, latitude handling, coherence bounds and null
  false-alarm rates, recovered phase and coherence).
- `README.en.md` (English README) linked from the Chinese README.
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
