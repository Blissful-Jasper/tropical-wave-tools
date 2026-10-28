# Tropical Wave Tools

[中文](README.md) | **English**

A Python toolkit for equatorial wave analysis and tropical climate diagnostics: Wheeler–Kiladis
wavenumber–frequency spectra, Kelvin / ER / MRG / IG / TD / MJO filtering, EOF and regression
analysis, event composites and a set of publication-style figures. It keeps the scientific logic of the
original `wave_tools` research code in a tested, packaged form, and its filter and spectrum
implementations are cross-checked against NCL (see [Validation](#validation)).

Documentation: <https://Blissful-Jasper.github.io/tropical-wave-tools/> (currently in Chinese).

## Install

```bash
git clone https://github.com/Blissful-Jasper/tropical-wave-tools.git
cd tropical-wave-tools
pip install -e ".[dev]"            # numpy, scipy, xarray, netCDF4, matplotlib, pytest, ruff
```

Optional extras: `docs` (mkdocs), `app` (Streamlit demo), `diagnostics` (MetPy), `advanced`
(dask, xeofs, land masks), `maps` (cartopy, for projected maps), `grid` (healpy, numba).
Python 3.9 or newer. Without cartopy the map plots fall back to plain longitude-latitude axes.

## Quick start

A one-year equatorial OLR sample ships with the package.

```python
from tropical_wave_tools.sample_data import open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.plotting import plot_wk_spectrum

olr = open_example_olr()

# Wheeler–Kiladis spectrum (96-day windows overlapping by 60 days, as in WK99)
result = analyze_wk_spectrum(olr)
figure, axes = plot_wk_spectrum(result)

# Kelvin-wave filtered field (3-20 days, zonal wavenumbers 2-14, equivalent depth 8-90 m)
kelvin = filter_wave_signal(olr, wave_name="kelvin", method="cckw", n_workers=1)
```

Command line:

```bash
tropical-wave-tools wk-spectrum --input src/tropical_wave_tools/data/olr_equatorial_1979.nc \
    --var olr --output-dir outputs/wk_demo
tropical-wave-tools filter-wave --input data/local/olr.day.mean.nc --var olr \
    --wave kelvin --method cckw --output outputs/kelvin_filtered.nc
```

An interactive demo: `streamlit run apps/streamlit_app.py` (needs the `app` extra).

## Conventions worth knowing

- **Window spacing** follows NCL `wkSpaceTime`: `SpectralConfig.window_skip_days` is the *gap* between
  windows and a negative value overlaps them, so window starts are `window_size_days + window_skip_days`
  apart. The default `-60` is the Wheeler & Kiladis (1999) setting.
- **Wavenumber sign**: positive zonal wavenumber is eastward propagation.
- **Kelvin phase** (`phase.calculate_kelvin_phase`) returns `-θ` for `x = A sin θ`, so crests sit at `-π/2`.
- **Case 10 regressions** use the *filtered OLR* at the base point (standardised, +1σ = suppressed
  convection) as predictor; the shading shows the convective proxy `-OLR`, so it is negative at the
  base point at lag 0. Positive lag means the field lags the reference.
- **Latitude order**: both ascending and descending latitude axes are supported.

## Validation

| Component | Checked against | Result |
| --- | --- | --- |
| Wave filter (`kf_filter`; Kelvin, ER, MRG, TD, MJO, EIG, WIG) | NCL `kf_filter`, 3 inputs | agrees to ~1e-7 (NCL output is single precision) |
| WK spectrum | NCL `wkSpaceTime` | agrees to ≲ 3e-7 for `nDaySkip = 30`; ~1e-6 median for overlapping windows, with one open low-frequency discrepancy |
| EOF, regression, correlation | eigen-decomposition, `scipy` | agree; missing-value handling fixed |
| Matsuno dispersion curves | the dispersion cubic | residual ≲ 1e-15 |
| Cross-spectrum, Kelvin phase composites, event detection, Case 10 regressions | plane waves / wave packets of known phase and speed | recover phase, speed and direction |

NCL cross-checks use committed input and reference files in `validation/`; running them does not
require NCL (only regenerating the references does, see `validation/README_ncl.md`). Details and the
list of components not yet independently validated are in `docs/validation.md` (Chinese).
`CHANGELOG.md` lists the behaviour changes that came out of this work, including the new default of
`window_skip_days`.

## Package layout

```text
src/tropical_wave_tools/
├── io.py  preprocessing.py  stats.py  diagnostics.py
├── spectral.py  filters.py  cross_spectrum.py  matsuno.py
├── eof.py  phase.py
├── plotting/   _style  _geo  _wind  spectrum  hovmoller  maps  eof  climatology
├── atlas/      _config  _paths  monthly  events  case10  wave_eof  workflow
└── cli.py  workflows.py  sample_data.py
```

`tropical_wave_tools.preprocess` is a deprecated alias of `tropical_wave_tools.preprocessing`.

## Development

```bash
pytest                                  # unit, reference and NCL cross-check tests
ruff check src tests apps scripts examples validation
python scripts/generate_gallery.py      # regenerate the documentation figures (needs data/local)
python scripts/optimize_assets.py       # lossless PNG recompression (pip install pyoxipng)
mkdocs build                            # documentation site
```

## Citation and license

If you use this software, please cite it via `CITATION.cff` and the methods papers it implements
(Wheeler & Kiladis, 1999; Kiladis et al., 2009). MIT license.
