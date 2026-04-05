# Case 02: WK Spectrum

<img src="../../assets/wk_spectrum.png" alt="WK spectrum example output">

## Minimal Code

```python
from tropical_wave_tools import open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.plotting import plot_wk_spectrum

result = analyze_wk_spectrum(open_example_olr())
fig, axes = plot_wk_spectrum(result, annotate_filter_boxes=True)
```

## Core Functions

- `analyze_wk_spectrum`
- `plot_wk_spectrum`

## Source Files

- [`src/tropical_wave_tools/spectral.py`](https://github.com/Blissful-Jasper/tropical-wave-tools/blob/main/src/tropical_wave_tools/spectral.py)
- [`src/tropical_wave_tools/plotting.py`](https://github.com/Blissful-Jasper/tropical-wave-tools/blob/main/src/tropical_wave_tools/plotting.py)
- [`examples/wk_spectrum_demo.py`](https://github.com/Blissful-Jasper/tropical-wave-tools/blob/main/examples/wk_spectrum_demo.py)
- [`scripts/generate_gallery.py`](https://github.com/Blissful-Jasper/tropical-wave-tools/blob/main/scripts/generate_gallery.py)
