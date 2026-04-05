# Quick Start

## 安装与常用命令

```bash
bash scripts/twave.sh setup
bash scripts/twave.sh test
bash scripts/twave.sh docs
bash scripts/twave.sh app
```

```bash
bash scripts/twave.sh docs-serve
# if the default port is busy
bash scripts/twave.sh docs-serve --port 8012
```

## 内置 OLR 样例

```python
from tropical_wave_tools import open_example_olr

olr = open_example_olr()
print(olr)
```

## WK 频谱

```python
from tropical_wave_tools import open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.plotting import plot_wk_spectrum

result = analyze_wk_spectrum(open_example_olr())
fig, axes = plot_wk_spectrum(result)
```

## Kelvin 滤波

```python
from tropical_wave_tools import open_example_olr
from tropical_wave_tools.filters import filter_wave_signal

olr = open_example_olr()
kelvin = filter_wave_signal(olr, wave_name="kelvin", method="cckw", n_workers=1)
```

## Gallery 产图

```bash
python scripts/generate_gallery.py
```

## CLI

```bash
tropical-wave-tools wk-spectrum \
  --input src/tropical_wave_tools/data/olr_equatorial_1979.nc \
  --var olr \
  --output-dir outputs/wk_demo
```

```bash
tropical-wave-tools filter-wave \
  --input src/tropical_wave_tools/data/olr_equatorial_1979.nc \
  --var olr \
  --wave kelvin \
  --method cckw \
  --output outputs/kelvin_demo.nc
```

## 下一页

- 想看结果导向的展示方式：去 [Gallery 与示例](examples.md)
- 想按模块理解函数：去 [API 与工作流](api/index.md)
- 想按原理理解整条分析链：去 [方法与原理](theory/index.md)
