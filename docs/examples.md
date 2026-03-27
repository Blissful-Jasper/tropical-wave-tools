# 示例与结果

## Python API 示例

```python
from tropical_wave_tools import open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.plotting import plot_wk_spectrum

data = open_example_olr()
result = analyze_wk_spectrum(data)
fig, axes = plot_wk_spectrum(result)
```

## CLI 示例

```bash
tropical-wave-tools wk-spectrum \
  --input src/tropical_wave_tools/data/olr_equatorial_1979_jja.nc \
  --output-dir outputs/wk_demo
```

## 输出结果

- `wk_spectrum.nc`: 频谱结果数据文件
- `wk_spectrum.png`: 频谱图
- `*_spatial_std_comparison.nc`: legacy / CCKW 对比结果
- `wave_filter_spatial_comparison_summary.csv`: 对比指标汇总

