# API 与工作流

## 模块

| 模块 | 主要函数/类 | 输入 | 输出 | 适用场景 |
|---|---|---|---|---|
| 数据读取与标准化 | `load_dataarray`, `load_dataset`, `standardize_data`, `open_example_olr` | NetCDF / `xarray` 对象 | 标准化 `DataArray` | 首次加载数据、统一坐标 |
| 预处理 | `select_region`, `select_time`, `compute_climatology`, `compute_anomaly` | `DataArray` | 区域子集、异常场、气候态 | 进入诊断前的数据准备 |
| 谱分析 | `analyze_wk_spectrum`, `WKSpectralAnalysis`, `plot_wk_spectrum` | 赤道带时空场 | WK 频谱结果对象与图 | 波动类型识别、谱空间分析 |
| 波动滤波 | `filter_wave_signal`, `WaveFilter`, `CCKWFilter` | OLR 等时空场 | 特定波段滤波结果 | Kelvin、ER、MJO 等信号提取 |
| 统计与模态 | `standard_deviation`, `variance`, `linear_trend`, `EOFAnalyzer`, `analyze_cross_spectrum` | 异常场或配对场 | 标准差、趋势、EOF、交叉谱等 | 结果解释、实验对比 |
| 绘图与导出 | `plot_latlon_field`, `plot_spatial_std_comparison`, `plot_time_series`, `save_figure` | 数据对象 | 论文/展示级图形 | 首页、gallery、论文插图 |
| 工作流与 CLI | `analyze_wk_spectrum_from_file`, `compare_filter_spatial_fields`, CLI 子命令 | 文件路径和参数 | NetCDF、PNG、CSV | 自动化批处理、演示脚本 |

## 核心函数

- `open_example_olr`
- `compute_anomaly`
- `analyze_wk_spectrum`
- `filter_wave_signal`
- `plot_wk_spectrum`
- `plot_spatial_std_comparison`

## 模板

- [函数展示模板](function-template.md)
- [方法与原理首页](../theory/index.md)
- [研究笔记模板](../notes/research-note-template.md)
