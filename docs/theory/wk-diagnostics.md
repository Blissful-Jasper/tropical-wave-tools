# WK 与波动诊断流程

这一页用一条科研分析主线，把方法原理、代码模块和展示页面串起来。

## 一条推荐主线

```text
原始 OLR 数据
  -> 坐标标准化与区域选择
  -> 去季节循环 / 构建异常
  -> 赤道对称 / 反对称分解
  -> WK 频率-波数谱分析
  -> 识别目标波段
  -> Kelvin / ER / MJO 等滤波
  -> Hovmoller / 空间分布 / 对比 / composite
```

## 这条主线在当前仓库中的映射

| 分析步骤 | 当前函数或类 |
|---|---|
| 数据读取与统一 | `load_dataarray`, `standardize_data`, `open_example_olr` |
| 区域与时间选择 | `select_region`, `select_time` |
| 背景态与异常 | `compute_climatology`, `compute_anomaly` |
| WK 诊断 | `analyze_wk_spectrum`, `WKSpectralAnalysis`, `plot_wk_spectrum` |
| 波动滤波 | `filter_wave_signal`, `WaveFilter`, `CCKWFilter` |
| 基础统计 | `standard_deviation`, `variance`, `linear_trend` |
| 结果展示 | `plot_latlon_field`, `plot_spatial_std_comparison`, `save_figure` |

## 编号方法笔记

如果需要从实现回到数学原理和物理意义，可以按编号查阅：

| 编号 | 主题 |
|---|---|
| M01 | [数据标准化与异常场](../notes/method-01-data-standardization.md) |
| M02 | [Wheeler-Kiladis 频率-波数谱](../notes/method-02-wk-spectrum.md) |
| M03 | [频率-波数滤波与波段提取](../notes/method-03-wave-filtering.md) |
| M04 | [Hovmoller、事件识别与滞后合成](../notes/method-04-hovmoller-composite.md) |
| M05 | [EOF 模态与 PC 回归](../notes/method-05-eof-regression.md) |
| M06 | [方差、季节循环与风场诊断](../notes/method-06-variance-wind-diagnostics.md) |




## 传播与对称性速记

- Kelvin、MJO 的主传播方向通常表现为东传
- ER、MRG、TD 的主传播方向通常表现为西传
- Kelvin、ER 更适合看赤道平均的经向投影
- MRG、EIG/WIG 这类跨赤道变号更明显的信号，更适合看反对称投影，而不是简单赤道平均

## 适合链接到这一页的地方

- 首页 hero 下方的工作流模块
- Gallery 的 OLR showcase 页面
- API 页里的 `analyze_wk_spectrum` 和 `filter_wave_signal`
- 研究笔记中的参数选择记录
