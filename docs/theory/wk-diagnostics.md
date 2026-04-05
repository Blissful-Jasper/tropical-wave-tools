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


## 每一步在页面上应强调什么

### 数据读取

- 强调开箱即用
- 强调坐标标准化与科研数据兼容性

### 异常构建

- 强调“为什么要去背景态”
- 强调月平均、日循环和季节平均的区别

### WK 频谱

- 强调频率-波数结构与波动识别
- 强调理论分散曲线只是辅助解释，不是唯一判断标准

### 波动滤波

- 强调“谱空间识别”如何转化为“时空信号提取”
- 强调 CCKW 与 legacy 的对比价值

### 后处理与解释

- 强调 Hovmoller、标准差分布、合成分析和相位分析的衔接关系

## 推荐补充的参考说明

后续这一页可继续扩展：

- Wheeler-Kiladis 谱图的坐标定义与归一化说明
- 对称/反对称分量的物理意义
- Matsuno 分散曲线在图上的解释方式
- Kelvin、ER、MRG、EIG、WIG 等波族在图上的典型位置

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
