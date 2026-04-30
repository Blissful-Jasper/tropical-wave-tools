# M02 Wheeler-Kiladis 频率-波数谱

## 对应代码

- `src/tropical_wave_tools/spectral.py`
- `src/tropical_wave_tools/preprocessing.py`
- `src/tropical_wave_tools/plotting.py`

## 解决的问题

WK 频率-波数谱用于识别赤道波信号在频率和纬向波数空间中的能量分布。它特别适合诊断 Kelvin、ER、MRG、IG、TD、MJO 等热带波动。

## 数学原理

先把赤道带扰动分解为关于赤道对称和反对称的部分：

```text
Xs(t, y, x) = 0.5 * [X(t, y, x) + X(t, -y, x)]
Xa(t, y, x) = 0.5 * [X(t, y, x) - X(t, -y, x)]
```

然后对每个时间窗口做二维傅里叶变换：

```text
F(omega, k, y) = FFT_time,lon [X(t, y, x)]
P(omega, k) = mean_or_sum_y |F(omega, k, y)|^2
```

当前默认配置为：

- 窗口长度：96 天
- 窗口步长：30 天
- 时间窗函数：Tukey window
- 背景谱平滑：1-2-1 smoother

## 物理意义

- 频率表示振荡周期。
- 纬向波数表示沿经度方向的空间尺度。
- 频率-波数象限可以区分东传和西传扰动。
- 对称谱更突出 Kelvin、ER、MJO 等偏对称结构。
- 反对称谱更突出 MRG、部分 IG 波等跨赤道反对称结构。

## 实现要点

当前实现会：

1. 对输入场去线性趋势。
2. 去掉低频背景，默认截止频率为 `1 / window_size_days`。
3. 构建对称/反对称布局。
4. 对每个滑动窗口做 `time-lon` 二维 FFT。
5. 计算功率谱并沿纬度汇总。
6. 平滑背景谱，用于绘制相对谱强度。

## 注意事项

- 纬度网格必须关于赤道成对，例如 `[-15, ..., 0, ..., 15]`。否则对称/反对称分解没有清晰物理意义。
- 频率-波数符号约定必须和绘图、滤波约定一起检查。换资料源或换经度排序后，建议用理想东传/西传波做一次 sanity check。
- WK 谱适合看统计能量分布，不等同于单个事件的相位传播图。
- 对非 OLR 变量也可以做 WK 谱，但物理解释要换成对应变量的含义。
