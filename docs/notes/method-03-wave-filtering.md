# M03 频率-波数滤波与波段提取

## 对应代码

- `src/tropical_wave_tools/config.py`
- `src/tropical_wave_tools/filters.py`
- `src/tropical_wave_tools/workflows.py`

## 解决的问题

频率-波数滤波用于从原始扰动场中提取某一类热带波信号，例如 Kelvin、ER、MJO、MRG 或 TD。它回答的问题是：某个变量中，有多少结构落在目标波的传播尺度和时间尺度内？

## 数学原理

对每个纬度的 `time-lon` 场做二维傅里叶变换：

```text
F(omega, k) = FFT_time,lon [X'(t, x)]
```

然后构造一个掩膜 `M(omega, k)`：

```text
F_filtered(omega, k) = F(omega, k) * M(omega, k)
X_filtered(t, x) = inverse_FFT [F_filtered(omega, k)]
```

掩膜由三类条件共同决定：

- 周期范围，例如 Kelvin 常用 3-20 天。
- 波数范围，例如 Kelvin 常用东传波数 2-14。
- 等效深度或理论色散曲线范围，例如 Kelvin、ER、MRG、IG。

## 默认波段

当前主要默认值在 `DEFAULT_WAVE_SPECS` 中定义：

| 波型 | 周期 | 波数 | 等效深度 | 说明 |
|---|---:|---:|---:|---|
| Kelvin | 3-20 天 | 2 到 14 | 8-90 m | 东传赤道 Kelvin 波 |
| ER | 9-72 天 | -10 到 -1 | 8-90 m | 西传赤道 Rossby 波 |
| MRG | 2.5-10 天 | -10 到 -1 | 8-90 m | 混合 Rossby-gravity 波 |
| TD | 2.5-5 天 | -20 到 -6 | 无 | 热带扰动经验波段 |
| MJO | 20-100 天 | 1 到 5 | 无 | 季节内东传扰动 |

## 物理意义

滤波结果不是新的观测变量，而是原变量中符合目标波段的扰动分量。对 OLR 来说，滤波负异常可解释为该波段相关的深对流增强；对 U850/V850 来说，滤波结果表示该波段相关的低层风异常。

## 实现要点

当前流程会：

1. 从日气候态中提取低阶谐波，构造平滑年循环。
2. 用原始场减去年循环得到异常场。
3. 对每个纬度做 `time-lon` FFT。
4. 根据目标波段和色散曲线清除不需要的频率-波数成分。
5. 逆变换回 `time, lat, lon`。

## 注意事项

- `--spd` 或 `obs_per_day` 必须与真实采样频率一致，否则周期边界和 Nyquist 频率会错。
- MJO、TD 这类经验盒式滤波不使用等效深度约束，物理解释应更谨慎。
- 滤波边界附近可能出现谱泄漏，长时间序列比短样本更稳。
- 对降水使用事件检测时通常应寻找正异常，不应沿用 OLR 的负异常默认解释。
