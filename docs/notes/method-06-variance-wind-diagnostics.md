# M06 方差、季节循环与风场诊断

## 对应代码

- `src/tropical_wave_tools/atlas.py`
- `src/tropical_wave_tools/diagnostics.py`
- `src/tropical_wave_tools/stats.py`

## 解决的问题

方差和 RMS 用于描述波动活跃度，风场散度和涡度用于解释低层环流结构。它们回答的问题是：某个波段在何时、何地更活跃？它对应的是低层辐合、辐散、气旋性还是反气旋性环流？

## 数学原理

月尺度 RMS：

```text
RMS_m(y, x) = sqrt(mean_{t in month m} X_filtered(t, y, x)^2)
```

滤波方差占比：

```text
fraction_m = 100 * var(X_filtered) / var(X_raw_anomaly)
```

面积加权平均：

```text
mean_area(X) = sum X(y, x) * cos(y) / sum cos(y)
```

水平散度和相对涡度：

```text
div = du/dx + dv/dy
zeta = dv/dx - du/dy
```

## 物理意义

- RMS 越大，说明该波段扰动振幅越强。
- 方差占比越高，说明该波段对总异常方差贡献越大。
- 低层散度为负通常对应辐合，有利于对流发展。
- 北半球正涡度通常表示气旋性旋转，南半球符号解释需结合所在纬度。

## 实现要点

当前代码会：

1. 对滤波后的 OLR/U850/V850 计算标准差或 RMS。
2. 按月份汇总季节循环。
3. 对指定区域做 `cos(lat)` 面积加权。
4. 使用 MetPy 计算散度和涡度；如果 MetPy 不可用，则回退到有限差分。
5. 对方差占比图，当前采用“先区域平均方差，再求比值”的方案。

## 注意事项

- 不同滤波波段可能重叠，方差占比不能简单相加为 100%。
- 先求格点比值再区域平均，与先区域平均方差再求比值，物理含义不同。
- 风场诊断要求 U/V 网格严格对齐。
- 有限差分回退适合常规规则经纬网；对复杂网格应使用更专业的矢量微分方案。
