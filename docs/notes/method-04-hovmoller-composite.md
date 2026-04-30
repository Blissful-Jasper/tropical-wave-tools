# M04 Hovmoller、事件识别与滞后合成

## 对应代码

- `src/tropical_wave_tools/atlas.py`
- `src/tropical_wave_tools/plotting.py`

## 解决的问题

Hovmoller 图和滞后合成用于从事件角度理解波的传播和结构演变。WK 谱告诉我们能量在哪里，Hovmoller 和 composite 告诉我们某次或某类事件如何随时间和经度移动。

## 数学原理

先把三维场投影到经度-时间平面：

```text
X_eq(t, x) = mean_lat in band [X(t, y, x)]
```

对于反对称波，可以使用南北差：

```text
X_asym(t, x) = mean_north[X(t, y, x)] - mean_south[X(t, y, x)]
```

事件检测通常在参考经度 `x0` 上进行：

```text
r(t) = X_projection(t, x0)
event if r(t) < -a * std(r)     # OLR 常用
event if r(t) >  a * std(r)     # 降水常用
```

滞后合成定义为：

```text
C(tau, y, x) = mean_events X_filtered(t_event + tau, y, x)
```

其中 `tau` 是事件前后的滞后天数。

## 物理意义

- Hovmoller 图突出传播方向和相速度。
- 滞后合成突出典型事件的空间结构演变。
- OLR 与 U850/V850 联合合成可以显示对流和低层环流的耦合。
- 参考经度和事件阈值决定了 composite 代表的是哪一区域、哪类强度的事件。

## 实现要点

当前 atlas 会：

1. 对滤波 OLR 做经向投影。
2. 在参考经度处寻找超过阈值的事件。
3. 对 OLR、U850、V850 分别做 lag composite。
4. 输出水平结构图、Hovmoller 图和 lead-lag 演变图。

## 注意事项

- OLR 默认检测负异常，因为深对流增强对应更低的长波辐射。
- 换成降水时应使用正异常事件。
- 换成风场时不应简单套用正/负阈值，最好用环流指数、EOF PC 或区域平均风异常定义事件。
- 事件数太少时 composite 不稳。当前代码在无事件时会用最强绝对事件作为 fallback，适合展示，不适合作为严格统计结论。
