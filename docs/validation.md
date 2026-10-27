# 方法验证

本页说明滤波与 WK 频谱实现如何被验证、验证到什么程度，以及如何与 NCL 对比。

## 已自动化的解析检验

`tests/test_filter_validation.py` 使用已知纬向波数 `k`、周期 `P` 的合成平面波（`k>0` 向东传播）：

| 检验 | 预期 |
| --- | --- |
| Kelvin (k=5, P=8d)、ER (k=-3, P=20d)、MJO (k=2, P=40d)、TD (k=-10, P=4d)、MRG (k=-4, P=5d) | 带内增益 ≈ 1（误差 < 2%） |
| Kelvin 反向传播、Kelvin 周期过长、ER 东传、MJO 西传 | 带外增益 < 2% |
| 经度含循环点 (0° 与 360°) | 输出与输入一致，首末列相等 |
| WK 谱峰值 | 东传波落在正波数，西传波落在负波数，频率误差小于一个频率分辨率 |

!!! note "已修复的问题"
    验证过程中发现：当经度轴含循环点时，旧版 `_kf_filter` 在错误的长度上做逆变换，
    输出被打乱（对 Kelvin 平面波 RMS 误差约 1.0，与信号同量级）。现已修复，并有回归测试。

## 与 NCL `kf_filter` 的对比（黄金文件）

输入是固定种子的随机数据（白噪声 + 已知 `k`、周期的平面波）和打包的 1979 年 OLR 赤道子集。
NCL 输出作为"黄金文件"提交在 `validation/reference/`，日常 `pytest` 只读它们，**不需要安装 NCL**。

### 实测结果

在 NCL 实跑的 Kelvin / ER / MRG / TD / MJO 五个波段 × 两个输入共 10 组上：

| | 修复前 | 修复后 |
| --- | --- | --- |
| 相关系数 | 0.947 – 0.99995 | 1 − corr ≲ 1e-14 |
| 归一化 RMSE | 1% – 34%（真实 OLR 的 MJO 最差） | ≲ 1e-7（NCL 输出为单精度） |

通过逐项排查，确认了两处与 NCL 不同并已修复：

1. **时间窗**：NCL 使用分裂余弦钟形窗（总共 5%，每端 `m = N·0.05/2` 点，权重 `½(1−cos(π(i+½)/m))`），
   而不是 `scipy` 的 Tukey 窗。两者端点约定不同，造成约 1% 的差异。
2. **周期截止的频率索引**：NCL 按**四舍五入**取整，旧实现用 `int()` 截断，会多保留（或少保留）
   一个处在 `[周期下限, 周期上限]` 之外的频率格点。对 365 天序列的 MJO（20–100 天），
   旧实现多保留了 121.7 天的格点。

色散曲线截止（`floor`/`ceil`）与 NCL 一致，无需修改。其余部分（线性去趋势、帧顺序、色散关系）
在对比中均吻合。

### 重新生成

在本机（需要 NCL，`kf_filter.ncl` 随 conda 安装的 NCL 自带）：

```bash
WAVES=kelvin,er,mrg,mjo,td,eig,wig bash validation/run_ncl_reference.sh
python validation/compare_with_ncl.py       # 逐组输出 1-corr / NRMSE / 振幅比
pytest tests/test_ncl_reference.py          # 存在黄金文件时自动执行
```

调用形式（逐纬度传入 `time×lon`、小写 `waveName`）取自作者现有的 CCKW 生产脚本；根据 `kf_filter.ncl`
头部说明，MJO/TD 这类无色散曲线的波段把 `hMin`/`hMax` 设为**缺失值**，`MRG` 与 `IG0` 是同一分支。
详见 `validation/README_ncl.md`。

## WK 频谱与 NCL `wkSpaceTime` 的对比

`ncl validation/ncl_wkspacetime.ncl` 在种子数据 `validation/wk_input.nc`（384 天 × 13 纬度 × 72 经度）
上调用 `wkSpaceTime`，谱以 netCDF 保存在 `validation/wk_ref/`；`python validation/compare_wk_with_ncl.py`
逐字段对比 `FIG_1_SYM/ASYM`（log10 的对称/反对称功率）、`FIG_3_BACK`（背景谱）和
`FIG_3_SYM/ASYM`（功率与背景之比）。

| 窗口设置 | 对比结果（log10 均方根差 / 比值均方根差） |
| --- | --- |
| `nDaySkip=30`（窗口起点相距 126 天，共 3 个窗口） | 全部字段 ≲ 3e-7，最大 1.6e-6 |
| `nDaySkip=-60`（起点相距 36 天，共 9 个窗口） | 中位数 ≲ 4e-6，均方根 3e-4 – 1e-3 |

排查中发现并修复了四处差异（修复前原始谱均方根差约 0.09，即 20% 量级）：

1. **窗口间隔语义**：NCL 的 `nDaySkip` 是窗口之间的**间隔**，负值表示重叠，相邻窗口起点相距 `nDayWin + nDaySkip`；
   旧实现把 `window_skip_days` 当作起点步长。现已按 NCL 的语义实现，默认值改为 `-60`（WK99：96 天窗口重叠 60 天）。
   原来的 `30`（步长 30 天）等价于现在的 `-66`。
2. **时间窗**：与滤波相同，NCL 使用分裂余弦钟形窗（窗口长度的 10%），不是 Tukey 窗。
3. **原始谱的频率平滑**：NCL 对原始对称/反对称谱做一次 1-2-1 频率平滑（不含零频与 Nyquist 行，端点复制），
   旧实现没有平滑，谱峰在频率方向没有按 ¼–½–¼ 展开。
4. **频率轴与背景平滑**：NCL 的频率轴包含 Nyquist（0.5 cpd）；背景谱平滑把 NaN 的零频行排除在外
   （旧实现用插值把它卷进平滑，影响最低几个频率行）。

!!! note "尚未解释的残差"
    重叠窗口（`nDaySkip=-60`）的结果仍有一点差异，且集中在最低的非零频率格点（周期等于窗口长度）。
    逐个替换单个窗口的预处理都无法消除，原因尚未确定；该情形的容差因此放宽到 3e-3。

## 待核实

- **分裂余弦窗的点数取整**：已有数据（N=360、365）下 `floor` 与 `round` 无法区分；
  `validation/odd_input.nc`（N=382，`m=9.55`）专门用来判定，需要用 NCL 重新生成该输入的参考文件。
- **`eig`（NCL `ig0`）与 `wig`（NCL `ig1`）两个分支**尚未与 NCL 对比，已加入 `WAVES` 列表。
- 上面"尚未解释的残差"。

## 尚未覆盖

- 与 Wheeler & Kiladis (1999) 原文图的定量对比；
- 非整数天采样（`obs_per_day > 1`）的专项检验。
