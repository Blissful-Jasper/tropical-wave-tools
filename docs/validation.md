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

在 NCL 实跑的 Kelvin / ER / MRG / TD / MJO / EIG(`ig0`) / WIG(`ig1`) 七个波段 × 三个输入（随机数据、382 天随机数据、真实 OLR）共 21 组上：

| | 修复前 | 修复后 |
| --- | --- | --- |
| 相关系数 | 0.947 – 0.99995 | 1 − corr ≲ 1e-14 |
| 归一化 RMSE | 1% – 34%（真实 OLR 的 MJO 最差） | ≲ 1e-7（NCL 输出为单精度） |

通过逐项排查，确认了两处与 NCL 不同并已修复：

1. **时间窗**：NCL 使用分裂余弦钟形窗，而不是 `scipy` 的 Tukey 窗（端点约定不同，约 1% 差异）。
   锥化点数为 `m = round(N·p) // 2`（先对总锥化点数取整，再整除 2），每端权重 `½(1−cos(π(i+½)/m))`；
   例如 382 个样本、p=5% 时 `m=9`（不是 10），96 个样本、p=10% 时 `m=5`。
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

## 其他模块的核对

下面的检验使用独立的参考（`scipy`、协方差特征分解、Matsuno 三次方程、已知相位的平面波、球面解析场），
测试位于 `tests/test_reference_checks.py` 与 `tests/test_diagnostics.py`。

| 模块 | 检验 | 结果 |
| --- | --- | --- |
| `stats`：回归、相关、t 检验、趋势 | 与 `scipy` 对比，含缺测值 | 干净数据一致；缺测值下**曾经错误**（相关系数 1.014），已修复 |
| `stats.linear_trend` 的显著性 | AR(1) 噪声的假阳性率 | 默认假设样本独立，ρ=0.9 时约 70%；新增 `adjust_for_autocorrelation` 后约 8% |
| `eof`：水平 EOF | 与协方差特征分解对比 | 特征值、模态一致（1e-6） |
| `eof`：缺测 / 陆地掩膜 | 一个格点为 NaN | **曾经**丢弃全部时间样本、返回 0 个模态，已修复 |
| `eof`：自由度（North 检验） | 与时间滞后自相关对比 | 垂直 EOF **曾经**沿堆叠轴取滞后（86387），已改为沿时间（39，真值约 38） |
| `matsuno` | 各分支是否满足三次方程 | ER / EIG / WIG（n=1,2）残差 ≲ 1e-15 |
| `cross_spectrum` | 已知波数、周期、相位差的平面波 | 峰值位置、符号、相干（1.0）、相位（0、−π/2、π）均正确 |
| `phase`：极值、影响范围、相位 | 与 scipy、已知相位的正弦波、半波分段对比 | 极值与影响范围一致；周期 20 个采样时相位误差 ≤ 0.07 rad（4°）；波峰处恒为 −π/2 |
| `phase`：相位合成、滞后合成 | 解析分箱均值、边界情形 | 分箱**曾经**不等宽且 ±π 被劈开；`lag_composite` **曾经**在 `lon_ref` 不在网格或只有一次事件时崩溃，均已修复 |
| `phase`：端到端相速度 | 已知速度的局部波包（10/15/20 m/s）与全球波数 3 的波 | 脊线追踪的 `Cp_ave` 误差 < 5%（局部波包 < 0.5%）；旧的直线拟合在波数 3 时失败 |
| `atlas`：事件检测（`detect_wave_events` / `detect_point_events`） | 已知正弦波的极值、子日采样、缺测、纬度降序 | 正常输入正确；`min_distance_days` **曾经**按采样点而非天计（6 小时、1 小时数据下相隔 3 天的事件未合并）；纬度降序**曾经**导致全 NaN、零事件，已修复 |
| `atlas`：滞后合成、Hovmöller 窗口、活动掩膜、投影 | 手算均值、边界位置、已知强弱时段 | 均正确；滞后合成现在报告每个滞后的事件数（`n_events`） |
| `atlas`：事件 → 合成的端到端 | 已知速度 15 m/s 的局部波包 | 找到每个波包中心，合成脊的速度误差 < 10% |
| `atlas`：方差分数 | 已知方差比、手算面积加权（含跨 0° 经度范围） | 精确一致（1e-14）；网格点比值再平均与区域比值定义不同（相差约 0.7 个百分点），已写入文档 |
| `atlas`：月 RMS 显著性 | 与 `scipy` t 检验对比 | 一致；首尾不完整年份**曾经**使年平均产生偏差（p 值 0.028 vs 0.016），现仅用完整年份 |
| `atlas`：Case 10 滞后回归整体流程 | 与 `scipy.linregress`（活动样本）对比；已知速度的波包 | 逐点一致（1e-15）；正滞后下脊线向东移动，速度误差 < 10% |
| `atlas`：波 EOF（`compute_wave_eof`） | 已知的对称/反对称模态；协方差特征分解；加权与还原 | 投影、方差解释率（1e-6）、降序纬度均正确；模态符号**曾经**取决于 SVD 例程，现已固定（最大载荷为正） |
| `atlas`：`regress_field_onto_pcs` | 精确关系；`scipy.linregress`（含缺测日） | 缺测日下**曾经**有偏（23% 缺测时 0.5%），现与 scipy 精确一致 |
| `cross_spectrum_analysis`：预处理 | 季节循环去除；掩膜 | 季节循环被去除；浮点（1/NaN）掩膜**曾经**被完全忽略，已修复 |
| `cross_spectrum`：纬度与对称分量 | 非对称、降序纬度 | 非对称网格**曾经**被静默接受，现在报错；降序输入与升序结果一致 |
| `cross_spectrum`：相干显著性 | 独立噪声下的假阳性率 | `dof = 2.667×段数` 是约定：经向完全相干时 5% 水平实际为 9%，经向独立时仅 0.2%；已加 `dof_per_segment` 参数并写明 |
| `diagnostics`：散度、涡度 | 刚体旋转与光滑解析场 | **曾经**漏掉曲率项（25°N 涡度偏低 50%），已修复；含经度接缝 |
| `io`：经度规范化 | 循环点（0° 与 360°、±180°）、舍入误差 | 循环点**曾经**在 `standardize_data` 之后留下重复的 0° 列，使 `_kf_filter` 的环绕判断失效；`-1e-14 % 360` **曾经**得到 360.0，均已修复 |
| `io`：`valid_range` | 打包数据（`scale_factor`/`add_offset`）与未打包数据 | CF 规定打包数据的 `valid_range` 以打包值计；**曾经**按物理值比较（该例保留 120 个点而非 27），已修复 |
| `io`：变量推断、保存 | 含 `crs`、`*_bnds` 的数据集；`None`/布尔/字典/含 `None` 的元组属性 | `infer_variable_name` **曾经**返回 `crs`；保存**曾经**因属性类型报 `TypeError`（这也使 `filter-wave --method legacy` 对 TD/MJO 无法保存），已修复 |
| `io` / `stats` / `atlas`：非标准日历 | CMIP 的 `noleap`、`360_day`（cftime） | `linear_trend` **曾经**在这些日历上崩溃，事件最小间隔换算同理，已修复；斜率单位为每天 |
| `io`：读取与选取 | 纬度降序文件、跨 0° 的经度范围、时间范围 | 选取正确，数值跟随各自坐标 |
| `workflows`：`compute_spatial_metrics` | 与 `numpy` 对比，含缺测 | 一致；平均值**曾经**在含缺测时各取各的有效点，`mean_bias` 与两个均值之差不符，现统一取共同有效点 |
| `workflows`：WK 流程、示例子集 | 与直接调用、保存文件对比 | 结果与直接调用一致，写出的 netCDF 与内存结果一致 |
| `workflows`：`compare_filter_spatial_fields` | RMSE、相关、输出文件 | 可运行并写出 PNG / NetCDF / CSV；两种方法共用同一核心，所以一致性**不是独立检验**，独立验证见上文 NCL 对比 |
| `filters`：无法分辨的波段 | `eig` / `wig` / `ig` 用日数据 | `CCKWFilter` 返回全零，**曾经**没有任何提示；现在发 `RuntimeWarning` 并在结果属性中写明（`method='legacy'` 会把波段截到 Nyquist） |
| `cli` | 各子命令端到端 | `prepare-sample-data` **曾经**只能覆盖包内样例，新增 `--output`；其余命令均通过 |
| `plotting`：色标范围 | 在已知位置放 10 倍于背景的强信号，读取渲染出的像素 | 色标由分位数决定，`extend="neither"` 时最强的信号**曾经**被画成空白：`plot_wk_spectrum`、`plot_hovmoller_comparison`、`plot_lag_longitude_evolution`、`plot_paper_style_hovmoller`（Case 10）、`plot_horizontal_structure`、`plot_lagged_horizontal_structure`、`plot_wind_diagnostics_panel`、交叉谱面板；现在按数据推断 `extend`，强信号以端点色绘出 |
| `plotting`：位置与颜色 | 渲染像素与 ContourSet 色带逐点对比（Hovmöller、经纬度图、WK 谱、Case 10；含 cartopy） | 一致：经度、纬度、时间、滞后方向正确，缺测格点留白 |
| `plotting`：维度顺序 | `(lon, lat)` 输入、纬度降序 | `(lon, lat)` 输入**曾经**画成转置的地图，风矢量**静默**用错分量（误差 30 m/s），显著性阴影线抛 `TypeError`；现在输出与维度顺序无关（逐像素相同） |
| `plotting`：WK 谱叠加 | 色散曲线对 Matsuno 无量纲关系；滤波框、CCKW 包络线对配置 | Kelvin、ER n=1、MRG、EIG n=0 残差 ≲ 1.5e-3，波数符号正确；滤波框与包络线与配置一致；周期标注 2.5 天**曾经**写成 "2d" |
| `plotting`：折线、趋势、显著性标记 | 与输入数据、`polyfit` 对比 | 一致 |
| `plotting`：辅助函数 | 色阶、经度标签、经度窗口、循环点 | 色阶覆盖数据且对称；359.6° **曾经**标成 "0W"；已含循环点的全球网格**曾经**再加一个循环点；全零 / 全缺测输入 `plot_spatial_std_comparison` **曾经**崩溃 |
| `eof`：xeofs 后端 | 对独立协方差特征分解与 SVD 后端；缺测、陆地掩膜 | 清洁数据一致（1e-8）；xeofs 自身拒绝任何 NaN，后端**曾经**因此在缺测时直接报错；结果字典**曾经**缺特征值、North 误差、掩膜；`vertical_eof_with_nan_handling` **曾经**只丢弃全缺测样本，地形以下缺测的层会崩溃；现与 SVD 后端共用缺测处理与结果组装 |
| `eof`：SVD 不收敛 | 传播波的对称分量（秩亏、特征值严格成对） | numpy 自带 LAPACK 的 `gesdd` 在某些环境下抛 "SVD did not converge"（Atlas 的波 EOF 因此失败）；现在回退到 `gesvd` |
| `atlas`：端到端工作流（`generate_local_wave_atlas`） | 合成的 Kelvin 波包：k=5、周期 6 天（12°/天），已知日期 | 事件全部落在波包内且每个波包都被找到；`std` 图等于独立滤波后的标准差；滞后合成与 Case 10 回归的传播速度均为 **12.00°/天**（真值 12）；WK 谱峰在 (k=5, 1/6 cpd)；波 EOF 为正交模态对（各 47.8%）；Case 10 阴影在基准点为负；文件齐全、图非空白；GPCP 与 OLR 两种 Case 05 来源 |

约定（Case 10）：预测量是基准点的**滤波 OLR**（标准化，+1σ = 对流被抑制），阴影是对流代理（−OLR）对它的回归，
所以阴影在基准点、零滞后处为**负**，等值线（滤波 OLR 对自身的回归）为正；正滞后表示场滞后于参考。
单位为"每个标准差"。

约定：事件检测的 `sign="negative"` 找 OLR 的极小值（对流中心），阈值为整条序列的 `threshold_std` 倍标准差；
反对称波的经向投影为 `北半球均值 − 南半球均值`（单侧振幅的两倍）。

约定：交叉谱的相位为 `arg(conj(X)·Y)`，Y 滞后 X 时为负；Kelvin 相位 `calculate_kelvin_phase` 对 `x = A sin θ` 返回 `−θ`，
即波峰在 −π/2、波谷在 +π/2。

!!! note "`composite_kw_phase` 的相速度估计器"
    默认的 `speed_method="ridge"` 从 `lon_ref` 出发沿合成图中最强的一条脊逐滞后追踪（经度按圆周处理、展开成连续值），
    再对脊线做直线拟合；多个同时存在的波峰（例如纬向波数 3 的波）和脊线越过 0°/360° 都不影响它。
    旧的 `"threshold_fit"`（对阈值以上所有点做一次直线拟合）仍可选用，但只适用于单条脊且不越过经度边界：
    波数 3 的波得到 −1.5 m/s（真值 +15），22 m/s 的单条脊读成 11.4 m/s。
    相位在噪声下的可定义比例也会下降：噪声为振幅的 20% 时，约 71% 的时次有定义相位。

!!! note "关于带边缘"
    端到端检验里，紧贴 Kelvin 带边缘的波（k=5、周期 10 天；h=8 m 的曲线在 k=5 对应 10.5 天）经 CCKW 滤波后，
    合成场测得的传播速度是 7.9°/天，而不是 7.2°/天：色散曲线把波包频谱偏低频的一半削掉了，剩下的中心频率偏高。
    这是滤波带宽的效应，不是流水线误差；把波放到带内（周期 6 天）后测得 12.00°/天，与真值一致。

!!! note "仍未做独立验证"
    图的美观与版式（只检查数据、颜色、位置和标注）；cartopy 的海岸线与陆地要素（需要 Natural Earth 数据，
    测试只检查数据在地图上的位置）；Streamlit 应用；`geocat-comp` / `metpy` 可选后端。
    此外 EOF 没有面积权重选项（水平 EOF 通常需要 √cos φ 加权，波 EOF 在 `compute_wave_eof` 里自行加权）。
    xeofs 与 cartopy 是可选依赖，相应测试在没装时自动跳过，CI 有一个装了它们的任务。

## 待核实

- 重叠窗口 WK 谱（`nDaySkip=-60`）在最低非零频率格点上仍有小差异，原因未明（见上文）。
- 锥化点数公式 `round(N·p)//2` 由 N=96、360、365、382 四个长度反推；更多长度可进一步确认。

## 尚未覆盖

- 与 Wheeler & Kiladis (1999) 原文图的定量对比；
- 非整数天采样（`obs_per_day > 1`）的专项检验。
