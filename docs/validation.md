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

本仓库**还没有**随附 NCL 输出，因此目前不能声称"与 NCL 逐点一致"。流程如下：

- 输入是固定种子的随机数据（白噪声 + 4 个已知 `k`、周期的平面波）和打包的 1979 年 OLR 赤道子集，各约 0.5 MB，已提交到 `validation/`。
- 在有 NCL 的机器上运行一次，生成"黄金文件"并提交；之后 `pytest` 只读取它们，**不需要安装 NCL**。

```bash
python validation/make_input.py             # 重新生成输入（仅在需要时）
ncl validation/ncl_generate_reference.ncl   # 需要 NCL 与 kf_filter.ncl，输出 validation/reference/ncl_*.nc
python validation/compare_with_ncl.py       # 逐波段输出 corr / NRMSE / 振幅比
pytest tests/test_ncl_reference.py          # 存在黄金文件时自动执行
```

!!! warning "NCL 脚本尚待核对"
    `ncl_generate_reference.ncl` 中 `kf_filter` 的参数顺序、"不使用"的哨兵值和 `waveName`
    取值需要对照 [NCL 官方文档](https://www.ncl.ucar.edu/Document/Functions/User_contributed/kf_filter.shtml)
    确认后再信任对比结果。

默认阈值：相关系数 ≥ 0.9999，归一化 RMSE ≤ 0.01，拿到实测结果后再收紧。若某个波段不达标，优先检查：

1. NCL `kf_filter` 的参数顺序与 `waveName` 命名；
2. 色散关系取整方式（`floor`/`ceil`）与边界是否含端点；
3. 去趋势与窗函数是否一致。

## 尚未覆盖

- WK 背景谱平滑与 NCL `wkSpaceTime` 的逐点对照；
- 与 Wheeler & Kiladis (1999) 原文图的定量对比；
- 非整数天采样（`obs_per_day > 1`）的专项检验。
