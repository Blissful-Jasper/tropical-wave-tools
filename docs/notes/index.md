# Notes

## 方法笔记索引

| 编号 | 笔记 | 核心问题 |
|---|---|---|
| M01 | [数据标准化与异常场](method-01-data-standardization.md) | 数据如何进入统一的 `time, lat, lon` 分析格式 |
| M02 | [Wheeler-Kiladis 频率-波数谱](method-02-wk-spectrum.md) | 如何用频率-波数空间识别赤道波 |
| M03 | [频率-波数滤波与波段提取](method-03-wave-filtering.md) | 如何提取 Kelvin、ER、MJO、MRG、TD 等波段 |
| M04 | [Hovmoller、事件识别与滞后合成](method-04-hovmoller-composite.md) | 如何从事件角度看传播和空间结构演变 |
| M05 | [EOF 模态与 PC 回归](method-05-eof-regression.md) | 如何提取主模态并诊断对应风场结构 |
| M06 | [方差、季节循环与风场诊断](method-06-variance-wind-diagnostics.md) | 如何解释活跃度、方差贡献、散度和涡度 |

## 类型

| 类型 | 内容 | 适合放什么 |
|---|---|---|
| Research Notes | 研究过程记录 | 参数试验、结果解读、论文图复现 |
| Method Notes | 方法说明 | 数学原理、流程解释、参考文献摘要 |
| Implementation Notes | 实现记录 | 重构原因、模块设计、性能与兼容性说明 |

## 命名

- `YYYY-MM-DD-topic-name.md`
- `method-wk-spectrum.md`
- `implementation-filter-refactor.md`
- `validation-legacy-vs-cckw.md`

## 模板

- [研究笔记模板](research-note-template.md)
- [函数展示模板](../api/function-template.md)
