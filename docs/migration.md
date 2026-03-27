# 迁移说明

## 保留与调整

为了尽量不破坏原有科研分析逻辑，这次重构遵循了两条原则：

1. 频谱、滤波和波段定义尽量保留原实现。
2. 模块边界和项目组织方式重构成更适合开源维护的结构。

## 主要映射关系

| 原始对象 | 新对象 | 调整原因 |
|---|---|---|
| `wave_tools.spectral.calculate_wk_spectrum` | `tropical_wave_tools.spectral.calculate_wk_spectrum` | 保留旧返回签名，便于迁移 |
| `wave_tools.spectral.WKSpectralAnalysis` | `tropical_wave_tools.spectral.WKSpectralAnalysis` | 保留类名，内部依赖解耦 |
| `wave_tools.filters.WaveFilter` | `tropical_wave_tools.filters.WaveFilter` | 保留核心滤波逻辑 |
| `wave_tools.filters.CCKWFilter` | `tropical_wave_tools.filters.CCKWFilter` | 保留原算法意图，清理 I/O 边界 |
| `wave_tools.compare_filter_spatial_fields.py` | `tropical_wave_tools.workflows.compare_filter_spatial_fields` | 从脚本提升为工作流 API 和 CLI |

## 这次主动修正的问题

- 去掉了原包里重复的 `create_cmap_from_string`
- 不再依赖包内循环导入
- 把路径写死和脚本式入口改成参数化工作流
- 把完整大文件样例和 GitHub 可发布样例分开管理
- 给核心函数补了类型注解和 docstring

