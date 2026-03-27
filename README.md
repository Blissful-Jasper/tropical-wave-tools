# Tropical Wave Tools

`tropical-wave-tools` 是基于你原有 `wave_tools` 研究代码重构出来的现代 Python 工具包骨架，目标是把零散脚本整理成一个可维护、可测试、可发布、可展示的开源项目。

这个新项目保留了原始科研逻辑里的核心概念和变量语义，尤其是：

- `WKSpectralAnalysis` / `calculate_wk_spectrum`
- `WaveFilter` / `CCKWFilter`
- `calculate_cross_spectrum` / `analyze_cross_spectrum`
- `EOFAnalyzer`
- Kelvin phase composite workflows
- Kelvin / ER / MJO 等波动参数定义
- 基于 `xarray` 的 `(time, lat, lon)` 数据流
- Wheeler-Kiladis 频谱分析与滤波工作流

## 为什么新建这个包

原始 `wave_tools` 已经具备很强的科研价值，但仍属于“研究脚本式聚合”：

- 计算、绘图、I/O、路径处理耦合较强
- `__init__.py` 暴露符号过多
- 模块之间存在循环依赖和重复函数
- 缺少标准打包、测试、文档和发布流程
- 大数据示例没有和 GitHub 发布方式解耦

这个重构版采用了更适合长期维护的布局：

- `src/` 包结构
- `pyproject.toml`
- `pytest` + GitHub Actions
- `mkdocs-material` 文档站点
- `Streamlit` 轻量交互演示
- 示例数据与本地大文件分离

## 核心模块

- `io`: 统一 Dataset / DataArray、维度别名和经纬度规范化
- `preprocess`: 区域选择、时间选择、气候态、异常、月平均、季节平均
- `diagnostics`: 区域平均、zonal/meridional mean、网格间距和 GMS 相关诊断
- `spectral`: Wheeler-Kiladis 频谱分析
- `cross_spectrum`: 交叉谱、相干平方、相位与向量分量
- `cross_spectrum_analysis`: 多试验批量交叉谱工作流
- `stats`: 趋势、相关、回归、方差、标准差
- `eof`: EOF/SVD 分解与垂直模态比较
- `phase`: Kelvin 相位识别、位相合成和时滞合成
- `plotting`: 频谱图、通用时序图、经纬度场图

## 项目结构

```text
tropical-wave-tools/
├── src/tropical_wave_tools/
├── tests/
├── examples/
├── scripts/
├── docs/
├── apps/
├── data/
│   ├── local/
│   └── samples/
└── .github/
```

## 快速开始

```bash
pip install -e ".[dev,docs,app]"
```

准备轻量示例数据：

```bash
tropical-wave-tools prepare-sample-data \
  --source /work/mh1498/m301257/code_project2/olr.day.mean.nc \
  --copy-full-data
```

计算 WK 频谱：

```bash
tropical-wave-tools wk-spectrum \
  --input data/local/olr.day.mean.nc \
  --var olr \
  --time-start 1979-01-01 \
  --time-end 1981-12-31 \
  --lat-min -15 \
  --lat-max 15 \
  --output-dir outputs/wk
```

运行 Kelvin 波滤波：

```bash
tropical-wave-tools filter-wave \
  --input data/local/olr.day.mean.nc \
  --var olr \
  --wave kelvin \
  --method cckw \
  --output outputs/kelvin_filtered.nc
```

## 与原项目的命名映射

| 原 `wave_tools` | 新包 `tropical_wave_tools` | 说明 |
|---|---|---|
| `spectral.py` | `spectral.py` | 保留主分析类与便捷函数 |
| `filters.py` | `filters.py` | 保留 `WaveFilter` 与 `CCKWFilter` |
| `cross_spectrum.py` | `cross_spectrum.py` | 保留交叉谱和 coherence/phase 定义 |
| `cross_spectrum_analysis.py` | `cross_spectrum_analysis.py` | 保留多试验交叉谱分析流程 |
| `eof.py` | `eof.py` | 保留 EOF 分析并补充可选 `xeofs` 后端 |
| `phase.py` | `phase.py` | 保留 Kelvin phase/composite 工作流 |
| `plotting.py` | `plotting.py` | 保留 WK 图与空间图绘制 |
| `matsuno.py` | `matsuno.py` | 保留理论色散曲线 |
| 一次性区域/异常/平均脚本 | `preprocess.py` | 通用化为函数接口 |
| 一次性趋势/相关/回归脚本 | `stats.py` | 标准化统计分析接口 |
| `compare_filter_spatial_fields.py` | `workflows.py` + CLI | 从脚本改为工作流和命令行入口 |
| `utils.load_data` | `io.load_dataarray` | 路径处理更通用、带类型注解 |

## 示例数据策略

你提供的原始测试数据 `olr.day.mean.nc` 大约 333 MB。为了兼顾科研复现和 GitHub 发布：

- `data/local/` 用于本地保留完整数据副本，默认 `.gitignore`
- `src/tropical_wave_tools/data/` 内置一个轻量整年赤道样例，便于测试、示例和网页展示
- `scripts/` 和 CLI 提供自动切分样例数据的能力

## 文档与网页展示

- `mkdocs-material` 用于 GitHub Pages 静态站点
- `apps/streamlit_app.py` 提供本地或 Streamlit Cloud 的轻量交互演示

## 开发

```bash
pytest
ruff check .
mkdocs serve
streamlit run apps/streamlit_app.py
```
