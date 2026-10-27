# 项目结构
## 当前结构

```text
tropical-wave-tools/
├── apps/                  # Streamlit 交互演示
├── data/
│   ├── local/             # 本地大文件缓存
│   └── samples/
├── docs/                  # 当前文档站点
├── examples/              # 最小可运行示例
├── scripts/               # 数据准备、建站、发布脚本
├── src/tropical_wave_tools/
│   ├── io.py
│   ├── preprocessing.py       # 选区、气候态/异常、去趋势、平滑（原 preprocess.py 已并入）
│   ├── spectral.py            # WK 频率-波数谱
│   ├── filters.py             # 波动滤波（对齐 NCL kf_filter）
│   ├── stats.py
│   ├── eof.py  phase.py  matsuno.py  cross_spectrum.py  diagnostics.py
│   ├── plotting/              # 按主题拆分的绘图包
│   │   ├── _style.py  _geo.py  _wind.py        # 共享的样式、地图、风矢量辅助
│   │   └── spectrum.py  hovmoller.py  maps.py  eof.py  climatology.py
│   ├── atlas/                 # 本地波动图集
│   │   ├── _config.py  _paths.py                # 各 Case 常量、数据路径
│   │   └── monthly.py  events.py  case10.py  wave_eof.py  workflow.py
│   ├── workflows.py
│   └── ...
└── tests/
```

## 推荐结构

```text
tropical-wave-tools/
├── apps/
├── data/
│   ├── local/
│   └── samples/
├── docs/
│   ├── assets/            # 首页和 gallery 结果图
│   ├── api/               # API 总览与旗舰函数页
│   ├── gallery/           # 展示型案例页
│   ├── notes/             # 研究笔记入口与模板
│   ├── theory/            # 方法原理与流程说明
│   ├── stylesheets/
│   ├── blueprint.md       # 总体改版蓝图
│   ├── getting-started.md
│   ├── index.md
│   └── ...
├── examples/
│   ├── quickstart/        # 极短示例
│   ├── workflows/         # 端到端工作流示例
│   └── figure_recipes/    # 单图类型示例
├── notebooks/
│   ├── tutorials/
│   └── case-studies/
├── notes/
│   ├── research/
│   ├── methods/
│   └── implementation/
├── scripts/
├── src/tropical_wave_tools/
│   ├── io.py
│   ├── preprocessing.py       # 选区、气候态/异常、去趋势、平滑（原 preprocess.py 已并入）
│   ├── spectral.py            # WK 频率-波数谱
│   ├── filters.py             # 波动滤波（对齐 NCL kf_filter）
│   ├── stats.py
│   ├── eof.py  phase.py  matsuno.py  cross_spectrum.py  diagnostics.py
│   ├── plotting/              # 按主题拆分的绘图包
│   │   ├── _style.py  _geo.py  _wind.py        # 共享的样式、地图、风矢量辅助
│   │   └── spectrum.py  hovmoller.py  maps.py  eof.py  climatology.py
│   ├── atlas/                 # 本地波动图集
│   │   ├── _config.py  _paths.py                # 各 Case 常量、数据路径
│   │   └── monthly.py  events.py  case10.py  wave_eof.py  workflow.py
│   ├── workflows.py
│   └── ...
└── tests/
```
