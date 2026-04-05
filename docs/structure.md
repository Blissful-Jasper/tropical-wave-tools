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
│   ├── preprocess.py
│   ├── spectral.py
│   ├── filters.py
│   ├── stats.py
│   ├── plotting.py
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
│   ├── preprocess.py
│   ├── spectral.py
│   ├── filters.py
│   ├── stats.py
│   ├── plotting.py
│   ├── workflows.py
│   └── ...
└── tests/
```
