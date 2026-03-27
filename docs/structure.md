# 项目结构

推荐目录树如下：

```text
tropical-wave-tools/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
├── apps/
│   └── streamlit_app.py
├── data/
│   ├── local/
│   └── samples/
├── docs/
│   ├── assets/
│   ├── examples.md
│   ├── getting-started.md
│   ├── index.md
│   ├── migration.md
│   └── deployment.md
├── examples/
├── scripts/
├── src/
│   └── tropical_wave_tools/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── easyxp.py
│       ├── exceptions.py
│       ├── filters.py
│       ├── diagnostics.py
│       ├── io.py
│       ├── matsuno.py
│       ├── plotting.py
│       ├── preprocess.py
│       ├── preprocessing.py
│       ├── sample_data.py
│       ├── spectral.py
│       ├── stats.py
│       ├── workflows.py
│       └── data/
└── tests/
```

## 每个目录的职责

- `src/tropical_wave_tools/`
  核心可发布包
- `tests/`
  核心行为测试
- `examples/`
  最小可运行示例
- `scripts/`
  数据准备和画廊生成脚本
- `docs/`
  GitHub Pages 文档站点
- `apps/`
  轻量交互演示
- `data/local/`
  本地大文件缓存，不进入 Git
