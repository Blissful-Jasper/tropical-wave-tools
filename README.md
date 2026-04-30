# Tropical Wave Tools

面向热带波和气候诊断的 Python 工具包，保留原始 `wave_tools` 的科研逻辑，并整理为可测试、可发布、可展示的项目结构。

更完整的使用说明见：

- `docs/user-guide.md`
- `docs/getting-started.md`
- `docs/examples.md`
- `docs/notes/index.md`

## Setup

```bash
bash scripts/twave.sh setup
```

或：

```bash
make setup
```

## Data

```bash
tropical-wave-tools prepare-sample-data \
  --source /work/mh1498/m301257/code_project2/olr.day.mean.nc \
  --copy-full-data
```

## Common Commands

```bash
bash scripts/twave.sh test
bash scripts/twave.sh docs
bash scripts/twave.sh docs-serve
# or pick a specific port if 8000 is occupied
bash scripts/twave.sh docs-serve --port 8012
bash scripts/twave.sh pages
bash scripts/twave.sh app
bash scripts/twave.sh build
bash scripts/twave.sh publish
bash scripts/twave.sh publish-test
```

或：

```bash
make test
make docs
make docs-serve
make pages
make app
make build
make publish
make publish-test
```

## Development and Deployment Workflow

建议在本地使用功能分支开发，确认无误后再合并到 `main`。GitHub Pages 只从 `main` 自动部署，避免开发分支内容直接发布到公开站点。

```bash
cd /Users/lipu/Desktop/tropical_tools/tropical-wave-tools

# 更新主分支
git switch main
git pull origin main

# 创建新的开发分支；也可以继续使用已有分支，例如 apple
git switch -c feature-name

# 修改代码后提交
git status
git add .
git commit -m "Describe the change"
git push -u origin feature-name
```

后续流程：

1. 在 GitHub 上从开发分支创建 Pull Request。
2. 检查 CI 和文档构建结果。
3. 合并 Pull Request 到 `main`。
4. `main` 更新后会触发 `.github/workflows/pages.yml`，自动构建并部署 GitHub Pages。

如果 Pages 部署被环境规则拦截，请在仓库的 `Settings -> Environments -> github-pages` 中确认 `Deployment branches and tags` 允许 `main` 部署。

## CLI Examples

```bash
tropical-wave-tools wk-spectrum \
  --input data/local/olr.day.mean.nc \
  --var olr \
  --time-start 1979-01-01 \
  --time-end 1981-12-31 \
  --lat-min -15 \
  --lat-max 15 \
  --output-dir outputs/wk

tropical-wave-tools filter-wave \
  --input data/local/olr.day.mean.nc \
  --var olr \
  --wave kelvin \
  --method cckw \
  --output outputs/kelvin_filtered.nc
```

## Notes

```bash
PYTHONNOUSERSITE=1
```

项目默认通过 `environment.yml` 管理环境，并在脚本内部使用 `mamba/conda run -n twave ...`，避免把包装到只读系统目录或混入 `~/.local` 的不兼容依赖。

离线环境下，如果 `twave` 环境无法创建，脚本会自动回退到当前系统 Python，并复用本机已有的 `pytest`、`mkdocs`、`streamlit`、`build`、`twine`。
