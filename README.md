# Tropical Wave Tools

面向热带波和气候诊断的 Python 工具包，保留原始 `wave_tools` 的科研逻辑，并整理为可测试、可发布、可展示的项目结构。

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
