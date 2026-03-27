# 快速开始

## 环境

```bash
bash scripts/twave.sh setup
```

```bash
bash scripts/twave.sh test
```

```bash
bash scripts/twave.sh docs
bash scripts/twave.sh docs-serve
```

```bash
bash scripts/twave.sh app
```

## 数据

```bash
tropical-wave-tools prepare-sample-data \
  --source /work/mh1498/m301257/code_project2/olr.day.mean.nc \
  --copy-full-data
```

## 运行 WK 频谱分析

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

## 运行波动滤波

```bash
tropical-wave-tools filter-wave \
  --input data/local/olr.day.mean.nc \
  --var olr \
  --wave kelvin \
  --method cckw \
  --output outputs/kelvin_filtered.nc
```
