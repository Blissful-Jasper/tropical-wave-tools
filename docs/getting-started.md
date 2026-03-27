# 快速开始

## 安装

```bash
pip install -e ".[dev,docs,app]"
```

## 准备示例数据

```bash
tropical-wave-tools prepare-sample-data \
  --source /work/mh1498/m301257/code_project2/olr.day.mean.nc \
  --copy-full-data
```

执行后会得到两类数据：

- `src/tropical_wave_tools/data/olr_equatorial_1979.nc`
  用于测试、示例和网页展示的轻量内置样例
- `data/local/olr.day.mean.nc`
  本地完整数据副本，默认不会加入 Git

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
