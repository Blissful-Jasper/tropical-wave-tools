# User Guide

这份指南面向第一次接触本项目的使用者，目标是把“如何运行、如何改图、如何接自己的数据、如何发布给别人学习”讲清楚。你可以把它当成项目的实操手册。

## 1. 项目能做什么

这个仓库主要面向热带波与相关气候诊断，核心能力包括：

- `WK spectrum` 频率-波数谱分析
- `Kelvin / ER / MJO / MRG / TD` 等波段过滤
- 传播、lead-lag、EOF、低层风场、季节循环等图形诊断
- 基于本地 NetCDF 数据生成一整套 `gallery` 示例图

项目既可以当作 Python 包调用，也可以直接用命令行运行。

## 2. 快速开始

推荐先准备环境：

```bash
bash scripts/twave.sh setup
```

常用命令：

```bash
bash scripts/twave.sh test
bash scripts/twave.sh docs
bash scripts/twave.sh docs-serve
bash scripts/twave.sh docs-serve --port 8012
bash scripts/twave.sh app
```

如果你已经有可用环境，也可以直接调用 CLI：

```bash
tropical-wave-tools --help
tropical-wave-tools wk-spectrum --help
tropical-wave-tools filter-wave --help
tropical-wave-tools local-wave-atlas --help
```

## 3. 数据要求

项目默认假设输入是日尺度 `NetCDF` 数据，并且至少能标准化成这三个维度：

- `time`
- `lat`
- `lon`

维度名不一定要完全一致。代码会自动把常见名字标准化，例如：

- `longitude -> lon`
- `latitude -> lat`
- `valid_time -> time`

如果数据里有单层额外维度，例如 `level=850`，项目也会自动压缩掉。

### 变量建议

常见默认本地文件：

- `data/local/olr.day.mean.nc`
- `data/local/uwnd_850hPa_1979-2024.nc`
- `data/local/vwnd_850hPa_1979-2024.nc`
- `data/local/GPCP_data_1997-2020-2.5x2.5_stand.nc`

其中：

- `OLR` 常用变量名通常是 `olr`
- `U850` 常用变量名通常是 `uwnd`
- `V850` 常用变量名通常是 `vwnd`
- `GPCP` 在当前本地文件中变量名是 `precipitation`

## 4. 最常用的三种调用方式

### 4.1 做一次 WK 谱分析

```bash
tropical-wave-tools wk-spectrum \
  --input data/local/olr.day.mean.nc \
  --var olr \
  --time-start 1998-01-01 \
  --time-end 2014-12-31 \
  --lat-min -15 \
  --lat-max 15 \
  --output-dir outputs/wk
```

### 4.2 过滤某一个波段

```bash
tropical-wave-tools filter-wave \
  --input data/local/olr.day.mean.nc \
  --var olr \
  --wave kelvin \
  --method cckw \
  --output outputs/kelvin_filtered.nc
```

### 4.3 生成整套本地 atlas

```bash
tropical-wave-tools local-wave-atlas \
  --output-dir outputs/local_wave_atlas \
  --waves kelvin er mjo mrg td \
  --time-start 1997-01-01 \
  --time-end 2014-12-31 \
  --lat-min -25 \
  --lat-max 30 \
  --hov-days 240 \
  --n-workers 1
```

这是最适合学习项目整体流程的入口，因为它会一口气生成多个案例图和诊断文件。

## 5. 如何修改结果

如果你想改图而不是只运行默认参数，最值得先看的文件是：

- `src/tropical_wave_tools/atlas.py`
- `src/tropical_wave_tools/plotting.py`
- `scripts/generate_gallery.py`

### `atlas.py` 负责什么

`atlas.py` 更像“案例编排层”。这里集中定义了：

- 每个案例包含哪些波型
- 每个波型的 `lag`
- 每个案例的经纬度范围、参考经度、事件基点
- Case 05 / 08 / 10 这类论文风格案例的专门配置

如果你想改：

- 某个案例的时间窗口
- 某个波动的经度范围
- 某个 Case 用哪种数据源

通常都先改这里。

### `plotting.py` 负责什么

`plotting.py` 更像“画图风格层”。这里集中定义了：

- colormap
- colorbar
- quiver 样式
- 子图布局
- 标题、注释、面板标号

如果你想改：

- 图更亮或更克制
- colorbar 更细、更宽、更远离主图
- 箭头更明显
- 标题、字体、面板间距更像期刊图

通常都先改这里。

### `generate_gallery.py` 负责什么

这个脚本负责把 atlas 生成的关键图复制到 `docs/assets`，也负责清理前台不再使用的旧图。

如果你：

- 增加了一个新案例
- 替换了某个前台图片
- 删除了旧案例图

记得同步改这个脚本。

## 6. 如何接自己的数据

最简单的方式是把自己的文件放到 `data/local/`，然后在命令行里显式指定输入文件和变量名。

例如你有自己的降水：

```bash
tropical-wave-tools filter-wave \
  --input /path/to/my_precip.nc \
  --var precipitation \
  --wave er \
  --method cckw \
  --output outputs/my_er_precip.nc
```

如果你要让整个 atlas 默认使用自己的数据，有两种方式：

### 方式 A：替换 `data/local` 中的默认文件

适合个人本地试验。

### 方式 B：在代码里改默认路径

看这里：

- `src/tropical_wave_tools/atlas.py` 里的 `DEFAULT_LOCAL_PATHS`

这种方式更适合把项目整理成固定工作流。

## 7. Case 05、08、10 怎么改

这三个案例最常被定制。

### Case 05

现在优先使用本地 `GPCP` 日降水，生成论文式的季节方差占比图。

主要看：

- `CASE05_WAVES`
- `CASE05_LAT_RANGE`
- `CASE05_REGION_RANGES`
- `compute_monthly_variance_fraction_samples`

当前实现是“先算区域月方差，再求滤波方差占比”，而不是先做格点比值再区域平均。如果你要复现实测论文图，这个区别很重要。

### Case 08

这是区域相位演变案例，最常改的是：

- 每个波的 `lag`
- 参考经度和纬带
- 风场箭头密度
- `OLR` 填色范围

主要看 `atlas.py` 中所有 `CASE08_*` 常量。

### Case 10

这是论文风格的 lagged-regression Hovmoller。

最常改的是：

- `CASE10_BASE_POINTS`
- `CASE10_LAGS`
- `CASE10_LON_WINDOWS`
- `CASE10_SHADING_RANGE_SCALE`
- `CASE10_SHADING_MIN_LIMIT`

如果你想让不同波的主传播带更清楚，这里通常是第一修改入口。

对于 `MRG`，还要额外注意不要把“快速西传相位”和“慢速东传群传播”混成一个判断。Lubis and Jacobi (2015) 里这两层结构是同时存在的。

## 8. 如何重生成文档图片

如果你改了案例逻辑或绘图风格，建议重新生成 gallery 资产：

```bash
python scripts/generate_gallery.py
```

如果你只想重算某一个案例，通常直接调用 `generate_local_wave_atlas(...)` 更省时间。

## 9. 如何发布给别人学习

建议把下面几部分一起维护好：

- `README.md`
- `docs/getting-started.md`
- 这份 `docs/user-guide.md`
- `docs/examples.md`

发布路径通常有三类：

- 本地 `mkdocs serve`
- GitHub Pages 文档站
- PyPI 包发布

对应脚本：

```bash
bash scripts/twave.sh docs-serve
bash scripts/twave.sh pages
bash scripts/twave.sh build
bash scripts/twave.sh publish-test
bash scripts/twave.sh publish
```

## 10. 常见问题

### 环境里有 ABI 冲突怎么办

如果你遇到 `numpy/pandas/xarray` 兼容问题，优先使用项目环境，不要混用 `~/.local`：

```bash
PYTHONNOUSERSITE=1
```

### 为什么有些图信号很弱

优先检查三类问题：

- 输入数据本身是否有缺测或坏值
- 当前波型的时间范围和基点是否合理
- 当前图是不是用了不适合这个波型的投影或纬带

### 为什么图改了但文档没变

通常是忘了同步：

- `scripts/generate_gallery.py`
- `docs/assets/`
- `mkdocs.yml`

## 11. 建议的学习顺序

如果你是第一次读这个项目，最推荐按这个顺序：

1. `README.md`
2. `docs/getting-started.md`
3. `docs/examples.md`
4. `src/tropical_wave_tools/atlas.py`
5. `src/tropical_wave_tools/plotting.py`

这样最容易建立“数据 -> 诊断 -> 图形 -> 文档展示”的整体心智模型。
