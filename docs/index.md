# Tropical Wave Tools

<section class="tw-hero">
  <div class="tw-hero-copy">
    <p class="tw-hero-kicker">Equatorial Wave Diagnostics</p>
    <p class="tw-hero-summary">
      面向热带波动与赤道波诊断的 Python 工具包。内置 OLR 样例、标准化 I/O、WK 频谱、波动滤波和
      local atlas 工作流，适合快速复现与继续扩展。
    </p>
    <div class="tw-hero-meta">
      <span class="tw-pill">xarray-first</span>
      <span class="tw-pill">WK spectrum</span>
      <span class="tw-pill">Wave filters</span>
      <span class="tw-pill">Local atlas</span>
    </div>
    <div class="tw-hero-actions">
      <a class="md-button md-button--primary" href="getting-started/">快速开始</a>
      <a class="md-button" href="examples/">案例</a>
      <a class="md-button" href="api/">API</a>
      <a class="md-button" href="theory/">方法</a>
    </div>
    <p class="tw-note">推荐从内置 OLR 样例开始，再进入谱分析、滤波和 atlas 诊断。</p>
  </div>
  <div class="tw-hero-visual">
    <img src="assets/wk_spectrum.png" alt="Wheeler-Kiladis spectrum from the built-in sample">
    <img src="assets/kelvin_hovmoller_triptych.png" alt="Kelvin-wave hovmoller diagnostics">
  </div>
</section>

<div class="tw-stat-grid">
  <article class="tw-stat">
    <p class="tw-stat-value">3</p>
    <p class="tw-stat-label">核心入口</p>
    <p class="tw-stat-text">WK 频谱、波动滤波、local atlas。</p>
  </article>
  <article class="tw-stat">
    <p class="tw-stat-value">1</p>
    <p class="tw-stat-label">内置样例</p>
    <p class="tw-stat-text">开箱即用的赤道 OLR 数据与展示页。</p>
  </article>
  <article class="tw-stat">
    <p class="tw-stat-value">API</p>
    <p class="tw-stat-label">按任务组织</p>
    <p class="tw-stat-text">从 I/O、统计到诊断与绘图都可直接调用。</p>
  </article>
</div>

## 你可以直接做什么

<div class="tw-grid tw-grid-3">
  <article class="tw-card">
    <p class="tw-card-label">Spectrum</p>
    <h3>计算 WK 频谱</h3>
    <p>快速查看频率-波数结构，并导出标准图件与 NetCDF 结果。</p>
  </article>
  <article class="tw-card">
    <p class="tw-card-label">Filtering</p>
    <h3>提取 Kelvin / ER / MJO 等波动</h3>
    <p>支持 legacy 与 CCKW 两类滤波入口，方便并行比较与诊断。</p>
  </article>
  <article class="tw-card">
    <p class="tw-card-label">Atlas</p>
    <h3>生成本地波动图谱</h3>
    <p>围绕 OLR、U850、V850 产出更接近论文工作流的结果页面。</p>
  </article>
</div>

## 最短路径

<div class="tw-workflow tw-workflow-compact">
  <article class="tw-step">
    <p class="tw-step-label">Step 1</p>
    <h3>读取并标准化样例</h3>
    <p><code>open_example_olr</code> · <code>load_dataarray</code></p>
    <p>统一时间、纬度和经度坐标。</p>
  </article>
  <article class="tw-step">
    <p class="tw-step-label">Step 2</p>
    <h3>先看谱，再看波</h3>
    <p><code>analyze_wk_spectrum</code> · <code>filter_wave_signal</code></p>
    <p>先定位频谱结构，再提取目标波段。</p>
  </article>
  <article class="tw-step">
    <p class="tw-step-label">Step 3</p>
    <h3>扩展到 atlas 与对比图</h3>
    <p><code>generate_local_wave_atlas</code> · <code>plot_wk_spectrum</code></p>
    <p>把分析结果整理成持续可复用的图件产物。</p>
  </article>
</div>

## 最小示例

```python
from tropical_wave_tools import analyze_wk_spectrum, open_example_olr, plot_wk_spectrum

olr = open_example_olr()
result = analyze_wk_spectrum(olr)
fig, _ = plot_wk_spectrum(result)
```

## 代表结果

<div class="tw-grid tw-grid-2">
  <article class="tw-card tw-gallery-card">
    <p class="tw-card-label">Sample</p>
    <h3>内置样例场与异常变率</h3>
    <p>先确认背景场和异常强度，再进入后续波动诊断。</p>
    <img src="assets/sample_mean_field.png" alt="Time-mean OLR field from the built-in sample">
    <img src="assets/monthly_anomaly_std.png" alt="Standard deviation of monthly anomalies">
  </article>
  <article class="tw-card tw-gallery-card">
    <p class="tw-card-label">Diagnostics</p>
    <h3>WK 频谱与代表波型结构</h3>
    <p>从频谱定位到空间响应，对接后续滤波、合成和 atlas 工作流。</p>
    <img src="assets/wk_spectrum.png" alt="WK spectrum from the built-in sample">
    <img src="assets/wave_spatial_compare_large_scale.png" alt="Filtered OLR standard deviation for large-scale tropical waves">
  </article>
</div>
