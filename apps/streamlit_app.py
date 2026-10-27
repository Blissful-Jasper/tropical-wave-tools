"""Interactive Tropical Wave Tools demo: WK spectrum and wave filtering."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import streamlit as st
import xarray as xr

from tropical_wave_tools.config import DEFAULT_WAVE_SPECS, SpectralConfig
from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.io import describe_dataarray, load_dataarray
from tropical_wave_tools.plotting import plot_wk_spectrum
from tropical_wave_tools.sample_data import get_sample_path, open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum

ACCENT = "#0f766e"
WAVE_CHOICES = ("kelvin", "er", "mrg", "mjo", "td")

st.set_page_config(
    page_title="Tropical Wave Tools",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; max-width: 1180px; }}
      .tw-hero {{
        padding: 1.4rem 1.6rem; margin-bottom: 1.2rem; border-radius: 1rem;
        border: 1px solid rgba(128,128,128,.25);
        background: radial-gradient(circle at top right, rgba(2,132,199,.16), transparent 40%),
                    linear-gradient(180deg, rgba(15,118,110,.10), rgba(15,118,110,.02));
      }}
      .tw-hero h1 {{ margin: 0 0 .35rem; font-size: 1.9rem; letter-spacing: -.01em; }}
      .tw-hero p {{ margin: 0; opacity: .78; line-height: 1.7; }}
      .tw-kicker {{ color: {ACCENT}; font-weight: 800; font-size: .74rem;
                    letter-spacing: .09em; text-transform: uppercase; }}
      div[data-testid="stMetric"] {{
        padding: .8rem 1rem; border-radius: .9rem;
        border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.05);
      }}
      .stButton > button[kind="primary"] {{
        background: {ACCENT}; border-color: {ACCENT}; border-radius: 999px; font-weight: 700;
      }}
    </style>
    <div class="tw-hero">
      <div class="tw-kicker">Equatorial wave diagnostics</div>
      <h1>Tropical Wave Tools</h1>
      <p>Wheeler–Kiladis 频率-波数谱与 Kelvin / ER / MRG / TD / MJO 波动滤波的交互演示。
      选择数据与参数，在下方标签页中运行分析并下载结果。</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="加载内置样例…")
def load_sample() -> xr.DataArray:
    return open_example_olr().load()


@st.cache_data(show_spinner="读取 NetCDF…")
def read_upload(payload: bytes, variable: str | None) -> tuple[list[str], xr.DataArray | None]:
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
        handle.write(payload)
        path = Path(handle.name)
    try:
        with xr.open_dataset(path) as dataset:
            names = list(dataset.data_vars)
        data = load_dataarray(path, variable=variable).load() if variable else None
    finally:
        path.unlink(missing_ok=True)
    return names, data


def fig_to_png(figure: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    return buffer.getvalue()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("数据")
    source = st.radio("来源", ("内置样例", "上传 NetCDF"), horizontal=True)
    data: xr.DataArray
    if source == "内置样例":
        data = load_sample()
        st.caption(f"样例文件：`{get_sample_path().name}`")
    else:
        uploaded = st.file_uploader("NetCDF 文件", type=["nc"])
        if uploaded is None:
            st.info("请上传 NetCDF 文件，或切换回内置样例。")
            st.stop()
        payload = uploaded.getvalue()
        names, _ = read_upload(payload, None)
        variable = st.selectbox("变量", names)
        try:
            _, loaded = read_upload(payload, variable)
        except Exception as exc:  # surface any reader error in the UI
            st.error(f"无法读取变量 `{variable}`：{exc}")
            st.stop()
        data = loaded

    lat_bounds = st.slider("纬度范围 (°)", -30, 30, (-15, 15))
    data = data.sortby("lat").sel(lat=slice(*lat_bounds))
    if data.sizes.get("lat", 0) < 2:
        st.error("所选纬度范围内网格点不足。")
        st.stop()

summary = describe_dataarray(data)
c1, c2, c3, c4 = st.columns(4)
c1.metric("变量", str(summary["name"]))
c2.metric("时间步", f"{data.sizes['time']}")
c3.metric("纬度点", f"{data.sizes['lat']}")
c4.metric("经度点", f"{data.sizes['lon']}")
with st.expander("数据摘要"):
    st.json(summary)

tab_wk, tab_filter = st.tabs(["WK 频谱", "波动滤波"])

# ---------------------------------------------------------------- WK spectrum
with tab_wk:
    left, right = st.columns([1, 3])
    with left:
        window = st.number_input("窗口长度 (天)", 32, 256, 96, step=8)
        skip = st.number_input("窗口步长 (天)", 8, 128, 30, step=2)
        max_wn = st.slider("最大波数", 5, 20, 15)
        max_freq = st.slider("最大频率 (cpd)", 0.2, 0.8, 0.5, step=0.05)
        run_wk = st.button("运行频谱分析", type="primary")
    with right:
        if run_wk:
            if data.sizes["time"] < window:
                st.error(f"时间长度 ({data.sizes['time']}) 小于窗口长度 ({window})。")
            else:
                with st.spinner("计算 WK 频谱…"):
                    result = analyze_wk_spectrum(
                        data,
                        config=SpectralConfig(
                            window_size_days=int(window), window_skip_days=int(skip)
                        ),
                    )
                    figure, _ = plot_wk_spectrum(result, max_wn=max_wn, max_freq=max_freq)
                st.pyplot(figure)
                st.download_button("下载 PNG", fig_to_png(figure), "wk_spectrum.png", "image/png")
                plt.close(figure)
        else:
            st.caption("设置参数后点击「运行频谱分析」。")

# ---------------------------------------------------------------- filtering
with tab_filter:
    left, right = st.columns([1, 3])
    with left:
        wave_name = st.selectbox("波动类型", WAVE_CHOICES)
        method = st.selectbox("滤波方法", ("cckw", "legacy"))
        spec = DEFAULT_WAVE_SPECS[wave_name]
        lo, hi = spec.period_days
        st.caption(
            f"周期 {lo}–{hi} 天 · 纬向波数 {spec.wavenumber[0]} 到 {spec.wavenumber[1]}"
            + (
                f" · 等效深度 {spec.equivalent_depth[0]}–{spec.equivalent_depth[1]} m"
                if spec.equivalent_depth[0] is not None
                else ""
            )
        )
        run_filter = st.button("运行滤波", type="primary")
    with right:
        if run_filter:
            with st.spinner("滤波中…"):
                filtered = filter_wave_signal(
                    data, wave_name=wave_name, method=method, n_workers=1
                )
                figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
                filtered.std("time").plot(
                    ax=axes[0], cmap="Spectral_r", cbar_kwargs={"label": "std"}
                )
                axes[0].set_title(f"{wave_name.upper()} filtered std")
                equator = filtered.sel(lat=0.0, method="nearest")
                equator.plot(ax=axes[1], x="lon", y="time", cmap="RdBu_r", center=0.0)
                axes[1].set_title(f"{wave_name.upper()} equatorial Hovmoller")
            st.pyplot(figure)
            plt.close(figure)
            st.download_button(
                "下载滤波结果 (NetCDF)",
                bytes(filtered.to_dataset(name=f"{wave_name}_filtered").to_netcdf()),
                f"{wave_name}_filtered.nc",
                "application/x-netcdf",
            )
        else:
            st.caption("选择波动类型后点击「运行滤波」。")
