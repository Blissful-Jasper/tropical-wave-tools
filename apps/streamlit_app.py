from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import xarray as xr

from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.io import describe_dataarray, load_dataarray
from tropical_wave_tools.plotting import plot_wk_spectrum
from tropical_wave_tools.sample_data import get_sample_path, open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

st.set_page_config(page_title="Tropical Wave Tools Demo", layout="wide")
st.title("Tropical Wave Tools Demo")
st.caption("一个轻量交互页面，用来快速体验 WK 频谱分析与波动滤波。")

source_mode = st.sidebar.radio("数据来源", ("内置样例", "上传 NetCDF"))

data: xr.DataArray
if source_mode == "内置样例":
    data = open_example_olr()
    st.sidebar.write(f"样例文件: `{get_sample_path().name}`")
else:
    uploaded = st.sidebar.file_uploader("上传 NetCDF 文件", type=["nc"])
    if uploaded is None:
        st.info("请先上传一个 NetCDF 文件，或者切换回内置样例。")
        st.stop()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as temp_file:
        temp_file.write(uploaded.getbuffer())
        temp_path = Path(temp_file.name)
    dataset = xr.open_dataset(temp_path)
    variable = st.sidebar.selectbox("变量", options=list(dataset.data_vars))
    data = load_dataarray(temp_path, variable=variable)

st.write("### 数据摘要")
st.json(describe_dataarray(data))

mode = st.sidebar.selectbox("分析模式", ("WK Spectrum", "Wave Filter"))
lat_bounds = st.sidebar.slider("纬度范围", min_value=-15, max_value=15, value=(-15, 15))

data = data.sel(lat=slice(lat_bounds[0], lat_bounds[1])).sortby("lat")

if mode == "WK Spectrum":
    if st.button("运行频谱分析"):
        result = analyze_wk_spectrum(data)
        figure, _ = plot_wk_spectrum(result)
        st.pyplot(figure)
        plt.close(figure)
else:
    wave_name = st.sidebar.selectbox("波动类型", ("kelvin", "er", "mjo", "td"))
    method = st.sidebar.selectbox("滤波方法", ("cckw", "legacy"))
    if st.button("运行滤波"):
        filtered = filter_wave_signal(data, wave_name=wave_name, method=method, n_workers=1)
        figure, axis = plt.subplots(figsize=(8, 4))
        filtered.std("time").plot(ax=axis, cmap="Spectral_r")
        axis.set_title(f"{wave_name.upper()} STD")
        st.pyplot(figure)
        plt.close(figure)

