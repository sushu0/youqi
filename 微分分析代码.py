# -*- coding: utf-8 -*-
"""基于重32公式的微分分析辅助脚本。"""

import os
from datetime import datetime

import lasio
import numpy as np
import pandas as pd

from chong32_formulas import CHONG32_LAYER_PARAMETERS, get_layer_key_from_surface


DEFAULT_LAYER_KEY = "G1"
MISSING_VALUE = -999.25


def get_flist(path, fmt_input):
    return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(fmt_input)]


def read_las(filepath):
    las_data = lasio.read(filepath)
    data = las_data.df()
    data.dropna(how="all", inplace=True)
    data.reset_index(inplace=True)
    return data


def read_data(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="gb2312")
    if path.lower().endswith(".las"):
        return read_las(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"无法识别的输入格式: {path}")


def write_las(data, cols, output_path, well):
    las = lasio.LASFile()
    las.well.DATE = datetime.today()
    las.well.WELL = well
    las.well.NULL = MISSING_VALUE
    las.well.UWI = well

    for col in cols:
        curve_name = "DEPT" if col == "#DEPTH" else col
        las.append_curve(curve_name, data[col])
    las.write(output_path, version=2)


def resolve_layer_key(layer_value, default_layer_key=DEFAULT_LAYER_KEY):
    if pd.isna(layer_value) or str(layer_value).strip() == "":
        return default_layer_key

    text = str(layer_value).strip()
    if text in CHONG32_LAYER_PARAMETERS:
        return text
    return get_layer_key_from_surface(text) or default_layer_key


def get_layer_params(layer_value):
    layer_key = resolve_layer_key(layer_value)
    return CHONG32_LAYER_PARAMETERS[layer_key]


def normalize_porosity(porosity):
    if pd.isna(porosity) or porosity in (MISSING_VALUE, 0):
        return np.nan
    if porosity > 1:
        return porosity / 100.0
    return porosity


def normalize_resistivity(resistivity):
    if pd.isna(resistivity) or resistivity in (MISSING_VALUE, 0):
        return np.nan
    return resistivity


def cal_line_w(porosity, layer_value=DEFAULT_LAYER_KEY):
    porosity = normalize_porosity(porosity)
    if pd.isna(porosity):
        return np.nan
    params = get_layer_params(layer_value)
    return -(params["a"] * params["m"] * params["rw"]) / np.power(porosity, params["m"] + 1)


def cal_line_o(porosity, layer_value=DEFAULT_LAYER_KEY):
    porosity = normalize_porosity(porosity)
    if pd.isna(porosity):
        return np.nan
    params = get_layer_params(layer_value)
    swi = max(0.0, 1.0 - params["so_cutoff"] / 100.0)
    return -(params["a"] * params["m"] * params["rw"]) / (
        np.power(porosity, params["m"] + 1) * np.power(swi, params["n"])
    )


def cal_line_r(porosity, resistivity, layer_value=DEFAULT_LAYER_KEY):
    porosity = normalize_porosity(porosity)
    resistivity = normalize_resistivity(resistivity)
    if pd.isna(porosity) or pd.isna(resistivity):
        return np.nan
    params = get_layer_params(layer_value)
    sw = np.power(
        (params["a"] * params["b"] * params["rw"])
        / (resistivity * np.power(porosity, params["m"])),
        1 / params["n"],
    )
    return -(params["a"] * params["m"] * params["rw"]) / (
        np.power(sw, params["n"]) * np.power(porosity, params["m"] + 1)
    )


def cal_line_r2(porosity, resistivity, layer_value=DEFAULT_LAYER_KEY):
    porosity = normalize_porosity(porosity)
    resistivity = normalize_resistivity(resistivity)
    if pd.isna(porosity) or pd.isna(resistivity):
        return np.nan
    params = get_layer_params(layer_value)
    sw = np.power(
        (params["a"] * params["b"] * params["rw"])
        / (resistivity * np.power(porosity, params["m"])),
        1 / params["n"],
    )
    return -(params["a"] * params["rw"]) / (
        np.power(sw, params["n"] + 1) * np.power(porosity, params["m"])
    )


if __name__ == "__main__":
    from local_project_paths import DIFFERENTIAL_DIR, ensure_output_dirs, first_existing_las_file

    ensure_output_dirs()
    input_file = first_existing_las_file()
    if input_file is None:
        raise SystemExit("未找到可用于微分分析的 LAS 文件，请先运行主流程生成解释结果。")

    file = str(input_file)
    outfold = str(DIFFERENTIAL_DIR)
    rt_curve = "RTCAL"
    por_curve = "POR_PRE"
    layer_curve = "FORM_LAYER"
    default_layer = "G1"

    os.makedirs(outfold, exist_ok=True)
    _, filename = os.path.split(file)
    outpath = os.path.join(outfold, filename)
    data = read_data(file)

    if por_curve in data.columns and rt_curve in data.columns:
        if "DEPT:1" in data.columns:
            data.rename(columns={"DEPT:1": "DEPT"}, inplace=True)

        keep_cols = ["DEPT", por_curve, rt_curve]
        if layer_curve in data.columns:
            keep_cols.append(layer_curve)
        data = data[keep_cols].copy()

        data["LINE_WATER"] = np.nan
        data["LINE_OIL"] = np.nan
        data["LINE_RESULT"] = np.nan

        for i in range(len(data)):
            layer_value = data.loc[i, layer_curve] if layer_curve in data.columns else default_layer
            data.loc[i, "LINE_WATER"] = cal_line_w(data.loc[i, por_curve], layer_value)
            data.loc[i, "LINE_OIL"] = cal_line_o(data.loc[i, por_curve], layer_value)
            data.loc[i, "LINE_RESULT"] = cal_line_r(data.loc[i, por_curve], data.loc[i, rt_curve], layer_value)

        cols = data.columns
        write_las(data, cols, outpath, filename.split(".")[0])
        print(f"微分分析完成: {outpath}")
