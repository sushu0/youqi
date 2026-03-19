# -*- coding: utf-8 -*-
"""饱和度与油层标志计算。"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from chong32_formulas import calculate_saturation_and_oil_flag


class SaturationCalculator:
    def __init__(self, config):
        self.config = config
        self.input_dir = config["input_dir"]
        self.output_dir = config["output_dir"]
        self.MISSING_VALUE = -999.25
        self.well_tops_data = self._load_tops_data()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")

    def _load_tops_data(self):
        tops_path = self.config.get("tops_path")
        if not tops_path:
            return None

        try:
            tops_data = pd.read_excel(tops_path)
        except FileNotFoundError:
            print(f"警告: 分层数据文件未找到: {tops_path}")
            return None
        except Exception as exc:
            print(f"加载分层数据失败: {exc}")
            return None

        if "Well" not in tops_data.columns or "MD" not in tops_data.columns:
            print(f"警告: 分层数据缺少 Well / MD 列: {tops_path}")
            return None

        tops_data = tops_data.copy()
        tops_data["Well"] = tops_data["Well"].astype(str)
        tops_data.rename(columns={"MD": "DEPTH_TOP"}, inplace=True)
        print(f"分层数据加载成功: {tops_path}")
        return tops_data

    def _get_surface_series(self, log_data, well_name, depth_curve):
        if self.well_tops_data is None:
            return pd.Series(np.nan, index=log_data.index)

        well_tops = self.well_tops_data[self.well_tops_data["Well"] == well_name].copy()
        if well_tops.empty:
            return pd.Series(np.nan, index=log_data.index)

        depth_df = pd.DataFrame(
            {
                "__row_id": log_data.index,
                depth_curve: pd.to_numeric(log_data[depth_curve], errors="coerce"),
            }
        ).sort_values(depth_curve)

        well_tops["DEPTH_TOP"] = pd.to_numeric(well_tops["DEPTH_TOP"], errors="coerce")
        well_tops = well_tops.dropna(subset=["DEPTH_TOP"]).sort_values("DEPTH_TOP")
        if well_tops.empty:
            return pd.Series(np.nan, index=log_data.index)

        merged = pd.merge_asof(
            depth_df,
            well_tops[["DEPTH_TOP", "Surface"]],
            left_on=depth_curve,
            right_on="DEPTH_TOP",
            direction="backward",
        )
        return merged.sort_values("__row_id").set_index("__row_id")["Surface"].reindex(log_data.index)

    def _fill_output_missing_values(self, dataframe):
        numeric_cols = list(dataframe.select_dtypes(include=[np.number]).columns)
        text_cols = [col for col in dataframe.columns if col not in numeric_cols]

        if numeric_cols:
            dataframe.loc[:, numeric_cols] = dataframe[numeric_cols].fillna(self.MISSING_VALUE)
        if text_cols:
            dataframe.loc[:, text_cols] = dataframe[text_cols].fillna("")
        return dataframe

    def _clip_valid_series(self, series, lower=None, upper=None):
        clipped = series.copy()
        valid_mask = clipped.notna() & (clipped != self.MISSING_VALUE)
        clipped.loc[valid_mask] = clipped.loc[valid_mask].clip(lower=lower, upper=upper)
        return clipped

    def _percent_series_to_fraction(self, series):
        fraction = series.copy()
        valid_mask = fraction.notna() & (fraction != self.MISSING_VALUE)
        fraction.loc[valid_mask] = fraction.loc[valid_mask] / 100.0
        fraction.loc[~valid_mask] = self.MISSING_VALUE
        return fraction

    def _calculate_sw_archie(self, rt, poro, rw):
        params = self.config["archie_params"]
        a, m, n = params["a"], params["m"], params["n"]
        poro_safe = np.where(poro > 0, poro, np.nan)
        formation_factor = a / (poro_safe**m)
        sw = (formation_factor * rw / rt) ** (1 / n)
        return np.where(poro > 0, sw, 1)

    def _calculate_sw_indonesia(self, rt, poro, vsh, rw):
        params = self.config["indonesia_params"]
        a, m, n, rsh = params["a"], params["m"], params["n"], params["rsh"]
        poro_safe = np.where(poro > 0, poro, np.nan)
        ro = a * rw / (poro_safe**m)
        c_term = 1 - (vsh * 0.5)
        b_term = (rt / ro) ** 0.5
        a_term = (vsh**c_term) / ((rsh / rt) ** 0.5)
        sw = (a_term + b_term) ** (-2 / n)
        return np.where(poro > 0, sw, 1)

    def run_batch_processing(self):
        all_files = [f for f in os.listdir(self.input_dir) if f.endswith(".csv")]
        process_bar = tqdm(all_files, desc="整体进度", leave=True, ascii=True, ncols=80)

        for filename in process_bar:
            well_name = filename.replace(".csv", "")
            process_bar.set_description(f"正在处理: {well_name}")
            try:
                self._process_single_well(filename, well_name)
            except Exception as exc:
                print(f"处理井 {well_name} 时发生错误: {exc}")

    def _process_single_well(self, filename, well_name):
        file_path = os.path.join(self.input_dir, filename)
        log_data = pd.read_csv(file_path, encoding="gb2312")
        log_data.replace(self.MISSING_VALUE, np.nan, inplace=True)

        depth_curve = log_data.columns[0]
        curves = self.config["required_curves"]
        model = self.config["calculation_model"]

        required_curve_keys = ["rt", "poro"]
        if model == "indonesia":
            required_curve_keys.append("vsh")

        if not all(curves[key] in log_data.columns for key in required_curve_keys):
            print(f"警告: 井 {well_name} 缺少必要曲线，跳过。")
            return

        rt = pd.to_numeric(log_data[curves["rt"]], errors="coerce")
        poro = pd.to_numeric(log_data[curves["poro"]], errors="coerce")
        vsh = None
        if "vsh" in curves and curves["vsh"] in log_data.columns:
            vsh = pd.to_numeric(log_data[curves["vsh"]], errors="coerce")

        if model == "chong32":
            surface_series = self._get_surface_series(log_data, well_name, depth_curve)
            if surface_series.isna().all():
                print(f"警告: 井 {well_name} 未找到可匹配分层，跳过重32公式计算。")
                return

            por_percent_curve = self.config.get("por_percent_curve")
            if por_percent_curve and por_percent_curve in log_data.columns:
                por_percent = pd.to_numeric(log_data[por_percent_curve], errors="coerce")
            else:
                por_percent = poro * 100.0

            result = calculate_saturation_and_oil_flag(
                rt_series=rt,
                por_percent_series=por_percent,
                surface_series=surface_series,
                surface_to_layer_map=self.config.get("surface_to_layer_map"),
                missing_value=self.MISSING_VALUE,
            )

            log_data["FORM_LAYER"] = result["layer_key"].fillna("")
            log_data["SW_PERCENT_PRE"] = result["sw_percent"]
            log_data["SO_PERCENT_PRE"] = result["so_percent"]
            log_data["SW_PRE"] = self._percent_series_to_fraction(result["sw_percent"])
            log_data["SO_PRE"] = self._percent_series_to_fraction(result["so_percent"])
            log_data["OIL_PRE"] = result["oil_flag"]
            log_data["JC_PRE"] = result["jc_flag"]
        elif model == "archie":
            rw = self.config["rw_fallback_rules"]["default"]
            sw = self._calculate_sw_archie(rt, poro, rw)
            log_data["SW_PRE"] = pd.Series(sw, index=log_data.index).clip(0, 1)
            log_data["SO_PRE"] = 1 - log_data["SW_PRE"]
        elif model == "indonesia":
            if vsh is None:
                print(f"警告: 井 {well_name} 缺少 Vsh 曲线，无法运行 Indonesia 模型。")
                return
            rw = self.config["rw_fallback_rules"]["default"]
            sw = self._calculate_sw_indonesia(rt, poro, vsh, rw)
            log_data["SW_PRE"] = pd.Series(sw, index=log_data.index).clip(0, 1)
            log_data["SO_PRE"] = 1 - log_data["SW_PRE"]
        else:
            raise ValueError(f"未知计算模型: {model}")

        if "SW_PRE" in log_data.columns:
            log_data["SW_PRE"] = self._clip_valid_series(log_data["SW_PRE"], 0, 1)
        if "SO_PRE" in log_data.columns:
            log_data["SO_PRE"] = self._clip_valid_series(log_data["SO_PRE"], 0, 1)
        if "SW_PERCENT_PRE" in log_data.columns:
            log_data["SW_PERCENT_PRE"] = self._clip_valid_series(log_data["SW_PERCENT_PRE"], 0, 100)
        if "SO_PERCENT_PRE" in log_data.columns:
            log_data["SO_PERCENT_PRE"] = self._clip_valid_series(log_data["SO_PERCENT_PRE"], 0, 80)

        log_data = self._fill_output_missing_values(log_data)
        output_path = os.path.join(self.output_dir, filename)
        log_data.to_csv(output_path, index=False, encoding="gb2312")


if __name__ == "__main__":
    from local_project_paths import (
        PETROPHYS_DIR,
        SATURATION_DIR,
        SURFACE_TO_LAYER_MAP,
        ensure_output_dirs,
        prepare_local_tops_workbook,
    )

    ensure_output_dirs()
    tops_path = prepare_local_tops_workbook()

    config = {
        "input_dir": str(PETROPHYS_DIR),
        "output_dir": str(SATURATION_DIR),
        "tops_path": str(tops_path),
        "calculation_model": "chong32",
        "required_curves": {
            "rt": "RTCAL",
            "poro": "POR_PRE",
            "vsh": "VSH_PRE",
        },
        "por_percent_curve": "POR_PERCENT_PRE",
        "surface_to_layer_map": SURFACE_TO_LAYER_MAP,
    }

    print("使用项目内默认路径运行重32饱和度计算")
    print(f"输入目录: {PETROPHYS_DIR}")
    print(f"分层文件: {tops_path}")
    print(f"输出目录: {SATURATION_DIR}")

    calculator = SaturationCalculator(config)
    calculator.run_batch_processing()

    print(f"饱和度计算完成，结果目录: {SATURATION_DIR}")
