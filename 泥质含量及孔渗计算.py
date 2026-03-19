# -*- coding: utf-8 -*-
"""泥质含量、孔隙度与渗透率计算。"""

import os

import numpy as np
import pandas as pd

from chong32_formulas import (
    calculate_lithology,
    calculate_permeability_from_porosity_percent,
    calculate_porosity_percent_from_density,
    calculate_vertical_permeability_from_perm,
)


class PetrophysicalCalculator:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.MISSING_VALUE = -999.25

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")

    def _calculate_vsh_from_gr(self, gr_series):
        gr = gr_series.copy()
        valid_gr = gr[gr != self.MISSING_VALUE]
        if valid_gr.empty:
            return pd.Series(np.full(len(gr), self.MISSING_VALUE), index=gr.index)

        gr_min, gr_max = valid_gr.min(), valid_gr.max()
        if gr_max == gr_min:
            result = pd.Series(np.zeros(len(gr)), index=gr.index, dtype=float)
            result[gr == self.MISSING_VALUE] = self.MISSING_VALUE
            return result

        gr_index = (gr - gr_min) / (gr_max - gr_min)
        vsh = (2 ** (2 * gr_index) - 1) / 3
        vsh[gr_series == self.MISSING_VALUE] = self.MISSING_VALUE
        return vsh

    def _calculate_lithology_from_rt_den(self, rt_series, density_series):
        return calculate_lithology(rt_series, density_series, self.MISSING_VALUE)

    def _calculate_por_from_density(self, density_series, lith_series):
        por_percent = calculate_porosity_percent_from_density(density_series, lith_series, self.MISSING_VALUE)
        por_fraction = por_percent.copy()
        valid_mask = por_fraction != self.MISSING_VALUE
        por_fraction.loc[valid_mask] = por_fraction.loc[valid_mask] / 100.0
        return por_fraction, por_percent

    def _calculate_perm_from_por(self, por_series):
        por_percent = por_series.copy()
        valid_mask = por_percent != self.MISSING_VALUE
        if valid_mask.any() and por_percent.loc[valid_mask].max() <= 1:
            por_percent.loc[valid_mask] = por_percent.loc[valid_mask] * 100.0
        return calculate_permeability_from_porosity_percent(por_percent, self.MISSING_VALUE)

    def process_files(
        self,
        vsh_curve,
        por_sonic_curve,
        por_vsh_curve,
        perm_por_curve,
        lith_rt_curve="RTCAL",
        lith_density_curve="DEN",
    ):
        del por_sonic_curve, por_vsh_curve

        filename_list = os.listdir(self.input_dir)
        error_files = []

        for filename in filename_list:
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(self.input_dir, filename)
            try:
                log_data = pd.read_csv(file_path)
                print(f"--- 正在处理文件: {filename} ---")

                if vsh_curve in log_data.columns:
                    print(f"计算泥质含量(Vsh)，使用曲线: {vsh_curve}")
                    log_data["VSH_PRE"] = self._calculate_vsh_from_gr(log_data[vsh_curve])
                    log_data.loc[(log_data["VSH_PRE"] > 1) & (log_data["VSH_PRE"] != self.MISSING_VALUE), "VSH_PRE"] = 1
                    log_data.loc[(log_data["VSH_PRE"] < 0) & (log_data["VSH_PRE"] != self.MISSING_VALUE), "VSH_PRE"] = 0
                else:
                    print(f"警告: 文件 {filename} 缺少曲线 {vsh_curve}，跳过 Vsh 计算。")

                if lith_rt_curve in log_data.columns and lith_density_curve in log_data.columns:
                    print(f"计算岩性与孔隙度，使用曲线: {lith_rt_curve}, {lith_density_curve}")
                    log_data["LITH_PRE"] = self._calculate_lithology_from_rt_den(
                        log_data[lith_rt_curve],
                        log_data[lith_density_curve],
                    )
                    por_fraction, por_percent = self._calculate_por_from_density(
                        log_data[lith_density_curve],
                        log_data["LITH_PRE"],
                    )
                    log_data["POR_PRE"] = por_fraction
                    log_data["POR_PERCENT_PRE"] = por_percent
                else:
                    print(
                        f"警告: 文件 {filename} 缺少重32公式所需曲线 {lith_rt_curve} / {lith_density_curve}，"
                        "跳过孔隙度计算。"
                    )

                if perm_por_curve in log_data.columns:
                    print(f"计算渗透率(PERM)，使用曲线: {perm_por_curve}")
                    log_data["PERM_PRE"] = self._calculate_perm_from_por(log_data[perm_por_curve])
                    log_data["PERMV_PRE"] = calculate_vertical_permeability_from_perm(
                        log_data["PERM_PRE"],
                        self.MISSING_VALUE,
                    )
                else:
                    print(f"警告: 文件 {filename} 缺少曲线 {perm_por_curve}，跳过渗透率计算。")

                output_file_path = os.path.join(self.output_dir, filename)
                log_data.to_csv(output_file_path, encoding="gb2312", index=False)
                print(f"结果已保存至: {output_file_path}")

            except Exception as exc:
                print(f"处理文件 {filename} 时发生错误: {exc}")
                error_files.append(filename)

        if error_files:
            print("\n--- 以下文件处理失败 ---")
            for failed_file in error_files:
                print(failed_file)
        else:
            print("\n所有文件处理成功完成。")


if __name__ == "__main__":
    from local_project_paths import PETROPHYS_DIR, RAW_CSV_DIR, ensure_output_dirs, ensure_raw_csv_exists

    ensure_output_dirs()
    ensure_raw_csv_exists()

    print("使用项目内默认路径运行重32孔渗计算")
    print(f"输入目录: {RAW_CSV_DIR}")
    print(f"输出目录: {PETROPHYS_DIR}")

    calculator = PetrophysicalCalculator(
        input_dir=str(RAW_CSV_DIR),
        output_dir=str(PETROPHYS_DIR),
    )
    calculator.process_files(
        vsh_curve="GR",
        por_sonic_curve=None,
        por_vsh_curve=None,
        perm_por_curve="POR_PERCENT_PRE",
        lith_rt_curve="RTCAL",
        lith_density_curve="DEN",
    )

    print(f"孔渗计算完成，结果目录: {PETROPHYS_DIR}")
