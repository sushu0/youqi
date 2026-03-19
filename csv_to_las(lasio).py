# -*- coding: utf-8 -*-
"""CSV 转 LAS。"""

import os
from datetime import datetime

import lasio
import pandas as pd
from tqdm import tqdm


if not hasattr(lasio.LASFile, "add_curve"):
    lasio.LASFile.add_curve = lasio.LASFile.append_curve


class CsvToLasConverter:
    def __init__(self, config):
        self.config = config
        self.input_dir = config["input_dir"]
        self.output_dir = config["output_dir"]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")

    def _fill_output_missing_values(self, dataframe):
        numeric_cols = list(dataframe.select_dtypes(include="number").columns)
        text_cols = [col for col in dataframe.columns if col not in numeric_cols]

        if numeric_cols:
            dataframe.loc[:, numeric_cols] = dataframe[numeric_cols].fillna(
                self.config.get("null_value", -999.25)
            )
        if text_cols:
            dataframe.loc[:, text_cols] = dataframe[text_cols].fillna("")
        return dataframe

    def _create_las_object(self, dataframe, well_name):
        las = lasio.LASFile()
        las.well.WELL = well_name
        las.well.UWI = self.config.get("well_uwi", well_name)
        las.well.DATE = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        las.well.NULL = self.config.get("null_value", -999.25)

        for key, value in self.config.get("well_header_info", {}).items():
            las.well[key] = value

        for curve_name in dataframe.columns:
            las.add_curve(curve_name, dataframe[curve_name].values, unit="")
        return las

    def run_batch_conversion(self):
        all_files = [f for f in os.listdir(self.input_dir) if f.endswith(".csv")]
        process_bar = tqdm(all_files, desc="整体进度", leave=True, ascii=True, ncols=80)

        for filename in process_bar:
            well_name = filename.replace(".csv", "")
            process_bar.set_description(f"正在处理: {well_name}")
            try:
                self._process_single_file(filename, well_name)
            except Exception as exc:
                print(f"处理文件 {filename} 时发生错误: {exc}")

    def _process_single_file(self, filename, well_name):
        file_path = os.path.join(self.input_dir, filename)
        data = pd.read_csv(file_path, encoding="gb2312")

        depth_curve = data.columns[0]
        data = data[data[depth_curve] != -999.25].copy()
        data.sort_values(by=depth_curve, ascending=True, inplace=True)

        curves_to_include = self.config.get("curves_to_include", [])
        if curves_to_include:
            final_curves = [depth_curve] + [c for c in curves_to_include if c in data.columns and c != depth_curve]
            data = data[final_curves]

        data = self._fill_output_missing_values(data)

        las_file = self._create_las_object(data, well_name)
        output_path = os.path.join(self.output_dir, f"{well_name}.las")
        las_file.write(output_path, version=2.0, fmt="%.3f")


if __name__ == "__main__":
    from local_project_paths import INTERPRET_DIR, SATURATION_DIR, SATURATION_LAS_DIR, ensure_output_dirs

    ensure_output_dirs()
    if any(INTERPRET_DIR.glob("*.csv")):
        input_dir = INTERPRET_DIR
        output_dir = INTERPRET_DIR
    else:
        input_dir = SATURATION_DIR
        output_dir = SATURATION_LAS_DIR

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "curves_to_include": [
            "GR",
            "SP",
            "CAL",
            "GR_b",
            "SP_b",
            "CAL_b",
            "RTCAL",
            "RTCALQ",
            "DT",
            "DEN",
            "NPHI",
            "DT_b",
            "DEN_b",
            "NPHI_b",
            "POR_PRE",
            "POR_PERCENT_PRE",
            "PERM_PRE",
            "PERMV_PRE",
            "SW_PRE",
            "SO_PRE",
            "SW_PERCENT_PRE",
            "SO_PERCENT_PRE",
            "VSH_PRE",
            "LITH_PRE",
            "FORM_LAYER",
            "OIL_PRE",
            "JC_PRE",
            "Sand_RT",
        ],
        "null_value": -999.25,
        "well_header_info": {
            "STEP": 0.1,
            "COMP": "LOCAL_RUN",
            "FLD": "CHONG32",
            "SRVC": "OPENAI-CODEX",
            "CTRY": "CHINA",
        },
        "well_uwi": "",
    }

    print("使用项目内默认路径执行 CSV 转 LAS")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    converter = CsvToLasConverter(config)
    converter.run_batch_conversion()

    print(f"LAS 导出完成，结果目录: {output_dir}")
