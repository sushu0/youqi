from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "_local_run"
RAW_CSV_DIR = OUTPUT_ROOT / "01_raw_csv"
PETROPHYS_DIR = OUTPUT_ROOT / "02_petrophysics"
SATURATION_DIR = OUTPUT_ROOT / "03_saturation"
SATURATION_LAS_DIR = OUTPUT_ROOT / "04_saturation_las"
SAND_RT_LAS_DIR = OUTPUT_ROOT / "05_sand_rt_las"
SAND_RT_CSV_DIR = OUTPUT_ROOT / "06_sand_rt_csv"
INTERPRET_DIR = OUTPUT_ROOT / "07_interpretation"
DIFFERENTIAL_DIR = OUTPUT_ROOT / "08_differential_analysis"
META_DIR = OUTPUT_ROOT / "meta"

SURFACE_TO_LAYER_MAP = {
    "J3q22-1": "G1",
    "J3q22-2": "G2",
    "J3q22-3": "G3",
    "G1": "G1",
    "G2": "G2",
    "G3": "G3",
}


def ensure_output_dirs():
    for path in [
        OUTPUT_ROOT,
        RAW_CSV_DIR,
        PETROPHYS_DIR,
        SATURATION_DIR,
        SATURATION_LAS_DIR,
        SAND_RT_LAS_DIR,
        SAND_RT_CSV_DIR,
        INTERPRET_DIR,
        DIFFERENTIAL_DIR,
        META_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def find_tops_workbook():
    exact_path = BASE_DIR / "01 基础数据" / "03 分层" / "重32分层（2-1&2-2分开）.xlsx"
    if exact_path.exists():
        return exact_path

    search_dir = BASE_DIR / "01 基础数据" / "03 分层"
    if not search_dir.exists():
        raise FileNotFoundError(f"未找到分层目录: {search_dir}")

    for xlsx_path in sorted(search_dir.glob("*.xlsx")):
        try:
            excel_file = pd.ExcelFile(xlsx_path)
        except Exception:
            continue
        if "最终GPT" in excel_file.sheet_names:
            return xlsx_path

    raise FileNotFoundError("未找到包含“最终GPT”工作表的分层文件。")


def prepare_local_tops_workbook():
    ensure_output_dirs()
    source_path = find_tops_workbook()
    tops_df = pd.read_excel(source_path, sheet_name="最终GPT")

    adapted = tops_df.rename(
        columns={
            "well": "Well",
            "层位": "Surface",
            "顶深": "MD",
            "底深": "Bottom",
            "厚度": "Thickness",
        }
    ).copy()
    adapted["Well"] = adapted["Well"].astype(str)
    adapted["MD"] = pd.to_numeric(adapted["MD"], errors="coerce")
    adapted = adapted.dropna(subset=["Well", "Surface", "MD"]).sort_values(["Well", "MD"])

    output_path = META_DIR / "tops_for_saturation.xlsx"
    adapted.to_excel(output_path, index=False)
    return output_path


def ensure_raw_csv_exists():
    ensure_output_dirs()
    if any(RAW_CSV_DIR.glob("*.csv")):
        return

    from run_local_pipeline import convert_las_to_csv

    convert_las_to_csv()


def first_existing_las_file():
    for folder in [INTERPRET_DIR, SATURATION_LAS_DIR]:
        las_files = sorted(folder.glob("*.las"))
        if las_files:
            return las_files[0]
    return None
