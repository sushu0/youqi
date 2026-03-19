from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import lasio
import numpy as np
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
META_DIR = OUTPUT_ROOT / "meta"


if not hasattr(lasio.LASFile, "add_curve"):
    lasio.LASFile.add_curve = lasio.LASFile.append_curve


def load_module(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, BASE_DIR / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_files(suffix: str):
    for path in BASE_DIR.rglob(f"*{suffix}"):
        if any(part.startswith(".venv") for part in path.parts):
            continue
        if OUTPUT_ROOT in path.parents:
            continue
        yield path


def reset_output_dirs():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    for path in [
        RAW_CSV_DIR,
        PETROPHYS_DIR,
        SATURATION_DIR,
        SATURATION_LAS_DIR,
        SAND_RT_LAS_DIR,
        SAND_RT_CSV_DIR,
        INTERPRET_DIR,
        META_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def prepare_tops_for_saturation() -> tuple[Path, pd.DataFrame]:
    source_path = None
    source_sheet = None
    tops_df = None

    for xlsx_path in sorted(project_files(".xlsx")):
        try:
            excel_file = pd.ExcelFile(xlsx_path)
        except Exception:
            continue
        if "最终GPT" in excel_file.sheet_names:
            source_path = xlsx_path
            source_sheet = "最终GPT"
            tops_df = pd.read_excel(xlsx_path, sheet_name=source_sheet)
            break

    if tops_df is None:
        raise FileNotFoundError("Could not find a workbook sheet named '最终GPT' for tops data.")

    required_cols = {"well", "层位", "顶深"}
    if not required_cols.issubset(tops_df.columns):
        raise ValueError(f"Tops sheet is missing required columns: {required_cols}")

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

    print(f"[tops] source={source_path} sheet={source_sheet} rows={len(adapted)} output={output_path}")
    return output_path, adapted


def convert_las_to_csv() -> pd.DataFrame:
    records = []

    for las_path in sorted(project_files(".LAS")):
        well_name = las_path.stem
        las = lasio.read(str(las_path), ignore_header_errors=True)
        df = las.df().reset_index()
        df.columns = [str(col).strip() for col in df.columns]

        if not len(df.columns):
            continue

        first_col = df.columns[0]
        if first_col.startswith("#"):
            df = df.rename(columns={first_col: "DEPT"})
            first_col = "DEPT"

        rename_map = {}
        for col in df.columns:
            clean = col.strip()
            if col != clean and clean not in df.columns:
                rename_map[col] = clean
        if rename_map:
            df = df.rename(columns=rename_map)

        # Normalize common resistivity naming so the later scripts can reuse a single curve name.
        if "RTCAL" not in df.columns:
            for candidate in ["RI", "RD", "LLD", "RILD", "RT"]:
                if candidate in df.columns:
                    df["RTCAL"] = df[candidate]
                    break

        df = df.replace([np.inf, -np.inf], np.nan).fillna(-999.25)
        output_path = RAW_CSV_DIR / f"{well_name}.csv"
        df.to_csv(output_path, index=False, encoding="gb2312")

        records.append(
            {
                "well": well_name,
                "source_las": str(las_path.relative_to(BASE_DIR)),
                "rows": len(df),
                "depth_curve": first_col,
                "has_gr": "GR" in df.columns,
                "has_ac": "AC" in df.columns,
                "has_den": "DEN" in df.columns,
                "has_rtcal": "RTCAL" in df.columns,
            }
        )

    summary = pd.DataFrame(records).sort_values("well").reset_index(drop=True)
    summary.to_csv(META_DIR / "las_to_csv_summary.csv", index=False, encoding="utf-8-sig")
    print(
        "[las->csv] files={} with_gr={} with_ac={} with_rtcal={}".format(
            len(summary),
            int(summary["has_gr"].sum()),
            int(summary["has_ac"].sum()),
            int(summary["has_rtcal"].sum()),
        )
    )
    return summary


def count_csv_with_columns(folder: Path, required_columns: list[str]) -> int:
    count = 0
    for csv_path in sorted(folder.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, encoding="gb2312", nrows=5)
        except Exception:
            continue
        if all(col in df.columns for col in required_columns):
            count += 1
    return count


def merge_sand_rt_into_csv():
    merged_count = 0
    for csv_path in sorted(SATURATION_DIR.glob("*.csv")):
        las_path = SAND_RT_LAS_DIR / f"{csv_path.stem}.las"
        if not las_path.exists():
            continue

        base_df = pd.read_csv(csv_path, encoding="gb2312")
        sand_df = lasio.read(str(las_path), ignore_header_errors=True).df().reset_index()
        sand_df.columns = [str(col).strip() for col in sand_df.columns]

        sand_col = next((col for col in sand_df.columns if col.upper() == "SAND_RT"), None)
        if sand_col is None:
            continue

        depth_col = base_df.columns[0]
        sand_depth_col = sand_df.columns[0]

        if (
            len(base_df) == len(sand_df)
            and depth_col == sand_depth_col
            and np.isclose(base_df.iloc[0, 0], sand_df.iloc[0, 0])
            and np.isclose(base_df.iloc[-1, 0], sand_df.iloc[-1, 0])
        ):
            base_df["Sand_RT"] = sand_df[sand_col].values
        else:
            merged = base_df.merge(
                sand_df[[sand_depth_col, sand_col]],
                left_on=depth_col,
                right_on=sand_depth_col,
                how="left",
            )
            base_df["Sand_RT"] = merged[sand_col].fillna(0)

        base_df.to_csv(SAND_RT_CSV_DIR / csv_path.name, index=False, encoding="gb2312")
        merged_count += 1

    print(f"[sand-rt-merge] merged={merged_count}")


def build_summary(tops_df: pd.DataFrame, las_summary: pd.DataFrame):
    tops_wells = set(tops_df["Well"].astype(str).unique())
    raw_wells = set(las_summary["well"].astype(str).unique())
    raw_full_wells = set(las_summary.loc[las_summary["has_gr"] & las_summary["has_ac"] & las_summary["has_rtcal"], "well"])

    summary = {
        "raw_las_files": int(len(las_summary)),
        "raw_csv_files": int(len(list(RAW_CSV_DIR.glob("*.csv")))),
        "tops_rows": int(len(tops_df)),
        "tops_wells": int(len(tops_wells)),
        "raw_wells": int(len(raw_wells)),
        "wells_with_gr_ac_rtcal": int(len(raw_full_wells)),
        "wells_with_gr_ac_rtcal_and_tops": int(len(raw_full_wells & tops_wells)),
        "petrophysics_csv_files": int(len(list(PETROPHYS_DIR.glob("*.csv")))),
        "petrophysics_with_por_perm": int(count_csv_with_columns(PETROPHYS_DIR, ["POR_PRE", "PERM_PRE"])),
        "saturation_csv_files": int(len(list(SATURATION_DIR.glob("*.csv")))),
        "saturation_with_sw": int(count_csv_with_columns(SATURATION_DIR, ["SW_PRE", "SO_PRE"])),
        "sand_rt_las_files": int(len(list(SAND_RT_LAS_DIR.glob("*.las")))),
        "sand_rt_csv_files": int(len(list(SAND_RT_CSV_DIR.glob("*.csv")))),
        "interpretation_csv_files": int(len(list(INTERPRET_DIR.glob("*.csv")))),
        "interpretation_las_files": int(len(list(INTERPRET_DIR.glob("*.las")))),
    }

    summary_path = META_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summary] {json.dumps(summary, ensure_ascii=False)}")
    print(f"[summary] written to {summary_path}")


def main():
    reset_output_dirs()

    tops_path, tops_df = prepare_tops_for_saturation()
    las_summary = convert_las_to_csv()

    petrophys_module = load_module("泥质含量及孔渗计算.py", "petrophys_module")
    saturation_module = load_module("含油饱和度计算.py", "saturation_module")
    csv_to_las_module = load_module("csv_to_las(lasio).py", "csv_to_las_module")
    processor_module = load_module("数据处理及油水解释.py", "processor_module")

    petrophys = petrophys_module.PetrophysicalCalculator(str(RAW_CSV_DIR), str(PETROPHYS_DIR))
    petrophys.process_files(
        vsh_curve="GR",
        por_sonic_curve=None,
        por_vsh_curve=None,
        perm_por_curve="POR_PERCENT_PRE",
        lith_rt_curve="RTCAL",
        lith_density_curve="DEN",
    )

    saturation = saturation_module.SaturationCalculator(
        {
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
            "surface_to_layer_map": {
                "J3q22-1": "G1",
                "J3q22-2": "G2",
                "J3q22-3": "G3",
                "G1": "G1",
                "G2": "G2",
                "G3": "G3",
            },
        }
    )
    saturation.run_batch_processing()

    las_converter = csv_to_las_module.CsvToLasConverter(
        {
            "input_dir": str(SATURATION_DIR),
            "output_dir": str(SATURATION_LAS_DIR),
            "curves_to_include": [
                "GR",
                "SP",
                "AC",
                "DEN",
                "CALI",
                "RI",
                "RTCAL",
                "VSH_PRE",
                "POR_PRE",
                "POR_PERCENT_PRE",
                "PERM_PRE",
                "PERMV_PRE",
                "SW_PRE",
                "SO_PRE",
                "SW_PERCENT_PRE",
                "SO_PERCENT_PRE",
                "OIL_PRE",
                "JC_PRE",
            ],
            "null_value": -999.25,
            "well_header_info": {
                "STEP": 0.125,
                "COMP": "LOCAL_RUN",
                "FLD": "CHONG32",
                "SRVC": "OPENAI-CODEX",
                "CTRY": "CHINA",
            },
            "well_uwi": "",
        }
    )
    las_converter.run_batch_conversion()

    processor = processor_module.WellLogProcessor()
    processor.sand_rt(
        input_dir=str(SATURATION_DIR),
        output_dir=str(SAND_RT_LAS_DIR),
        rt_log="RTCAL",
        rt_threshold=1.0,
        filter_scale=4,
    )

    merge_sand_rt_into_csv()

    for csv_path in sorted(SAND_RT_CSV_DIR.glob("*.csv")):
        shutil.copy2(csv_path, INTERPRET_DIR / csv_path.name)

    interpretation_converter = csv_to_las_module.CsvToLasConverter(
        {
            "input_dir": str(INTERPRET_DIR),
            "output_dir": str(INTERPRET_DIR),
            "curves_to_include": [
                "RTCAL",
                "DEN",
                "LITH_PRE",
                "POR_PRE",
                "POR_PERCENT_PRE",
                "PERM_PRE",
                "PERMV_PRE",
                "SW_PRE",
                "SO_PRE",
                "SW_PERCENT_PRE",
                "SO_PERCENT_PRE",
                "OIL_PRE",
                "JC_PRE",
                "Sand_RT",
            ],
            "null_value": -999.25,
            "well_header_info": {
                "STEP": 0.125,
                "COMP": "LOCAL_RUN",
                "FLD": "CHONG32",
                "SRVC": "OPENAI-CODEX",
                "CTRY": "CHINA",
            },
            "well_uwi": "",
        }
    )
    interpretation_converter.run_batch_conversion()

    build_summary(tops_df, las_summary)


if __name__ == "__main__":
    main()
