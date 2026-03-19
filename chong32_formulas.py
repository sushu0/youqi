from __future__ import annotations

import re

import numpy as np
import pandas as pd


MISSING_VALUE = -999.25


CHONG32_LAYER_PARAMETERS = {
    "G1": {
        "a": 0.926,
        "b": 0.947,
        "m": 1.39,
        "n": 2.003,
        "rw": 1.2,
        "rt_cutoff": 50.0,
        "por_cutoff": 23.0,
        "so_cutoff": 55.0,
    },
    "G2": {
        "a": 0.865,
        "b": 0.901,
        "m": 1.488,
        "n": 1.983,
        "rw": 1.2,
        "rt_cutoff": 25.0,
        "por_cutoff": 23.0,
        "so_cutoff": 50.0,
    },
    "G3": {
        "a": 0.99,
        "b": 0.938,
        "m": 1.372,
        "n": 2.013,
        "rw": 1.2,
        "rt_cutoff": 30.0,
        "por_cutoff": 23.0,
        "so_cutoff": 50.0,
    },
}


DEFAULT_SURFACE_TO_LAYER_MAP = {
    "G1": "G1",
    "G2": "G2",
    "G3": "G3",
    "J3Q22-1": "G1",
    "J3Q221": "G1",
    "J3Q2-2-1": "G1",
    "J3Q22-2": "G2",
    "J3Q222": "G2",
    "J3Q2-2-2": "G2",
    "J3Q22-3": "G3",
    "J3Q223": "G3",
    "J3Q2-2-3": "G3",
}


def normalize_surface_name(surface) -> str:
    return re.sub(r"\s+", "", str(surface).upper())


def get_layer_key_from_surface(surface, surface_to_layer_map=None):
    mapping = surface_to_layer_map or DEFAULT_SURFACE_TO_LAYER_MAP
    normalized = normalize_surface_name(surface)

    if normalized in mapping:
        return mapping[normalized]

    normalized_mapping = {normalize_surface_name(key): value for key, value in mapping.items()}
    if normalized in normalized_mapping:
        return normalized_mapping[normalized]

    for alias, layer_key in normalized_mapping.items():
        if alias and alias in normalized:
            return layer_key
    return None


def build_layer_parameter_frame(surface_series, surface_to_layer_map=None):
    layer_keys = surface_series.map(lambda value: get_layer_key_from_surface(value, surface_to_layer_map))
    params = pd.DataFrame(index=surface_series.index)
    params["LAYER_KEY"] = layer_keys

    for param_name in ["a", "b", "m", "n", "rw", "rt_cutoff", "por_cutoff", "so_cutoff"]:
        params[param_name] = layer_keys.map(
            lambda layer_key: CHONG32_LAYER_PARAMETERS.get(layer_key, {}).get(param_name, np.nan)
        )
    return params


def calculate_lithology(rt_series, density_series, missing_value=MISSING_VALUE):
    rt = pd.to_numeric(rt_series, errors="coerce").replace(missing_value, np.nan)
    den = pd.to_numeric(density_series, errors="coerce").replace(missing_value, np.nan)

    result = pd.Series(np.full(len(rt), missing_value), index=rt_series.index, dtype=float)
    valid_mask = rt.notna() & den.notna()
    if not valid_mask.any():
        return result

    rt_valid = rt.loc[valid_mask]
    den_valid = den.loc[valid_mask]
    result.loc[valid_mask] = np.where(
        rt_valid <= 9,
        1,
        np.where(rt_valid <= 20, 2, np.where(den_valid <= 2.35, 3, 4)),
    )
    return result


def calculate_porosity_percent_from_density(density_series, lith_series=None, missing_value=MISSING_VALUE):
    density = pd.to_numeric(density_series, errors="coerce").replace(missing_value, np.nan)

    result = pd.Series(np.full(len(density), missing_value), index=density_series.index, dtype=float)
    valid_mask = density.notna()
    if not valid_mask.any():
        return result

    por_percent = -74.877 * density.loc[valid_mask] + 198.86
    por_percent = por_percent.clip(lower=0.01, upper=36.0)
    result.loc[valid_mask] = por_percent

    if lith_series is not None:
        lith = pd.to_numeric(lith_series, errors="coerce").replace(missing_value, np.nan)
        mud_mask = valid_mask & lith.eq(1)
        result.loc[mud_mask] = 5.0
    return result


def calculate_permeability_from_porosity_percent(por_percent_series, missing_value=MISSING_VALUE):
    por_percent = pd.to_numeric(por_percent_series, errors="coerce").replace(missing_value, np.nan)

    result = pd.Series(np.full(len(por_percent), missing_value), index=por_percent_series.index, dtype=float)
    valid_mask = por_percent.notna()
    if not valid_mask.any():
        return result

    perm = 0.00005 * np.exp(0.5653 * por_percent.loc[valid_mask])
    result.loc[valid_mask] = np.minimum(perm, 10000.0)
    return result


def calculate_vertical_permeability_from_perm(perm_series, missing_value=MISSING_VALUE):
    perm = pd.to_numeric(perm_series, errors="coerce").replace(missing_value, np.nan)

    result = pd.Series(np.full(len(perm), missing_value), index=perm_series.index, dtype=float)
    valid_mask = perm.notna()
    if not valid_mask.any():
        return result

    permv = 0.9187 * np.power(perm.loc[valid_mask], 0.9789)
    result.loc[valid_mask] = np.minimum(permv, 100000.0)
    return result


def calculate_saturation_and_oil_flag(
    rt_series,
    por_percent_series,
    surface_series,
    surface_to_layer_map=None,
    missing_value=MISSING_VALUE,
):
    rt = pd.to_numeric(rt_series, errors="coerce")
    rt = rt.replace(missing_value, np.nan)
    por_percent = pd.to_numeric(por_percent_series, errors="coerce").replace(missing_value, np.nan)
    params = build_layer_parameter_frame(surface_series, surface_to_layer_map)

    sw_percent = pd.Series(np.full(len(rt), missing_value), index=rt_series.index, dtype=float)
    so_percent = pd.Series(np.full(len(rt), missing_value), index=rt_series.index, dtype=float)
    oil_flag = pd.Series(np.full(len(rt), missing_value), index=rt_series.index, dtype=float)
    jc_flag = pd.Series(np.full(len(rt), missing_value), index=rt_series.index, dtype=float)

    valid_mask = (
        rt.notna()
        & por_percent.notna()
        & (rt > 0)
        & (por_percent > 0)
        & params["a"].notna()
        & params["b"].notna()
        & params["m"].notna()
        & params["n"].notna()
        & params["rw"].notna()
    )
    if not valid_mask.any():
        return {
            "sw_percent": sw_percent,
            "so_percent": so_percent,
            "oil_flag": oil_flag,
            "jc_flag": jc_flag,
            "layer_key": params["LAYER_KEY"],
        }

    por_fraction = por_percent.loc[valid_mask] / 100.0
    sw_raw = np.power(
        (params.loc[valid_mask, "a"] * params.loc[valid_mask, "b"] * params.loc[valid_mask, "rw"])
        / (rt.loc[valid_mask] * np.power(por_fraction, params.loc[valid_mask, "m"])),
        1.0 / params.loc[valid_mask, "n"],
    ) * 100.0

    so_valid = (100.0 - sw_raw).clip(lower=0.0, upper=80.0)
    sw_valid = 100.0 - so_valid

    sw_percent.loc[valid_mask] = sw_valid
    so_percent.loc[valid_mask] = so_valid

    oil_valid = (
        (rt.loc[valid_mask] >= params.loc[valid_mask, "rt_cutoff"])
        & (por_percent.loc[valid_mask] >= params.loc[valid_mask, "por_cutoff"])
        & (so_valid >= params.loc[valid_mask, "so_cutoff"])
    ).astype(float)

    oil_flag.loc[valid_mask] = oil_valid
    jc_flag.loc[valid_mask] = np.where(oil_valid == 1, 0.0, 1.0)

    return {
        "sw_percent": sw_percent,
        "so_percent": so_percent,
        "oil_flag": oil_flag,
        "jc_flag": jc_flag,
        "layer_key": params["LAYER_KEY"],
    }
