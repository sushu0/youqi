from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.mixture import GaussianMixture


RANDOM_STATE = 42
MISSING_VALUE = -999.25

DEFAULT_CLUSTER_FEATURES = [
    "GR",
    "DEN",
    "AC",
    "RTCAL",
    "VSH_PRE",
    "LITH_PRE",
    "POR_PERCENT_PRE",
    "PERM_PRE",
    "SW_PERCENT_PRE",
    "Sand_RT",
    "FORM_LAYER",
]

DEFAULT_CLASSIFICATION_FEATURES = [
    "GR",
    "DEN",
    "AC",
    "RTCAL",
    "VSH_PRE",
    "LITH_PRE",
    "POR_PERCENT_PRE",
    "PERM_PRE",
    "SW_PERCENT_PRE",
    "Sand_RT",
    "FORM_LAYER",
]

DEFAULT_REGRESSION_FEATURES = [
    "GR",
    "DEN",
    "AC",
    "RTCAL",
    "VSH_PRE",
    "LITH_PRE",
    "POR_PERCENT_PRE",
    "SW_PERCENT_PRE",
    "Sand_RT",
    "FORM_LAYER",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified ML/DL workbench for clustering, recognition, and fitting."
    )
    parser.add_argument(
        "--task",
        choices=["cluster", "classify", "regress", "all"],
        default="all",
        help="Task to run. Default runs all three.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("_local_run") / "07_interpretation",
        help="Preprocessed interpretation CSV directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_local_run") / "09_ml_suite",
        help="Directory used to store ML outputs.",
    )
    parser.add_argument(
        "--classification-target",
        default="OIL_PRE",
        help="Target column for recognition/classification.",
    )
    parser.add_argument(
        "--regression-target",
        default="PERM_PRE",
        help="Target column for fitting/regression.",
    )
    parser.add_argument(
        "--cluster-features",
        default=",".join(DEFAULT_CLUSTER_FEATURES),
        help="Comma-separated features for clustering.",
    )
    parser.add_argument(
        "--classification-features",
        default=",".join(DEFAULT_CLASSIFICATION_FEATURES),
        help="Comma-separated features for classification.",
    )
    parser.add_argument(
        "--regression-features",
        default=",".join(DEFAULT_REGRESSION_FEATURES),
        help="Comma-separated features for regression.",
    )
    parser.add_argument(
        "--cluster-samples",
        type=int,
        default=15000,
        help="Maximum sampled rows for clustering.",
    )
    parser.add_argument(
        "--supervised-samples",
        type=int,
        default=20000,
        help="Maximum sampled rows for classification/regression.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Cluster number used by algorithms that require it.",
    )
    return parser.parse_args()


def parse_feature_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def ensure_interpretation_exists(input_dir: Path) -> None:
    input_dir = Path(input_dir)
    if any(input_dir.glob("*.csv")):
        return

    print(f"[prepare] {input_dir} does not contain interpretation CSV files, running preprocessing first.")
    from run_local_pipeline import main as run_local_pipeline_main

    run_local_pipeline_main()

    if not any(input_dir.glob("*.csv")):
        raise FileNotFoundError(f"Unable to prepare interpretation data in {input_dir}")


def load_interpretation_rows(input_dir: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    data_frames: list[pd.DataFrame] = []
    required_columns = set(required_columns)
    required_columns.update({"DEPT", "FORM_LAYER"})

    for csv_path in sorted(Path(input_dir).glob("*.csv")):
        df = pd.read_csv(
            csv_path,
            encoding="gb2312",
            usecols=lambda column: column in required_columns,
        )
        df["WELL_NAME"] = csv_path.stem
        data_frames.append(df)

    if not data_frames:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    merged = pd.concat(data_frames, ignore_index=True)
    if "DEPT" not in merged.columns:
        merged["DEPT"] = np.nan
    if "FORM_LAYER" not in merged.columns:
        merged["FORM_LAYER"] = ""
    return merged


def clean_numeric_column(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace(MISSING_VALUE, np.nan)


def clean_categorical_column(series: pd.Series) -> pd.Series:
    values = series.astype(str).replace({"nan": "", "None": "", str(MISSING_VALUE): ""})
    values = values.str.strip()
    return values.replace("", np.nan)


def normalize_task_frame(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = raw_df.copy()
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for column in feature_columns:
        if column not in frame.columns:
            continue
        if column == "FORM_LAYER" or frame[column].dtype == object:
            frame[column] = clean_categorical_column(frame[column])
            if frame[column].notna().any():
                categorical_features.append(column)
        else:
            frame[column] = clean_numeric_column(frame[column])
            if frame[column].notna().any():
                numeric_features.append(column)

    if target_column and target_column in frame.columns:
        if frame[target_column].dtype == object and target_column != "FORM_LAYER":
            frame[target_column] = clean_categorical_column(frame[target_column])
        else:
            frame[target_column] = clean_numeric_column(frame[target_column])

    if not numeric_features and not categorical_features:
        raise ValueError("No valid features remain after cleaning.")

    return frame, numeric_features, categorical_features


def sample_rows(
    df: pd.DataFrame,
    max_samples: int | None,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df.copy()

    stratify = None
    if stratify_col and stratify_col in df.columns:
        stratify_values = df[stratify_col]
        if stratify_values.nunique(dropna=True) > 1:
            stratify = stratify_values

    try:
        sampled, _ = train_test_split(
            df,
            train_size=max_samples,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
        return sampled.copy()
    except ValueError:
        return df.sample(n=max_samples, random_state=RANDOM_STATE).copy()


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def profile_dataset(raw_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    profile = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "files": int(raw_df["WELL_NAME"].nunique()) if "WELL_NAME" in raw_df.columns else 0,
        "rows": int(len(raw_df)),
        "columns": list(raw_df.columns),
        "classification_target": args.classification_target,
        "regression_target": args.regression_target,
    }
    save_json(output_dir / "dataset_profile.json", profile)


def run_clustering(
    raw_df: pd.DataFrame,
    output_dir: Path,
    feature_columns: list[str],
    n_clusters: int,
    max_samples: int,
) -> None:
    task_dir = output_dir / "clustering"
    task_dir.mkdir(parents=True, exist_ok=True)

    frame, numeric_features, categorical_features = normalize_task_frame(raw_df, feature_columns)
    model_df = frame[["WELL_NAME", "DEPT"] + numeric_features + categorical_features].copy()
    model_df = model_df.dropna(how="all", subset=numeric_features + categorical_features)
    model_df = sample_rows(model_df, max_samples=max_samples)

    X = model_df[numeric_features + categorical_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_transformed = preprocessor.fit_transform(X)

    models = {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "birch": Birch(n_clusters=n_clusters),
        "gaussian_mixture": GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE),
        "dbscan": DBSCAN(eps=1.2, min_samples=25),
    }

    metrics_rows: list[dict] = []
    label_frame = model_df[["WELL_NAME", "DEPT"]].copy()

    for model_name, model in models.items():
        print(f"[cluster] running {model_name}")
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_transformed)
        else:
            labels = model.fit(X_transformed).predict(X_transformed)

        label_frame[model_name] = labels
        valid_mask = labels != -1
        unique_labels = np.unique(labels[valid_mask]) if valid_mask.any() else np.array([])

        metrics = {
            "model": model_name,
            "samples": int(len(labels)),
            "clusters_found": int(len(np.unique(labels))),
            "core_clusters": int(len(unique_labels)),
            "noise_samples": int(np.sum(labels == -1)),
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

        if len(unique_labels) >= 2:
            X_metric = X_transformed[valid_mask] if np.any(~valid_mask) else X_transformed
            labels_metric = labels[valid_mask] if np.any(~valid_mask) else labels
            metrics["silhouette"] = float(silhouette_score(X_metric, labels_metric))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X_metric, labels_metric))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_metric, labels_metric))

        metrics_rows.append(metrics)

    pd.DataFrame(metrics_rows).sort_values("model").to_csv(
        task_dir / "cluster_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    label_frame.to_csv(task_dir / "cluster_labels_sample.csv", index=False, encoding="utf-8-sig")

    save_json(
        task_dir / "cluster_config.json",
        {
            "features": feature_columns,
            "used_numeric_features": numeric_features,
            "used_categorical_features": categorical_features,
            "n_clusters": n_clusters,
            "samples": int(len(model_df)),
        },
    )


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": np.nan,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = np.nan
    return metrics


def run_classification(
    raw_df: pd.DataFrame,
    output_dir: Path,
    feature_columns: list[str],
    target_column: str,
    max_samples: int,
) -> None:
    task_dir = output_dir / "classification"
    task_dir.mkdir(parents=True, exist_ok=True)

    if target_column not in raw_df.columns:
        raise ValueError(f"Target column {target_column} not found in dataset.")

    frame, numeric_features, categorical_features = normalize_task_frame(raw_df, feature_columns, target_column)
    model_df = frame[["WELL_NAME", "DEPT"] + numeric_features + categorical_features + [target_column]].copy()
    model_df = model_df.dropna(subset=[target_column])
    model_df = model_df[model_df[target_column] != MISSING_VALUE]
    model_df[target_column] = model_df[target_column].astype(int)
    model_df = sample_rows(model_df, max_samples=max_samples, stratify_col=target_column)

    X = model_df[numeric_features + categorical_features]
    y = model_df[target_column]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        model_df[["WELL_NAME", "DEPT"]],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "linear_svm": LinearSVC(random_state=RANDOM_STATE, max_iter=10000),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "dnn_mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            random_state=RANDOM_STATE,
            max_iter=200,
            early_stopping=True,
        ),
    }

    metrics_rows: list[dict] = []
    prediction_frame = meta_test.reset_index(drop=True).copy()
    prediction_frame["actual"] = y_test.reset_index(drop=True)

    for model_name, estimator in models.items():
        print(f"[classification] running {model_name}")
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_score = None
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(X_test)

        metrics = {"model": model_name, "samples": int(len(model_df))}
        metrics.update(classification_metrics(y_test, y_pred, y_score))
        metrics_rows.append(metrics)
        prediction_frame[model_name] = y_pred

    pd.DataFrame(metrics_rows).sort_values("f1", ascending=False).to_csv(
        task_dir / "classification_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_frame.to_csv(task_dir / "classification_predictions.csv", index=False, encoding="utf-8-sig")

    save_json(
        task_dir / "classification_config.json",
        {
            "target": target_column,
            "features": feature_columns,
            "used_numeric_features": numeric_features,
            "used_categorical_features": categorical_features,
            "samples": int(len(model_df)),
            "positive_ratio": float(y.mean()),
        },
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run_regression(
    raw_df: pd.DataFrame,
    output_dir: Path,
    feature_columns: list[str],
    target_column: str,
    max_samples: int,
) -> None:
    task_dir = output_dir / "regression"
    task_dir.mkdir(parents=True, exist_ok=True)

    if target_column not in raw_df.columns:
        raise ValueError(f"Target column {target_column} not found in dataset.")

    frame, numeric_features, categorical_features = normalize_task_frame(raw_df, feature_columns, target_column)
    model_df = frame[["WELL_NAME", "DEPT"] + numeric_features + categorical_features + [target_column]].copy()
    model_df = model_df.dropna(subset=[target_column])
    model_df = model_df[model_df[target_column] != MISSING_VALUE]
    model_df = sample_rows(model_df, max_samples=max_samples)

    X = model_df[numeric_features + categorical_features]
    y_original = model_df[target_column].astype(float)

    use_log_target = target_column in {"PERM_PRE", "PERMV_PRE"}
    y_model = np.log1p(y_original) if use_log_target else y_original.copy()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_original_train,
        y_test_original,
        meta_train,
        meta_test,
    ) = train_test_split(
        X,
        y_model,
        y_original,
        model_df[["WELL_NAME", "DEPT"]],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "linear_svm": LinearSVR(random_state=RANDOM_STATE, max_iter=20000),
        "dnn_mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            random_state=RANDOM_STATE,
            max_iter=200,
            early_stopping=True,
        ),
    }

    metrics_rows: list[dict] = []
    prediction_frame = meta_test.reset_index(drop=True).copy()
    prediction_frame["actual"] = y_test_original.reset_index(drop=True)

    for model_name, estimator in models.items():
        print(f"[regression] running {model_name}")
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred_model = pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_model) if use_log_target else y_pred_model

        metrics = {"model": model_name, "samples": int(len(model_df))}
        metrics.update(regression_metrics(y_test_original.to_numpy(), y_pred))
        metrics_rows.append(metrics)
        prediction_frame[model_name] = y_pred

    pd.DataFrame(metrics_rows).sort_values("r2", ascending=False).to_csv(
        task_dir / "regression_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_frame.to_csv(task_dir / "regression_predictions.csv", index=False, encoding="utf-8-sig")

    save_json(
        task_dir / "regression_config.json",
        {
            "target": target_column,
            "features": feature_columns,
            "used_numeric_features": numeric_features,
            "used_categorical_features": categorical_features,
            "samples": int(len(model_df)),
            "log_target": use_log_target,
        },
    )


def main() -> None:
    args = parse_args()
    ensure_interpretation_exists(args.input_dir)

    cluster_features = parse_feature_list(args.cluster_features)
    classification_features = parse_feature_list(args.classification_features)
    regression_features = parse_feature_list(args.regression_features)

    required_columns = set(cluster_features + classification_features + regression_features)
    required_columns.update({args.classification_target, args.regression_target})

    raw_df = load_interpretation_rows(args.input_dir, required_columns=required_columns)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_dataset(raw_df, args.output_dir, args)

    if args.task in {"cluster", "all"}:
        run_clustering(
            raw_df,
            args.output_dir,
            feature_columns=cluster_features,
            n_clusters=args.n_clusters,
            max_samples=args.cluster_samples,
        )

    if args.task in {"classify", "all"}:
        run_classification(
            raw_df,
            args.output_dir,
            feature_columns=classification_features,
            target_column=args.classification_target,
            max_samples=args.supervised_samples,
        )

    if args.task in {"regress", "all"}:
        run_regression(
            raw_df,
            args.output_dir,
            feature_columns=regression_features,
            target_column=args.regression_target,
            max_samples=args.supervised_samples,
        )

    print(f"[done] results written to {args.output_dir}")


if __name__ == "__main__":
    main()
