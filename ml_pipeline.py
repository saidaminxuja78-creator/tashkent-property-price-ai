from __future__ import annotations

import io
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import KFold, cross_validate, train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

RANDOM_STATE = 42
TARGET = "price_uzs"

DATA_CANDIDATES = [
    "olx_massive_real_estate.xlsx",
    "olx_massive_real_estate(1).xlsx",
    "rf_model.pkl",  # original upload had an Excel workbook with this misleading name
    "data/olx_massive_real_estate.xlsx",
]

RAW_COLUMNS_EXPECTED = [
    "id", "title", "description", "url", "created_time", "last_refresh_time", "valid_to_time", "pushup_time",
    "category_id", "category_name", "region", "district", "latitude", "longitude", "seller_id", "seller_name",
    "photos_count", "status", "type_of_market", "number_of_rooms", "total_living_area", "total_area", "floor",
    "total_floors", "house_type", "year_of_construction_sale", "wc", "furnished", "more", "repairs", "comission",
    "price", "layout", "near_is", "ceiling_height", "kitchen_area"
]

CATEGORICAL_FEATURES = [
    "region", "district", "type_of_market", "house_type", "wc", "furnished", "repairs", "comission", "layout"
]
NUMERIC_FEATURES = [
    "number_of_rooms", "total_area", "total_living_area", "floor", "total_floors", "building_age",
    "ceiling_height", "kitchen_area", "photos_count", "area_per_room", "floor_ratio", "is_top_floor",
    "is_first_floor", "amenities_count", "nearby_count", "title_length", "description_length"
]
PREFERRED_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def find_dataset_path(base: str | Path = ".") -> Optional[Path]:
    base = Path(base)
    for name in DATA_CANDIDATES:
        p = base / name
        if p.exists():
            return p
    for p in base.glob("*.xlsx"):
        if "olx" in p.name.lower() or "real" in p.name.lower() or "estate" in p.name.lower():
            return p
    return None


def _read_any_tabular(file_or_path: Any) -> pd.DataFrame:
    """Read Excel/CSV even when the extension is wrong. Raise a useful error if it is code/text."""
    if hasattr(file_or_path, "read"):
        data = file_or_path.read()
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        head = data[:200]
        bio = io.BytesIO(data)
        # Excel zip header
        if data[:2] == b"PK":
            return pd.read_excel(bio)
        text_head = head.decode("utf-8", errors="ignore").lower()
        if "import pandas" in text_head or "pd.read_excel" in text_head:
            raise ValueError("The uploaded file looks like Python code, not a dataset. Upload the Excel workbook containing OLX rows.")
        try:
            bio.seek(0)
            return pd.read_excel(bio)
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio)

    path = Path(file_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = path.read_bytes()[:300]
    text_head = raw.decode("utf-8", errors="ignore").lower()
    if "import pandas" in text_head or "pd.read_excel" in text_head:
        raise ValueError(f"{path.name} is not the OLX dataset. It appears to be Python code saved with an Excel-like name.")
    if raw[:2] == b"PK" or path.suffix.lower() in {".xlsx", ".xls", ".pkl"}:
        try:
            return pd.read_excel(path)
        except Exception as exc:
            if path.suffix.lower() == ".pkl":
                return pd.read_pickle(path)
            raise ValueError(f"Could not read {path.name} as an Excel workbook. Upload a valid .xlsx dataset.") from exc
    return pd.read_csv(path)


def load_raw_data(path: str | Path | None = None, uploaded_file: Any | None = None) -> pd.DataFrame:
    if uploaded_file is not None:
        return _read_any_tabular(uploaded_file)
    if path is None:
        path = find_dataset_path()
        if path is None:
            raise FileNotFoundError("Dataset not found. Expected olx_massive_real_estate.xlsx in the project root, or upload a dataset in the app.")
    return _read_any_tabular(path)


def _extract_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).replace("\xa0", " ").replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def parse_price_to_uzs(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    digits = re.sub(r"\D", "", str(value))
    return float(digits) if digits else np.nan


def parse_ceiling_height(value: Any) -> float:
    x = _extract_number(value)
    if pd.isna(x):
        return np.nan
    if x > 20:
        x = x / 100.0
    return x


def parse_year_midpoint(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", str(value))]
    if not years:
        return np.nan
    return float(np.mean(years[:2]))


def count_items(value: Any) -> int:
    if pd.isna(value) or str(value).strip() == "":
        return 0
    return len([x for x in str(value).split(",") if x.strip()])


def clean_real_estate_data(raw: pd.DataFrame, tashkent_only: bool = True, remove_developer_market: bool = True) -> pd.DataFrame:
    df = raw.copy()
    # Guarantee expected columns so the app does not crash when a scraped export changes.
    for col in RAW_COLUMNS_EXPECTED:
        if col not in df.columns:
            df[col] = np.nan

    df[TARGET] = df["price"].apply(parse_price_to_uzs)
    for col in ["total_area", "total_living_area", "kitchen_area"]:
        df[col] = df[col].apply(_extract_number)
    df["ceiling_height"] = df["ceiling_height"].apply(parse_ceiling_height)
    built_year = df["year_of_construction_sale"].apply(parse_year_midpoint)
    df["building_age"] = 2026 - built_year

    for col in ["number_of_rooms", "floor", "total_floors", "photos_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["title_length"] = df["title"].fillna("").astype(str).str.len()
    df["description_length"] = df["description"].fillna("").astype(str).str.len()
    df["amenities_count"] = df["more"].apply(count_items)
    df["nearby_count"] = df["near_is"].apply(count_items)

    if tashkent_only and "region" in df.columns:
        mask = df["region"].fillna("").astype(str).str.contains("Ташкент", case=False, na=False)
        if mask.sum() > 100:
            df = df[mask].copy()
    if remove_developer_market and "type_of_market" in df.columns:
        df = df[df["type_of_market"].fillna("") != "От застройщика"].copy()

    # Target and core sanity filters.
    df = df[df[TARGET].notna() & df["total_area"].notna()].copy()
    df = df[(df[TARGET] >= 100_000_000) & (df[TARGET] <= 10_000_000_000)].copy()
    df = df[(df["total_area"] >= 15) & (df["total_area"] <= 500)].copy()

    df["number_of_rooms"] = df["number_of_rooms"].clip(1, 12)
    df["floor"] = df["floor"].clip(1, 60)
    df["total_floors"] = df["total_floors"].clip(1, 80)
    # Ensure floor <= total_floors where total_floors exists; keep robust for bad scraped rows.
    df.loc[df["floor"] > df["total_floors"], "floor"] = df.loc[df["floor"] > df["total_floors"], "total_floors"]

    df["floor_ratio"] = df["floor"] / df["total_floors"].replace(0, np.nan)
    df["is_top_floor"] = (df["floor"] >= df["total_floors"]).astype(float)
    df["is_first_floor"] = (df["floor"] == 1).astype(float)
    df["area_per_room"] = df["total_area"] / df["number_of_rooms"].replace(0, np.nan)
    df["price_per_sqm"] = df[TARGET] / df["total_area"]

    # Robust price-per-square-metre trim; less destructive than absolute-price IQR filtering.
    low, high = df["price_per_sqm"].quantile([0.01, 0.99])
    df = df[(df["price_per_sqm"] >= low) & (df["price_per_sqm"] <= high)].copy()

    for col in PREFERRED_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[PREFERRED_FEATURES].copy(), df[TARGET].copy()


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), NUMERIC_FEATURES),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25))]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_candidates() -> Dict[str, Any]:
    return {
        "Dummy median": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=70, max_depth=16, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=80, max_depth=18, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=80, learning_rate=0.06, max_depth=3, random_state=RANDOM_STATE),
    }


def make_pipeline(model: Any) -> TransformedTargetRegressor:
    pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", model)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1)
    ape = np.abs(y_true - y_pred) / denom
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MedianAE": float(medae),
        "R2": float(r2),
        "MAPE": float(np.mean(ape)),
        "Within_10pct": float(np.mean(ape <= 0.10)),
        "Within_20pct": float(np.mean(ape <= 0.20)),
    }


def evaluate_models(df: pd.DataFrame, mode: str = "quick", max_rows: Optional[int] = None) -> Dict[str, Any]:
    data = df.copy()
    if max_rows and len(data) > max_rows:
        data = data.sample(max_rows, random_state=RANDOM_STATE)
    X, y = split_xy(data)
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    folds = 3 if mode == "quick" else 5
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    leaderboard = []
    fold_rows = []
    predictions = {}
    fitted_models = {}

    scorers = {
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
        "r2": "r2",
        "neg_mape": "neg_mean_absolute_percentage_error",
    }

    for name, model in model_candidates().items():
        estimator = make_pipeline(model)
        cv_res = cross_validate(estimator, X_train, y_train, scoring=scorers, cv=cv, n_jobs=1, return_train_score=False)
        for i in range(folds):
            fold_rows.append({
                "Model": name,
                "Fold": i + 1,
                "MAE": -float(cv_res["test_neg_mae"][i]),
                "RMSE": -float(cv_res["test_neg_rmse"][i]),
                "R2": float(cv_res["test_r2"][i]),
                "MAPE": -float(cv_res["test_neg_mape"][i]),
            })
        fitted = make_pipeline(model)
        fitted.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        metrics.update({
            "Model": name,
            "CV_MAE_mean": -float(np.mean(cv_res["test_neg_mae"])),
            "CV_MAE_std": float(np.std(-cv_res["test_neg_mae"])),
            "CV_RMSE_mean": -float(np.mean(cv_res["test_neg_rmse"])),
            "CV_R2_mean": float(np.mean(cv_res["test_r2"])),
        })
        leaderboard.append(metrics)
        predictions[name] = pd.DataFrame({"actual": y_test.values, "predicted": y_pred}, index=y_test.index)
        fitted_models[name] = fitted

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(["MAE", "RMSE"], ascending=True).reset_index(drop=True)
    best_model_name = str(leaderboard_df.iloc[0]["Model"])
    best_model = fitted_models[best_model_name]
    best_pred = predictions[best_model_name]

    return {
        "mode": mode,
        "n_rows_used": len(data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "folds": folds,
        "leaderboard": leaderboard_df,
        "fold_metrics": pd.DataFrame(fold_rows),
        "predictions": predictions,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_predictions": best_pred,
        "feature_columns": PREFERRED_FEATURES,
        "trained_at_note": "Generated in current Streamlit session; download the evidence pack to preserve results.",
    }


def market_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(df)),
        "features": int(len(PREFERRED_FEATURES)),
        "median_price": float(df[TARGET].median()),
        "mean_price": float(df[TARGET].mean()),
        "median_price_per_sqm": float(df["price_per_sqm"].median()),
        "districts": int(df["district"].nunique()) if "district" in df.columns else 0,
        "median_area": float(df["total_area"].median()),
        "rooms_median": float(df["number_of_rooms"].median()),
    }


def district_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    if "district" not in df.columns:
        return pd.DataFrame()
    out = df.groupby("district").agg(
        listings=(TARGET, "size"),
        median_price=(TARGET, "median"),
        median_price_sqm=("price_per_sqm", "median"),
        median_area=("total_area", "median"),
    ).reset_index().sort_values("median_price", ascending=False)
    return out


def predict_single(model: Any, values: Dict[str, Any]) -> float:
    row = pd.DataFrame([{col: values.get(col, np.nan) for col in PREFERRED_FEATURES}])
    for col in CATEGORICAL_FEATURES:
        row[col] = row[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        row[col] = pd.to_numeric(row[col], errors="coerce")
    return float(model.predict(row)[0])


def prepare_prediction_features(df: pd.DataFrame, user_values: Dict[str, Any]) -> Dict[str, Any]:
    vals = {}
    for col in PREFERRED_FEATURES:
        if col in user_values:
            vals[col] = user_values[col]
        elif col in CATEGORICAL_FEATURES:
            vals[col] = str(df[col].mode().iloc[0]) if col in df.columns and not df[col].mode().empty else "Unknown"
        else:
            vals[col] = float(df[col].median()) if col in df.columns and not pd.isna(df[col].median()) else np.nan
    # Derived feature consistency.
    total_area = float(vals.get("total_area") or 0)
    rooms = float(vals.get("number_of_rooms") or 1)
    floor = float(vals.get("floor") or 1)
    total_floors = float(vals.get("total_floors") or max(floor, 1))
    vals["area_per_room"] = total_area / max(rooms, 1)
    vals["floor_ratio"] = floor / max(total_floors, 1)
    vals["is_top_floor"] = 1.0 if floor >= total_floors else 0.0
    vals["is_first_floor"] = 1.0 if floor == 1 else 0.0
    return vals


def model_feature_importance(model: Any, X: pd.DataFrame, y: pd.Series, max_rows: int = 1200) -> pd.DataFrame:
    if len(X) > max_rows:
        Xs = X.sample(max_rows, random_state=RANDOM_STATE)
        ys = y.loc[Xs.index]
    else:
        Xs, ys = X, y
    perm = permutation_importance(model, Xs, ys, scoring="neg_mean_absolute_error", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    return pd.DataFrame({
        "feature": Xs.columns,
        "importance_mae_increase": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance_mae_increase", ascending=False)


def bootstrap_metric_ci(y_true: Iterable[float], y_pred: Iterable[float], n_boot: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    rows = []
    n = len(y_true)
    for metric_name, fn in [
        ("MAE", lambda a, b: mean_absolute_error(a, b)),
        ("RMSE", lambda a, b: math.sqrt(mean_squared_error(a, b))),
        ("R2", lambda a, b: r2_score(a, b)),
        ("MAPE", lambda a, b: np.mean(np.abs(a - b) / np.maximum(np.abs(a), 1))),
    ]:
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            vals.append(float(fn(y_true[idx], y_pred[idx])))
        rows.append({
            "metric": metric_name,
            "estimate": float(fn(y_true, y_pred)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        })
    return pd.DataFrame(rows)


def learning_curve_table(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    X, y = split_xy(df)
    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return pd.DataFrame({
        "train_size": sizes,
        "train_mae": -train_scores.mean(axis=1),
        "validation_mae": -val_scores.mean(axis=1),
    })


def dataset_fingerprint(df: pd.DataFrame) -> str:
    sig = f"{len(df)}-{len(df.columns)}-{int(df[TARGET].median())}-{int(df['total_area'].median())}"
    return str(abs(hash(sig)))[:12]


def environment_metadata() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": __import__("sklearn").__version__,
    }


def save_model(model: Any) -> bytes:
    buff = io.BytesIO()
    joblib.dump(model, buff)
    return buff.getvalue()


def evidence_pack(results: Dict[str, Any], df: pd.DataFrame) -> bytes:
    import zipfile
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("leaderboard.csv", results["leaderboard"].to_csv(index=False))
        zf.writestr("fold_metrics.csv", results["fold_metrics"].to_csv(index=False))
        zf.writestr("best_predictions.csv", results["best_predictions"].to_csv(index=True))
        zf.writestr("district_benchmark.csv", district_benchmark(df).to_csv(index=False))
        zf.writestr("metadata.json", json.dumps({
            "best_model": results["best_model_name"],
            "mode": results["mode"],
            "rows_used": results["n_rows_used"],
            "dataset_fingerprint": dataset_fingerprint(df),
            "environment": environment_metadata(),
        }, indent=2))
        zf.writestr("best_model.joblib", save_model(results["best_model"]))
    buff.seek(0)
    return buff.getvalue()
