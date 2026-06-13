from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
PASS_THRESHOLD = 10

DATA_CANDIDATES = (
    "student-mat.csv",
    "student.csv",
    "student-performance.csv",
    "student_performance.csv",
    "data/student-mat.csv",
    "data/student.csv",
    "data/student-performance.csv",
    "data/student_performance.csv",
)


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-separated CSV without silently mangling columns."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            df = pd.read_csv(path)

    df.columns = [str(col).strip() for col in df.columns]
    unnamed = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _find_dataset(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Dataset topilmadi: {path}")

    root = Path(__file__).resolve().parent
    for candidate in DATA_CANDIDATES:
        path = root / candidate
        if path.exists():
            return path

    student_csvs = sorted(
        p for p in root.rglob("*.csv") if "student" in p.name.lower()
    )
    if student_csvs:
        return student_csvs[0]

    raise FileNotFoundError(
        "Dataset topilmadi. Repository ichiga student-mat.csv faylini qo'ying "
        "yoki data/student-mat.csv yo'lidan foydalaning."
    )


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the student dataset and create binary target when needed."""
    dataset_path = _find_dataset(path)
    df = _read_csv_robust(dataset_path)

    if "target" not in df.columns:
        if "G3" not in df.columns:
            raise ValueError(
                "Datasetda target ham, G3 ham yo'q. target ustunini qo'shing "
                "yoki UCI student-mat.csv faylidan foydalaning."
            )
        df["target"] = (pd.to_numeric(df["G3"], errors="raise") >= PASS_THRESHOLD).astype(int)
    else:
        target = df["target"]
        if target.dtype == "object":
            normalized = target.astype(str).str.strip().str.lower()
            mapping = {
                "pass": 1,
                "passed": 1,
                "yes": 1,
                "true": 1,
                "1": 1,
                "fail": 0,
                "failed": 0,
                "no": 0,
                "false": 0,
                "0": 0,
            }
            mapped = normalized.map(mapping)
            if mapped.isna().any():
                raise ValueError("target ustunida tanilmagan qiymatlar bor.")
            df["target"] = mapped.astype(int)
        else:
            df["target"] = pd.to_numeric(target, errors="raise").astype(int)

    classes = set(df["target"].dropna().unique().tolist())
    if not classes.issubset({0, 1}) or len(classes) != 2:
        raise ValueError("target ikkilik klass bo'lishi kerak: 0 va 1.")

    return df.reset_index(drop=True)


def prepare_features(
    df: pd.DataFrame,
    include_prior_grades: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build X/y. G3 is always excluded because it defines the target.
    By default G1 and G2 are excluded for an early-warning model.
    """
    drop_cols = ["target", "G3"]
    if not include_prior_grades:
        drop_cols.extend(["G1", "G2"])

    existing = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=existing).copy()
    y = df["target"].to_numpy(dtype=int)

    if X.empty:
        raise ValueError("Feature ustunlari qolmadi.")
    return X, y


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [col for col in X.columns if col not in cat_cols]
    return cat_cols, num_cols


def get_preprocessor(
    cat_cols: list[str],
    num_cols: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, cat_cols),
            ("num", numeric_pipe, num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_models() -> OrderedDict[str, tuple[Any, dict[str, list[Any]]]]:
    """Six classifiers with deliberately compact grids for Streamlit Cloud."""
    return OrderedDict(
        {
            "Logistic Regression": (
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
                {
                    "clf__C": [0.01, 0.1, 1.0, 10.0],
                    "clf__l1_ratio": [0.0, 1.0],
                },
            ),
            "Decision Tree": (
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                {
                    "clf__max_depth": [3, 5, None],
                    "clf__min_samples_leaf": [1, 3],
                },
            ),
            "Random Forest": (
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                {
                    "clf__n_estimators": [100, 200],
                    "clf__max_depth": [5, None],
                    "clf__min_samples_leaf": [1, 3],
                },
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                {
                    "clf__n_estimators": [100, 150],
                    "clf__learning_rate": [0.05, 0.1],
                    "clf__max_depth": [2, 3],
                },
            ),
            "SVM RBF": (
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                {
                    "clf__C": [0.1, 1.0, 10.0],
                    "clf__gamma": ["scale", "auto"],
                },
            ),
            "KNN": (
                KNeighborsClassifier(),
                {
                    "clf__n_neighbors": [5, 9, 15],
                    "clf__weights": ["uniform", "distance"],
                },
            ),
        }
    )


def _positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    scores = estimator.decision_function(X)
    return 1.0 / (1.0 + np.exp(-np.asarray(scores)))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("clf__"): value for key, value in params.items()}


def train_models(
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    outer_splits: int = 5,
    inner_splits: int = 5,
    progress_callback: Callable[[str, int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run true nested CV and fit one final tuned pipeline per model."""
    models = get_models()
    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results: dict[str, Any] = {}
    f1_arrays: dict[str, np.ndarray] = {}
    fold_metrics: list[dict[str, Any]] = []
    best_pipes: dict[str, Pipeline] = {}
    oof: dict[str, dict[str, np.ndarray]] = {}

    for model_index, (name, (estimator, param_grid)) in enumerate(models.items(), start=1):
        metric_values = {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1": [],
            "AUC-ROC": [],
        }
        oof_pred = np.zeros(len(y), dtype=int)
        oof_prob = np.zeros(len(y), dtype=float)

        for fold_index, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            if progress_callback is not None:
                progress_callback(
                    name,
                    model_index,
                    len(models),
                    fold_index,
                    outer_splits,
                )

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            inner_cv = StratifiedKFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=RANDOM_STATE + fold_index,
            )
            pipe = Pipeline(
                steps=[
                    ("pre", clone(preprocessor)),
                    ("clf", clone(estimator)),
                ]
            )
            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="f1",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)

            y_pred = search.best_estimator_.predict(X_test)
            y_prob = _positive_probability(search.best_estimator_, X_test)
            oof_pred[test_idx] = y_pred
            oof_prob[test_idx] = y_prob

            fold_result = {
                "Model": name,
                "Fold": fold_index,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "AUC-ROC": roc_auc_score(y_test, y_prob),
            }
            fold_metrics.append(fold_result)
            for metric in metric_values:
                metric_values[metric].append(float(fold_result[metric]))

        final_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        final_pipe = Pipeline(
            steps=[
                ("pre", clone(preprocessor)),
                ("clf", clone(estimator)),
            ]
        )
        final_search = GridSearchCV(
            estimator=final_pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=final_cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        final_search.fit(X, y)

        results[name] = {
            metric: _summary(values)
            for metric, values in metric_values.items()
        }
        results[name]["best_params"] = _clean_params(final_search.best_params_)
        f1_arrays[name] = np.asarray(metric_values["F1"], dtype=float)
        best_pipes[name] = final_search.best_estimator_
        oof[name] = {
            "y_true": y.copy(),
            "y_pred": oof_pred,
            "y_prob": oof_prob,
        }

    best_model_name = max(results, key=lambda model: results[model]["F1"]["mean"])

    return {
        "results": results,
        "f1_arrays": f1_arrays,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "best_pipes": best_pipes,
        "oof": oof,
        "best_model_name": best_model_name,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
    }


def fit_default_model(
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Fast fallback model used before the full nested-CV training is run."""
    model = LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline(
        steps=[
            ("pre", clone(preprocessor)),
            ("clf", model),
        ]
    )
    pipe.fit(X, y)
    return pipe


def build_optuna_estimator(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            solver="liblinear",
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Noma'lum model: {model_name}")
