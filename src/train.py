"""
train.py
--------
Step 3 + 4: build a leakage-safe Pipeline (ColumnTransformer for mixed
numeric/categorical features) and evaluate it with TimeSeriesSplit
cross-validation on the 2018-2025 data.

Why TimeSeriesSplit (not KFold)?
  F1 is temporal. We must always train on PAST races and validate on
  FUTURE ones. Random KFold would let the model "see the future",
  inflating scores and lying about real-world performance.

Run:
    python src/train.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from features import build_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "f1_raw.csv")

NUM_FEATURES = [
    "grid", "quali_pos",
    "driver_avg_finish", "driver_recent_finish", "driver_points_rate",
    "constructor_recent_finish", "season_points_so_far",
]
CAT_FEATURES = ["constructorId", "circuitId"]


def make_pipeline(model=None):
    """Build the preprocessing + model pipeline."""
    if model is None:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])
    return Pipeline([("prep", prep), ("clf", model)])


def load_train_test(csv_path=DEFAULT_DATA):
    """Build features, then split: 2018-2025 = train, 2026 = held-out test."""
    df, _ = build_features(csv_path)
    # CRITICAL: sort by time so TimeSeriesSplit sees races in order
    df = df.sort_values(["year", "round", "grid"]).reset_index(drop=True)

    train = df[df.year <= 2025].reset_index(drop=True)
    test = df[df.year == 2026].reset_index(drop=True)

    cols = NUM_FEATURES + CAT_FEATURES
    X_train, y_train = train[cols], train["points_finish"]
    X_test, y_test = test[cols], test["points_finish"]
    return X_train, y_train, X_test, y_test, train, test


def run_cv(X_train, y_train, n_splits=5):
    """TimeSeriesSplit cross-validation with multiple metrics."""
    pipe = make_pipeline()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = cross_validate(
        pipe, X_train, y_train,
        cv=tscv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=True,
    )
    print(f"=== TimeSeriesSplit CV ({n_splits} folds) on 2018-2025 ===\n")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        tr = results[f"train_{m}"].mean()
        te = results[f"test_{m}"].mean()
        print(f"{m:10s} train={tr:.3f}  val={te:.3f}  (gap {tr - te:+.3f})")

    # show per-fold roc_auc to see learning as train window grows
    print("\nPer-fold validation roc_auc:",
          np.round(results["test_roc_auc"], 3))
    return results


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _, _ = load_train_test()
    print(f"Train rows (2018-2025): {len(X_train)}")
    print(f"Test rows  (2026):      {len(X_test)}\n")
    run_cv(X_train, y_train)
