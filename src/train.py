"""
train.py
--------
Build a leakage-safe Pipeline (ColumnTransformer for mixed numeric /
categorical features) and evaluate it with SEASON-BY-SEASON walk-forward
cross-validation on the 2018-2025 data.

Why walk-forward by season (not KFold, and not plain TimeSeriesSplit)?
  * F1 is temporal: we must always train on PAST races and validate on
    FUTURE ones. Random KFold lets the model "see the future".
  * sklearn's TimeSeriesSplit cuts by ROW count. On this data all 5 of its
    fold boundaries fell in the middle of a race, so part of a race was in
    training and the rest in validation.
  * Splitting on whole seasons keeps every race intact and matches how the
    model would really be used: train on everything so far, predict the
    next season.

Run:
    python src/train.py
"""
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from features import build_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "f1_raw.csv")

NUM_FEATURES = [
    "grid", "quali_pos",
    "driver_avg_finish", "driver_recent_finish", "driver_points_rate",
    "constructor_recent_finish", "season_points_so_far",
]
CAT_FEATURES = ["constructorId", "circuitId"]

TEST_SEASON = 2026
FIRST_CV_SEASON = 2020   # need >= 2 seasons of history before the first fold


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


def season_splits(years, first_test=FIRST_CV_SEASON):
    """Walk-forward CV folds: train on all seasons < s, validate on season s.

    `years` is the per-row season array. Returns a list of (train_idx,
    val_idx) that sklearn's cross_validate / GridSearchCV accept as `cv=`.
    """
    years = np.asarray(years)
    return [
        (np.where(years < s)[0], np.where(years == s)[0])
        for s in sorted(set(years))
        if s >= first_test
    ]


def load_train_test(csv_path=DEFAULT_DATA):
    """Build features, then split: 2018-2025 = train, 2026 = held-out test."""
    df, _ = build_features(csv_path)
    df = df.sort_values(["year", "round", "grid"]).reset_index(drop=True)

    train = df[df.year < TEST_SEASON].reset_index(drop=True)
    test = df[df.year == TEST_SEASON].reset_index(drop=True)

    cols = NUM_FEATURES + CAT_FEATURES
    X_train, y_train = train[cols], train["points_finish"]
    X_test, y_test = test[cols], test["points_finish"]
    return X_train, y_train, X_test, y_test, train, test


def run_cv(X_train, y_train, train_df):
    """Season walk-forward cross-validation with multiple metrics."""
    pipe = make_pipeline()
    folds = season_splits(train_df["year"])
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = cross_validate(pipe, X_train, y_train, cv=folds,
                             scoring=metrics, return_train_score=True)
    seasons = [int(train_df["year"].iloc[v[0]]) for _, v in folds]
    print(f"=== Walk-forward CV by season ({len(folds)} folds: {seasons}) ===\n")
    for m in metrics:
        tr = results[f"train_{m}"].mean()
        te = results[f"test_{m}"].mean()
        print(f"{m:10s} train={tr:.3f}  val={te:.3f}  (gap {tr - te:+.3f})")
    print("\nPer-season validation roc_auc:",
          {s: round(float(v), 3) for s, v in zip(seasons, results["test_roc_auc"])})
    return results


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, train_df, _ = load_train_test()
    print(f"Train rows (2018-2025): {len(X_train)}")
    print(f"Test rows  (2026):      {len(X_test)}\n")
    run_cv(X_train, y_train, train_df)
