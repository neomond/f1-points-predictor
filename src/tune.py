"""
tune.py
-------
Step 5: hyperparameter tuning with GridSearchCV, using TimeSeriesSplit
as the cross-validation strategy (so tuning stays leakage-safe), and
comparing three models:
    - Logistic Regression (linear baseline)
    - Random Forest
    - Gradient Boosting

Key idea: the `step__param` naming reaches into the Pipeline.
    clf__C                          -> the classifier's C
    prep__num__impute__strategy     -> imputer inside the numeric branch

Run:
    python src/tune.py
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from train import make_pipeline, load_train_test


def build_search_space():
    """One entry per model: (estimator, param_grid)."""
    tscv = TimeSeriesSplit(n_splits=5)

    spaces = {
        "logreg": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {
                "clf__C": [0.01, 0.1, 1, 10],
                "prep__num__impute__strategy": ["mean", "median"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [4, 8, None],
                "clf__min_samples_leaf": [1, 5],
            },
        ),
        "grad_boost": (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [150, 300],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
    }
    return spaces, tscv


def tune_all(X_train, y_train):
    spaces, tscv = build_search_space()
    best = {}
    for name, (model, grid) in spaces.items():
        pipe = make_pipeline(model)
        search = GridSearchCV(
            pipe, grid,
            cv=tscv,
            scoring="roc_auc",   # threshold-independent, good for imbalance
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best[name] = search
        print(f"\n=== {name} ===")
        print(f"best CV roc_auc: {search.best_score_:.3f}")
        print(f"best params: {search.best_params_}")
    return best


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _, _ = load_train_test()
    best = tune_all(X_train, y_train)

    print("\n" + "=" * 45)
    print("MODEL COMPARISON (mean CV roc_auc on 2018-2025)")
    print("=" * 45)
    ranking = sorted(best.items(), key=lambda kv: kv[1].best_score_, reverse=True)
    for name, search in ranking:
        print(f"{name:15s} {search.best_score_:.3f}")
    winner = ranking[0][0]
    print(f"\nWinner: {winner}")
