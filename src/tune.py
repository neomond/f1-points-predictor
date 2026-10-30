"""
tune.py
-------
Hyperparameter tuning with GridSearchCV, using season walk-forward folds
(so tuning stays leakage-safe), comparing three models:
    - Logistic Regression (linear baseline)
    - Random Forest
    - Gradient Boosting

Key idea: the `step__param` naming reaches into the Pipeline.
    clf__C                          -> the classifier's C
    prep__num__impute__strategy     -> imputer inside the numeric branch

Multiple-testing note: every configuration tried is a "trial". The more
trials, the more likely the best CV score is partly luck. We print the
total so it can be recorded, and we only trust a model that beats the
grid baseline by more than the run-to-run noise (see evaluate.py).

Run:
    python src/tune.py
"""
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from train import make_pipeline, load_train_test, season_splits


def build_search_space():
    """One entry per model: (estimator, param_grid)."""
    return {
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


def tune_all(X_train, y_train, train_df):
    folds = season_splits(train_df["year"])
    spaces = build_search_space()
    n_trials = sum(len(ParameterGrid(grid)) for _, grid in spaces.values())
    print(f"Total configurations tried: {n_trials} "
          f"(x {len(folds)} folds = {n_trials * len(folds)} fits)")

    best = {}
    for name, (model, grid) in spaces.items():
        search = GridSearchCV(make_pipeline(model), grid, cv=folds,
                              scoring="roc_auc", n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best[name] = search
        i = search.best_index_
        sd = search.cv_results_["std_test_score"][i]
        print(f"\n=== {name} ===")
        print(f"best CV roc_auc: {search.best_score_:.3f} (± {sd:.3f} across seasons)")
        print(f"best params: {search.best_params_}")
    return best


if __name__ == "__main__":
    X_train, y_train, _, _, train_df, _ = load_train_test()
    best = tune_all(X_train, y_train, train_df)

    print("\n" + "=" * 45)
    print("MODEL COMPARISON (mean walk-forward roc_auc, 2020-2025)")
    print("=" * 45)
    ranking = sorted(best.items(), key=lambda kv: kv[1].best_score_, reverse=True)
    for name, search in ranking:
        print(f"{name:15s} {search.best_score_:.3f}")
    print("\nDifferences smaller than the ± across seasons are not meaningful;")
    print("prefer the simplest model among those within that band.")
