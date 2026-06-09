"""
evaluate.py
-----------
Step 6: final evaluation of the chosen model on the 2026 held-out test
set (data the model has NEVER seen), plus the distribution-shift story.

What this produces:
  1. CV score (2018-2025) vs TEST score (2026) -> the shift gap
  2. Confusion matrix + classification report on 2026
  3. ROC-AUC on 2026
  4. Feature coefficients (what drives a points finish)
  5. Threshold analysis (precision/recall trade-off)
  6. Saves the fitted model to models/model.joblib

Run:
    python src/evaluate.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve,
)

from train import make_pipeline, load_train_test, NUM_FEATURES, CAT_FEATURES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.joblib")


def final_evaluate():
    X_train, y_train, X_test, y_test, _, test_df = load_train_test()

    # Winning model from Step 5: regularized logistic regression.
    model = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    pipe = make_pipeline(model)

    # --- 1. CV score on the training era (for the shift comparison) ---
    tscv = TimeSeriesSplit(n_splits=5)
    cv_auc = cross_val_score(pipe, X_train, y_train,
                             cv=tscv, scoring="roc_auc").mean()

    # --- Fit on ALL of 2018-2025, predict 2026 ---
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)

    print("=" * 55)
    print("DISTRIBUTION SHIFT: training era vs 2026 (post-reg-reset)")
    print("=" * 55)
    print(f"CV roc_auc  (2018-2025): {cv_auc:.3f}")
    print(f"TEST roc_auc (2026):     {test_auc:.3f}")
    print(f"Shift gap:               {cv_auc - test_auc:+.3f}")
    print("\nA drop here = the 2026 regulation reset changed the")
    print("feature->outcome relationship the model had learned.\n")

    # --- 2 & 3. Confusion matrix + report on 2026 ---
    print("=" * 55)
    print("2026 HELD-OUT TEST PERFORMANCE")
    print("=" * 55)
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, digits=3))

    # --- 4. What drives a points finish? (coefficients) ---
    print("=" * 55)
    print("FEATURE INFLUENCE (logistic regression coefficients)")
    print("=" * 55)
    ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
    feat_names = NUM_FEATURES + cat_names
    coefs = pipe.named_steps["clf"].coef_[0]
    imp = pd.Series(coefs, index=feat_names).sort_values()
    print("\nMost NEGATIVE (push toward NO points):")
    print(imp.head(5).round(3).to_string())
    print("\nMost POSITIVE (push toward points):")
    print(imp.tail(5).round(3).to_string())

    # --- 5. Threshold analysis ---
    print("\n" + "=" * 55)
    print("THRESHOLD ANALYSIS (precision/recall trade-off)")
    print("=" * 55)
    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_t = (y_proba >= t).astype(int)
        tp = ((pred_t == 1) & (y_test == 1)).sum()
        fp = ((pred_t == 1) & (y_test == 0)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (y_test == 1).sum()
        print(f"threshold={t:.1f}  precision={p:.3f}  recall={r:.3f}")

    # --- 6. Save fitted model ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved fitted pipeline -> {MODEL_PATH}")
    return pipe


if __name__ == "__main__":
    final_evaluate()
