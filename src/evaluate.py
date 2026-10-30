"""
evaluate.py
-----------
Final evaluation of the chosen model, judged against a simple baseline and
with honest uncertainty.

What this produces:
  1. Walk-forward table, season by season (2020-2026): model vs the grid
     baseline, with 95% race-bootstrap confidence intervals.
  2. Distribution-shift check: does 2026 (regulation reset) look different
     from normal season-to-season variation? 2022 (also a regulation reset)
     serves as a control.
  3. 2026 held-out details: confusion matrix, report, precision@10,
     calibration (Brier), threshold trade-off.
  4. Feature coefficients (what drives a points finish).
  5. Saves the model fitted on 2018-2025 to models/model.joblib.

Run:
    python src/evaluate.py
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss

from train import (make_pipeline, load_train_test, NUM_FEATURES, CAT_FEATURES,
                   DEFAULT_DATA, TEST_SEASON, FIRST_CV_SEASON)
from features import build_features
from metrics import grid_score, precision_at_10, race_bootstrap, auc, auc_lift

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.joblib")
COLS = NUM_FEATURES + CAT_FEATURES
REG_RESETS = {2022: "ground-effect regs", 2026: "new PU + chassis regs"}


def chosen_model():
    """Winner of tune.py: strongly regularised logistic regression.

    (All three model families were within the season-to-season noise of
    each other, so the simplest, most interpretable one is used.)
    """
    pipe = make_pipeline(LogisticRegression(C=0.01, max_iter=1000, class_weight="balanced"))
    return pipe.set_params(prep__num__impute__strategy="mean")


def fit_final_model():
    """Fit the chosen model on every training season (used by the app)."""
    X_train, y_train, *_ = load_train_test()
    return chosen_model().fit(X_train, y_train)


def walk_forward(df, first=FIRST_CV_SEASON, last=TEST_SEASON, n_boot=2000):
    """Train on all seasons < s, predict season s, for every s in [first, last]."""
    rows, preds = [], []
    for s in range(first, last + 1):
        train, test = df[df.year < s], df[df.year == s].copy()
        model = chosen_model().fit(train[COLS], train["points_finish"])
        test["p"] = model.predict_proba(test[COLS])[:, 1]
        test["grid_score"] = grid_score(test)

        m_auc, m_lo, m_hi = race_bootstrap(test, lambda d: auc(d, "p"), n_boot)
        lift, l_lo, l_hi = race_bootstrap(test, auc_lift, n_boot)
        rows.append({
            "season": s, "races": int(test.groupby(["year", "round"]).ngroups),
            "model_auc": m_auc, "auc_lo": m_lo, "auc_hi": m_hi,
            "grid_auc": auc(test, "grid_score"),
            "lift": lift, "lift_lo": l_lo, "lift_hi": l_hi,
            "p10_model": precision_at_10(test, "p"),
            "p10_grid": precision_at_10(test, "grid_score"),
        })
        preds.append(test)
    return pd.DataFrame(rows), pd.concat(preds)


def print_walk_forward(wf):
    print("=" * 92)
    print("WALK-FORWARD BY SEASON  (train on all earlier seasons; 95% CI = race bootstrap)")
    print("=" * 92)
    print(f"{'season':>6} {'races':>5}  {'model AUC [95% CI]':>22}  {'grid AUC':>8}  "
          f"{'lift [95% CI]':>24}  {'P@10 model/grid':>15}")
    for r in wf.itertuples():
        tag = f"  <- {REG_RESETS[r.season]}" if r.season in REG_RESETS else ""
        print(f"{r.season:>6} {r.races:>5}  {r.model_auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]  "
              f"{r.grid_auc:8.3f}  {r.lift:+.3f} [{r.lift_lo:+.3f}, {r.lift_hi:+.3f}]  "
              f"{r.p10_model:.3f} / {r.p10_grid:.3f}{tag}")


def print_shift_check(wf):
    """Is 2026 outside normal season-to-season variation?"""
    hist = wf[wf.season < TEST_SEASON]
    cur = wf[wf.season == TEST_SEASON].iloc[0]
    print("\n" + "=" * 92)
    print("DISTRIBUTION-SHIFT CHECK")
    print("=" * 92)
    print(f"Model AUC, 2020-2025: mean {hist.model_auc.mean():.3f}, "
          f"range {hist.model_auc.min():.3f}-{hist.model_auc.max():.3f}")
    print(f"Model AUC, 2026:      {cur.model_auc:.3f}  95% CI [{cur.auc_lo:.3f}, {cur.auc_hi:.3f}]  "
          f"({int(cur.races)} races)")
    inside = cur.auc_lo <= hist.model_auc.mean() <= cur.auc_hi
    print(f"-> the historical mean {'IS' if inside else 'is NOT'} inside the 2026 interval: "
          f"{'no statistically clear degradation yet' if inside else 'a real drop'}.")
    print(f"Grid baseline AUC moved too: {hist.grid_auc.mean():.3f} -> {cur.grid_auc:.3f}, "
          "so part of any drop is races being harder to predict for everyone.")
    if 2022 in set(wf.season):
        r22 = wf[wf.season == 2022].iloc[0]
        print(f"Control: 2022 (also a regulation reset) model AUC {r22.model_auc:.3f}, "
              f"lift {r22.lift:+.3f} -> a reset does not automatically break the model.")


def final_evaluate():
    X_train, y_train, X_test, y_test, _, test_df = load_train_test()
    df, _ = build_features(DEFAULT_DATA)

    # --- 1 & 2. Walk-forward with baseline and uncertainty ---
    wf, _ = walk_forward(df)
    print_walk_forward(wf)
    print_shift_check(wf)

    # --- Fit on ALL of 2018-2025, predict 2026 ---
    pipe = fit_final_model()
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    # --- 3. 2026 details ---
    print("\n" + "=" * 92)
    print(f"{TEST_SEASON} HELD-OUT TEST")
    print("=" * 92)
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Calibration: mean predicted {proba.mean():.3f} vs actual rate {y_test.mean():.3f}; "
          f"Brier score {brier_score_loss(y_test, proba):.3f} (lower is better)")

    print("\nThreshold trade-off:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_t = proba >= t
        tp = int((pred_t & (y_test == 1)).sum())
        p = tp / pred_t.sum() if pred_t.sum() else 0.0
        r = tp / (y_test == 1).sum()
        print(f"  threshold={t:.1f}  precision={p:.3f}  recall={r:.3f}")

    # --- 4. Coefficients ---
    print("\n" + "=" * 92)
    print("FEATURE INFLUENCE (standardised logistic regression coefficients)")
    print("=" * 92)
    ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    names = NUM_FEATURES + list(ohe.get_feature_names_out(CAT_FEATURES))
    imp = pd.Series(pipe.named_steps["clf"].coef_[0], index=names).sort_values()
    print("Most NEGATIVE (push toward NO points):")
    print(imp.head(5).round(3).to_string())
    print("\nMost POSITIVE (push toward points):")
    print(imp.tail(5).round(3).to_string())

    # --- 5. Save ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved fitted pipeline -> {MODEL_PATH}")
    return pipe


if __name__ == "__main__":
    final_evaluate()
