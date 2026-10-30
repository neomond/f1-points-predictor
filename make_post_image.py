"""
make_post_image.py
------------------
Generates polished result visuals for the LinkedIn post / README,
in a McLaren-inspired dark + papaya-orange theme.

Outputs:
  - linkedin_results.png  (1200x627 landscape, two-panel)
  - feature_importance.png (1080x1080 square)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, "src")
from features import build_features
from train import NUM_FEATURES, CAT_FEATURES, DEFAULT_DATA
from evaluate import walk_forward, fit_final_model

# ---- Theme ----
BG = "#15151e"
ORANGE = "#ff8000"
LIGHT = "#ffffff"
MUTED = "#9a9aa5"
BLUE = "#3b82f6"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "text.color": LIGHT,
    "axes.labelcolor": LIGHT, "xtick.color": LIGHT,
    "ytick.color": LIGHT, "axes.edgecolor": MUTED,
    "font.size": 12,
})

# ---- Recompute the numbers fresh ----
df, _ = build_features(DEFAULT_DATA)
wf, _ = walk_forward(df)
pipe = fit_final_model()

# coefficients
ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
feat_names = NUM_FEATURES + cat_names
coefs = pipe.named_steps["clf"].coef_[0]
imp = pd.Series(coefs, index=feat_names).sort_values()

# ============================================================
# FIGURE 1: two-panel landscape (1200x627)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.27), dpi=100)

# Panel 1: walk-forward, model vs grid baseline, with 95% race-bootstrap CI
x = wf["season"].to_numpy()
yerr = [wf["model_auc"] - wf["auc_lo"], wf["auc_hi"] - wf["model_auc"]]
ax1.errorbar(x, wf["model_auc"], yerr=yerr, color=ORANGE, marker="o", lw=2.5,
             capsize=4, label="model (95% CI)")
ax1.plot(x, wf["grid_auc"], color=BLUE, marker="s", lw=2, ls="--",
         label="baseline: grid position only")
for s, name in [(2022, "2022 regs"), (2026, "2026 regs")]:
    ax1.axvline(s, color=MUTED, lw=0.8, ls=":")
    ax1.text(s, 0.955, name, color=MUTED, fontsize=10, ha="center")
ax1.set_ylim(0.65, 0.97)
ax1.set_xticks(x)
ax1.set_ylabel("ROC-AUC (season predicted from all earlier seasons)",
               fontsize=11, fontweight="bold")
ax1.set_title("Walk-forward by season:\nmodel vs a one-feature baseline",
              fontsize=15, fontweight="bold", color=ORANGE, pad=15)
ax1.legend(loc="lower left", frameon=False, fontsize=10)
ax1.spines[["top", "right"]].set_visible(False)

# Panel 2: top feature influences
top = pd.concat([imp.head(4), imp.tail(4)])
colors = [BLUE if v < 0 else ORANGE for v in top.values]
labels = [n.replace("constructorId_", "team: ").replace("_", " ")
          for n in top.index]
ax2.barh(range(len(top)), top.values, color=colors, edgecolor="none")
ax2.set_yticks(range(len(top)))
ax2.set_yticklabels(labels, fontsize=11)
ax2.axvline(0, color=MUTED, lw=0.8)
ax2.set_xlabel("← less likely to score    |    more likely to score →",
               fontsize=11)
ax2.set_title("What drives a points finish?\n(model coefficients)",
              fontsize=15, fontweight="bold", color=ORANGE, pad=15)
ax2.spines[["top", "right", "left"]].set_visible(False)

fig.suptitle("F1 Points-Finish Predictor  ·  walk-forward validation + race bootstrap",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("linkedin_results.png", bbox_inches="tight", dpi=100)
print("Saved linkedin_results.png")

# ============================================================
# FIGURE 2: square feature-importance (1080x1080)
# ============================================================
fig2, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
allimp = imp.copy()
labels2 = [n.replace("constructorId_", "team: ")
            .replace("circuitId_", "circuit: ").replace("_", " ")
           for n in allimp.index]
colors2 = [BLUE if v < 0 else ORANGE for v in allimp.values]
ax.barh(range(len(allimp)), allimp.values, color=colors2, edgecolor="none")
ax.set_yticks(range(len(allimp)))
ax.set_yticklabels(labels2, fontsize=11)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("← less likely to score        more likely to score →",
              fontsize=13)
ax.set_title("Will an F1 driver finish in the points?\n"
             "What the model learned",
             fontsize=18, fontweight="bold", color=ORANGE, pad=20)
ax.spines[["top", "right", "left"]].set_visible(False)
fig2.text(0.5, 0.02, "Logistic regression  ·  trained 2018-2025  ·  "
          "leakage-safe features", ha="center", color=MUTED, fontsize=11)
fig2.tight_layout(rect=[0, 0.04, 1, 1])
fig2.savefig("feature_importance.png", bbox_inches="tight", dpi=100)
print("Saved feature_importance.png")
