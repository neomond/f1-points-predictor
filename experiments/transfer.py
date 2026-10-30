"""
transfer.py
===========
Transfer-learning experiment for the F1 points-finish predictor.

Tests the project's headline claim — "concept drift around regulation resets
is the real lever" — by measuring whether ADAPTING a model to a new
regulation era beats training on it cold.

Eras
----
  OLD era : 2018..2024  (large, stable-ish regulation history)
  NEW era : 2025         (start of the modern reset; the adaptation target)
  TEST    : 2026         (held out — true forward test)

Four models, all evaluated on 2026:
  A. OLD-only        : train on OLD, never see NEW            (no adaptation)
  B. NEW-cold        : train on NEW only, from scratch        (small data)
  C. Frozen-extractor: pretrain OLD, FREEZE body, train new head on NEW
  D. Fine-tuned      : pretrain OLD, UNFREEZE, low-LR train on NEW

The freeze / low-LR mechanics in C and D are exactly the transfer-learning
levers asked about in ML interviews.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from build_dataset import get_dataset
from train import F1Dataset, FEATURE_COLS, TARGET_COL, device

OLD_MAX = 2024      # OLD era = seasons <= 2024
NEW_SEASON = 2025   # adaptation target
TEST_SEASON = 2026  # held-out forward test


# ---------------------------------------------------------------------------
# Model split into BODY (reusable feature extractor) + HEAD (task-specific).
# Splitting this way is what makes freezing/replacing the head clean.
# ---------------------------------------------------------------------------
class TransferNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # BODY: learns a general representation of a driver-race.
        self.body = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        # HEAD: maps representation -> single logit (points finish or not).
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(1)

    def freeze_body(self):
        for p in self.body.parameters():
            p.requires_grad = False     # optimiser will skip these

    def reset_head(self):
        # Fresh, randomly-initialised head for the new task/era.
        self.head = nn.Linear(16, 1).to(next(self.parameters()).device)


# ---------------------------------------------------------------------------
# A single, reusable training routine. `params` lets the caller pass only the
# trainable parameters (e.g. just the head when the body is frozen).
# ---------------------------------------------------------------------------
def fit(model, Xtr, ytr, *, lr, epochs, params=None, seed=0, batch=64):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(F1Dataset(Xtr, ytr), batch_size=batch,
                        shuffle=True, generator=g)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params or model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def auc(model, X, y):
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    p = torch.sigmoid(model(xb)).cpu().numpy()
    return roc_auc_score(y, p)


def run_once(seed, Xold, yold, Xnew, ynew, Xtest, ytest, n_feat):
    results = {}

    # --- A. OLD-only: train on old era, no adaptation to new ---
    m = TransferNet(n_feat).to(device)
    fit(m, Xold, yold, lr=1e-3, epochs=40, seed=seed)
    results["A_old_only"] = auc(m, Xtest, ytest)

    # --- B. NEW-cold: train on new era only, from scratch (small data) ---
    m = TransferNet(n_feat).to(device)
    fit(m, Xnew, ynew, lr=1e-3, epochs=60, seed=seed)
    results["B_new_cold"] = auc(m, Xtest, ytest)

    # --- C. Frozen extractor: pretrain old, freeze body, train fresh head on new ---
    m = TransferNet(n_feat).to(device)
    fit(m, Xold, yold, lr=1e-3, epochs=40, seed=seed)   # pretrain
    m.reset_head()                                       # fresh head for new era
    m.freeze_body()                                      # body becomes fixed extractor
    head_params = [p for p in m.parameters() if p.requires_grad]
    fit(m, Xnew, ynew, lr=1e-3, epochs=60, params=head_params, seed=seed)
    results["C_frozen"] = auc(m, Xtest, ytest)

    # --- D. Fine-tuned: pretrain old, then low-LR train ALL weights on new ---
    m = TransferNet(n_feat).to(device)
    fit(m, Xold, yold, lr=1e-3, epochs=40, seed=seed)   # pretrain
    fit(m, Xnew, ynew, lr=1e-4, epochs=60, seed=seed)   # low LR => nudge, don't destroy
    results["D_finetuned"] = auc(m, Xtest, ytest)

    return results


def main(seeds=(0, 1, 2, 3, 4)):
    df = get_dataset().dropna(subset=FEATURE_COLS + [TARGET_COL])

    old = df[df.season <= OLD_MAX]
    new = df[df.season == NEW_SEASON]
    test = df[df.season == TEST_SEASON]
    print(f"OLD (<= {OLD_MAX}): {len(old):,} rows | "
          f"NEW ({NEW_SEASON}): {len(new):,} rows | "
          f"TEST ({TEST_SEASON}): {len(test):,} rows")

    # Scale on OLD only — the pretraining distribution defines the scaler,
    # and new/test eras are transformed with it (no leakage from the future).
    scaler = StandardScaler().fit(old[FEATURE_COLS].values)
    Xold = scaler.transform(old[FEATURE_COLS].values)
    Xnew = scaler.transform(new[FEATURE_COLS].values)
    Xtest = scaler.transform(test[FEATURE_COLS].values)
    yold = old[TARGET_COL].values.astype(np.float32)
    ynew = new[TARGET_COL].values.astype(np.float32)
    ytest = test[TARGET_COL].values.astype(np.float32)

    rows = [run_once(s, Xold, yold, Xnew, ynew, Xtest, ytest, len(FEATURE_COLS))
            for s in seeds]

    labels = {
        "A_old_only":  "A. OLD-only (no adaptation)   ",
        "B_new_cold":  "B. NEW-cold (from scratch)    ",
        "C_frozen":    "C. Frozen extractor + new head",
        "D_finetuned": "D. Fine-tuned (low-LR on NEW) ",
    }
    print(f"\n{'='*56}\n  {TEST_SEASON} held-out ROC-AUC  (mean +/- std, n={len(seeds)})\n{'='*56}")
    means = {}
    for key, lab in labels.items():
        vals = np.array([r[key] for r in rows])
        means[key] = vals.mean()
        print(f"  {lab} : {vals.mean():.3f} +/- {vals.std():.3f}")
    print("=" * 56)

    best = max(means, key=means.get)
    adapt_gain = max(means["C_frozen"], means["D_finetuned"]) - means["A_old_only"]
    print(f"\n  Best: {labels[best].strip()}")
    if adapt_gain > 0.01:
        print(f"  Adapting to the new era beats OLD-only by {adapt_gain:+.3f} AUC.\n"
              "  => Concept drift is real and measurable: a model that updates\n"
              "     for the regulation reset outperforms one frozen in the past.")
    else:
        print(f"  Adaptation gain is small ({adapt_gain:+.3f}).\n"
              "  => On these features the representation already transfers across\n"
              "     eras; the engineered features are era-robust. Also a finding.")


if __name__ == "__main__":
    main()
