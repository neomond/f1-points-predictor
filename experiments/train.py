"""
train.py
========
Stage 2 + 3 of the F1 Points-Finish NN extension.

What it demonstrates (each maps to a PyTorch concept block):
  * Custom Dataset + DataLoader                       -> batching
  * nn.Module model returning LOGITS                  -> tensors / forward
  * BCEWithLogitsLoss + Adam                          -> loss / optimiser
  * train()/eval() modes, zero_grad/backward/step     -> autograd loop
  * torch.no_grad() validation + checkpointing        -> best-model saving
  * Honest comparison vs a logistic-regression baseline (your existing winner)

Split mirrors your deployed project: train on 2018..2025, hold out 2026.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from build_dataset import get_dataset, EVAL_SEASON

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT = PROJECT_ROOT / "best_model.pt"

FEATURE_COLS = [
    "grid_position",
    "roll_avg_points_5",
    "roll_points_rate_5",
    "prev_finish_pos",
    "prev_grid_to_finish",
    "team_roll_avg_points_10",
    "career_races_so_far",
]
TARGET_COL = "points_finish"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Dataset: turns one row of the feature table into one (features, label) pair.
# ---------------------------------------------------------------------------
class F1Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # float32 because model weights are float32 by default.
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:     # how many samples exist. The DataLoader needs this to know when an epoch ends.
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model: a small MLP. forward() returns RAW LOGITS (no sigmoid) because
# BCEWithLogitsLoss applies the sigmoid internally in a stable way.
# ---------------------------------------------------------------------------
class PointsNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),          # active in train(), frozen in eval()
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)   # shape (batch,) to match labels


# ---------------------------------------------------------------------------
# Training loop with validation + checkpointing + early stopping.
# ---------------------------------------------------------------------------
def train_model(train_loader, val_loader, n_features,
                n_epochs=60, lr=1e-3, patience=8, seed=42, verbose=True,
                ckpt_path=None):
    torch.manual_seed(seed)              # makes each run independently seeded
    if ckpt_path is None:
        ckpt_path = CKPT
    model = PointsNet(n_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()            # 1. clear accumulated grads
            logits = model(xb)               # 2. forward (builds graph)
            loss = criterion(logits, yb)     #    compare to truth
            loss.backward()                  # 3. fill .grad via chain rule
            optimizer.step()                 # 4. nudge weights
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # ---- validate ----
        model.eval()                         # dropout off
        val_loss = 0.0
        with torch.no_grad():                # no graph -> faster, less memory
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)   # keep BEST, not last
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or improved):
            flag = "  <- best" if improved else ""
            print(f"  epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}{flag}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no val improvement in {patience}).")
            break

    model.load_state_dict(torch.load(ckpt_path))   # restore best checkpoint
    return model


@torch.no_grad()
def nn_predict_proba(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    return torch.sigmoid(model(xb)).cpu().numpy()   # logits -> probabilities


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(seeds=(0, 1, 2, 3, 4)):
    df = get_dataset().dropna(subset=FEATURE_COLS + [TARGET_COL])

    train_df = df[df.season < EVAL_SEASON]
    eval_df = df[df.season == EVAL_SEASON]
    print(f"\nTrain rows: {len(train_df):,} (2018..{EVAL_SEASON-1})   "
          f"Eval rows: {len(eval_df):,} ({EVAL_SEASON} held out)")

    X_train_raw = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_eval_raw = eval_df[FEATURE_COLS].values
    y_eval = eval_df[TARGET_COL].values.astype(np.float32)

    # Scale features (fit on TRAIN only — no leakage from eval).
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_eval = scaler.transform(X_eval_raw)

    # Carve a validation slice out of train (last 15% by time order).
    cut = int(len(X_train) * 0.85)
    Xtr, Xval = X_train[:cut], X_train[cut:]
    ytr, yval = y_train[:cut], y_train[cut:]

    val_loader = DataLoader(F1Dataset(Xval, yval), batch_size=256, shuffle=False)

    # ---- Neural net: train once PER SEED to measure run-to-run variance ----
    print(f"\n=== Training PyTorch MLP across {len(seeds)} seeds ===")
    nn_aucs = []
    for seed in seeds:
        # Re-seed the DataLoader shuffle too, so the whole run varies by seed.
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(F1Dataset(Xtr, ytr), batch_size=64,
                                  shuffle=True, generator=g)
        model = train_model(train_loader, val_loader,
                            n_features=len(FEATURE_COLS),
                            seed=seed, verbose=False,
                            ckpt_path=PROJECT_ROOT / f"best_model_seed{seed}.pt")
        auc = roc_auc_score(y_eval, nn_predict_proba(model, X_eval))
        nn_aucs.append(auc)
        print(f"  seed {seed}: ROC-AUC {auc:.3f}")

    nn_aucs = np.array(nn_aucs)

    # ---- Logistic baseline: deterministic, so a single fit is enough ----
    print("\n=== Logistic Regression baseline ===")
    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    lr_auc = roc_auc_score(y_eval, logreg.predict_proba(X_eval)[:, 1])
    print(f"  ROC-AUC {lr_auc:.3f}")

    # ---- Honest, variance-aware comparison ----
    print("\n" + "=" * 52)
    print(f"  {EVAL_SEASON} held-out ROC-AUC")
    print(f"    Logistic Regression : {lr_auc:.3f}  (deterministic)")
    print(f"    PyTorch MLP         : {nn_aucs.mean():.3f} +/- {nn_aucs.std():.3f}"
          f"  (n={len(seeds)} seeds)")
    print("=" * 52)

    # Is the logistic score inside the MLP's spread? If so, they're tied.
    lo, hi = nn_aucs.mean() - nn_aucs.std(), nn_aucs.mean() + nn_aucs.std()
    if lo <= lr_auc <= hi:
        print("  The logistic score sits INSIDE the MLP's +/-1 std band.\n"
              "  => statistically indistinguishable. The interpretable model\n"
              "     carries no accuracy cost, so it is the right production\n"
              "     choice. Concept drift, not model class, is the real lever.")
    elif lr_auc > hi:
        print("  Logistic beats the MLP's whole +/-1 std band — a clear win for\n"
              "  the simpler model on these features.")
    else:
        print("  MLP's band sits above logistic. Worth checking whether the gain\n"
              "  is worth the lost interpretability before shipping it.")


if __name__ == "__main__":
    main()
