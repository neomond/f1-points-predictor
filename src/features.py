"""
features.py
-----------
Turns raw per-driver-per-race rows into a leakage-safe feature table
for the "Will this driver finish in the points?" classifier.

GOLDEN RULE: every feature for a given race may only use information
available BEFORE that race. We enforce this by sorting chronologically
and using .shift(1) before any rolling/expanding aggregation, so a row
never sees its own result or any future result.

Usage:
    from features import build_features
    df = build_features("data/f1_raw.csv")
"""
import pandas as pd


def build_features(csv_path, recent_window=3):
    df = pd.read_csv(csv_path)

    # --- chronological order is essential for all time-aware features ---
    # a global race index so we can sort every row by when it happened
    df = df.sort_values(["year", "round", "driverId"]).reset_index(drop=True)
    df["race_id"] = df["year"] * 100 + df["round"]   # e.g. 202606

    # --- TARGET: did they finish in the points (top 10)? ---
    df["points_finish"] = (df["position"] <= 10).astype(int)

    # ============================================================
    # DRIVER FORM FEATURES (computed from each driver's PAST races)
    # ============================================================
    df = df.sort_values(["driverId", "year", "round"]).reset_index(drop=True)
    g = df.groupby("driverId", group_keys=False)

    # .shift(1) => exclude the current race, so we only see the past.
    # expanding().mean() => career-to-date average finishing position.
    df["driver_avg_finish"] = g["position"].apply(
        lambda s: s.shift(1).expanding().mean())

    # rolling recent form: average finish over last N races (shifted).
    df["driver_recent_finish"] = g["position"].apply(
        lambda s: s.shift(1).rolling(recent_window, min_periods=1).mean())

    # historic points-finish rate (fraction of past races in the points).
    df["driver_points_rate"] = g["points_finish"].apply(
        lambda s: s.shift(1).expanding().mean())

    # ============================================================
    # CONSTRUCTOR FORM (computed from the team's PAST races)
    # ============================================================
    df = df.sort_values(["constructorId", "year", "round"]).reset_index(drop=True)
    gc = df.groupby("constructorId", group_keys=False)
    df["constructor_recent_finish"] = gc["position"].apply(
        lambda s: s.shift(1).rolling(recent_window * 2, min_periods=1).mean())

    # ============================================================
    # SEASON POINTS SO FAR (cumulative within a season, excl. current race)
    # ============================================================
    df = df.sort_values(["driverId", "year", "round"]).reset_index(drop=True)
    df["season_points_so_far"] = (
        df.groupby(["driverId", "year"], group_keys=False)["points"]
          .apply(lambda s: s.shift(1).cumsum().fillna(0))
    )

    # ============================================================
    # Restore race order and finalize
    # ============================================================
    df = df.sort_values(["year", "round", "grid"]).reset_index(drop=True)

    # First-ever appearances have no history -> NaN. We KEEP them and let
    # the pipeline's imputer handle NaNs (more realistic than dropping).
    feature_cols = [
        "grid", "quali_pos",                      # qualifying (pre-race, safe)
        "driver_avg_finish", "driver_recent_finish", "driver_points_rate",
        "constructor_recent_finish",
        "season_points_so_far",
        "constructorId", "circuitId",             # categoricals
    ]
    return df, feature_cols


if __name__ == "__main__":
    df, cols = build_features("data/f1_raw.csv")
    print("Feature columns:", cols)
    print("\nShape:", df.shape)
    print("\nTarget balance (points_finish):")
    print(df["points_finish"].value_counts(normalize=True).round(3))
    print("\nSample (2026, round 6):")
    show = df[(df.year == 2026) & (df["round"] == 6)][
        ["driverId", "constructorId", "grid", "driver_recent_finish",
         "driver_points_rate", "season_points_so_far", "points_finish"]
    ]
    print(show.head(10).to_string(index=False))
