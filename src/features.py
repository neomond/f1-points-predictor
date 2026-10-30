"""
features.py
-----------
Turns raw per-driver-per-race rows into a leakage-safe feature table
for the "Will this driver finish in the points?" classifier.

GOLDEN RULE: every feature for a given race may only use information
available BEFORE that race starts.

How this is enforced:
  * Driver features   -> grouped by driver, each driver has exactly ONE row
                         per race, so shifting by one row = shifting by one race.
  * Team features     -> a team has TWO rows per race (two drivers). Shifting
                         by one row would let driver #2 see driver #1's result
                         from the SAME race. So we first collapse to one row
                         per (team, race), shift by one RACE, then merge back.
  * tests/test_features.py proves it: changing the results of race r, or
    deleting every race after r, must not change any feature of race r.

Usage:
    from features import build_features
    df, feature_cols = build_features("data/f1_raw.csv")
"""
import pandas as pd

from validate import validate_raw

RACE_KEYS = ["year", "round"]

FEATURE_COLS = [
    "grid", "quali_pos",                      # qualifying (pre-race, safe)
    "driver_avg_finish", "driver_recent_finish", "driver_points_rate",
    "constructor_recent_finish",
    "season_points_so_far",
    "constructorId", "circuitId",             # categoricals
]


def _clean_grid(df):
    """grid == 0 means a PIT-LANE start in the Ergast/Jolpica data.

    Left as 0 the model reads it as *better than pole position*. A pit-lane
    starter actually starts behind everyone, so map it to (field size + 1).
    The original value is kept in `pit_lane_start` for transparency.
    """
    field_size = df.groupby(RACE_KEYS)["driverId"].transform("size")
    df["pit_lane_start"] = (df["grid"] == 0).astype(int)
    df["grid"] = df["grid"].where(df["grid"] > 0, field_size + 1)
    return df


def _driver_features(df, recent_window):
    """Career and recent form from each driver's PAST races only.

    Vectorised: cumsum/cumcount instead of groupby().apply(lambda ...).
    The row's own result is subtracted out, so only earlier races count.
    """
    df = df.sort_values(["driverId", *RACE_KEYS]).reset_index(drop=True)
    g = df.groupby("driverId")

    n_prev = g.cumcount()                                    # races before this one
    prev_pos_sum = g["position"].cumsum() - df["position"]   # excludes current race
    prev_pts_sum = g["points_finish"].cumsum() - df["points_finish"]

    df["driver_avg_finish"] = (prev_pos_sum / n_prev).where(n_prev > 0)
    df["driver_points_rate"] = (prev_pts_sum / n_prev).where(n_prev > 0)

    prev_pos = g["position"].shift(1)
    df["driver_recent_finish"] = (
        prev_pos.groupby(df["driverId"])
                .rolling(recent_window, min_periods=1).mean()
                .reset_index(level=0, drop=True)
    )

    # Championship points before this race (resets each season).
    gs = df.groupby(["driverId", "year"])
    df["season_points_so_far"] = gs["points"].cumsum() - df["points"]
    return df


def _constructor_features(df, recent_races):
    """Team form from the team's PAST races only.

    Step 1: one row per (team, race) = mean finishing position of its cars.
    Step 2: shift by one RACE (not one row) and take a rolling mean.
    Step 3: merge back, so both drivers get the same pre-race value.
    """
    team_race = (df.groupby(["constructorId", *RACE_KEYS], as_index=False)["position"]
                   .mean()
                   .sort_values(["constructorId", *RACE_KEYS]))
    prev = team_race.groupby("constructorId")["position"].shift(1)
    team_race["constructor_recent_finish"] = (
        prev.groupby(team_race["constructorId"])
            .rolling(recent_races, min_periods=1).mean()
            .reset_index(level=0, drop=True)
    )
    return df.merge(
        team_race[["constructorId", *RACE_KEYS, "constructor_recent_finish"]],
        on=["constructorId", *RACE_KEYS], how="left", validate="many_to_one",
    )


def build_features(csv_path, recent_window=3):
    raw = pd.read_csv(csv_path)
    validate_raw(raw)                       # fail fast on bad data

    df = raw.copy()
    df["race_id"] = df["year"] * 100 + df["round"]          # e.g. 202606
    df["points_finish"] = (df["position"] <= 10).astype(int)  # TARGET
    df = _clean_grid(df)
    df = _driver_features(df, recent_window)
    df = _constructor_features(df, recent_races=recent_window)

    # Restore race order. First-ever appearances have no history -> NaN;
    # we keep them and let the pipeline's imputer handle it.
    df = df.sort_values([*RACE_KEYS, "grid"]).reset_index(drop=True)
    return df, list(FEATURE_COLS)


if __name__ == "__main__":
    df, cols = build_features("data/f1_raw.csv")
    print("Feature columns:", cols)
    print("\nShape:", df.shape)
    print("Pit-lane starts remapped:", int(df["pit_lane_start"].sum()))
    print("\nTarget balance (points_finish):")
    print(df["points_finish"].value_counts(normalize=True).round(3))
    print("\nSample (2026, round 6):")
    show = df[(df.year == 2026) & (df["round"] == 6)][
        ["driverId", "constructorId", "grid", "driver_recent_finish",
         "constructor_recent_finish", "season_points_so_far", "points_finish"]
    ]
    print(show.head(10).to_string(index=False))
