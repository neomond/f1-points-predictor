"""
Tests that PROVE the features are leakage-safe instead of just claiming it.

Two independent checks, both based on one rule:
    a feature for race r may depend only on races BEFORE r.

1. Perturbation test: scramble the results of race r. If any feature of
   race r (or earlier) changes, that feature was reading race r's outcome.
   This is the test that caught the original teammate leak.
2. Truncation test: delete every race after r. If any feature of race r
   changes, it was reading the future.

Run:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from conftest import DATA
from features import build_features, FEATURE_COLS
from validate import validate_raw, DataValidationError
from train import season_splits

KEY = ["year", "round", "driverId"]
NUMERIC = [c for c in FEATURE_COLS if c not in ("constructorId", "circuitId")]
# a spread of races: early, mid-season, a reg-reset opener, and recent
RACES = [(2018, 5), (2021, 12), (2022, 1), (2024, 20), (2026, 3)]


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def baseline(raw):
    df, _ = build_features(DATA)
    return df.set_index(KEY).sort_index()


def _features(raw_df, tmp_path, name):
    path = tmp_path / name
    raw_df.to_csv(path, index=False)
    df, _ = build_features(path)
    return df.set_index(KEY).sort_index()


def _upto(df, year, rnd):
    y = df.index.get_level_values("year")
    r = df.index.get_level_values("round")
    return df[(y < year) | ((y == year) & (r <= rnd))]


@pytest.mark.parametrize("year,rnd", RACES)
def test_scrambling_a_race_does_not_change_its_own_features(raw, baseline, tmp_path, year, rnd):
    scrambled = raw.copy()
    race = (scrambled.year == year) & (scrambled["round"] == rnd)
    rng = np.random.default_rng(0)
    for col in ["position", "points"]:
        scrambled.loc[race, col] = rng.permutation(scrambled.loc[race, col].values)

    after = _features(scrambled, tmp_path, "scrambled.csv")
    before, changed = _upto(baseline, year, rnd), _upto(after, year, rnd)
    pd.testing.assert_frame_equal(before[NUMERIC], changed[NUMERIC], check_exact=False)


@pytest.mark.parametrize("year,rnd", RACES)
def test_deleting_the_future_does_not_change_the_past(raw, baseline, tmp_path, year, rnd):
    past_only = raw[(raw.year < year) | ((raw.year == year) & (raw["round"] <= rnd))]
    after = _features(past_only, tmp_path, "truncated.csv")
    pd.testing.assert_frame_equal(_upto(baseline, year, rnd)[NUMERIC], after[NUMERIC],
                                  check_exact=False)


def test_teammates_share_one_pre_race_team_value(baseline):
    per_race_team = baseline.groupby(["year", "round", "constructorId"])[
        "constructor_recent_finish"].nunique(dropna=False)
    assert (per_race_team == 1).all()


def test_pit_lane_starts_go_to_the_back(baseline):
    pit = baseline[baseline["pit_lane_start"] == 1]
    assert len(pit) > 0
    field = baseline.groupby(["year", "round"]).size()
    for (y, r, _), row in pit.iterrows():
        assert row["grid"] == field.loc[(y, r)] + 1


def test_first_appearance_has_no_history(baseline):
    first = baseline.reset_index().sort_values(["year", "round"]).groupby("driverId").head(1)
    assert first["driver_avg_finish"].isna().all()
    assert first["driver_points_rate"].isna().all()


def test_validation_rejects_duplicate_rows(raw):
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_raw(pd.concat([raw, raw.head(1)]))


def test_validation_rejects_gap_in_positions(raw):
    bad = raw.copy()
    bad.loc[bad.index[0], "position"] = 99
    with pytest.raises(DataValidationError, match="positions"):
        validate_raw(bad)


def test_season_splits_never_cut_a_race_or_look_ahead(baseline):
    df = baseline.reset_index()
    for train_idx, val_idx in season_splits(df["year"]):
        assert df.loc[train_idx, "year"].max() < df.loc[val_idx, "year"].min()
        assert df.loc[val_idx, "year"].nunique() == 1
