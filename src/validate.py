"""
validate.py
-----------
Data-quality checks on the raw per-driver-per-race table, run before any
feature is built. A model is only as correct as the data under it, and API
data can be silently wrong (duplicate rows, missing drivers, odd encodings).

Hard errors (raise) -> the data cannot be trusted, stop the pipeline.
Soft warnings (print) -> known quirks that the pipeline handles.

Run standalone:
    python src/validate.py data/f1_raw.csv
"""
import sys
import pandas as pd

REQUIRED = ["year", "round", "driverId", "constructorId", "grid",
            "position", "points", "status", "circuitId", "quali_pos"]
RACE = ["year", "round"]


class DataValidationError(ValueError):
    pass


def validate_raw(df, verbose=False):
    errors, warnings = [], []

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    # 1. One row per driver per race.
    dup = df.duplicated(RACE + ["driverId"]).sum()
    if dup:
        errors.append(f"{dup} duplicate (year, round, driverId) rows")

    # 2. Finishing positions in each race are exactly 1..N with no gaps.
    def positions_ok(g):
        return sorted(g["position"].tolist()) == list(range(1, len(g) + 1))
    bad_pos = [k for k, g in df.groupby(RACE) if not positions_ok(g)]
    if bad_pos:
        errors.append(f"{len(bad_pos)} races with non-contiguous positions, e.g. {bad_pos[:3]}")

    # 3. Plausible field size (F1 grids since 2018: 20 cars, rarely fewer).
    size = df.groupby(RACE).size()
    odd = size[(size < 15) | (size > 24)]
    if len(odd):
        errors.append(f"{len(odd)} races with implausible field size: {odd.to_dict()}")

    # 4. At most two cars per team per race.
    per_team = df.groupby(RACE + ["constructorId"]).size()
    if (per_team > 2).any():
        errors.append(f"{int((per_team > 2).sum())} (race, team) pairs with >2 cars")

    # 5. Value ranges.
    if (df["points"] < 0).any():
        errors.append("negative points")
    if (df["grid"] < 0).any():
        errors.append("negative grid position")

    # Soft: known quirks the pipeline handles explicitly.
    n_pit = int((df["grid"] == 0).sum())
    if n_pit:
        warnings.append(f"{n_pit} pit-lane starts (grid == 0) -> remapped in features.py")
    n_q = int(df["quali_pos"].isna().sum())
    if n_q:
        warnings.append(f"{n_q} missing qualifying positions -> imputed in the pipeline")

    if errors:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(errors))
    if verbose:
        for w in warnings:
            print("WARNING:", w)
        print(f"OK: {len(df)} rows, {df.groupby(RACE).ngroups} races passed all checks")
    return warnings


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/f1_raw.csv"
    validate_raw(pd.read_csv(path), verbose=True)
