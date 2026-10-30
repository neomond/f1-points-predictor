"""
build_dataset.py
================
Stage 1 of the F1 Points-Finish NN extension.

Responsibilities:
  1. Fetch race results from the Jolpica F1 API (2018..2026).
  2. Load them into a SQL database (Postgres if DATABASE_URL is set,
     otherwise a local SQLite file — the feature SQL is portable across both).
  3. Execute sql/features.sql to produce the leakage-aware feature table.
  4. Return a pandas DataFrame of features + target.

Design notes
------------
* PROJECT_ROOT-relative paths so the script is portable.
* Network is optional: if Jolpica is unreachable, we synthesise a
  realistic dataset so the rest of the pipeline (and your learning) still runs.
* The SAME features.sql runs on Postgres and SQLite. SQLite >= 3.25 supports
  window functions, so ROW_NUMBER / LAG / framed aggregates work identically.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"

SEASONS = list(range(2018, 2027))           # 2018..2026 inclusive
EVAL_SEASON = 2026                          # held out, mirrors your existing split
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


# ---------------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------------
def fetch_jolpica(seasons=SEASONS, timeout=10) -> pd.DataFrame | None:
    """Fetch results for each season from Jolpica. Returns None on any failure
    so the caller can fall back to synthetic data."""
    try:
        import requests
    except ImportError:
        return None

    rows = []
    for season in seasons:
        # Jolpica paginates; results endpoint returns up to `limit` per call.
        offset, limit = 0, 100
        got_any = False
        while True:
            url = f"{JOLPICA_BASE}/{season}/results/?limit={limit}&offset={offset}"
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
            except Exception:
                # Network blocked or season missing — bail to synthetic.
                return None if not got_any and season == seasons[0] else _finalise(rows)
            data = r.json()
            races = data["MRData"]["RaceTable"]["Races"]
            if not races:
                break
            for race in races:
                rnd = int(race["round"])
                date = race.get("date")
                for res in race.get("Results", []):
                    got_any = True
                    rows.append({
                        "season": season,
                        "round": rnd,
                        "race_date": date,
                        "driver": res["Driver"]["driverId"],
                        "constructor": res["Constructor"]["constructorId"],
                        "grid": int(res["grid"]),
                        "position": int(res["position"]) if res["position"].isdigit() else None,
                        "points": float(res["points"]),
                        "status": res.get("status"),
                    })
            total = int(data["MRData"]["total"])
            offset += limit
            if offset >= total:
                break
            time.sleep(0.2)   # be polite to the public API
    return _finalise(rows)


def _finalise(rows) -> pd.DataFrame | None:
    if not rows:
        return None
    return pd.DataFrame(rows)


def synthesize(seasons=SEASONS, seed=42) -> pd.DataFrame:
    """Generate a plausible results table when the API is unavailable.
    Grid position drives finish probability (mirrors the real dominant
    predictor) so the pipeline behaves realistically for learning."""
    rng = np.random.default_rng(seed)
    drivers = [f"driver_{i:02d}" for i in range(20)]
    teams = [f"team_{i:02d}" for i in range(10)]
    driver_team = {d: teams[i // 2] for i, d in enumerate(drivers)}
    # latent skill per driver — stable across the era
    skill = {d: rng.normal(0, 1) for d in drivers}

    rows = []
    for season in seasons:
        n_rounds = 22 if season != EVAL_SEASON else 10
        # a small regulation-reset shock in the eval season -> concept drift
        drift = rng.normal(0, 1.2, size=len(drivers)) if season == EVAL_SEASON else np.zeros(len(drivers))
        for rnd in range(1, n_rounds + 1):
            # qualifying: grid order influenced by skill + per-season drift + noise
            drift_map = dict(zip(drivers, drift))
            quali_score = (
                np.array([skill[d] for d in drivers])
                + np.array([drift_map[d] for d in drivers])
                + rng.normal(0, 1, len(drivers))
            )
            grid_order = np.argsort(-quali_score)
            grid_pos = {drivers[grid_order[i]]: i + 1 for i in range(len(drivers))}
            # race finish: grid + skill + race noise
            race_score = np.array([
                -grid_pos[d] * 0.6 + skill[d] * 0.8 + rng.normal(0, 1.0)
                for d in drivers
            ])
            finish_order = np.argsort(-race_score)
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                          6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            for fp, idx in enumerate(finish_order, start=1):
                d = drivers[idx]
                rows.append({
                    "season": season, "round": rnd,
                    "race_date": f"{season}-{rnd:02d}-01",
                    "driver": d, "constructor": driver_team[d],
                    "grid": grid_pos[d], "position": fp,
                    "points": float(points_map.get(fp, 0)),
                    "status": "Finished",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2 + 3. Load to SQL and run the feature query
# ---------------------------------------------------------------------------
def build_features(results: pd.DataFrame) -> pd.DataFrame:
    """Load `results` into SQL, execute features.sql, return feature frame."""
    sql_text = (SQL_DIR / "features.sql").read_text()
    db_url = os.environ.get("DATABASE_URL")

    if db_url:  # ---- Postgres path (production-shaped) ----
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        results.to_sql("results", engine, if_exists="replace", index=False)
        with engine.begin() as conn:
            # split off the CREATE INDEX statements (run them, ignore failures
            # if indexes already exist) then run the final SELECT.
            *ddl, select_stmt = _split_sql(sql_text)
            for stmt in ddl:
                conn.execute(text(stmt))
            feats = pd.read_sql(text(select_stmt), conn)
        return feats

    # ---- SQLite fallback (zero setup, same window-function SQL) ----
    import sqlite3
    con = sqlite3.connect(":memory:")
    results.to_sql("results", con, if_exists="replace", index=False)
    *ddl, select_stmt = _split_sql(sql_text, sqlite=True)
    for stmt in ddl:
        try:
            con.execute(stmt)
        except sqlite3.OperationalError:
            pass  # e.g. IF NOT EXISTS already handled / dialect quirk
    feats = pd.read_sql_query(select_stmt, con)
    con.close()
    return feats


def _split_sql(sql_text: str, sqlite: bool = False) -> list[str]:
    """Return individual statements with comments stripped. Handles both
    full-line and inline `--` comments. When sqlite=True, translate the two
    Postgres-isms our features.sql uses (`::type` casts) into SQLite form."""
    out_lines = []
    for line in sql_text.splitlines():
        # drop inline comment (our SQL has no -- inside string literals)
        code = line.split("--", 1)[0]
        if code.strip():
            out_lines.append(code)
    cleaned = "\n".join(out_lines)
    if sqlite:
        # Postgres `expr::float` -> SQLite `CAST(expr AS REAL)`.
        import re
        cleaned = re.sub(r"(\w+)::float", r"CAST(\1 AS REAL)", cleaned)
    return [s.strip() for s in cleaned.split(";") if s.strip()]


def get_dataset() -> pd.DataFrame:
    print("Fetching results from Jolpica ...")
    results = fetch_jolpica()
    if results is None or results.empty:
        print("  API unavailable — synthesising a realistic dataset instead.")
        results = synthesize()
    else:
        print(f"  Fetched {len(results):,} result rows from the API.")
    print("Running SQL feature layer ...")
    feats = build_features(results)
    print(f"  Feature table: {feats.shape[0]:,} rows x {feats.shape[1]} cols.")
    return feats


if __name__ == "__main__":
    df = get_dataset()
    print(df.head(10).to_string(index=False))
