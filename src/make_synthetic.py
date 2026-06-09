"""
make_synthetic.py
-----------------
Generates a realistic synthetic F1 dataset with the SAME schema as
fetch_data.py output, so we can build & test the ML pipeline before
running the real API fetch.

Relationships baked in (so the model has real signal to learn):
  - Starting grid position strongly predicts finishing position
  - Stronger constructors finish higher on average
  - ~12% chance of a DNF (Did Not Finish) per driver per race
  - Qualifying position correlates with grid (with occasional penalties)

Output columns match the real fetcher exactly.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Constructor "strength" rating (lower finishing offset = better).
# 2018-2025 = stable era order.
CONSTRUCTORS = {
    "redbull": -4.5, "ferrari": -3.0, "mercedes": -2.8, "mclaren": -1.5,
    "aston": -0.5, "alpine": 0.5, "williams": 1.5, "rb": 1.8,
    "sauber": 2.5, "haas": 2.8,
}

# 2026 regulation reset: new engines + sustainable fuel + new chassis
# reshuffles the competitive order. Some teams rise, some fall.
# This simulates the DISTRIBUTION SHIFT we'll analyze in the report.
CONSTRUCTORS_2026 = {
    "mclaren": -4.0, "mercedes": -3.5, "ferrari": -1.0, "williams": -2.0,
    "redbull": 0.5, "aston": 1.5, "sauber": -0.5, "alpine": 2.0,
    "rb": 2.5, "haas": 3.0,
}
# 2 drivers per constructor
DRIVERS = {f"{c}_d{i}": c for c in CONSTRUCTORS for i in (1, 2)}
CIRCUITS = ["bahrain", "jeddah", "albert_park", "imola", "monaco",
            "catalunya", "silverstone", "spa", "monza", "suzuka",
            "americas", "interlagos"]

DRIVER_IDS = list(DRIVERS.keys())  # 20 drivers


def simulate_season(year):
    rows = []
    # Pick the strength map: 2026 uses the post-reset order
    strengths = CONSTRUCTORS_2026 if year >= 2026 else CONSTRUCTORS
    # 2026 is in progress: only ~6 rounds completed so far
    circuits = CIRCUITS[:6] if year >= 2026 else CIRCUITS
    for rnd, circuit in enumerate(circuits, start=1):
        # Each driver gets a "race pace" = constructor strength + driver skill + noise
        driver_skill = {d: rng.normal(0, 1.2) for d in DRIVER_IDS}
        # latent pace -> grid order
        pace = {}
        for d in DRIVER_IDS:
            pace[d] = strengths[DRIVERS[d]] + driver_skill[d] + rng.normal(0, 1.5)
        # qualifying order = sort by pace (best pace = pole = position 1)
        quali_order = sorted(DRIVER_IDS, key=lambda d: pace[d])
        quali_pos = {d: i + 1 for i, d in enumerate(quali_order)}
        # grid usually = quali, but ~10% get a grid penalty
        grid = {}
        for d in DRIVER_IDS:
            base = quali_pos[d]
            if rng.random() < 0.10:
                base = min(20, base + rng.integers(3, 8))  # penalty drops them back
            grid[d] = base
        # race result: finishing pace = grid influence + fresh race noise
        race_score = {}
        dnf = {}
        for d in DRIVER_IDS:
            dnf[d] = rng.random() < 0.12   # 12% DNF chance
            race_score[d] = grid[d] + rng.normal(0, 3.5)
        # finishing order among non-DNF
        finishers = [d for d in DRIVER_IDS if not dnf[d]]
        finishers.sort(key=lambda d: race_score[d])
        finish_pos = {d: i + 1 for i, d in enumerate(finishers)}
        # DNFs get positions after all finishers
        dnf_drivers = [d for d in DRIVER_IDS if dnf[d]]
        for j, d in enumerate(dnf_drivers):
            finish_pos[d] = len(finishers) + j + 1

        points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                      6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        for d in DRIVER_IDS:
            pos = finish_pos[d]
            pts = 0.0 if dnf[d] else float(points_map.get(pos, 0))
            rows.append({
                "year": year,
                "round": rnd,
                "date": f"{year}-{rnd:02d}-01",
                "circuitId": circuit,
                "driverId": d,
                "constructorId": DRIVERS[d],
                "grid": grid[d],
                "quali_pos": quali_pos[d],
                "position": pos,
                "points": pts,
                "status": "Retired" if dnf[d] else "Finished",
            })
    return rows


def main(start=2018, end=2026, out="data/f1_raw.csv"):
    all_rows = []
    for year in range(start, end + 1):
        all_rows.extend(simulate_season(year))
    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} synthetic rows ({start}-{end}) -> {out}")
    return df


if __name__ == "__main__":
    df = main()
    print(df.head(12).to_string())
