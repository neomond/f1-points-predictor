"""
fetch_data.py
-------------
Pulls real Formula 1 race + qualifying data from the Jolpica API
(the free successor to the deprecated Ergast API) and saves a flat CSV.

Run this on YOUR machine (not in a restricted sandbox):
    python src/fetch_data.py --start 2018 --end 2024 --out data/f1_raw.csv

Jolpica is rate-limited (~4 req/sec, 500/hr for anon). We sleep between
calls to stay polite. Pulling ~7 seasons takes a few minutes.

Docs: https://github.com/jolpica/jolpica-f1
"""
import argparse
import time
import requests
import pandas as pd

BASE = "https://api.jolpi.ca/ergast/f1"


def _get(url, retries=3):
    """GET with basic retry + rate-limit backoff."""
    for attempt in range(retries):
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:          # too many requests
            time.sleep(2 ** attempt)      # exponential backoff
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def get_season_rounds(year):
    data = _get(f"{BASE}/{year}.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    return [(int(r["round"]), r["raceName"], r["Circuit"]["circuitId"],
             r["date"]) for r in races]


def get_results(year, rnd):
    """Race finishing results for one round."""
    data = _get(f"{BASE}/{year}/{rnd}/results.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    rows = []
    for res in races[0]["Results"]:
        rows.append({
            "year": year,
            "round": rnd,
            "driverId": res["Driver"]["driverId"],
            "constructorId": res["Constructor"]["constructorId"],
            "grid": int(res["grid"]),                 # starting position
            "position": int(res["position"]),         # finishing position
            "points": float(res["points"]),
            "status": res["status"],                  # 'Finished', '+1 Lap', 'Accident'...
        })
    return rows


def get_qualifying(year, rnd):
    """Qualifying results -> best quali position per driver."""
    data = _get(f"{BASE}/{year}/{rnd}/qualifying.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races or "QualifyingResults" not in races[0]:
        return {}
    out = {}
    for q in races[0]["QualifyingResults"]:
        out[q["Driver"]["driverId"]] = int(q["position"])
    return out


def main(start, end, out):
    all_rows = []
    for year in range(start, end + 1):
        rounds = get_season_rounds(year)
        print(f"{year}: {len(rounds)} rounds")
        for rnd, name, circuit, date in rounds:
            results = get_results(year, rnd)
            quali = get_qualifying(year, rnd)
            for row in results:
                row["circuitId"] = circuit
                row["date"] = date
                row["quali_pos"] = quali.get(row["driverId"], None)
                all_rows.append(row)
            time.sleep(0.3)   # be polite to the free API
    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2018)
    p.add_argument("--end", type=int, default=2024)
    p.add_argument("--out", default="data/f1_raw.csv")
    args = p.parse_args()
    main(args.start, args.end, args.out)
