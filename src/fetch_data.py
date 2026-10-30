"""
fetch_data.py
-------------
Pulls real Formula 1 race + qualifying data from the Jolpica API
(the free successor to the deprecated Ergast API) and saves a flat CSV.

    # incremental (recommended): only fetch races not already in the CSV
    python src/fetch_data.py --update

    # full rebuild
    python src/fetch_data.py --start 2018 --end 2026

Rate limits: Jolpica allows a few requests per second and a limited number
per hour for anonymous users. This script therefore:
  * waits a fixed interval between EVERY request (not just between rounds),
  * on HTTP 429 honours the server's Retry-After header, else backs off
    exponentially (up to ~2 minutes), with several retries,
  * saves progress after every season, so a failure never loses finished
    work and a re-run with --update resumes where it stopped.

Docs: https://github.com/jolpica/jolpica-f1
"""
import argparse
import datetime as dt
import os
import time

import pandas as pd
import requests

BASE = "https://api.jolpi.ca/ergast/f1"
REQUEST_INTERVAL = 0.6        # seconds between requests (~1.7 req/s)
MAX_RETRIES = 6
KEY = ["year", "round", "driverId"]

_session = requests.Session()
_last_request = 0.0


def _get(url):
    """GET with polite pacing, Retry-After support and exponential backoff."""
    global _last_request
    for attempt in range(MAX_RETRIES):
        wait = REQUEST_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            r = _session.get(url, timeout=30)
        except requests.RequestException as e:          # network blip
            delay = min(2 ** attempt * 2, 120)
            print(f"  network error ({e.__class__.__name__}), retry in {delay}s")
            time.sleep(delay)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            retry_after = r.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() \
                else min(2 ** attempt * 5, 120)
            print(f"  HTTP {r.status_code}, waiting {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def get_season_rounds(year):
    data = _get(f"{BASE}/{year}.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    return [(int(r["round"]), r["raceName"], r["Circuit"]["circuitId"],
             r["date"]) for r in races]


def get_results(year, rnd):
    """Race finishing results for one round ([] if not raced yet)."""
    data = _get(f"{BASE}/{year}/{rnd}/results.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    return [{
        "year": year,
        "round": rnd,
        "driverId": res["Driver"]["driverId"],
        "constructorId": res["Constructor"]["constructorId"],
        "grid": int(res["grid"]),                 # 0 = pit-lane start
        "position": int(res["position"]),         # finishing position
        "points": float(res["points"]),
        "status": res["status"],
    } for res in races[0]["Results"]]


def get_qualifying(year, rnd):
    """Qualifying results -> quali position per driver."""
    data = _get(f"{BASE}/{year}/{rnd}/qualifying.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races or "QualifyingResults" not in races[0]:
        return {}
    return {q["Driver"]["driverId"]: int(q["position"])
            for q in races[0]["QualifyingResults"]}


def fetch_season(year, skip_rounds=()):
    """All completed rounds of a season, except rounds in skip_rounds."""
    rows = []
    rounds = get_season_rounds(year)
    todo = [r for r in rounds if r[0] not in skip_rounds]
    print(f"{year}: {len(rounds)} rounds on calendar, {len(todo)} to fetch")
    for rnd, name, circuit, date in todo:
        if dt.date.fromisoformat(date) >= dt.date.today():
            print(f"  round {rnd} ({name}) is on {date}: not raced yet, stopping season")
            break
        results = get_results(year, rnd)
        if not results:                  # past date but no results: cancelled/not yet published
            print(f"  round {rnd} ({name}): no results published, skipping")
            continue
        quali = get_qualifying(year, rnd)
        for row in results:
            row["circuitId"] = circuit
            row["date"] = date
            row["quali_pos"] = quali.get(row["driverId"])
            rows.append(row)
        print(f"  round {rnd} ({name}): {len(results)} drivers")
    return rows


def _save(existing, new_rows, out):
    """Merge, de-duplicate, sort and write atomically (temp file + rename)."""
    df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    df = (df.drop_duplicates(KEY, keep="last")
            .sort_values(["year", "round", "position"])
            .reset_index(drop=True))
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    return df


def main(start, end, out, update):
    if update and os.path.exists(out):
        existing = pd.read_csv(out)
        start = int(existing["year"].max())       # resume from latest season
        print(f"Update mode: {len(existing)} rows on disk, "
              f"latest = {start} round {existing.loc[existing.year == start, 'round'].max()}")
    else:
        existing = pd.DataFrame()

    for year in range(start, end + 1):
        have = set(existing.loc[existing["year"] == year, "round"]) if len(existing) else set()
        rows = fetch_season(year, skip_rounds=have)
        if rows:
            existing = _save(existing, rows, out)   # checkpoint after each season
            print(f"  saved -> {out} ({len(existing)} rows)")
    print(f"Done: {len(existing)} rows in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2018)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--out", default="data/f1_raw.csv")
    p.add_argument("--update", action="store_true",
                   help="only fetch races missing from --out (resumable)")
    args = p.parse_args()
    main(args.start, args.end, args.out, args.update)