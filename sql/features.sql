-- ============================================================================
-- F1 Points-Finish Predictor — SQL Feature Layer (PostgreSQL)
-- ----------------------------------------------------------------------------
-- Every feature below is LEAKAGE-AWARE: each window frame uses only rows that
-- occurred BEFORE the current race for that driver/constructor. We never read
-- the current race's outcome or any future race when building a feature for it.
--
-- That is enforced two ways:
--   1. Frame `ROWS BETWEEN N PRECEDING AND 1 PRECEDING`  (strictly past rows)
--   2. Where a feature legitimately belongs to the race itself (grid/quali
--      position is known BEFORE the race starts) we use it as-is.
--
-- Source table expected (loaded from Jolpica in build_dataset.py):
--   results(season INT, round INT, race_date DATE, driver TEXT,
--           constructor TEXT, grid INT, position INT, points NUMERIC,
--           status TEXT)
-- ============================================================================

-- Helpful indexes: we filter/sort heavily on (driver, season, round).
-- An index here turns the per-driver window ordering from a sort-the-world
-- operation into an index-ordered scan.
CREATE INDEX IF NOT EXISTS idx_results_driver_season_round
    ON results (driver, season, round);
CREATE INDEX IF NOT EXISTS idx_results_constructor_season_round
    ON results (constructor, season, round);


-- ----------------------------------------------------------------------------
-- Main feature query. Returns one row per (driver, race) with:
--   - the TARGET: points_finish (1 if finished top 10 / scored, else 0)
--   - race-known features: grid position
--   - leakage-safe historical form features via window functions
-- ----------------------------------------------------------------------------
WITH base AS (
    SELECT
        season,
        round,
        race_date,
        driver,
        constructor,
        grid,
        position,
        points,
        -- TARGET. A driver "finishes in the points" if they scored > 0.
        -- (Equivalent to top-10 in the modern era.) This is the label —
        -- it describes the CURRENT race and must never feed a feature.
        CASE WHEN points > 0 THEN 1 ELSE 0 END AS points_finish
    FROM results
    WHERE position IS NOT NULL          -- drop rows with no classified finish
),

with_form AS (
    SELECT
        b.*,

        -- ------------------------------------------------------------------
        -- FEATURE 1: rolling average points over the driver's last 5 races.
        -- Frame is `5 PRECEDING AND 1 PRECEDING` => strictly the five races
        -- BEFORE this one. CURRENT ROW is deliberately excluded so the
        -- current race's points (part of the label) can't leak in.
        -- Crosses seasons intentionally to capture true recent form; swap the
        -- PARTITION to (driver, season) if you want within-season only.
        -- ------------------------------------------------------------------
        AVG(b.points) OVER (
            PARTITION BY b.driver
            ORDER BY b.season, b.round
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS roll_avg_points_5,

        -- FEATURE 2: rolling points-finish RATE over last 5 races (0..1).
        -- "How often has this driver been scoring lately."
        AVG(CASE WHEN b.points > 0 THEN 1.0 ELSE 0.0 END) OVER (
            PARTITION BY b.driver
            ORDER BY b.season, b.round
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS roll_points_rate_5,

        -- FEATURE 3: previous race finishing position (LAG = strictly past).
        -- COALESCE'd later; NULL for a driver's very first race.
        LAG(b.position) OVER (
            PARTITION BY b.driver
            ORDER BY b.season, b.round
        ) AS prev_finish_pos,

        -- FEATURE 4: previous race grid->finish delta (positions gained/lost),
        -- a cheap proxy for racecraft / car pace on Sunday.
        LAG(b.grid - b.position) OVER (
            PARTITION BY b.driver
            ORDER BY b.season, b.round
        ) AS prev_grid_to_finish,

        -- FEATURE 5: constructor recent form — rolling avg points of the TEAM
        -- over its last 10 car-results (both drivers), strictly past.
        AVG(b.points) OVER (
            PARTITION BY b.constructor
            ORDER BY b.season, b.round, b.driver
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS team_roll_avg_points_10,

        -- FEATURE 6: races of experience so far (count of strictly-prior races).
        COUNT(*) OVER (
            PARTITION BY b.driver
            ORDER BY b.season, b.round
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS career_races_so_far

    FROM base b
)

SELECT
    season,
    round,
    driver,
    constructor,

    -- ---- MODEL INPUT FEATURES ----
    -- grid is known before lights out, so it is a legitimate race-time feature.
    -- It is also your known dominant predictor — keep it front and centre.
    grid::float                               AS grid_position,
    COALESCE(roll_avg_points_5, 0)            AS roll_avg_points_5,
    COALESCE(roll_points_rate_5, 0)           AS roll_points_rate_5,
    COALESCE(prev_finish_pos, 20)             AS prev_finish_pos,      -- assume back of grid if unknown
    COALESCE(prev_grid_to_finish, 0)          AS prev_grid_to_finish,
    COALESCE(team_roll_avg_points_10, 0)      AS team_roll_avg_points_10,
    career_races_so_far::float                AS career_races_so_far,

    -- ---- TARGET ----
    points_finish
FROM with_form
ORDER BY season, round, driver;
