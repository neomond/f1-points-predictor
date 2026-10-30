"""
metrics.py
----------
Evaluation helpers that respect the structure of F1 data.

* Grid baseline      -> "the driver starting higher finishes higher".
                        Any model must be judged by its LIFT over this.
* precision_at_10    -> exactly 10 drivers score per race, so the natural
                        question is "of my top-10 picks, how many scored?"
* race_bootstrap     -> rows inside one race are not independent (one
                        driver's result pushes another's), so we resample
                        whole RACES, not rows, to get honest confidence
                        intervals (a block bootstrap).
"""
import numpy as np
from scipy.stats import rankdata

RACE = ["year", "round"]


def grid_score(df):
    """Baseline score: better (lower) grid -> higher score."""
    return -df["grid"].to_numpy()


def precision_at_10(df, score_col):
    """Mean over races of: share of the 10 highest-scored drivers who scored."""
    return float(np.mean([
        g.nlargest(10, score_col)["points_finish"].mean()
        for _, g in df.groupby(RACE)
    ]))


def race_bootstrap(df, stat_fn, n_boot=2000, seed=0):
    """Point estimate + 95% CI of stat_fn(df), resampling whole races.

    stat_fn takes a DataFrame and returns a float. Because the same races
    are drawn for every term inside stat_fn, a difference such as
    (model AUC - baseline AUC) gets a correctly PAIRED interval.
    """
    rng = np.random.default_rng(seed)
    # row positions of each race, computed once (no DataFrame concat per draw)
    races = list(df.groupby(RACE).indices.values())
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([races[i] for i in rng.integers(0, len(races), len(races))])
        stats.append(stat_fn(df.iloc[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return stat_fn(df), lo, hi


def fast_auc(y, score):
    """ROC-AUC via the Mann-Whitney U statistic (same value as sklearn's
    roc_auc_score, including tie handling, ~10x faster for bootstrapping)."""
    y = np.asarray(y)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    ranks = rankdata(score)              # average ranks -> ties count as 0.5
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc(df, score_col):
    return fast_auc(df["points_finish"].to_numpy(), df[score_col].to_numpy())


def auc_lift(df, score_col="p", base_col="grid_score"):
    return auc(df, score_col) - auc(df, base_col)
