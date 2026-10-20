"""Time-based train / out-of-time (OOT) split.

The target looks 12 months ahead, so an observation at month t uses information up to t + 12.
To avoid leakage, the last training period must be at least `gap_periods` before the first OOT period.
"""
import numpy as np


def time_based_split(df, time_col, oot_periods=6, gap_periods=12):
    periods = np.sort(df[time_col].unique())
    if oot_periods < 1:
        raise ValueError("oot_periods must be at least 1")
    n_train = len(periods) - oot_periods - gap_periods
    if n_train < 1:
        raise ValueError(
            f"Not enough periods ({len(periods)}) for oot_periods={oot_periods} and "
            f"gap_periods={gap_periods}. Load more origination years or reduce the gap."
        )
    train_periods = periods[:n_train]
    oot_periods_set = periods[-oot_periods:]
    train = df[df[time_col].isin(train_periods)].copy()
    oot = df[df[time_col].isin(oot_periods_set)].copy()
    return train, oot
