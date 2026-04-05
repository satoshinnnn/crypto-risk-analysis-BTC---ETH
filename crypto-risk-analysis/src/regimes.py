from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regimes(volatility: pd.Series) -> pd.Series:
    clean = volatility.dropna()
    if clean.empty:
        return pd.Series(index=volatility.index, dtype="object")

    q1, q2, q3 = clean.quantile([0.25, 0.50, 0.75]).tolist()

    def label(v):
        if pd.isna(v):
            return np.nan
        if v <= q1:
            return "Deep Calm"
        if v <= q2:
            return "Calm"
        if v <= q3:
            return "Turbulent"
        return "Stress Turbulent"

    return volatility.apply(label)
