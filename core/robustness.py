"""Estimation-noise lab: how much do 'optimal' weights move when the data are resampled?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize import annualised_inputs, max_sharpe


def bootstrap_weights(rets: pd.DataFrame, rf: float, cap: float = 1.0, n_boot: int = 60,
                      shrink: float = 0.0, ledoit_wolf: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        sample = rets.iloc[rng.integers(0, len(rets), len(rets))]
        mu, cov = annualised_inputs(sample, shrink, ledoit_wolf)
        rows.append(max_sharpe(mu, cov, rf, cap).values)
    return pd.DataFrame(rows, columns=rets.columns)


def stability_summary(W: pd.DataFrame) -> dict:
    """Average weight swing = mean absolute change of the whole weight vector vs. the average weights."""
    avg = W.mean()
    swing = float((W - avg).abs().sum(axis=1).mean())
    return {"avg_weight_swing": swing, "max_asset_std": float(W.std().max()), "n": len(W)}
