"""Portfolio return series, performance metrics, and forward-looking scenario analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def portfolio_returns(returns: pd.DataFrame, w: pd.Series, rebalance: str = "none") -> pd.Series:
    """Daily portfolio returns. rebalance='none' = buy-and-hold from day 0; 'daily' = constant weights."""
    w = w.reindex(returns.columns).fillna(0.0)
    if rebalance == "daily":
        return (returns * w).sum(axis=1).rename("portfolio")
    growth = (1 + returns).cumprod()
    value = (growth * w).sum(axis=1)
    out = value.pct_change()
    out.iloc[0] = value.iloc[0] - 1.0
    return out.rename("portfolio")


def max_drawdown(rets: pd.Series) -> float:
    curve = (1 + rets).cumprod()
    return float((curve / curve.cummax() - 1).min())


def summarise(rets: pd.Series, rf_annual: float) -> dict:
    n = len(rets)
    total = float((1 + rets).prod() - 1)
    ann_ret = (1 + total) ** (TRADING_DAYS / n) - 1 if n else np.nan
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    rf_d = rf_annual / TRADING_DAYS
    sharpe = float((rets.mean() - rf_d) * TRADING_DAYS / vol) if vol > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(rets - rf_d, 0) ** 2)) * np.sqrt(TRADING_DAYS)
    sortino = float((rets.mean() - rf_d) * TRADING_DAYS / downside) if downside > 0 else np.nan
    return {
        "Total return": total,
        "Annualised return": float(ann_ret),
        "Annualised volatility": vol,
        "Sharpe ratio": sharpe,
        "Sortino ratio": sortino,
        "Max drawdown": max_drawdown(rets),
        "Trading days": n,
    }


def scenario_paths(
    rets: pd.Series,
    horizon: int = TRADING_DAYS,
    n_paths: int = 5000,
    drift_haircut: float = 0.5,
    block: int = 5,
    seed: int = 42,
) -> dict:
    """Block-bootstrap scenarios of the portfolio over `horizon` trading days.

    These are SCENARIOS, not forecasts. Optimised portfolios are chosen *because* they looked
    good in-sample, so we de-mean part of the historical drift (`drift_haircut`) to avoid
    projecting that selection bias forward.
    """
    rng = np.random.default_rng(seed)
    x = rets.dropna().values
    x = x - drift_haircut * x.mean()
    n = len(x)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n - block, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon]
    paths = np.cumprod(1 + x[idx], axis=1)
    terminal = paths[:, -1] - 1
    var95 = float(np.percentile(terminal, 5))
    return {
        "terminal": terminal,
        "paths": paths,
        "p05": var95,
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
        "prob_loss": float((terminal < 0).mean()),
        "cvar95": float(terminal[terminal <= var95].mean()),
    }
