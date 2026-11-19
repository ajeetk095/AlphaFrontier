"""Walk-forward back-test: re-estimate on a rolling window, rebalance on schedule, pay transaction costs.

No look-ahead: weights applied from day t use only returns strictly before t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import TRADING_DAYS
from .models import MODELS
from .performance import summarise


@dataclass
class BTConfig:
    lookback: int = 504           # trading days used to estimate each set of weights
    rebalance: str = "M"          # "M" monthly, "Q" quarterly
    cost_bps: float = 10.0        # one-way cost per unit of turnover, in basis points
    cap: float = 0.15
    rf: float = 0.04
    shrink: float = 0.0
    ledoit_wolf: bool = False
    model_kwargs: dict = field(default_factory=dict)


def rebalance_positions(index: pd.DatetimeIndex, lookback: int, rule: str) -> list[int]:
    """Integer positions of the first trading day of each period, once `lookback` days exist."""
    period = index.to_period("M" if rule == "M" else "Q")
    firsts = np.flatnonzero(np.r_[True, period[1:] != period[:-1]])
    return [int(i) for i in firsts if i >= lookback]


def run_strategy(rets: pd.DataFrame, name: str, cfg: BTConfig) -> dict:
    model = MODELS[name]
    pos = rebalance_positions(rets.index, cfg.lookback, cfg.rebalance)
    if not pos:
        raise ValueError("Not enough history for the chosen look-back window.")
    start = pos[0]
    R = rets.values
    n = R.shape[1]
    w = np.zeros(n)
    out, turn, weights_hist = [], [], {}
    pos_set = set(pos)
    for i in range(start, len(R)):
        cost = 0.0
        if i in pos_set:
            window = rets.iloc[i - cfg.lookback:i]  # strictly before day i
            target = model(window, cap=cfg.cap, rf=cfg.rf, shrink=cfg.shrink, ledoit_wolf=cfg.ledoit_wolf).reindex(rets.columns).values
            t = float(np.abs(target - w).sum()) if w.any() else 1.0  # first trade buys the whole book
            cost = t * cfg.cost_bps / 1e4
            turn.append((rets.index[i], t))
            weights_hist[rets.index[i]] = target
            w = target
        gross = float(w @ R[i])
        net = (1 + gross) * (1 - cost) - 1
        out.append(net)
        w = w * (1 + R[i]) / (1 + gross)  # drift between rebalances
    idx = rets.index[start:]
    turnover = pd.Series(dict(turn)).rename("turnover")
    return {
        "returns": pd.Series(out, index=idx, name=name),
        "turnover": turnover,
        "weights": pd.DataFrame(weights_hist, index=rets.columns).T,
    }


def run_backtest(rets: pd.DataFrame, strategies: list[str], cfg: BTConfig) -> dict:
    results = {s: run_strategy(rets, s, cfg) for s in strategies}
    start = max(r["returns"].index[0] for r in results.values())
    for r in results.values():
        r["returns"] = r["returns"].loc[start:]
    return results


def backtest_table(results: dict, rf: float, cost_bps: float, bench: pd.Series | None = None) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        s = summarise(r["returns"], rf)
        years = len(r["returns"]) / TRADING_DAYS
        turn_total = float(r["turnover"].sum())
        rows[name] = {
            "CAGR": s["Annualised return"], "Volatility": s["Annualised volatility"],
            "Sharpe": s["Sharpe ratio"], "Sortino": s["Sortino ratio"], "Max drawdown": s["Max drawdown"],
            "Turnover / yr": turn_total / years, "Cost drag / yr": turn_total / years * cost_bps / 1e4,
        }
    if bench is not None:
        first = next(iter(results.values()))["returns"].index
        b = bench.pct_change().dropna().reindex(first).dropna()
        s = summarise(b, rf)
        rows["S&P 500 (benchmark)"] = {
            "CAGR": s["Annualised return"], "Volatility": s["Annualised volatility"], "Sharpe": s["Sharpe ratio"],
            "Sortino": s["Sortino ratio"], "Max drawdown": s["Max drawdown"], "Turnover / yr": np.nan, "Cost drag / yr": np.nan,
        }
    return pd.DataFrame(rows).T


def sharpe_diff_test(a: pd.Series, b: pd.Series, rf: float, n_boot: int = 2000, block: int = 21, seed: int = 0) -> dict:
    """Paired block-bootstrap of Sharpe(a) - Sharpe(b). Returns the difference and a 90% interval."""
    d = pd.concat([a, b], axis=1).dropna().values
    T = len(d)
    rf_d = rf / TRADING_DAYS
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :T]
    samp = d[idx]                                         # (n_boot, T, 2)
    ex = samp - rf_d
    sharpe = ex.mean(1) / ex.std(1, ddof=1) * np.sqrt(TRADING_DAYS)
    diff = sharpe[:, 0] - sharpe[:, 1]
    ex0 = d - rf_d
    point = ex0[:, 0].mean() / ex0[:, 0].std(ddof=1) - ex0[:, 1].mean() / ex0[:, 1].std(ddof=1)
    lo, hi = np.percentile(diff, [5, 95])
    return {"diff": float(point * np.sqrt(TRADING_DAYS)), "lo": float(lo), "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def rolling_sharpe(r: pd.Series, rf: float, window: int = TRADING_DAYS) -> pd.Series:
    ex = r - rf / TRADING_DAYS
    return (ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(TRADING_DAYS)).dropna()
