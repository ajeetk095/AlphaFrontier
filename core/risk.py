"""Risk analytics: VaR/CVaR, drawdowns, risk contributions, historical and beta-shock stress tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import TRADING_DAYS

STRESS_EVENTS = {
    "Q4 2018 sell-off (20 Sep – 24 Dec 2018)": ("2018-09-20", "2018-12-24"),
    "COVID crash (19 Feb – 23 Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 bear market (3 Jan – 12 Oct 2022)": ("2022-01-03", "2022-10-12"),
    "Autumn 2023 correction (31 Jul – 27 Oct 2023)": ("2023-07-31", "2023-10-27"),
    "Spring 2025 tariff shock (19 Feb – 8 Apr 2025)": ("2025-02-19", "2025-04-08"),
}


def var_cvar_historical(r: pd.Series, level: float = 0.95, horizon: int = 1) -> tuple[float, float]:
    """Loss thresholds as positive numbers. Multi-day uses overlapping compounded windows."""
    x = r.dropna()
    if horizon > 1:
        x = (1 + x).rolling(horizon).apply(np.prod, raw=True).dropna() - 1
    q = np.percentile(x, (1 - level) * 100)
    tail = x[x <= q]
    return float(-q), float(-tail.mean())


def var_cvar_parametric(r: pd.Series, level: float = 0.95, horizon: int = 1) -> tuple[float, float]:
    """Normal-distribution VaR/CVaR, scaled by sqrt(horizon). Underestimates fat tails."""
    mu, sd = r.mean() * horizon, r.std() * np.sqrt(horizon)
    z = norm.ppf(1 - level)
    var = -(mu + z * sd)
    cvar = -(mu - sd * norm.pdf(z) / (1 - level))
    return float(var), float(cvar)


def drawdown_series(r: pd.Series) -> pd.Series:
    curve = (1 + r).cumprod()
    return curve / curve.cummax() - 1


def rolling_vol(r: pd.Series, window: int = 63) -> pd.Series:
    return (r.rolling(window).std() * np.sqrt(TRADING_DAYS)).dropna()


def replay_event(prices: pd.DataFrame, w: pd.Series, start: str, end: str, bench: pd.Series | None = None) -> dict | None:
    """Buy-and-hold the portfolio through a historical episode. Uses only holdings that traded throughout."""
    sub = prices.loc[start:end]
    if len(sub) < 5:
        return None
    ok = [c for c in prices.columns if sub[c].notna().all() and w.get(c, 0) > 0]
    if not ok:
        return None
    ww = w[ok] / w[ok].sum()
    growth = (sub[ok] / sub[ok].iloc[0])
    curve = (growth * ww).sum(axis=1)
    out = {"portfolio": float(curve.iloc[-1] - 1), "worst_day_dd": float((curve / curve.cummax() - 1).min()),
           "coverage": float(w[ok].sum() / w.sum()), "holdings_used": len(ok)}
    if bench is not None and bench.loc[start:end].notna().sum() > 4:
        b = bench.loc[start:end].dropna()
        out["benchmark"] = float(b.iloc[-1] / b.iloc[0] - 1)
    return out


def portfolio_beta(rets: pd.DataFrame, w: pd.Series, mkt: pd.Series) -> float:
    port = (rets * w.reindex(rets.columns).fillna(0)).sum(axis=1)
    df = pd.concat([port, mkt], axis=1, join="inner").dropna()
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())
