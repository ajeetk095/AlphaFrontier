"""Mean-variance optimisation (convex, solved with cvxpy) and efficient frontiers."""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .data import TRADING_DAYS


class InfeasibleError(ValueError):
    pass


def annualised_inputs(returns: pd.DataFrame, shrink: float = 0.0, ledoit_wolf: bool = False):
    """Annualised mean vector and covariance matrix.

    shrink      : 0..1 pull of each asset's mean toward the cross-asset average (tames estimation noise).
    ledoit_wolf : use Ledoit-Wolf shrinkage covariance instead of the raw sample covariance.
    """
    mu = returns.mean() * TRADING_DAYS
    if shrink > 0:
        mu = (1 - shrink) * mu + shrink * mu.mean()
    if ledoit_wolf:
        from sklearn.covariance import LedoitWolf

        cov = pd.DataFrame(
            LedoitWolf().fit(returns.values).covariance_ * TRADING_DAYS,
            index=returns.columns, columns=returns.columns,
        )
    else:
        cov = returns.cov() * TRADING_DAYS
    return mu, cov


def _check_cap(n: int, cap: float) -> None:
    if cap * n < 1 - 1e-9:
        raise InfeasibleError(
            f"A {cap:.0%} cap on {n} assets cannot sum to 100% (need cap >= {1/n:.1%})."
        )


def _clean(w: np.ndarray, cap: float) -> np.ndarray:
    w = np.where(w < 1e-7, 0.0, w)
    w = np.minimum(w, cap)
    return w / w.sum()


def _best_capped_excess(excess: np.ndarray, cap: float) -> float:
    """Largest achievable portfolio excess return under the cap (greedy fill of the best assets)."""
    remaining, total = 1.0, 0.0
    for e in np.sort(excess)[::-1]:
        take = min(cap, remaining)
        total += take * e
        remaining -= take
        if remaining <= 1e-12:
            break
    return total


def _solve(prob: cp.Problem) -> None:
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise InfeasibleError(f"Optimiser status: {prob.status}")


def min_variance(mu: pd.Series, cov: pd.DataFrame, cap: float = 1.0) -> pd.Series:
    n = len(mu)
    _check_cap(n, cap)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov.values))),
        [cp.sum(w) == 1, w >= 0, w <= cap],
    )
    _solve(prob)
    return pd.Series(_clean(w.value, cap), index=mu.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float, cap: float = 1.0) -> pd.Series:
    """Maximum-Sharpe (tangency) portfolio, long-only, optional per-asset cap.

    Sharpe maximisation is non-convex as written, so we use the standard exact convex
    reformulation (Cornuejols & Tutuncu): minimise y'Sy s.t. (mu-rf)'y = 1, 1'y = k,
    0 <= y <= cap*k, k >= 0, then w = y/k.
    """
    n = len(mu)
    _check_cap(n, cap)
    excess = (mu - rf).values
    if _best_capped_excess(excess, cap) <= 1e-12:
        return min_variance(mu, cov, cap)  # no allowed mix beats cash: fall back to the least-risk mix
    y, k = cp.Variable(n), cp.Variable(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov.values))),
        [excess @ y == 1, cp.sum(y) == k, y >= 0, y <= cap * k],
    )
    _solve(prob)
    return pd.Series(_clean(y.value / k.value, cap), index=mu.index)


@dataclass
class Stats:
    ret: float
    vol: float
    sharpe: float
    hhi: float
    eff_n: float
    n_hold: int


def portfolio_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float) -> Stats:
    wv = w.values
    ret = float(mu.values @ wv)
    vol = float(np.sqrt(wv @ cov.values @ wv))
    active = wv[wv > 1e-6]
    hhi = float((active ** 2).sum())
    return Stats(ret, vol, (ret - rf) / vol if vol > 0 else np.nan, hhi, 1 / hhi, int(len(active)))


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, cap: float = 1.0, n_points: int = 40) -> pd.DataFrame:
    """Efficient part of the long-only (capped) frontier: min variance for each target return."""
    n = len(mu)
    _check_cap(n, cap)
    w_mv = min_variance(mu, cov, cap)
    r_lo = float(mu.values @ w_mv.values)
    w = cp.Variable(n)
    top = cp.Problem(cp.Maximize(mu.values @ w), [cp.sum(w) == 1, w >= 0, w <= cap])
    _solve(top)
    r_hi = float(top.value)
    rows = []
    for t in r_lo + (r_hi - r_lo) * np.linspace(0, 0.9999, n_points):
        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov.values))),
            [cp.sum(w) == 1, w >= 0, w <= cap, mu.values @ w >= t],
        )
        try:
            _solve(prob)
            rows.append((np.sqrt(prob.value), t))
        except InfeasibleError:
            continue
    return pd.DataFrame(rows, columns=["vol", "ret"])
