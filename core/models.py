"""Portfolio construction models beyond mean-variance: 1/N, risk parity, HRP, min-CVaR.

All functions take a window of daily returns and return long-only weights summing to 1.
Exact cap handling: max-Sharpe, min-variance, min-CVaR. Heuristic cap handling (clip & redistribute): 1/N-capped, risk parity, HRP.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .optimize import InfeasibleError, _check_cap, _clean, _solve, annualised_inputs, max_sharpe, min_variance
from .simulation import project_to_cap

EXACT_CAP = {"Max Sharpe", "Min variance", "Min CVaR"}


def _cap(w: np.ndarray, cap: float) -> np.ndarray:
    if cap >= 1.0:
        return w
    return project_to_cap(w[None, :], cap)[0]


def equal_weight(rets: pd.DataFrame, cap: float = 1.0, **_) -> pd.Series:
    n = rets.shape[1]
    _check_cap(n, cap)
    return pd.Series(np.full(n, 1.0 / n), index=rets.columns)


def risk_parity(rets: pd.DataFrame, cap: float = 1.0, ledoit_wolf: bool = False, **_) -> pd.Series:
    """Equal risk contribution via the convex log-barrier problem min 0.5 y'Sy - (1/n) sum log y."""
    n = rets.shape[1]
    _check_cap(n, cap)
    _, cov = annualised_inputs(rets, 0.0, ledoit_wolf)
    y = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(cov.values)) - cp.sum(cp.log(y)) / n))
    _solve(prob)
    w = np.maximum(y.value, 0)
    return pd.Series(_cap(w / w.sum(), cap), index=rets.columns)


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp(rets: pd.DataFrame, cap: float = 1.0, **_) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado, 2016): cluster by correlation, split risk recursively."""
    n = rets.shape[1]
    _check_cap(n, cap)
    cov = rets.cov().values
    corr = rets.corr().values
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters if len(c) > 1 for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))]
        for k in range(0, len(clusters), 2):
            left, right = clusters[k], clusters[k + 1]
            vl, vr = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return pd.Series(_cap(w / w.sum(), cap), index=rets.columns)


def min_cvar(rets: pd.DataFrame, cap: float = 1.0, level: float = 0.95, **_) -> pd.Series:
    """Minimise historical CVaR (Rockafellar-Uryasev linear program), long-only with exact cap."""
    T, n = rets.shape
    _check_cap(n, cap)
    w, t, z = cp.Variable(n), cp.Variable(), cp.Variable(T, nonneg=True)
    loss = -rets.values @ w
    prob = cp.Problem(
        cp.Minimize(t + cp.sum(z) / ((1 - level) * T)),
        [z >= loss - t, cp.sum(w) == 1, w >= 0, w <= cap],
    )
    _solve(prob)
    return pd.Series(_clean(w.value, cap), index=rets.columns)


def max_sharpe_model(rets, cap=1.0, rf=0.04, shrink=0.0, ledoit_wolf=False, **_):
    mu, cov = annualised_inputs(rets, shrink, ledoit_wolf)
    return max_sharpe(mu, cov, rf, cap)


def min_variance_model(rets, cap=1.0, shrink=0.0, ledoit_wolf=False, **_):
    mu, cov = annualised_inputs(rets, shrink, ledoit_wolf)
    return min_variance(mu, cov, cap)


MODELS = {
    "Max Sharpe": max_sharpe_model,
    "Min variance": min_variance_model,
    "Risk parity": risk_parity,
    "HRP": hrp,
    "Min CVaR": min_cvar,
    "Equal weight (1/N)": equal_weight,
}

MODEL_BLURBS = {
    "Max Sharpe": "Best reward per unit of risk. Powerful, but leans hard on noisy return estimates.",
    "Min variance": "Lowest-risk mix. Ignores expected returns, so it's far more stable.",
    "Risk parity": "Every stock contributes the same amount of risk. Diversifies risk rather than money.",
    "HRP": "Clusters similar stocks, then spreads risk across clusters. No matrix inversion, so very stable.",
    "Min CVaR": "Minimises the average loss in the worst 5% of days. Focuses on crashes, not everyday wiggles.",
    "Equal weight (1/N)": "The same amount in everything. Hard to beat once estimation error is counted.",
}


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of total portfolio variance contributed by each holding (sums to 1)."""
    wv = w.values
    mrc = cov.values @ wv
    return pd.Series(wv * mrc / (wv @ mrc), index=w.index)


__all__ = ["MODELS", "MODEL_BLURBS", "EXACT_CAP", "InfeasibleError", "risk_contributions"]
