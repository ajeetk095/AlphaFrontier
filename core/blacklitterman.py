"""Black-Litterman: blend market-implied returns with the investor's own views."""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_returns(cov: pd.DataFrame, w_prior: pd.Series, delta: float) -> pd.Series:
    """Reverse optimisation: the excess returns that make the prior weights optimal (pi = delta * Sigma * w)."""
    return pd.Series(delta * cov.values @ w_prior.reindex(cov.index).values, index=cov.index)


def posterior(cov: pd.DataFrame, pi: pd.Series, views: dict[str, tuple[float, float]], tau: float = 0.05):
    """Posterior excess returns and covariance.

    views: {asset: (expected_excess_return, confidence in (0,1))} as absolute views.
    Omega_ii = (1/confidence - 1) * tau * (P Sigma P')_ii, so 50% confidence puts the view on equal
    footing with the prior; higher confidence trusts the view more.
    """
    if not views:
        return pi.copy(), cov.copy()
    assets = list(cov.index)
    P = np.zeros((len(views), len(assets)))
    Q = np.zeros(len(views))
    conf = np.zeros(len(views))
    for k, (a, (q, c)) in enumerate(views.items()):
        P[k, assets.index(a)] = 1.0
        Q[k], conf[k] = q, float(np.clip(c, 0.01, 0.99))
    S = cov.values
    tS = tau * S
    omega = np.diag((1 / conf - 1) * np.diag(P @ tS @ P.T))
    inv_tS, inv_om = np.linalg.inv(tS), np.linalg.inv(omega)
    M = np.linalg.inv(inv_tS + P.T @ inv_om @ P)
    mu_post = M @ (inv_tS @ pi.values + P.T @ inv_om @ Q)
    return pd.Series(mu_post, index=cov.index), pd.DataFrame(S + M, index=cov.index, columns=cov.columns)


def fetch_market_caps(tickers: list[str]) -> pd.Series | None:
    """Live market caps from Yahoo; None if unavailable (caller falls back to equal weights)."""
    try:
        import yfinance as yf

        caps = {t: float(yf.Ticker(t).fast_info["market_cap"]) for t in tickers}
        s = pd.Series(caps)
        return s if s.notna().all() and (s > 0).all() else None
    except Exception:
        return None
