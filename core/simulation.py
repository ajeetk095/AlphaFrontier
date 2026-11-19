"""Monte Carlo (random-weight) search vs. exact optimisation, with and without a position cap."""
from __future__ import annotations

import numpy as np
import pandas as pd

CHUNK = 200_000


def project_to_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Clip weights at `cap` and redistribute the excess proportionally to unclipped assets."""
    w = w.copy()
    for _ in range(w.shape[1]):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = np.where(over, w - cap, 0.0).sum(axis=1, keepdims=True)
        w = np.where(over, cap, w)
        room = np.where(w < cap - 1e-12, w, 0.0)
        tot = room.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        w = w + excess * room / tot
    return w


def draw(n_assets: int, n: int, cap: float, rng, method: str = "rejection", alphas=(1.0, 0.3)):
    """Draw n candidate long-only, fully-invested portfolios.

    Candidates mix Dirichlet concentrations (alpha<1 reaches the edges of the simplex, where
    optima with zero weights live). 'rejection' discards draws above the cap (as in the report);
    'projection' clips and redistributes so every draw is valid.
    """
    parts = np.array_split(np.arange(n), len(alphas))
    W = np.vstack([rng.dirichlet(np.full(n_assets, a), size=len(p)) for a, p in zip(alphas, parts)])
    if cap >= 1.0:
        return W, n
    if method == "projection":
        return project_to_cap(W, cap), n
    keep = (W <= cap + 1e-12).all(axis=1)
    return W[keep], int(keep.sum())


def best_portfolio(mu, cov, rf, n_assets, n, cap, rng, method="rejection", alphas=(1.0, 0.3)):
    """Best-Sharpe portfolio among n random candidates (processed in chunks to save memory)."""
    best_s, best_w, valid = -np.inf, None, 0
    mu_v, cov_v = np.asarray(mu), np.asarray(cov)
    left = n
    while left > 0:
        m = min(CHUNK, left)
        W, v = draw(n_assets, m, cap, rng, method, alphas)
        left -= m
        valid += v
        if len(W) == 0:
            continue
        ret = W @ mu_v
        vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov_v, W))
        s = (ret - rf) / vol
        i = int(np.argmax(s))
        if s[i] > best_s:
            best_s, best_w = float(s[i]), W[i]
    return best_s, best_w, valid


def convergence_study(mu, cov, rf, w_opt, sharpe_opt, cap, sizes, reps=10, method="rejection",
                      seed=0, alphas=(1.0, 0.3)) -> pd.DataFrame:
    """For each simulation budget N, repeat the search `reps` times and compare with the optimiser."""
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    rows = []
    for N in sizes:
        gaps, l1, mx, valid = [], [], [], []
        for _ in range(reps):
            s, w, v = best_portfolio(mu, cov, rf, n_assets, N, cap, rng, method, alphas)
            if w is None:
                continue
            gaps.append(sharpe_opt - s)
            diff = np.abs(w - np.asarray(w_opt))
            l1.append(diff.sum())
            mx.append(diff.max())
            valid.append(v / N)
        rows.append({
            "Simulations": N,
            "Valid draws (%)": 100 * float(np.mean(valid)) if valid else np.nan,
            "Sharpe gap (median)": float(np.median(gaps)) if gaps else np.nan,
            "Sharpe gap (worst)": float(np.max(gaps)) if gaps else np.nan,
            "Max weight error, pp (median)": 100 * float(np.median(mx)) if mx else np.nan,
            "Total weight error, pp (median)": 100 * float(np.median(l1)) if l1 else np.nan,
        })
    return pd.DataFrame(rows)


def simulations_needed(df: pd.DataFrame, tol_pp: float = 1.0):
    """Smallest N whose median worst-asset weight error is below tol_pp percentage points."""
    ok = df[df["Max weight error, pp (median)"] <= tol_pp]
    return int(ok["Simulations"].iloc[0]) if len(ok) else None
