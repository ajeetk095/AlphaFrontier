"""Fama-French five-factor regressions: OLS (HAC) + robust (Huber) with a chronological train/test split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import FACTORS


def compound(df: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    return (1 + df).resample(rule).apply(lambda s: s.prod() - 1).dropna()


def factor_correlation(ff: pd.DataFrame) -> pd.DataFrame:
    return ff[FACTORS].corr()


def factor_summary(ff: pd.DataFrame) -> pd.DataFrame:
    f = ff[FACTORS]
    return pd.DataFrame(
        {"Mean (%/day)": f.mean() * 100, "Std dev (%/day)": f.std() * 100,
         "Skewness": f.skew(), "Excess kurtosis": f.kurt()}
    )


def _table(res, names) -> pd.DataFrame:
    return pd.DataFrame(
        {"coef": res.params, "std_err": res.bse, "t": res.tvalues, "p": res.pvalues}
    ).loc[names]


def run_ff5(port_ret: pd.Series, ff: pd.DataFrame, freq: str = "D", train_frac: float = 0.7,
            hac_lags: int = 5) -> dict:
    """Regress portfolio excess returns on the five factors.

    OLS uses Newey-West (HAC) standard errors; the robust model is Huber M-estimation.
    The split is chronological (first `train_frac` of observations train, the rest test)
    because returns are time-ordered and a random split would leak the future.
    """
    df = pd.concat([port_ret.rename("rp"), ff], axis=1, join="inner").dropna()
    if freq == "W":
        df = compound(df)
    if len(df) < 40:
        raise ValueError(f"Only {len(df)} overlapping observations; need at least 40.")
    y = df["rp"] - df["RF"]
    X = sm.add_constant(df[FACTORS])
    names = ["const"] + FACTORS

    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    rob = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()

    n_tr = int(train_frac * len(df))
    Xtr, Xte, ytr, yte = X.iloc[:n_tr], X.iloc[n_tr:], y.iloc[:n_tr], y.iloc[n_tr:]
    ols_tr = sm.OLS(ytr, Xtr).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    rob_tr = sm.RLM(ytr, Xtr, M=sm.robust.norms.HuberT()).fit()

    def oos(model):
        pred = model.predict(Xte)
        sse = float(((yte - pred) ** 2).sum())
        sst = float(((yte - ytr.mean()) ** 2).sum())  # benchmark: "predict the training mean"
        return {"R2_oos": 1 - sse / sst, "RMSE": float(np.sqrt(sse / len(yte)))}

    ann = 252 if freq == "D" else 52
    return {
        "n_obs": len(df), "n_train": n_tr, "n_test": len(df) - n_tr, "freq": freq,
        "start": df.index[0], "end": df.index[-1],
        "train_end": df.index[n_tr - 1], "test_start": df.index[n_tr],
        "ols": _table(ols, names), "robust": _table(rob, names),
        "ols_train": _table(ols_tr, names), "robust_train": _table(rob_tr, names),
        "r2": float(ols.rsquared), "adj_r2": float(ols.rsquared_adj),
        "oos_ols": oos(ols_tr), "oos_robust": oos(rob_tr),
        "alpha_annual": float(ols.params["const"] * ann),
        "resid_vol_annual": float(ols.resid.std() * np.sqrt(ann)),
    }


def is_india_portfolio(tickers) -> bool:
    """True when most holdings are NSE/BSE-listed (US Fama-French factors are then only indicative)."""
    t = list(tickers)
    return bool(t) and sum(x.endswith((".NS", ".BO")) for x in t) / len(t) >= 0.5


def run_market_model(port_ret: pd.Series, mkt_ret: pd.Series, rf_annual: float, train_frac: float = 0.7,
                     hac_lags: int = 5) -> dict:
    """Single-index (CAPM-style) regression vs a local market index, with OLS(HAC) + Huber and a chronological split."""
    df = pd.concat([port_ret.rename("rp"), mkt_ret.rename("mkt")], axis=1, join="inner").dropna()
    if len(df) < 40:
        raise ValueError(f"Only {len(df)} overlapping observations; need at least 40.")
    rf_d = rf_annual / 252
    y, x = df["rp"] - rf_d, df["mkt"] - rf_d
    X = sm.add_constant(x.rename("MKT_RF"))
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    rob = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    n_tr = int(train_frac * len(df))
    tr = sm.OLS(y.iloc[:n_tr], X.iloc[:n_tr]).fit()
    pred = tr.predict(X.iloc[n_tr:])
    sse = float(((y.iloc[n_tr:] - pred) ** 2).sum())
    sst = float(((y.iloc[n_tr:] - y.iloc[:n_tr].mean()) ** 2).sum())
    names = ["const", "MKT_RF"]
    return {"n_obs": len(df), "ols": _table(ols, names), "robust": _table(rob, names), "r2": float(ols.rsquared),
            "r2_oos": 1 - sse / sst, "alpha_annual": float(ols.params["const"] * 252),
            "start": df.index[0], "end": df.index[-1]}
