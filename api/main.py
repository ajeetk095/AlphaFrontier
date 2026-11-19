"""Portfolio Lab REST API (FastAPI). Run: uvicorn api.main:app --reload"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import data as D
from core.models import MODELS
from core.optimize import InfeasibleError, annualised_inputs, max_sharpe, min_variance, portfolio_stats

app = FastAPI(title="Portfolio Lab API", version="1.0.0",
              description="Educational portfolio-construction endpoints. Not investment advice.")


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=3, max_length=30, examples=[["AAPL", "XOM", "WMT", "BAC"]])
    years: float = Field(2.0, ge=0.5, le=10)
    model: Literal["Max Sharpe", "Min variance", "Risk parity", "HRP", "Min CVaR", "Equal weight (1/N)"] = "Max Sharpe"
    cap: float = Field(1.0, ge=0.02, le=1.0)
    shrink: float = Field(0.0, ge=0, le=1)
    ledoit_wolf: bool = False
    rf: float | None = Field(None, description="Annual risk-free rate; defaults to the live 13-week T-bill average.")


class MomentsRequest(BaseModel):
    names: list[str]
    mu: list[float]
    cov: list[list[float]]
    rf: float = 0.04
    cap: float = Field(1.0, ge=0.02, le=1.0)
    objective: Literal["max_sharpe", "min_variance"] = "max_sharpe"


def _stats_dict(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float) -> dict:
    s = portfolio_stats(w, mu, cov, rf)
    return {"expected_return": s.ret, "volatility": s.vol, "sharpe": s.sharpe, "effective_n": s.eff_n, "holdings": s.n_hold}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": list(MODELS)}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    tickers = [t.upper() for t in req.tickers]
    end = date.today()
    start = end - timedelta(days=int(365 * req.years))
    try:
        prices, source, as_of = D.get_prices(tickers, start, end)
    except D.DataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    rets = prices.pct_change().dropna()
    if len(rets) < 60:
        raise HTTPException(status_code=422, detail="Not enough overlapping history for these tickers.")
    rf = req.rf if req.rf is not None else D.fetch_rf_annual(start, end)[0]
    try:
        w = MODELS[req.model](rets, cap=req.cap, rf=rf, shrink=req.shrink, ledoit_wolf=req.ledoit_wolf)
    except InfeasibleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    mu, cov = annualised_inputs(rets, req.shrink, req.ledoit_wolf)
    return {"weights": w.round(6).to_dict(), "stats": _stats_dict(w, mu, cov, rf), "data_source": source,
            "as_of": str(pd.Timestamp(as_of).date()), "observations": len(rets), "risk_free": rf}


@app.post("/optimize/from-moments")
def optimize_from_moments(req: MomentsRequest):
    """Pure maths: supply your own expected returns and covariance (annualised)."""
    n = len(req.names)
    if len(req.mu) != n or len(req.cov) != n or any(len(r) != n for r in req.cov):
        raise HTTPException(status_code=422, detail="Dimensions of names, mu and cov must agree.")
    mu = pd.Series(req.mu, index=req.names)
    cov = pd.DataFrame(np.array(req.cov), index=req.names, columns=req.names)
    if not np.allclose(cov.values, cov.values.T, atol=1e-10) or np.linalg.eigvalsh(cov.values).min() < -1e-10:
        raise HTTPException(status_code=422, detail="cov must be symmetric positive semi-definite.")
    try:
        w = max_sharpe(mu, cov, req.rf, req.cap) if req.objective == "max_sharpe" else min_variance(mu, cov, req.cap)
    except InfeasibleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"weights": w.round(6).to_dict(), "stats": _stats_dict(w, mu, cov, req.rf)}
