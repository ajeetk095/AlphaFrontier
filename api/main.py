"""Portfolio Lab REST API (FastAPI). Run: uvicorn api.main:app --reload"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat, model_validator

from core import data as D
from core.models import MODELS
from core.optimize import InfeasibleError, annualised_inputs, max_sharpe, min_variance, portfolio_stats

MAX_ASSETS = 30

app = FastAPI(title="Portfolio Lab API", version="1.0.0",
              description="Educational portfolio-construction endpoints. Not investment advice.")


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=3, max_length=MAX_ASSETS, examples=[["AAPL", "XOM", "WMT", "BAC"]])
    years: FiniteFloat = Field(2.0, ge=0.5, le=10)
    model: Literal["Max Sharpe", "Min variance", "Risk parity", "HRP", "Min CVaR", "Equal weight (1/N)"] = "Max Sharpe"
    cap: FiniteFloat = Field(1.0, ge=0.02, le=1.0)
    shrink: FiniteFloat = Field(0.0, ge=0, le=1)
    ledoit_wolf: bool = False
    rf: FiniteFloat | None = Field(None, description="Annual risk-free rate; defaults to the live 13-week T-bill average.")

    @model_validator(mode="after")
    def tickers_are_unique(self):
        normalized = [ticker.strip().upper() for ticker in self.tickers]
        if any(not ticker for ticker in normalized):
            raise ValueError("Tickers must not be blank.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Tickers must be unique (case-insensitive).")
        self.tickers = normalized
        return self


class MomentsRequest(BaseModel):
    names: list[str] = Field(..., min_length=3, max_length=MAX_ASSETS)
    mu: list[FiniteFloat] = Field(..., min_length=3, max_length=MAX_ASSETS)
    cov: list[list[FiniteFloat]] = Field(..., min_length=3, max_length=MAX_ASSETS)
    rf: FiniteFloat = 0.04
    cap: FiniteFloat = Field(1.0, ge=0.02, le=1.0)
    objective: Literal["max_sharpe", "min_variance"] = "max_sharpe"

    @model_validator(mode="after")
    def validate_dimensions_and_names(self):
        n = len(self.names)
        if len(self.mu) != n or len(self.cov) != n or any(len(row) != n for row in self.cov):
            raise ValueError("Dimensions of names, mu and cov must agree; cov must be square.")
        if any(not name.strip() for name in self.names):
            raise ValueError("Asset names must not be blank.")
        if len(set(self.names)) != n:
            raise ValueError("Asset names must be unique.")
        return self


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
    end = date.today()
    start = end - timedelta(days=int(365 * req.years))
    try:
        prices, source, as_of = D.get_prices(req.tickers, start, end)
    except D.DataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
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
    mu = pd.Series(req.mu, index=req.names)
    cov_values = np.asarray(req.cov, dtype=float)
    if not np.allclose(cov_values, cov_values.T, atol=1e-10):
        raise HTTPException(status_code=422, detail="cov must be symmetric positive semi-definite.")
    try:
        if np.linalg.eigvalsh(cov_values).min() < -1e-10:
            raise HTTPException(status_code=422, detail="cov must be symmetric positive semi-definite.")
    except np.linalg.LinAlgError as exc:
        raise HTTPException(status_code=422, detail="cov must be a valid positive semi-definite matrix.") from exc
    cov = pd.DataFrame(cov_values, index=req.names, columns=req.names)
    try:
        w = max_sharpe(mu, cov, req.rf, req.cap) if req.objective == "max_sharpe" else min_variance(mu, cov, req.cap)
    except InfeasibleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"weights": w.round(6).to_dict(), "stats": _stats_dict(w, mu, cov, req.rf)}
