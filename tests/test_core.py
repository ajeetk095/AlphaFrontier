import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from core.data import DEFAULT_TICKERS, make_demo_data
from core.factors import run_ff5
from core.optimize import InfeasibleError, annualised_inputs, max_sharpe, min_variance, portfolio_stats
from core.performance import portfolio_returns, scenario_paths, summarise
from core.simulation import convergence_study, project_to_cap

RF = 0.045


@pytest.fixture(scope="module")
def market():
    prices, bench, ff = make_demo_data(DEFAULT_TICKERS, "2023-07-03", "2025-09-30")
    rets = prices.pct_change().dropna()
    return rets.loc[:"2025-06-30"], rets.loc["2025-07-01":], ff


def _slsqp(mu, cov, cap):
    n = len(mu)
    r = minimize(lambda w: -(mu.values @ w - RF) / np.sqrt(w @ cov.values @ w), np.ones(n) / n,
                 bounds=[(0, cap)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP")
    return -r.fun


@pytest.mark.parametrize("cap", [1.0, 0.3, 0.15])
def test_max_sharpe_matches_independent_solver(market, cap):
    tr, _, _ = market
    mu, cov = annualised_inputs(tr)
    w = max_sharpe(mu, cov, RF, cap)
    assert w.sum() == pytest.approx(1) and (w >= 0).all() and (w <= cap + 1e-9).all()
    assert portfolio_stats(w, mu, cov, RF).sharpe == pytest.approx(_slsqp(mu, cov, cap), abs=1e-4)


def test_cap_cannot_improve_sharpe(market):
    tr, _, _ = market
    mu, cov = annualised_inputs(tr)
    free = portfolio_stats(max_sharpe(mu, cov, RF), mu, cov, RF).sharpe
    capped = portfolio_stats(max_sharpe(mu, cov, RF, 0.15), mu, cov, RF).sharpe
    assert capped <= free + 1e-9


def test_infeasible_cap(market):
    tr, _, _ = market
    mu, cov = annualised_inputs(tr)
    with pytest.raises(InfeasibleError):
        min_variance(mu, cov, 0.05)


def test_projection_respects_cap():
    rng = np.random.default_rng(1)
    W = project_to_cap(rng.dirichlet(np.ones(5) * 0.3, 5000), 0.3)
    assert np.allclose(W.sum(1), 1) and W.max() <= 0.3 + 1e-9 and W.min() >= -1e-12


def test_simulation_error_shrinks(market):
    tr, _, _ = market
    sub = ["TSLA", "WMT", "BAC", "AAPL", "XOM"]
    mu, cov = annualised_inputs(tr[sub])
    w = max_sharpe(mu, cov, RF)
    s = portfolio_stats(w, mu, cov, RF).sharpe
    df = convergence_study(mu.values, cov.values, RF, w.values, s, 1.0, [1_000, 50_000], reps=4)
    assert df["Sharpe gap (median)"].iloc[1] <= df["Sharpe gap (median)"].iloc[0] + 1e-12


def test_buy_and_hold_total_return(market):
    _, te, _ = market
    w = pd.Series(1 / 10, index=te.columns)
    r = portfolio_returns(te, w, "none")
    expected = ((1 + te).prod() * w).sum() - 1
    assert summarise(r, RF)["Total return"] == pytest.approx(expected)


def test_factor_regression_recovers_market_beta(market):
    tr, _, ff = market
    mu, cov = annualised_inputs(tr)
    p = portfolio_returns(tr, max_sharpe(mu, cov, RF, 0.15), "daily")
    res = run_ff5(p, ff)
    assert 0.3 < res["ols"].loc["MKT_RF", "coef"] < 1.8
    assert res["n_train"] + res["n_test"] == res["n_obs"]
    assert res["train_end"] < res["test_start"]


def test_scenarios_shape(market):
    tr, _, _ = market
    r = portfolio_returns(tr, pd.Series(0.1, index=tr.columns), "daily")
    sc = scenario_paths(r, n_paths=500)
    assert sc["p05"] <= sc["p50"] <= sc["p95"]


def test_max_sharpe_falls_back_when_cap_blocks_every_beat_of_cash():
    """Only 2 of 10 assets beat cash and a 15% cap forces money into losers: must not raise."""
    names = [f"S{i}" for i in range(10)]
    mu = pd.Series([0.10, 0.09] + [-0.05] * 8, index=names)
    cov = pd.DataFrame(np.eye(10) * 0.04, index=names, columns=names)
    w = max_sharpe(mu, cov, 0.04, cap=0.15)
    assert w.sum() == pytest.approx(1) and (w <= 0.15 + 1e-9).all()
    w_mv = min_variance(mu, cov, 0.15)
    assert np.allclose(w.values, w_mv.values, atol=1e-6)
