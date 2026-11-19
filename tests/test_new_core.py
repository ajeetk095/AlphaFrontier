import numpy as np
import pandas as pd
import pytest

from core import data as D
from core.backtest import BTConfig, backtest_table, rebalance_positions, run_backtest, sharpe_diff_test
from core.blacklitterman import implied_returns, posterior
from core.models import MODELS, hrp, min_cvar, risk_contributions, risk_parity
from core.optimize import annualised_inputs
from core.risk import drawdown_series, portfolio_beta, replay_event, var_cvar_historical, var_cvar_parametric
from core.robustness import bootstrap_weights, stability_summary

RF = 0.04


@pytest.fixture(scope="module")
def rets():
    prices, bench, ff = D.make_demo_data(D.DEFAULT_TICKERS, "2016-01-04", "2025-09-30")
    return prices.pct_change().dropna(), prices, bench


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("cap", [1.0, 0.2])
def test_models_valid_weights(rets, name, cap):
    r, _, _ = rets
    w = MODELS[name](r.iloc[-504:], cap=cap, rf=RF)
    assert w.sum() == pytest.approx(1, abs=1e-6)
    assert (w >= -1e-9).all() and (w <= cap + 1e-6).all()


def test_risk_parity_equalises_risk(rets):
    r, _, _ = rets
    w = risk_parity(r.iloc[-504:])
    _, cov = annualised_inputs(r.iloc[-504:])
    rc = risk_contributions(w, cov)
    assert rc.std() / rc.mean() < 0.02


def test_min_cvar_beats_equal_weight_on_its_own_metric(rets):
    r, _, _ = rets
    win = r.iloc[-504:]
    w = min_cvar(win)
    ew = pd.Series(1 / 10, index=win.columns)
    assert var_cvar_historical((win * w).sum(axis=1))[1] <= var_cvar_historical((win * ew).sum(axis=1))[1] + 1e-9


def test_hrp_runs_and_diversifies(rets):
    r, _, _ = rets
    w = hrp(r.iloc[-504:])
    assert (w > 0).all()


def test_backtest_no_lookahead(rets):
    """Changing a future return must not change earlier strategy returns."""
    r, _, _ = rets
    r = r.iloc[-900:]
    cfg = BTConfig(lookback=300, cap=0.3, cost_bps=0)
    a = run_backtest(r, ["Min variance"], cfg)["Min variance"]["returns"]
    r2 = r.copy()
    r2.iloc[-20:] *= 3.0
    b = run_backtest(r2, ["Min variance"], cfg)["Min variance"]["returns"]
    cut = a.index[-21]
    assert np.allclose(a.loc[:cut], b.loc[:cut])


def test_costs_reduce_returns(rets):
    r, _, _ = rets
    r = r.iloc[-800:]
    lo = run_backtest(r, ["Equal weight (1/N)"], BTConfig(lookback=300, cost_bps=0))["Equal weight (1/N)"]["returns"]
    hi = run_backtest(r, ["Equal weight (1/N)"], BTConfig(lookback=300, cost_bps=50))["Equal weight (1/N)"]["returns"]
    assert (1 + hi).prod() < (1 + lo).prod()


def test_equal_weight_zero_cost_matches_manual(rets):
    r, _, _ = rets
    r = r.iloc[-600:]
    cfg = BTConfig(lookback=250, rebalance="M", cost_bps=0)
    res = run_backtest(r, ["Equal weight (1/N)"], cfg)["Equal weight (1/N)"]["returns"]
    # monthly-rebalanced equal weight, computed independently
    pos = rebalance_positions(r.index, 250, "M")
    exp, w = [], None
    for i in range(pos[0], len(r)):
        if i in pos:
            w = np.full(10, 0.1)
        g = float(w @ r.values[i])
        exp.append(g)
        w = w * (1 + r.values[i]) / (1 + g)
    assert np.allclose(res.values, exp)


def test_backtest_table_and_significance(rets):
    r, prices, bench = rets
    r = r.iloc[-700:]
    res = run_backtest(r, ["Min variance", "Equal weight (1/N)"], BTConfig(lookback=250, cap=0.3))
    tab = backtest_table(res, RF, 10, bench)
    assert {"CAGR", "Sharpe", "Turnover / yr"} <= set(tab.columns) and len(tab) == 3
    t = sharpe_diff_test(res["Min variance"]["returns"], res["Equal weight (1/N)"]["returns"], RF, n_boot=300)
    assert t["lo"] <= t["hi"]


def test_black_litterman_no_views_recovers_prior(rets):
    r, _, _ = rets
    _, cov = annualised_inputs(r.iloc[-504:])
    w_prior = pd.Series(np.random.default_rng(1).dirichlet(np.ones(10)), index=cov.index)
    pi = implied_returns(cov, w_prior, 2.5)
    mu_post, _ = posterior(cov, pi, {})
    # reverse optimisation: unconstrained utility maximiser gives the prior back
    w = np.linalg.solve(2.5 * cov.values, mu_post.values)
    assert np.allclose(w / w.sum(), w_prior.values, atol=1e-8)


def test_black_litterman_view_moves_return_toward_view(rets):
    r, _, _ = rets
    _, cov = annualised_inputs(r.iloc[-504:])
    pi = implied_returns(cov, pd.Series(0.1, index=cov.index), 2.5)
    view = float(pi["AAPL"]) + 0.10
    lo, _ = posterior(cov, pi, {"AAPL": (view, 0.2)})
    hi, _ = posterior(cov, pi, {"AAPL": (view, 0.9)})
    assert pi["AAPL"] < lo["AAPL"] < hi["AAPL"] <= view + 1e-9


def test_var_ordering_and_drawdown(rets):
    r, _, _ = rets
    p = r.mean(axis=1)
    v95, c95 = var_cvar_historical(p, 0.95)
    v99, c99 = var_cvar_historical(p, 0.99)
    assert 0 < v95 <= v99 and c95 >= v95 and c99 >= v99
    assert var_cvar_parametric(p, 0.99)[0] > var_cvar_parametric(p, 0.95)[0]
    assert drawdown_series(p).max() <= 0


def test_replay_and_beta(rets):
    r, prices, bench = rets
    w = pd.Series(0.1, index=prices.columns)
    out = replay_event(prices, w, "2020-02-19", "2020-03-23", bench)
    assert out and out["coverage"] == pytest.approx(1.0)
    assert portfolio_beta(r, w, bench.pct_change().dropna()) > 0


def test_bootstrap_stability_shrinkage_helps(rets):
    r, _, _ = rets
    win = r.iloc[-504:]
    raw = stability_summary(bootstrap_weights(win, RF, 1.0, n_boot=25, seed=1))
    shr = stability_summary(bootstrap_weights(win, RF, 1.0, n_boot=25, shrink=0.8, ledoit_wolf=True, seed=1))
    assert shr["avg_weight_swing"] < raw["avg_weight_swing"]
