# Portfolio Lab: technical documentation

## 0. What's new in 2.0
Walk-forward back-test, Black-Litterman, risk & stress, model comparison, estimation-noise lab, learning track, snapshot fallback, India mode, PDF export, REST API, Docker. See `CHANGELOG.md`.

## 1. Architecture
```
app.py                      Home: market snapshot, headline portfolio, guidance
helpers.py                  cached data access, data windows, shared sidebar, layout
core/
  data.py                   Yahoo Finance prices, French factors, synthetic demo data
  optimize.py               max-Sharpe / min-variance (cvxpy), efficient frontier, stats
  performance.py            buy-and-hold returns, metrics, bootstrap scenarios
  factors.py                FF5 regression: OLS (HAC) + Huber, chronological split
  simulation.py             random-weight search, cap handling, convergence study
  models.py                 1/N, risk parity, HRP, min-CVaR (+ max-Sharpe/min-variance wrappers)
  backtest.py               walk-forward engine, costs, turnover, Sharpe-difference bootstrap
  blacklitterman.py         implied returns, posterior, market caps
  risk.py                   VaR/CVaR, drawdown, replay, beta
  robustness.py             bootstrap weight stability
  insights.py / glossary.py plain-language text, tooltips
  pdf_report.py             PDF export
pages/                      Learn · Optimizer · Back-test · Black-Litterman · Risk · Models · Factors · Simulation · Report · About
api/main.py                 FastAPI service
scripts/refresh_snapshot.py daily data snapshot (GitHub Action)
tests/                      core, platform (fallback/API/PDF) and headless page tests
```
`core/` has no Streamlit dependency, so it can be reused in notebooks or batch jobs.

## 2. Data
| Item | Source | Notes |
|---|---|---|
| Prices | Yahoo Finance (`yfinance`), `auto_adjust=True` | Daily split/dividend-adjusted closes; end date made inclusive |
| Risk-free | `^IRX` 13-week T-bill, mean over estimation window | Fallback 4% if unavailable (flagged in UI) |
| Benchmark / mood | `^GSPC`, `^VIX` | |
| Factors | Kenneth French Data Library, `F-F_Research_Data_5_Factors_2x3_daily` | Percent → decimal; **published with ~1-2 month lag** |
"Real time" here means *latest daily close*. Nothing is tick-level.

## 3. Methods
**Inputs.** μ = 252·mean(daily returns); Σ = 252·sample covariance. Optional: shrink μ toward the cross-asset mean (λ), and Ledoit-Wolf covariance.

**Max-Sharpe (tangency).** Maximise (wᵀμ − r_f)/√(wᵀΣw) s.t. 1ᵀw = 1, 0 ≤ w ≤ cap. Non-convex as written, so we solve the exact convex equivalent: minimise yᵀΣy s.t. (μ − r_f)ᵀy = 1, 1ᵀy = k, 0 ≤ y ≤ cap·k, k ≥ 0, then w = y/k. Tests compare the result to SciPy SLSQP (agreement ≈ 1e-4 in Sharpe or better).

**Min-variance & frontier.** Quadratic programs; the frontier solves min wᵀΣw s.t. wᵀμ ≥ target across targets from the min-variance return up to the maximum feasible return (efficient part only). A cap only removes feasible portfolios, so the capped frontier lies on or inside the uncapped one.

**Out-of-sample test.** Weights are estimated on the training window and held (buy-and-hold) through the test window. Reported: total return, geometric annualised return, annualised vol, Sharpe, Sortino, max drawdown. Short windows (≈63 days) make annualised numbers noisy.

**Scenarios.** Block bootstrap (5-day blocks, 5,000 paths, 252 days) of the portfolio's own daily returns, with a *drift haircut* (default 50%) because optimised portfolios are selected for looking good in-sample. Outputs 5th/50th/95th percentile, P(loss), expected shortfall. **Scenarios, not forecasts.**

**Factor model.** r_p − r_f = α + β₁(Mkt−RF) + β₂SMB + β₃HML + β₄RMW + β₅CMA + ε, with r_f the daily RF from the factor file. OLS with Newey-West HAC (5 lags); robust fit by Huber M-estimation. Chronological split (default 70/30); out-of-sample R² = 1 − SSE/SST with SST taken around the *training mean*.

**Simulation vs optimisation.** Candidates: Dirichlet draws mixing α = 1 and α = 0.3 (the latter reaches simplex edges where zero-weight optima live). Cap handling: *rejection* (discard) or *projection* (clip & redistribute). For each budget N the search is repeated `reps` times; we report median/worst Sharpe gap and weight error vs the optimiser. "Simulations needed" = smallest N with median worst-asset weight error ≤ tolerance. For 5 assets, a uniform draw satisfies a 30% cap with probability only ≈ 5.5% (inclusion-exclusion: 1 − 5·0.7⁴ + 10·0.4⁴ − 10·0.1⁴), so rejection wastes ~94% of draws.

## 3b. Methods added in 2.0
**Walk-forward back-test.** At each rebalance day *t* (first trading day of each month/quarter) the model sees returns in `[t-L, t)` only, produces target weights, trades to them paying `turnover × bps`, then drifts buy-and-hold until the next rebalance. A unit test changes future returns and proves earlier results don't move. Turnover = Σ|w_target − w_drifted|; the first trade counts as 100%.
**Significance.** Paired block bootstrap (21-day blocks, 2,000 resamples) of Sharpe(strategy) − Sharpe(1/N); the 90% interval excluding 0 is reported as "distinguishable from luck". It tests noise only, not hindsight or survivorship bias.
**Black-Litterman.** π = δΣw_prior; absolute views (P = rows of I, Q = view − r_f); Ω_ii = (1/c_i − 1)·τ·(PΣPᵀ)_ii; μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹π + PᵀΩ⁻¹Q]; Σ_BL = Σ + M. With no views and a loose cap, max-Sharpe on π returns the prior (tested).
**Risk parity** minimises ½yᵀΣy − (1/n)Σlog yᵢ (convex), giving equal risk contributions (tested). **HRP** follows López de Prado (2016): single-linkage clustering on √(½(1−ρ)), recursive bisection by inverse-variance cluster risk. **Min-CVaR** is the Rockafellar-Uryasev LP. Caps are exact for max-Sharpe/min-variance/min-CVaR and heuristic (clip & redistribute) for 1/N, risk parity, HRP.
**Risk.** Historical and normal VaR/CVaR; replays of five S&P 500 episodes (holdings that didn't yet trade are dropped and the rest re-scaled, with coverage shown); beta shock = β·shock.
**India mode.** If ≥50% of holdings end in `.NS`/`.BO`, the factor page runs a single-index regression vs the Nifty 50 (OLS-HAC, Huber, chronological split). India-specific size/value/quality factors are **not** included.
**Data robustness.** Yahoo calls retry 3× (1.5 s, 3 s). On failure the app reads `data/snapshot_prices.csv.gz` if it holds every requested ticker and labels the page "Snapshot".
**Bug fixed during development.** With a position cap, max-Sharpe could be infeasible when no allowed mix beats cash; it now falls back to min-variance (regression-tested).

## 4. Metric glossary
Sharpe = (return − r_f)/volatility · Sortino penalises only downside · HHI = Σw² over holdings, effective N = 1/HHI · Max drawdown = worst peak-to-trough fall · Alpha = return unexplained by factors · R² = share of variance explained.

## 5. Limitations (read these)
- Mean-variance is highly sensitive to estimated returns; 2 years of data is short. Use the shrinkage controls.
- Constant-weight, no transaction costs, taxes or slippage; long-only.
- Factor regressions describe *past* exposure. HML/SMB etc. are US-market factors, so for Indian stocks (`.NS`) results are only indicative.
- Yahoo data is unofficial and can have gaps/revisions; rate limits apply.
- Educational use only: not investment advice.

## 6. Extending
Add Black-Litterman (views → posterior μ), risk-parity or CVaR objectives in `core/optimize.py`; add regional factor sets in `core/data.py`; add rebalancing back-tests in `core/performance.py`.

## 7. Testing
`PYTHONPATH=. pytest -q` runs solver cross-checks (vs SLSQP), cap/feasibility checks, buy-and-hold arithmetic, regression split integrity, simulation convergence, and renders every page headlessly in demo mode.
The live-data code paths (Yahoo, French library) need internet and were not exercised in the build sandbox; run the app once locally to confirm.
