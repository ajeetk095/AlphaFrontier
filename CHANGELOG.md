# Changelog

## 2.0.0
- Walk-forward back-test (rolling estimation, scheduled rebalancing, transaction costs, 1/N / min-variance / risk-parity / HRP / min-CVaR benchmarks, paired block-bootstrap Sharpe test).
- Black-Litterman page (market-implied prior, absolute views with confidence).
- Risk & stress page (VaR/CVaR, drawdowns, risk contributions, historical replays, beta shock).
- Model comparison and estimation-noise lab.
- Learn & Quiz track, risk-profile quiz, glossary tooltips, quizzes on every page.
- Data robustness: retry with backoff, daily-refreshed snapshot fallback, data-provenance badge.
- India mode: Nifty 50 market model for NSE-listed portfolios.
- PDF summary export, shareable URL parameters, currency/amount personalisation.
- FastAPI service, Dockerfiles, pre-commit, ruff, coverage in CI.
- Fix: max-Sharpe now falls back to min-variance when a position cap makes beating cash impossible (previously infeasible).

## 1.0.0
- Optimizer, factor analysis, simulation vs optimisation, report explanation, developer page.
