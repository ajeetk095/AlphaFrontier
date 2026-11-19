import streamlit as st

import helpers as H

H.page_setup("Report & Methodology", "📘")
st.title("📘 The research report, explained simply + methodology")
st.caption("MScFE 652 Portfolio Management · Group Work Project 1 · WorldQuant University · Group 13321 "
           "(Ajeet Kumar, Rajkumar S, Yijing Hong)")

st.markdown(
    """
## The big picture
A portfolio manager has **10 large US stocks** (Tesla, Walmart, Bank of America, Goldman Sachs, Eli Lilly, Merck, Alphabet, Meta, Apple, ExxonMobil)
and asks three questions. The report answers each with data from **1 Jul 2023 – 30 Jun 2025**, then tests the result on **1 Jul – 30 Sep 2025**.

### Part 1 · How should we split the money?
- **Step 1, free choice.** Using Markowitz's mean-variance method, find the weights with the best reward per unit of risk, with no short-selling.
  The result was *concentrated*: roughly 38% Apple and 21% ExxonMobil.
- **Step 2, 15% cap.** The client doesn't want any stock above 15%. The method re-spreads money across more stocks. Returns and risk both rise slightly
  versus Step 1 in the report's figures, and the portfolio is more evenly spread.
- **Step 3, risk/return curves.** Plotting the best achievable portfolios with and without the cap shows the **capped curve is never better**. A rule can only remove choices.
  The price of the rule is small compared with the protection it buys against one stock going wrong.
- **Step 4, test on new data.** Both portfolios are tried over Jul–Sep 2025 without re-tuning, against the S&P 500.

### Part 2 · What is the portfolio actually exposed to?
The client's chosen portfolio is the **15%-capped** one. Using the **Fama-French five-factor** model, the report regresses its returns on:
the market, company **size** (SMB), **value** vs growth (HML), **profitability** (RMW) and **investment style** (CMA).
- Main finding: **market exposure is the biggest driver** (beta ≈ 0.67, a calmer-than-market portfolio) and there's a significant **value tilt** (HML ≈ 0.34).
- Size, profitability and investment exposures were not statistically distinguishable from zero.
- Both a standard regression (OLS) and an outlier-resistant one (Huber) are used; data is split in time (earlier for training, later for testing).

### Part 3 · Can random simulation replace the maths?
Five stocks (Tesla, Walmart, Bank of America, Apple, ExxonMobil) are optimised exactly, then by **randomly generating portfolios** and keeping the best.
- Without a cap, random search gets *close* after tens of thousands of tries.
- With a **30% cap**, many random portfolios break the rule and must be discarded, so convergence is slower and weights stay a bit off.
- **Conclusion:** use exact optimisation to decide; use simulation to explore "what if" scenarios.

## What to take away
1. Concentration is the optimiser's default habit; **caps are cheap insurance**.
2. Most of a diversified US stock portfolio's behaviour comes from the **market itself**; style tilts are secondary.
3. **Exact beats random** for weights, **random shines for scenarios**.
4. Everything is estimated from a short, unusually strong market period, so **treat historical returns as description, not prediction**.
"""
)

with st.expander("🔧 Methodology notes: how this app extends the report"):
    st.markdown(
        """
The app is a live, re-runnable version of the report's workflow. Where practical it tightens the method:
- **Objective:** weights maximise the **Sharpe ratio** using an exact convex reformulation (and are cross-checked against an independent numerical solver).
- **Out-of-sample metrics** are computed **from the test-period returns themselves** (buy-and-hold, no rebalancing), including a geometric annualised return, Sharpe and Sortino.
- **Excess returns** in the factor regression use the **daily risk-free rate from the Fama-French file**, not a single constant.
- **Standard errors** are Newey-West (HAC); the robust model is Huber M-estimation; train/test split is chronological.
- **Simulation** uses repeated trials at each budget (so one lucky run can't mislead) and measures error in *weights*, not just Sharpe.
- **Live mode** re-estimates on the latest two years, so numbers differ from the report's fixed-window figures: that's expected.
        """
    )
st.markdown("## Methodology & limitations")
st.markdown(
    """
**What the app does**
- *Optimisation*: exact convex programs (cvxpy). Max-Sharpe via the Cornuejols-Tutuncu transformation; min-variance and min-CVaR as quadratic/linear programs; cross-checked against an independent solver in the test-suite.
- *Walk-forward*: weights at date *t* use only returns before *t*; proportional trading costs; buy-and-hold drift between rebalances; paired block-bootstrap test of Sharpe differences.
- *Black-Litterman*: reverse-optimised prior, absolute views, He-Litterman posterior.
- *Risk*: historical and normal VaR/CVaR, drawdowns, risk contributions, historical replays and a beta shock.
- *Factors*: OLS with Newey-West errors and Huber regression on the Fama-French five factors, chronological train/test split. Indian portfolios use a Nifty 50 market model instead.

**What it cannot do**
- Predict markets. Scenarios and back-tests describe *possible* outcomes from historical patterns.
- Remove **hindsight bias**: the default stocks are today's well-known winners.
- Capture taxes, bid-ask spreads, market impact, or the behaviour of real investors in a crash.
- Replace a licensed adviser. This is an educational tool.

**Data**: Yahoo Finance (unofficial, occasionally gappy), Kenneth French Data Library (factors published with a delay). A daily-refreshed snapshot is used if Yahoo is unreachable, and every page shows which source it is using.
"""
)
H.footer()
